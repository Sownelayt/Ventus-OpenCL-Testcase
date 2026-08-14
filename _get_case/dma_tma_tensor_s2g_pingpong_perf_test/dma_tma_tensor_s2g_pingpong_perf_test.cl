/*
 * DMA/TMA S2G ping-pong double-buffer performance microbench.
 *
 * Manual path: global -> register -> shared, compute in shared, then
 * shared -> register -> global. It uses the same ping-pong buffer order as
 * the TMA path, but the next-tile load is ordinary synchronous code.
 *
 * S2G paths replace the final shared -> register -> global store with
 * lane0-issued CP_ASYNC_BULK_S2G or CP_ASYNC_TENSOR_S2G writeback. The manual
 * S2G paths commit each writeback as an async group and use wait_group to reuse
 * shared output buffers without draining all outstanding S2G writes.
 */

#include "ventus_tma_v2_opencl.h"

#define DESC_WORDS 32u
#define COORD_VECTOR_WORDS 32u
#define COORD_PHASE_WORD 36u
#define COORD_WORDS 40u
#ifndef TILE_ROWS
#define TILE_ROWS 16u
#endif
#ifndef TILE_COLS
#define TILE_COLS 16u
#endif
#define TILE_WORDS (TILE_ROWS * TILE_COLS)
#define MAX_STAGES 16u
#define WG_SIZE 32u

kernel void
setup_desc_kernel(__global uint *g2s_desc,
                  __global uint *s2g_desc,
                  __global const uint *input,
                  __global uint *output)
{
  if (get_global_id(0) == 0) {
    g2s_desc[2] = (uint)input;
    s2g_desc[2] = (uint)output;
  }
}

static uint
min_u32(uint a, uint b)
{
  return a < b ? a : b;
}

static void
load_tile_manual(__local uint *dst, volatile __global const uint *input,
                 uint stage, uint lid)
{
  uint base = stage * TILE_WORDS;
  for (uint idx = lid; idx < TILE_WORDS; idx += WG_SIZE) {
    dst[idx] = input[base + idx];
  }
}

static void
compute_tile(__local uint *src, __local uint *dst, uint lid)
{
  for (uint idx = lid; idx < TILE_WORDS; idx += WG_SIZE) {
    float x = as_float(src[idx]);
    float y = x * 1.0009765625f + 0.000244140625f;
    dst[idx] = as_uint(y);
  }
}

static void
set_stage_coords(__local uint *coords, uint stage, uint lid)
{
  for (uint i = lid; i < COORD_VECTOR_WORDS; i += WG_SIZE) {
    coords[i] = 0u;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0u) {
    coords[0] = 0u;
    coords[1] = stage * TILE_ROWS;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
}

static __local uint *
align_local_8(__local uint *base)
{
  return (__local uint *)(((uint)base + 7u) & ~7u);
}

static __local uint *
align_local_128(__local uint *base)
{
  return (__local uint *)(((uint)base + 127u) & ~127u);
}

#define ISSUE_TMA_G2S_BUF(BUF, DESC, COORDS) do {                           \
  VENTUS_TMA_LOAD_COORDS_V12((COORDS));                                      \
  if (lid == 0u) {                                                           \
    __local uint *issue_mbarrier =                                           \
      align_local_8((COORDS) + COORD_VECTOR_WORDS);                          \
    if ((COORDS)[1] == 0u) {                                                 \
      VENTUS_TMA_MBARRIER_INIT(issue_mbarrier, 1u);                          \
    }                                                                        \
    (COORDS)[COORD_PHASE_WORD] = ((COORDS)[1] / TILE_ROWS) & 1u;             \
    VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(issue_mbarrier, TILE_WORDS * 4u);   \
    VENTUS_TMA_TENSOR_G2S((BUF), (DESC));                                    \
  }                                                                          \
} while (0)

#define ISSUE_TENSOR_S2G_BUF(BUF, DESC, COORDS, STAGE, LID, BASE) do {      \
  barrier(CLK_LOCAL_MEM_FENCE);                                             \
  set_stage_coords((COORDS), (STAGE), (LID));                               \
  VENTUS_TMA_LOAD_COORDS_V12((COORDS));                                     \
  if ((LID) == 0u) {                                                        \
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();                                  \
    VENTUS_TMA_TENSOR_S2G((BUF), (DESC));                                   \
  }                                                                         \
  barrier(CLK_LOCAL_MEM_FENCE);                                             \
} while (0)

static void
store_tile_global(__local uint *src, volatile __global uint *output,
                  uint stage, uint lid)
{
  uint base = stage * TILE_WORDS;
  for (uint idx = lid; idx < TILE_WORDS; idx += WG_SIZE) {
    output[base + idx] = src[idx];
  }
}

#define TMA_WAIT_ALL() do {                                                   \
  if (lid == 0u) {                                                            \
    __local uint *wait_mbarrier =                                             \
      align_local_8(coords + COORD_VECTOR_WORDS);                             \
    uint wait_phase = coords[COORD_PHASE_WORD];                               \
    VENTUS_TMA_MBARRIER_WAIT(wait_mbarrier, wait_phase);                      \
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();                                    \
  }                                                                           \
  barrier(CLK_LOCAL_MEM_FENCE);                                               \
} while (0)

kernel void
manual_roundtrip_pipeline_kernel(__global const uint *input,
                                 __global uint *output,
                                 uint buffers,
                                 uint stages)
{
  __local uint in0[TILE_WORDS];
  __local uint in1[TILE_WORDS];
  __local uint out0[TILE_WORDS];
  volatile __global const uint *vin = (volatile __global const uint *)input;
  uint lid = get_local_id(0);

  stages = min_u32(stages, MAX_STAGES);
  if (buffers != 2u || stages == 0u) return;

  load_tile_manual(in0, vin, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint stage = 0u; stage < stages; stage++) {
    if ((stage & 1u) == 0u) {
      if (stage + 1u < stages) {
        load_tile_manual(in1, vin, stage + 1u, lid);
        barrier(CLK_LOCAL_MEM_FENCE);
      }
      compute_tile(in0, out0, lid);
    } else {
      if (stage + 1u < stages) {
        load_tile_manual(in0, vin, stage + 1u, lid);
        barrier(CLK_LOCAL_MEM_FENCE);
      }
      compute_tile(in1, out0, lid);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    store_tile_global(out0, output, stage, lid);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}


#define STORE_TILE_CONST(SRC, OUTPUT, STAGE, LID) do {                       \
  uint tma_out_addr = (uint)(OUTPUT) + ((uint)(STAGE) * TILE_WORDS * 4u);     \
  __asm__ volatile("mv %[addr], %[addr]\n\t"                               \
                   : [addr] "+r"(tma_out_addr) :: "memory");                 \
  volatile __global uint *tma_vout = (volatile __global uint *)tma_out_addr;  \
  for (uint tma_idx = (LID); tma_idx < TILE_WORDS; tma_idx += WG_SIZE) {      \
    tma_vout[tma_idx] = (SRC)[tma_idx];                                      \
  }                                                                          \
} while (0)

#define LOAD_TILE_CONST(DST, INPUT, STAGE, LID) do {                         \
  uint load_addr = (uint)(INPUT) + ((uint)(STAGE) * TILE_WORDS * 4u);         \
  __asm__ volatile("mv %[addr], %[addr]\n\t"                               \
                   : [addr] "+r"(load_addr) :: "memory");                  \
  volatile __global const uint *load_ptr =                                   \
      (volatile __global const uint *)load_addr;                             \
  for (uint load_idx = (LID); load_idx < TILE_WORDS; load_idx += WG_SIZE) {   \
    (DST)[load_idx] = load_ptr[load_idx];                                    \
  }                                                                          \
} while (0)

#define RUN_MANUAL_STAGE(STAGE) do {                                         \
  LOAD_TILE_CONST(in0, vin, (STAGE), lid);                                   \
  barrier(CLK_LOCAL_MEM_FENCE);                                              \
  compute_tile(in0, out0, lid);                                              \
  barrier(CLK_LOCAL_MEM_FENCE);                                              \
  STORE_TILE_CONST(out0, output, (STAGE), lid);                              \
  barrier(CLK_LOCAL_MEM_FENCE);                                              \
} while (0)

kernel void
manual_pingpong_b2_s1_kernel(__global const uint *input,
                             __global uint *output,
                             uint buffers,
                             uint stages)
{
  __local uint in0[TILE_WORDS];
  __local uint out0[TILE_WORDS];
  volatile __global const uint *vin = (volatile __global const uint *)input;
  uint lid = get_local_id(0);
  if (buffers != 2u || stages != 1u) return;
  RUN_MANUAL_STAGE(0u);
}

kernel void
manual_pingpong_b2_s2_kernel(__global const uint *input,
                             __global uint *output,
                             uint buffers,
                             uint stages)
{
  __local uint in0[TILE_WORDS];
  __local uint out0[TILE_WORDS];
  volatile __global const uint *vin = (volatile __global const uint *)input;
  uint lid = get_local_id(0);
  if (buffers != 2u || stages != 2u) return;
  RUN_MANUAL_STAGE(0u);
  RUN_MANUAL_STAGE(1u);
}

kernel void
manual_pingpong_b2_s3_kernel(__global const uint *input,
                             __global uint *output,
                             uint buffers,
                             uint stages)
{
  __local uint in0[TILE_WORDS];
  __local uint out0[TILE_WORDS];
  volatile __global const uint *vin = (volatile __global const uint *)input;
  uint lid = get_local_id(0);
  if (buffers != 2u || stages != 3u) return;
  RUN_MANUAL_STAGE(0u);
  RUN_MANUAL_STAGE(1u);
  RUN_MANUAL_STAGE(2u);
}

kernel void
manual_pingpong_b2_s4_kernel(__global const uint *input,
                             __global uint *output,
                             uint buffers,
                             uint stages)
{
  __local uint in0[TILE_WORDS];
  __local uint out0[TILE_WORDS];
  volatile __global const uint *vin = (volatile __global const uint *)input;
  uint lid = get_local_id(0);
  if (buffers != 2u || stages != 4u) return;
  RUN_MANUAL_STAGE(0u);
  RUN_MANUAL_STAGE(1u);
  RUN_MANUAL_STAGE(2u);
  RUN_MANUAL_STAGE(3u);
}

kernel void
manual_pingpong_b2_s8_kernel(__global const uint *input,
                             __global uint *output,
                             uint buffers,
                             uint stages)
{
  __local uint in0[TILE_WORDS];
  __local uint out0[TILE_WORDS];
  volatile __global const uint *vin = (volatile __global const uint *)input;
  uint lid = get_local_id(0);
  if (buffers != 2u || stages != 8u) return;
  RUN_MANUAL_STAGE(0u);
  RUN_MANUAL_STAGE(1u);
  RUN_MANUAL_STAGE(2u);
  RUN_MANUAL_STAGE(3u);
  RUN_MANUAL_STAGE(4u);
  RUN_MANUAL_STAGE(5u);
  RUN_MANUAL_STAGE(6u);
  RUN_MANUAL_STAGE(7u);
}


kernel void
manual_pingpong_b2_s16_kernel(__global const uint *input,
                              __global uint *output,
                              uint buffers,
                              uint stages)
{
  __local uint in0[TILE_WORDS];
  __local uint out0[TILE_WORDS];
  volatile __global const uint *vin = (volatile __global const uint *)input;
  uint lid = get_local_id(0);
  if (buffers != 2u || stages != 16u) return;
  RUN_MANUAL_STAGE(0u);
  RUN_MANUAL_STAGE(1u);
  RUN_MANUAL_STAGE(2u);
  RUN_MANUAL_STAGE(3u);
  RUN_MANUAL_STAGE(4u);
  RUN_MANUAL_STAGE(5u);
  RUN_MANUAL_STAGE(6u);
  RUN_MANUAL_STAGE(7u);
  RUN_MANUAL_STAGE(8u);
  RUN_MANUAL_STAGE(9u);
  RUN_MANUAL_STAGE(10u);
  RUN_MANUAL_STAGE(11u);
  RUN_MANUAL_STAGE(12u);
  RUN_MANUAL_STAGE(13u);
  RUN_MANUAL_STAGE(14u);
  RUN_MANUAL_STAGE(15u);
}

#define ISSUE_TILE_S2G_CONST(SRC, OUTPUT, STAGE, LID) do {                  \
  barrier(CLK_LOCAL_MEM_FENCE);                                             \
  if ((LID) == 0u) {                                                        \
    uint s2g_dst_addr = (uint)(OUTPUT) + ((uint)(STAGE) * TILE_WORDS * 4u); \
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();                                  \
    VENTUS_TMA_BULK_S2G(s2g_dst_addr, (SRC), TILE_WORDS * 4u);              \
  }                                                                         \
  barrier(CLK_LOCAL_MEM_FENCE);                                             \
} while (0)

#define S2G_WAIT_ALL(LID) do {                                               \
  if ((LID) == 0u) {                                                        \
    VENTUS_TMA_S2G_WAIT_GROUP0();                                           \
  }                                                                         \
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);                      \
} while (0)

#define S2G_COMMIT_GROUP(LID) do {                                           \
  if ((LID) == 0u) {                                                        \
    VENTUS_TMA_S2G_COMMIT_GROUP();                                          \
  }                                                                         \
} while (0)

#define S2G_WAIT_GROUP2(LID) do {                                            \
  if ((LID) == 0u) {                                                        \
    VENTUS_TMA_S2G_WAIT_GROUP2();                                           \
  }                                                                         \
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);                      \
} while (0)

#define S2G_WAIT_GROUP0(LID) do {                                            \
  if ((LID) == 0u) {                                                        \
    VENTUS_TMA_S2G_WAIT_GROUP0();                                           \
  }                                                                         \
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);                      \
} while (0)

#define S2G_WAIT_GROUP1(LID) do {                                            \
  if ((LID) == 0u) {                                                        \
    VENTUS_TMA_S2G_WAIT_GROUP1();                                           \
  }                                                                         \
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);                      \
} while (0)

#define STORE_TILE_S2G_CONST(SRC, OUTPUT, STAGE, LID) do {                  \
  ISSUE_TILE_S2G_CONST((SRC), (OUTPUT), (STAGE), (LID));                    \
  S2G_COMMIT_GROUP((LID));                                                  \
  S2G_WAIT_ALL((LID));                                                      \
} while (0)

#define RUN_MANUAL_S2G_DELAY_STAGE(BUF, STAGE) do {                         \
  LOAD_TILE_CONST((BUF), vin, (STAGE), lid);                                \
  barrier(CLK_LOCAL_MEM_FENCE);                                             \
  compute_tile((BUF), (BUF), lid);                                          \
  ISSUE_TILE_S2G_CONST((BUF), output, (STAGE), lid);                        \
  S2G_COMMIT_GROUP(lid);                                                    \
} while (0)

#define RUN_MANUAL_S2G_DELAY_STAGE_AFTER_WAIT(BUF, STAGE) do {              \
  S2G_WAIT_GROUP2(lid);                                                     \
  RUN_MANUAL_S2G_DELAY_STAGE((BUF), (STAGE));                               \
} while (0)

kernel void
manual_s2g_b2_s1_kernel(__global const uint *input,
                        __global uint *output,
                        uint buffers,
                        uint stages)
{
  __local uint out0[TILE_WORDS];
  volatile __global const uint *vin = (volatile __global const uint *)input;
  uint lid = get_local_id(0);
  if (buffers != 2u || stages != 1u) return;
  RUN_MANUAL_S2G_DELAY_STAGE(out0, 0u);
  S2G_WAIT_ALL(lid);
}

kernel void
manual_s2g_b2_s2_kernel(__global const uint *input,
                        __global uint *output,
                        uint buffers,
                        uint stages)
{
  __local uint out0[TILE_WORDS];
  __local uint out1[TILE_WORDS];
  volatile __global const uint *vin = (volatile __global const uint *)input;
  uint lid = get_local_id(0);
  if (buffers != 2u || stages != 2u) return;
  RUN_MANUAL_S2G_DELAY_STAGE(out0, 0u);
  RUN_MANUAL_S2G_DELAY_STAGE(out1, 1u);
  S2G_WAIT_ALL(lid);
}

kernel void
manual_s2g_b2_s4_kernel(__global const uint *input,
                        __global uint *output,
                        uint buffers,
                        uint stages)
{
  __local uint out0[TILE_WORDS];
  __local uint out1[TILE_WORDS];
  __local uint out2[TILE_WORDS];
  volatile __global const uint *vin = (volatile __global const uint *)input;
  uint lid = get_local_id(0);
  if (buffers != 2u || stages != 4u) return;
  RUN_MANUAL_S2G_DELAY_STAGE(out0, 0u);
  RUN_MANUAL_S2G_DELAY_STAGE(out1, 1u);
  RUN_MANUAL_S2G_DELAY_STAGE(out2, 2u);
  RUN_MANUAL_S2G_DELAY_STAGE_AFTER_WAIT(out0, 3u);
  S2G_WAIT_ALL(lid);
}

kernel void
manual_s2g_b2_s8_kernel(__global const uint *input,
                        __global uint *output,
                        uint buffers,
                        uint stages)
{
  __local uint out0[TILE_WORDS];
  __local uint out1[TILE_WORDS];
  __local uint out2[TILE_WORDS];
  volatile __global const uint *vin = (volatile __global const uint *)input;
  uint lid = get_local_id(0);
  if (buffers != 2u || stages != 8u) return;
  RUN_MANUAL_S2G_DELAY_STAGE(out0, 0u);
  RUN_MANUAL_S2G_DELAY_STAGE(out1, 1u);
  RUN_MANUAL_S2G_DELAY_STAGE(out2, 2u);
  RUN_MANUAL_S2G_DELAY_STAGE_AFTER_WAIT(out0, 3u);
  RUN_MANUAL_S2G_DELAY_STAGE(out1, 4u);
  RUN_MANUAL_S2G_DELAY_STAGE(out2, 5u);
  RUN_MANUAL_S2G_DELAY_STAGE_AFTER_WAIT(out0, 6u);
  RUN_MANUAL_S2G_DELAY_STAGE(out1, 7u);
  S2G_WAIT_ALL(lid);
}

kernel void
manual_s2g_b2_s16_kernel(__global const uint *input,
                         __global uint *output,
                         uint buffers,
                         uint stages)
{
  __local uint out0[TILE_WORDS];
  __local uint out1[TILE_WORDS];
  __local uint out2[TILE_WORDS];
  volatile __global const uint *vin = (volatile __global const uint *)input;
  uint lid = get_local_id(0);
  if (buffers != 2u || stages != 16u) return;
  RUN_MANUAL_S2G_DELAY_STAGE(out0, 0u);
  RUN_MANUAL_S2G_DELAY_STAGE(out1, 1u);
  RUN_MANUAL_S2G_DELAY_STAGE(out2, 2u);
  RUN_MANUAL_S2G_DELAY_STAGE_AFTER_WAIT(out0, 3u);
  RUN_MANUAL_S2G_DELAY_STAGE(out1, 4u);
  RUN_MANUAL_S2G_DELAY_STAGE(out2, 5u);
  RUN_MANUAL_S2G_DELAY_STAGE_AFTER_WAIT(out0, 6u);
  RUN_MANUAL_S2G_DELAY_STAGE(out1, 7u);
  RUN_MANUAL_S2G_DELAY_STAGE(out2, 8u);
  RUN_MANUAL_S2G_DELAY_STAGE_AFTER_WAIT(out0, 9u);
  RUN_MANUAL_S2G_DELAY_STAGE(out1, 10u);
  RUN_MANUAL_S2G_DELAY_STAGE(out2, 11u);
  RUN_MANUAL_S2G_DELAY_STAGE_AFTER_WAIT(out0, 12u);
  RUN_MANUAL_S2G_DELAY_STAGE(out1, 13u);
  RUN_MANUAL_S2G_DELAY_STAGE(out2, 14u);
  RUN_MANUAL_S2G_DELAY_STAGE_AFTER_WAIT(out0, 15u);
  S2G_WAIT_ALL(lid);
}

#define RUN_MANUAL_TENSOR_S2G_DELAY_STAGE(BUF, STAGE) do {                  \
  LOAD_TILE_CONST((BUF), vin, (STAGE), lid);                                \
  barrier(CLK_LOCAL_MEM_FENCE);                                             \
  compute_tile((BUF), (BUF), lid);                                          \
  ISSUE_TENSOR_S2G_BUF((BUF), s2g_desc, coords, (STAGE), lid, output);       \
  S2G_COMMIT_GROUP(lid);                                                    \
} while (0)

#define RUN_MANUAL_TENSOR_S2G_DELAY_STAGE_AFTER_WAIT(BUF, STAGE) do {       \
  S2G_WAIT_GROUP2(lid);                                                     \
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE((BUF), (STAGE));                        \
} while (0)

#define PREPARE_S2G_DESCRIPTOR(DESC, OUTPUT, LID) do {                       \
  if ((LID) == 0u) {                                                        \
    (DESC)[2] = (uint)(OUTPUT);                                             \
    __asm__ volatile("fence rw, rw" ::: "memory");                         \
    VENTUS_TMA_INVALIDATE_TENSORMAP((DESC));                                \
  }                                                                         \
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);                      \
} while (0)

kernel void
manual_tensor_s2g_b2_s1_kernel(__global const uint *input,
                               __global uint *output,
                               __global uint *s2g_desc,
                               uint buffers,
                               uint stages)
{
  __local uint out0_raw[TILE_WORDS + 32u];
  __local uint *out0 = align_local_128(out0_raw);
  __local uint coords[COORD_WORDS];
  volatile __global const uint *vin = (volatile __global const uint *)input;
  uint lid = get_local_id(0);
  if (buffers != 2u || stages != 1u) return;
  PREPARE_S2G_DESCRIPTOR(s2g_desc, output, lid);
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE(out0, 0u);
  S2G_WAIT_ALL(lid);
}

kernel void
manual_tensor_s2g_b2_s2_kernel(__global const uint *input,
                               __global uint *output,
                               __global uint *s2g_desc,
                               uint buffers,
                               uint stages)
{
  __local uint out0_raw[TILE_WORDS + 32u];
  __local uint out1_raw[TILE_WORDS + 32u];
  __local uint *out0 = align_local_128(out0_raw);
  __local uint *out1 = align_local_128(out1_raw);
  __local uint coords[COORD_WORDS];
  volatile __global const uint *vin = (volatile __global const uint *)input;
  uint lid = get_local_id(0);
  if (buffers != 2u || stages != 2u) return;
  PREPARE_S2G_DESCRIPTOR(s2g_desc, output, lid);
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE(out0, 0u);
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE(out1, 1u);
  S2G_WAIT_ALL(lid);
}

kernel void
manual_tensor_s2g_b2_s4_kernel(__global const uint *input,
                               __global uint *output,
                               __global uint *s2g_desc,
                               uint buffers,
                               uint stages)
{
  __local uint out0_raw[TILE_WORDS + 32u];
  __local uint out1_raw[TILE_WORDS + 32u];
  __local uint out2_raw[TILE_WORDS + 32u];
  __local uint *out0 = align_local_128(out0_raw);
  __local uint *out1 = align_local_128(out1_raw);
  __local uint *out2 = align_local_128(out2_raw);
  __local uint coords[COORD_WORDS];
  volatile __global const uint *vin = (volatile __global const uint *)input;
  uint lid = get_local_id(0);
  if (buffers != 2u || stages != 4u) return;
  PREPARE_S2G_DESCRIPTOR(s2g_desc, output, lid);
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE(out0, 0u);
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE(out1, 1u);
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE(out2, 2u);
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE_AFTER_WAIT(out0, 3u);
  S2G_WAIT_ALL(lid);
}

kernel void
manual_tensor_s2g_b2_s8_kernel(__global const uint *input,
                               __global uint *output,
                               __global uint *s2g_desc,
                               uint buffers,
                               uint stages)
{
  __local uint out0_raw[TILE_WORDS + 32u];
  __local uint out1_raw[TILE_WORDS + 32u];
  __local uint out2_raw[TILE_WORDS + 32u];
  __local uint *out0 = align_local_128(out0_raw);
  __local uint *out1 = align_local_128(out1_raw);
  __local uint *out2 = align_local_128(out2_raw);
  __local uint coords[COORD_WORDS];
  volatile __global const uint *vin = (volatile __global const uint *)input;
  uint lid = get_local_id(0);
  if (buffers != 2u || stages != 8u) return;
  PREPARE_S2G_DESCRIPTOR(s2g_desc, output, lid);
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE(out0, 0u);
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE(out1, 1u);
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE(out2, 2u);
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE_AFTER_WAIT(out0, 3u);
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE(out1, 4u);
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE(out2, 5u);
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE_AFTER_WAIT(out0, 6u);
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE(out1, 7u);
  S2G_WAIT_ALL(lid);
}

kernel void
manual_tensor_s2g_b2_s16_kernel(__global const uint *input,
                                __global uint *output,
                                __global uint *s2g_desc,
                                uint buffers,
                                uint stages)
{
  __local uint out0_raw[TILE_WORDS + 32u];
  __local uint out1_raw[TILE_WORDS + 32u];
  __local uint out2_raw[TILE_WORDS + 32u];
  __local uint *out0 = align_local_128(out0_raw);
  __local uint *out1 = align_local_128(out1_raw);
  __local uint *out2 = align_local_128(out2_raw);
  __local uint coords[COORD_WORDS];
  volatile __global const uint *vin = (volatile __global const uint *)input;
  uint lid = get_local_id(0);
  if (buffers != 2u || stages != 16u) return;
  PREPARE_S2G_DESCRIPTOR(s2g_desc, output, lid);
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE(out0, 0u);
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE(out1, 1u);
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE(out2, 2u);
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE_AFTER_WAIT(out0, 3u);
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE(out1, 4u);
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE(out2, 5u);
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE_AFTER_WAIT(out0, 6u);
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE(out1, 7u);
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE(out2, 8u);
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE_AFTER_WAIT(out0, 9u);
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE(out1, 10u);
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE(out2, 11u);
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE_AFTER_WAIT(out0, 12u);
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE(out1, 13u);
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE(out2, 14u);
  RUN_MANUAL_TENSOR_S2G_DELAY_STAGE_AFTER_WAIT(out0, 15u);
  S2G_WAIT_ALL(lid);
}

#define TMA_S2G_PRELOAD0(BUF) do {                                          \
  if (lid == 0u) {                                                          \
    g2s_desc[2] = (uint)input;                                              \
  }                                                                         \
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);                      \
  set_stage_coords(coords, 0u, lid);                                        \
  ISSUE_TMA_G2S_BUF((BUF), g2s_desc, coords);                               \
  TMA_WAIT_ALL();                                                           \
} while (0)

#define RUN_TMA_S2G_PRELOAD_STORE(NEXT_STAGE, NEXT_BUF, CUR_BUF, STORE_STAGE) do { \
  set_stage_coords(coords, (NEXT_STAGE), lid);                              \
  ISSUE_TMA_G2S_BUF((NEXT_BUF), g2s_desc, coords);                          \
  compute_tile((CUR_BUF), out0, lid);                                       \
  STORE_TILE_S2G_CONST(out0, output, (STORE_STAGE), lid);                   \
  TMA_WAIT_ALL();                                                           \
} while (0)

#define RUN_TMA_S2G_STORE_FINAL(CUR_BUF, STORE_STAGE) do {                  \
  compute_tile((CUR_BUF), out0, lid);                                       \
  STORE_TILE_S2G_CONST(out0, output, (STORE_STAGE), lid);                   \
} while (0)

kernel void
tma_s2g_b2_s1_kernel(__global uint *g2s_desc,
                     __global const uint *input,
                     __global uint *output)
{
  __local uint in0[TILE_WORDS];
  __local uint in1[TILE_WORDS];
  __local uint out0[TILE_WORDS];
  __local uint coords[COORD_WORDS];
  uint lid = get_local_id(0);

  TMA_S2G_PRELOAD0(in0);
  RUN_TMA_S2G_STORE_FINAL(in0, 0u);
}

kernel void
tma_s2g_b2_s2_kernel(__global uint *g2s_desc,
                     __global const uint *input,
                     __global uint *output)
{
  __local uint in0[TILE_WORDS];
  __local uint in1[TILE_WORDS];
  __local uint out0[TILE_WORDS];
  __local uint coords[COORD_WORDS];
  uint lid = get_local_id(0);

  TMA_S2G_PRELOAD0(in0);
  RUN_TMA_S2G_PRELOAD_STORE(1u, in1, in0, 0u);
  RUN_TMA_S2G_STORE_FINAL(in1, 1u);
}

kernel void
tma_s2g_b2_s4_kernel(__global uint *g2s_desc,
                     __global const uint *input,
                     __global uint *output)
{
  __local uint in0[TILE_WORDS];
  __local uint in1[TILE_WORDS];
  __local uint out0[TILE_WORDS];
  __local uint coords[COORD_WORDS];
  uint lid = get_local_id(0);

  TMA_S2G_PRELOAD0(in0);
  RUN_TMA_S2G_PRELOAD_STORE(1u, in1, in0, 0u);
  RUN_TMA_S2G_PRELOAD_STORE(2u, in0, in1, 1u);
  RUN_TMA_S2G_PRELOAD_STORE(3u, in1, in0, 2u);
  RUN_TMA_S2G_STORE_FINAL(in1, 3u);
}

kernel void
tma_s2g_b2_s8_kernel(__global uint *g2s_desc,
                     __global const uint *input,
                     __global uint *output)
{
  __local uint in0[TILE_WORDS];
  __local uint in1[TILE_WORDS];
  __local uint out0[TILE_WORDS];
  __local uint coords[COORD_WORDS];
  uint lid = get_local_id(0);

  TMA_S2G_PRELOAD0(in0);
  RUN_TMA_S2G_PRELOAD_STORE(1u, in1, in0, 0u);
  RUN_TMA_S2G_PRELOAD_STORE(2u, in0, in1, 1u);
  RUN_TMA_S2G_PRELOAD_STORE(3u, in1, in0, 2u);
  RUN_TMA_S2G_PRELOAD_STORE(4u, in0, in1, 3u);
  RUN_TMA_S2G_PRELOAD_STORE(5u, in1, in0, 4u);
  RUN_TMA_S2G_PRELOAD_STORE(6u, in0, in1, 5u);
  RUN_TMA_S2G_PRELOAD_STORE(7u, in1, in0, 6u);
  RUN_TMA_S2G_STORE_FINAL(in1, 7u);
}

kernel void
tma_s2g_b2_s16_kernel(__global uint *g2s_desc,
                      __global const uint *input,
                      __global uint *output)
{
  __local uint in0[TILE_WORDS];
  __local uint in1[TILE_WORDS];
  __local uint out0[TILE_WORDS];
  __local uint coords[COORD_WORDS];
  uint lid = get_local_id(0);

  TMA_S2G_PRELOAD0(in0);
  RUN_TMA_S2G_PRELOAD_STORE(1u, in1, in0, 0u);
  RUN_TMA_S2G_PRELOAD_STORE(2u, in0, in1, 1u);
  RUN_TMA_S2G_PRELOAD_STORE(3u, in1, in0, 2u);
  RUN_TMA_S2G_PRELOAD_STORE(4u, in0, in1, 3u);
  RUN_TMA_S2G_PRELOAD_STORE(5u, in1, in0, 4u);
  RUN_TMA_S2G_PRELOAD_STORE(6u, in0, in1, 5u);
  RUN_TMA_S2G_PRELOAD_STORE(7u, in1, in0, 6u);
  RUN_TMA_S2G_PRELOAD_STORE(8u, in0, in1, 7u);
  RUN_TMA_S2G_PRELOAD_STORE(9u, in1, in0, 8u);
  RUN_TMA_S2G_PRELOAD_STORE(10u, in0, in1, 9u);
  RUN_TMA_S2G_PRELOAD_STORE(11u, in1, in0, 10u);
  RUN_TMA_S2G_PRELOAD_STORE(12u, in0, in1, 11u);
  RUN_TMA_S2G_PRELOAD_STORE(13u, in1, in0, 12u);
  RUN_TMA_S2G_PRELOAD_STORE(14u, in0, in1, 13u);
  RUN_TMA_S2G_PRELOAD_STORE(15u, in1, in0, 14u);
  RUN_TMA_S2G_STORE_FINAL(in1, 15u);
}

#define RUN_TMA_TENSOR_S2G_PRELOAD_STORE_NO_S2G_WAIT(NEXT_STAGE, NEXT_BUF, CUR_BUF, STORE_STAGE) do { \
  set_stage_coords(coords, (NEXT_STAGE), lid);                              \
  ISSUE_TMA_G2S_BUF((NEXT_BUF), g2s_desc, coords);                          \
  compute_tile((CUR_BUF), (CUR_BUF), lid);                                  \
  ISSUE_TENSOR_S2G_BUF((CUR_BUF), s2g_desc, coords, (STORE_STAGE), lid, output); \
  S2G_COMMIT_GROUP(lid);                                                    \
  S2G_WAIT_GROUP1(lid);                                                     \
  TMA_WAIT_ALL();                                                           \
} while (0)

#define RUN_TMA_TENSOR_S2G_PRELOAD_STORE_AFTER_WAIT(NEXT_STAGE, NEXT_BUF, CUR_BUF, STORE_STAGE) do { \
  S2G_WAIT_GROUP1(lid);                                                     \
  set_stage_coords(coords, (NEXT_STAGE), lid);                              \
  ISSUE_TMA_G2S_BUF((NEXT_BUF), g2s_desc, coords);                          \
  compute_tile((CUR_BUF), (CUR_BUF), lid);                                  \
  ISSUE_TENSOR_S2G_BUF((CUR_BUF), s2g_desc, coords, (STORE_STAGE), lid, output); \
  S2G_COMMIT_GROUP(lid);                                                    \
  S2G_WAIT_GROUP1(lid);                                                     \
  TMA_WAIT_ALL();                                                           \
} while (0)

#define RUN_TMA_TENSOR_S2G_STORE_FINAL_NO_S2G_WAIT(CUR_BUF, STORE_STAGE) do { \
  compute_tile((CUR_BUF), (CUR_BUF), lid);                                  \
  ISSUE_TENSOR_S2G_BUF((CUR_BUF), s2g_desc, coords, (STORE_STAGE), lid, output); \
  S2G_COMMIT_GROUP(lid);                                                    \
  S2G_WAIT_GROUP0(lid);                                                     \
} while (0)

#define RUN_TMA_TENSOR_S2G_STORE_FINAL(CUR_BUF, STORE_STAGE) do {           \
  compute_tile((CUR_BUF), (CUR_BUF), lid);                                  \
  ISSUE_TENSOR_S2G_BUF((CUR_BUF), s2g_desc, coords, (STORE_STAGE), lid, output); \
  S2G_COMMIT_GROUP(lid);                                                    \
  S2G_WAIT_GROUP0(lid);                                                     \
} while (0)

kernel void
tma_tensor_s2g_b2_s1_kernel(__global uint *g2s_desc,
                            __global uint *s2g_desc,
                            __global const uint *input,
                            __global uint *output)
{
  __local uint in0[TILE_WORDS];
  __local uint in1[TILE_WORDS];
  __local uint in2[TILE_WORDS];
  __local uint coords[COORD_WORDS];
  uint lid = get_local_id(0);
  (void)output;

  TMA_S2G_PRELOAD0(in0);
  RUN_TMA_TENSOR_S2G_STORE_FINAL_NO_S2G_WAIT(in0, 0u);
}

kernel void
tma_tensor_s2g_b2_s2_kernel(__global uint *g2s_desc,
                            __global uint *s2g_desc,
                            __global const uint *input,
                            __global uint *output)
{
  __local uint in0[TILE_WORDS];
  __local uint in1[TILE_WORDS];
  __local uint in2[TILE_WORDS];
  __local uint coords[COORD_WORDS];
  uint lid = get_local_id(0);
  (void)output;

  TMA_S2G_PRELOAD0(in0);
  RUN_TMA_TENSOR_S2G_PRELOAD_STORE_NO_S2G_WAIT(1u, in1, in0, 0u);
  RUN_TMA_TENSOR_S2G_STORE_FINAL(in1, 1u);
}

kernel void
tma_tensor_s2g_b2_s4_kernel(__global uint *g2s_desc,
                            __global uint *s2g_desc,
                            __global const uint *input,
                            __global uint *output)
{
  __local uint in0[TILE_WORDS];
  __local uint in1[TILE_WORDS];
  __local uint in2[TILE_WORDS];
  __local uint coords[COORD_WORDS];
  uint lid = get_local_id(0);
  (void)output;

  TMA_S2G_PRELOAD0(in0);
  RUN_TMA_TENSOR_S2G_PRELOAD_STORE_NO_S2G_WAIT(1u, in1, in0, 0u);
  RUN_TMA_TENSOR_S2G_PRELOAD_STORE_NO_S2G_WAIT(2u, in2, in1, 1u);
  RUN_TMA_TENSOR_S2G_PRELOAD_STORE_AFTER_WAIT(3u, in0, in2, 2u);
  RUN_TMA_TENSOR_S2G_STORE_FINAL(in0, 3u);
}

kernel void
tma_tensor_s2g_b2_s8_kernel(__global uint *g2s_desc,
                            __global uint *s2g_desc,
                            __global const uint *input,
                            __global uint *output)
{
  __local uint in0[TILE_WORDS];
  __local uint in1[TILE_WORDS];
  __local uint in2[TILE_WORDS];
  __local uint coords[COORD_WORDS];
  uint lid = get_local_id(0);
  (void)output;

  TMA_S2G_PRELOAD0(in0);
  RUN_TMA_TENSOR_S2G_PRELOAD_STORE_NO_S2G_WAIT(1u, in1, in0, 0u);
  RUN_TMA_TENSOR_S2G_PRELOAD_STORE_NO_S2G_WAIT(2u, in2, in1, 1u);
  RUN_TMA_TENSOR_S2G_PRELOAD_STORE_AFTER_WAIT(3u, in0, in2, 2u);
  RUN_TMA_TENSOR_S2G_PRELOAD_STORE_AFTER_WAIT(4u, in1, in0, 3u);
  RUN_TMA_TENSOR_S2G_PRELOAD_STORE_AFTER_WAIT(5u, in2, in1, 4u);
  RUN_TMA_TENSOR_S2G_PRELOAD_STORE_AFTER_WAIT(6u, in0, in2, 5u);
  RUN_TMA_TENSOR_S2G_PRELOAD_STORE_AFTER_WAIT(7u, in1, in0, 6u);
  RUN_TMA_TENSOR_S2G_STORE_FINAL(in1, 7u);
}

kernel void
tma_tensor_s2g_b2_s16_kernel(__global uint *g2s_desc,
                             __global uint *s2g_desc,
                             __global const uint *input,
                             __global uint *output)
{
  __local uint in0[TILE_WORDS];
  __local uint in1[TILE_WORDS];
  __local uint in2[TILE_WORDS];
  __local uint coords[COORD_WORDS];
  uint lid = get_local_id(0);
  (void)output;

  TMA_S2G_PRELOAD0(in0);
  RUN_TMA_TENSOR_S2G_PRELOAD_STORE_NO_S2G_WAIT(1u, in1, in0, 0u);
  RUN_TMA_TENSOR_S2G_PRELOAD_STORE_NO_S2G_WAIT(2u, in2, in1, 1u);
  RUN_TMA_TENSOR_S2G_PRELOAD_STORE_AFTER_WAIT(3u, in0, in2, 2u);
  RUN_TMA_TENSOR_S2G_PRELOAD_STORE_AFTER_WAIT(4u, in1, in0, 3u);
  RUN_TMA_TENSOR_S2G_PRELOAD_STORE_AFTER_WAIT(5u, in2, in1, 4u);
  RUN_TMA_TENSOR_S2G_PRELOAD_STORE_AFTER_WAIT(6u, in0, in2, 5u);
  RUN_TMA_TENSOR_S2G_PRELOAD_STORE_AFTER_WAIT(7u, in1, in0, 6u);
  RUN_TMA_TENSOR_S2G_PRELOAD_STORE_AFTER_WAIT(8u, in2, in1, 7u);
  RUN_TMA_TENSOR_S2G_PRELOAD_STORE_AFTER_WAIT(9u, in0, in2, 8u);
  RUN_TMA_TENSOR_S2G_PRELOAD_STORE_AFTER_WAIT(10u, in1, in0, 9u);
  RUN_TMA_TENSOR_S2G_PRELOAD_STORE_AFTER_WAIT(11u, in2, in1, 10u);
  RUN_TMA_TENSOR_S2G_PRELOAD_STORE_AFTER_WAIT(12u, in0, in2, 11u);
  RUN_TMA_TENSOR_S2G_PRELOAD_STORE_AFTER_WAIT(13u, in1, in0, 12u);
  RUN_TMA_TENSOR_S2G_PRELOAD_STORE_AFTER_WAIT(14u, in2, in1, 13u);
  RUN_TMA_TENSOR_S2G_PRELOAD_STORE_AFTER_WAIT(15u, in0, in2, 14u);
  RUN_TMA_TENSOR_S2G_STORE_FINAL(in0, 15u);
}

kernel void
tma_pingpong_b2_s1_kernel(__global uint *g2s_desc,
                           __global const uint *input,
                           __global uint *output)
{
  __local uint in0[TILE_WORDS];
  __local uint in1[TILE_WORDS];
  __local uint out0[TILE_WORDS];
  __local uint coords[COORD_WORDS];
  uint lid = get_local_id(0);

  if (lid == 0u) {
    g2s_desc[2] = (uint)input;
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  set_stage_coords(coords, 0u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  TMA_WAIT_ALL();
  compute_tile(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_CONST(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void
tma_pingpong_b2_s2_kernel(__global uint *g2s_desc,
                           __global const uint *input,
                           __global uint *output)
{
  __local uint in0[TILE_WORDS];
  __local uint in1[TILE_WORDS];
  __local uint out0[TILE_WORDS];
  __local uint coords[COORD_WORDS];
  uint lid = get_local_id(0);

  if (lid == 0u) {
    g2s_desc[2] = (uint)input;
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  set_stage_coords(coords, 0u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  TMA_WAIT_ALL();

  set_stage_coords(coords, 1u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  compute_tile(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_CONST(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  TMA_WAIT_ALL();
  compute_tile(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_CONST(out0, output, 1u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void
tma_pingpong_b2_s3_kernel(__global uint *g2s_desc,
                           __global const uint *input,
                           __global uint *output)
{
  __local uint in0[TILE_WORDS];
  __local uint in1[TILE_WORDS];
  __local uint out0[TILE_WORDS];
  __local uint coords[COORD_WORDS];
  uint lid = get_local_id(0);

  if (lid == 0u) {
    g2s_desc[2] = (uint)input;
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  set_stage_coords(coords, 0u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  TMA_WAIT_ALL();

  set_stage_coords(coords, 1u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  compute_tile(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_CONST(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  TMA_WAIT_ALL();

  set_stage_coords(coords, 2u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  compute_tile(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_CONST(out0, output, 1u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  TMA_WAIT_ALL();
  compute_tile(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_CONST(out0, output, 2u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void
tma_pingpong_b2_s4_kernel(__global uint *g2s_desc,
                           __global const uint *input,
                           __global uint *output)
{
  __local uint in0[TILE_WORDS];
  __local uint in1[TILE_WORDS];
  __local uint out0[TILE_WORDS];
  __local uint coords[COORD_WORDS];
  uint lid = get_local_id(0);

  if (lid == 0u) {
    g2s_desc[2] = (uint)input;
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  set_stage_coords(coords, 0u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  TMA_WAIT_ALL();

  set_stage_coords(coords, 1u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  compute_tile(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_CONST(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  TMA_WAIT_ALL();

  set_stage_coords(coords, 2u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  compute_tile(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_CONST(out0, output, 1u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  TMA_WAIT_ALL();

  set_stage_coords(coords, 3u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  compute_tile(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_CONST(out0, output, 2u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  TMA_WAIT_ALL();
  compute_tile(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_CONST(out0, output, 3u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void
tma_pingpong_b2_s8_kernel(__global uint *g2s_desc,
                           __global const uint *input,
                           __global uint *output)
{
  __local uint in0[TILE_WORDS];
  __local uint in1[TILE_WORDS];
  __local uint out0[TILE_WORDS];
  __local uint coords[COORD_WORDS];
  uint lid = get_local_id(0);

  if (lid == 0u) {
    g2s_desc[2] = (uint)input;
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  set_stage_coords(coords, 0u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  TMA_WAIT_ALL();

  set_stage_coords(coords, 1u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  compute_tile(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_CONST(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  TMA_WAIT_ALL();

  set_stage_coords(coords, 2u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  compute_tile(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_CONST(out0, output, 1u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  TMA_WAIT_ALL();

  set_stage_coords(coords, 3u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  compute_tile(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_CONST(out0, output, 2u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  TMA_WAIT_ALL();

  set_stage_coords(coords, 4u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  compute_tile(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_CONST(out0, output, 3u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  TMA_WAIT_ALL();

  set_stage_coords(coords, 5u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  compute_tile(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_CONST(out0, output, 4u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  TMA_WAIT_ALL();

  set_stage_coords(coords, 6u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  compute_tile(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_CONST(out0, output, 5u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  TMA_WAIT_ALL();

  set_stage_coords(coords, 7u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  compute_tile(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_CONST(out0, output, 6u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  TMA_WAIT_ALL();
  compute_tile(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_CONST(out0, output, 7u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
}

#define RUN_TMA_PRELOAD_STORE(NEXT_STAGE, NEXT_BUF, CUR_BUF, STORE_STAGE) do { \
  set_stage_coords(coords, (NEXT_STAGE), lid);                              \
  ISSUE_TMA_G2S_BUF((NEXT_BUF), g2s_desc, coords);                          \
  compute_tile((CUR_BUF), out0, lid);                                        \
  barrier(CLK_LOCAL_MEM_FENCE);                                              \
  STORE_TILE_CONST(out0, output, (STORE_STAGE), lid);                        \
  barrier(CLK_LOCAL_MEM_FENCE);                                              \
  TMA_WAIT_ALL();                                                            \
} while (0)

#define RUN_TMA_STORE_FINAL(CUR_BUF, STORE_STAGE) do {                       \
  compute_tile((CUR_BUF), out0, lid);                                        \
  barrier(CLK_LOCAL_MEM_FENCE);                                              \
  STORE_TILE_CONST(out0, output, (STORE_STAGE), lid);                        \
  barrier(CLK_LOCAL_MEM_FENCE);                                              \
} while (0)

kernel void
tma_pingpong_b2_s16_kernel(__global uint *g2s_desc,
                            __global const uint *input,
                            __global uint *output)
{
  __local uint in0[TILE_WORDS];
  __local uint in1[TILE_WORDS];
  __local uint out0[TILE_WORDS];
  __local uint coords[COORD_WORDS];
  uint lid = get_local_id(0);

  if (lid == 0u) {
    g2s_desc[2] = (uint)input;
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  set_stage_coords(coords, 0u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  TMA_WAIT_ALL();

  RUN_TMA_PRELOAD_STORE(1u, in1, in0, 0u);
  RUN_TMA_PRELOAD_STORE(2u, in0, in1, 1u);
  RUN_TMA_PRELOAD_STORE(3u, in1, in0, 2u);
  RUN_TMA_PRELOAD_STORE(4u, in0, in1, 3u);
  RUN_TMA_PRELOAD_STORE(5u, in1, in0, 4u);
  RUN_TMA_PRELOAD_STORE(6u, in0, in1, 5u);
  RUN_TMA_PRELOAD_STORE(7u, in1, in0, 6u);
  RUN_TMA_PRELOAD_STORE(8u, in0, in1, 7u);
  RUN_TMA_PRELOAD_STORE(9u, in1, in0, 8u);
  RUN_TMA_PRELOAD_STORE(10u, in0, in1, 9u);
  RUN_TMA_PRELOAD_STORE(11u, in1, in0, 10u);
  RUN_TMA_PRELOAD_STORE(12u, in0, in1, 11u);
  RUN_TMA_PRELOAD_STORE(13u, in1, in0, 12u);
  RUN_TMA_PRELOAD_STORE(14u, in0, in1, 13u);
  RUN_TMA_PRELOAD_STORE(15u, in1, in0, 14u);
  RUN_TMA_STORE_FINAL(in1, 15u);
}
