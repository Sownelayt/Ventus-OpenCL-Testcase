/*
 * TMA G2S ping-pong double-buffer performance microbench.
 *
 * Manual path: global -> register -> shared, compute in shared, then
 * shared -> register -> global. It uses the same ping-pong buffer order as
 * the TMA path, but the next-tile load is ordinary synchronous code.
 *
 * TMA path: issue tile0 and wait once. For each tile, issue the next tile to
 * the other shared buffer, compute and store the current tile, then wait for
 * the next tile before consuming it in the following iteration.
 */

#define DESC_WORDS 32u
#define COORD_WORDS 32u
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
                  __global const uint *input)
{
  if (get_global_id(0) == 0) {
    g2s_desc[2] = (uint)input;
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
  for (uint i = lid; i < COORD_WORDS; i += WG_SIZE) {
    coords[i] = 0u;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0u) {
    coords[0] = 0u;
    coords[1] = stage * TILE_ROWS;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
}

#define ISSUE_TMA_G2S_BUF(BUF, DESC, COORDS) do {                         \
  uint tma_smem = (uint)(BUF);                                             \
  uint tma_desc = (uint)(DESC);                                             \
  uint tma_coords = (uint)(COORDS);                                         \
  __asm__ volatile(                                                         \
    "vid.v v12\n\t"                                                       \
    "vsll.vi v12, v12, 2\n\t"                                             \
    "vadd.vx v12, v12, %[coords]\n\t"                                     \
    "vlw12.v v12, 0(v12)\n\t"                                             \
    "mv x10, %[smem]\n\t"                                                 \
    "mv x11, %[desc]\n\t"                                                 \
    ".word 0x00C5A542\n\t"                                                \
    :                                                                       \
    : [smem] "r"(tma_smem), [desc] "r"(tma_desc),                         \
      [coords] "r"(tma_coords)                                             \
    : "x10", "x11", "memory"                                             \
  );                                                                        \
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

#define TMA_WAIT_ALL() do {                                                  \
  __asm__ volatile(".word 0x00006042\n\t" ::: "memory");                  \
  barrier(CLK_LOCAL_MEM_FENCE);                                              \
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

