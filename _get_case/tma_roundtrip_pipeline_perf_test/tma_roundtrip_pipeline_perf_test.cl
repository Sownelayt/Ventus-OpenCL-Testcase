/*
 * TMA G2S pipeline performance microbench.
 *
 * Manual path: global -> register -> shared, compute in shared, then
 * shared -> register -> global.
 *
 * TMA path: descriptor TMA G2S fills shared input buffers, compute writes
 * one reusable shared output buffer, then the result is written back with
 * the same ordinary shared/register/global-store path used by the manual
 * baseline. S2G TMA is intentionally not used in this version.
 *
 * Current Ventus directed path is only read-safe for at most two G2S TMA
 * copies before a wait when the consumer is ordinary shared-memory code, so
 * larger buffer-count cases are split into two-copy subgroups.
 */

#define DESC_WORDS 32u
#define COORD_WORDS 32u
#define TILE_ROWS 16u
#define TILE_COLS 16u
#define TILE_WORDS (TILE_ROWS * TILE_COLS)
#define MAX_BUFFERS 8u
#define MAX_STAGES 8u
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

#define COMPUTE_TILE_BUF(SRC, DST, LID) do {                                \
  for (uint tma_idx = (LID); tma_idx < TILE_WORDS; tma_idx += WG_SIZE) {     \
    float tma_x = as_float((SRC)[tma_idx]);                                  \
    float tma_y = tma_x * 1.0009765625f + 0.000244140625f;                   \
    (DST)[tma_idx] = as_uint(tma_y);                                         \
  }                                                                          \
} while (0)

#define STORE_TILE_BUF(SRC, OUTPUT, STAGE, LID) do {                         \
  uint tma_out_addr = (uint)(OUTPUT) + ((uint)(STAGE) * TILE_WORDS * 4u);     \
  __asm__ volatile("mv %[addr], %[addr]\n\t"                               \
                   : [addr] "+r"(tma_out_addr) :: "memory");                 \
  volatile __global uint *tma_vout = (volatile __global uint *)tma_out_addr;  \
  for (uint tma_idx = (LID); tma_idx < TILE_WORDS; tma_idx += WG_SIZE) {      \
    tma_vout[tma_idx] = (SRC)[tma_idx];                                      \
  }                                                                          \
} while (0)

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
  __local uint in_storage[MAX_BUFFERS * TILE_WORDS];
  __local uint out_storage[MAX_BUFFERS * TILE_WORDS];
  volatile __global const uint *vin = (volatile __global const uint *)input;
  volatile __global uint *vout = (volatile __global uint *)output;
  uint lid = get_local_id(0);
  __local uint *in_buf = in_storage;
  __local uint *out_buf = out_storage;

  buffers = min_u32(buffers, MAX_BUFFERS);
  stages = min_u32(stages, MAX_STAGES);
  if (buffers == 0u || stages == 0u) return;

  for (uint base = 0u; base < stages; base += buffers) {
    uint chunk = min_u32(buffers, stages - base);

    for (uint b = 0u; b < chunk; b++) {
      uint stage = base + b;
      __local uint *in_tile = in_buf + b * TILE_WORDS;
      for (uint idx = lid; idx < TILE_WORDS; idx += WG_SIZE) {
        in_tile[idx] = vin[stage * TILE_WORDS + idx];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint b = 0u; b < chunk; b++) {
      compute_tile(in_buf + b * TILE_WORDS, out_buf + b * TILE_WORDS, lid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint b = 0u; b < chunk; b++) {
      uint stage = base + b;
      __local uint *out_tile = out_buf + b * TILE_WORDS;
      for (uint idx = lid; idx < TILE_WORDS; idx += WG_SIZE) {
        vout[stage * TILE_WORDS + idx] = out_tile[idx];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

kernel void
tma_roundtrip_b2_s1_kernel(__global uint *g2s_desc,
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
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void
tma_roundtrip_b2_s2_kernel(__global uint *g2s_desc,
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
  set_stage_coords(coords, 1u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 1u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void
tma_roundtrip_b2_s3_kernel(__global uint *g2s_desc,
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
  set_stage_coords(coords, 1u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 1u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  set_stage_coords(coords, 2u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 2u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void
tma_roundtrip_b2_s4_kernel(__global uint *g2s_desc,
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
  set_stage_coords(coords, 1u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 1u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  set_stage_coords(coords, 2u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  set_stage_coords(coords, 3u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 2u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 3u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void
tma_roundtrip_b2_s8_kernel(__global uint *g2s_desc,
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
  set_stage_coords(coords, 1u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 1u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  set_stage_coords(coords, 2u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  set_stage_coords(coords, 3u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 2u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 3u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  set_stage_coords(coords, 4u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  set_stage_coords(coords, 5u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 4u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 5u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  set_stage_coords(coords, 6u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  set_stage_coords(coords, 7u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 6u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 7u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void
tma_roundtrip_b3_s1_kernel(__global uint *g2s_desc,
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
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void
tma_roundtrip_b3_s2_kernel(__global uint *g2s_desc,
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
  set_stage_coords(coords, 1u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 1u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void
tma_roundtrip_b3_s3_kernel(__global uint *g2s_desc,
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
  set_stage_coords(coords, 1u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 1u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  set_stage_coords(coords, 2u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 2u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void
tma_roundtrip_b3_s4_kernel(__global uint *g2s_desc,
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
  set_stage_coords(coords, 1u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 1u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  set_stage_coords(coords, 2u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  set_stage_coords(coords, 3u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 2u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 3u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void
tma_roundtrip_b3_s8_kernel(__global uint *g2s_desc,
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
  set_stage_coords(coords, 1u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 1u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  set_stage_coords(coords, 2u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  set_stage_coords(coords, 3u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 2u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 3u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  set_stage_coords(coords, 4u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  set_stage_coords(coords, 5u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 4u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 5u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  set_stage_coords(coords, 6u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  set_stage_coords(coords, 7u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 6u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 7u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void
tma_roundtrip_b4_s1_kernel(__global uint *g2s_desc,
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
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void
tma_roundtrip_b4_s2_kernel(__global uint *g2s_desc,
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
  set_stage_coords(coords, 1u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 1u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void
tma_roundtrip_b4_s3_kernel(__global uint *g2s_desc,
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
  set_stage_coords(coords, 1u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 1u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  set_stage_coords(coords, 2u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 2u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void
tma_roundtrip_b4_s4_kernel(__global uint *g2s_desc,
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
  set_stage_coords(coords, 1u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 1u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  set_stage_coords(coords, 2u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  set_stage_coords(coords, 3u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 2u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 3u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void
tma_roundtrip_b4_s8_kernel(__global uint *g2s_desc,
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
  set_stage_coords(coords, 1u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 1u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  set_stage_coords(coords, 2u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  set_stage_coords(coords, 3u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 2u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 3u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  set_stage_coords(coords, 4u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  set_stage_coords(coords, 5u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 4u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 5u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  set_stage_coords(coords, 6u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  set_stage_coords(coords, 7u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 6u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 7u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void
tma_roundtrip_b6_s1_kernel(__global uint *g2s_desc,
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
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void
tma_roundtrip_b6_s2_kernel(__global uint *g2s_desc,
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
  set_stage_coords(coords, 1u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 1u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void
tma_roundtrip_b6_s3_kernel(__global uint *g2s_desc,
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
  set_stage_coords(coords, 1u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 1u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  set_stage_coords(coords, 2u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 2u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void
tma_roundtrip_b6_s4_kernel(__global uint *g2s_desc,
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
  set_stage_coords(coords, 1u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 1u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  set_stage_coords(coords, 2u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  set_stage_coords(coords, 3u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 2u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 3u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void
tma_roundtrip_b6_s8_kernel(__global uint *g2s_desc,
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
  set_stage_coords(coords, 1u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 1u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  set_stage_coords(coords, 2u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  set_stage_coords(coords, 3u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 2u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 3u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  set_stage_coords(coords, 4u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  set_stage_coords(coords, 5u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 4u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 5u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  set_stage_coords(coords, 6u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  set_stage_coords(coords, 7u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 6u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 7u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void
tma_roundtrip_b8_s1_kernel(__global uint *g2s_desc,
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
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void
tma_roundtrip_b8_s2_kernel(__global uint *g2s_desc,
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
  set_stage_coords(coords, 1u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 1u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void
tma_roundtrip_b8_s3_kernel(__global uint *g2s_desc,
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
  set_stage_coords(coords, 1u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 1u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  set_stage_coords(coords, 2u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 2u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void
tma_roundtrip_b8_s4_kernel(__global uint *g2s_desc,
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
  set_stage_coords(coords, 1u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 1u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  set_stage_coords(coords, 2u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  set_stage_coords(coords, 3u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 2u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 3u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
}

kernel void
tma_roundtrip_b8_s8_kernel(__global uint *g2s_desc,
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
  set_stage_coords(coords, 1u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 1u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  set_stage_coords(coords, 2u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  set_stage_coords(coords, 3u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 2u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 3u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  set_stage_coords(coords, 4u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  set_stage_coords(coords, 5u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 4u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 5u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  set_stage_coords(coords, 6u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  set_stage_coords(coords, 7u, lid);
  ISSUE_TMA_G2S_BUF(in1, g2s_desc, coords);
  TMA_WAIT_ALL();
  COMPUTE_TILE_BUF(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 6u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  COMPUTE_TILE_BUF(in1, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_BUF(out0, output, 7u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
}

