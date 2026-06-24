/*
 * TMA serial movement performance microbench.
 *
 * One work-group executes one tile in strict order:
 * movement -> ordinary shared-memory compute -> ordinary global writeback.
 *
 * The host uses tma_pingpong_pipeline_perf_test.cl directly for the manual
 * b2/s1 baseline. This file contains only the blocking serial TMA counterpart:
 * issue current tile, wait for completion, compute, then write back. Cycle
 * breakdown is recovered out-of-band from GVM retire-log PC timestamps and the
 * compiled object; the kernel itself has no measurement marker instructions.
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
#define WG_SIZE 32u


kernel void
setup_desc_kernel(__global uint *g2s_desc,
                  __global const uint *input)
{
  if (get_global_id(0) == 0) {
    g2s_desc[2] = (uint)input;
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

#define STORE_TILE_CONST(SRC, OUTPUT, STAGE, LID) do {                       \
  uint tma_out_addr = (uint)(OUTPUT) + ((uint)(STAGE) * TILE_WORDS * 4u);     \
  __asm__ volatile("mv %[addr], %[addr]\n\t"                               \
                   : [addr] "+r"(tma_out_addr) :: "memory");               \
  volatile __global uint *tma_vout = (volatile __global uint *)tma_out_addr;  \
  for (uint tma_idx = (LID); tma_idx < TILE_WORDS; tma_idx += WG_SIZE) {      \
    tma_vout[tma_idx] = (SRC)[tma_idx];                                       \
  }                                                                           \
} while (0)


#define ISSUE_TMA_G2S_BUF(BUF, DESC, COORDS) do {                            \
  uint tma_smem = (uint)(BUF);                                                \
  uint tma_desc = (uint)(DESC);                                               \
  uint tma_coords = (uint)(COORDS);                                           \
  __asm__ volatile(                                                           \
    "vid.v v12\n\t"                                                         \
    "vsll.vi v12, v12, 2\n\t"                                               \
    "vadd.vx v12, v12, %[coords]\n\t"                                      \
    "vlw12.v v12, 0(v12)\n\t"                                               \
    "mv x10, %[smem]\n\t"                                                   \
    "mv x11, %[desc]\n\t"                                                   \
    ".word 0x00C5A542\n\t"                                                  \
    :                                                                         \
    : [smem] "r"(tma_smem), [desc] "r"(tma_desc),                           \
      [coords] "r"(tma_coords)                                               \
    : "x10", "x11", "memory"                                               \
  );                                                                          \
} while (0)

#define TMA_WAIT_ALL() do {                                                  \
  __asm__ volatile(".word 0x00006042\n\t" ::: "memory");                  \
  barrier(CLK_LOCAL_MEM_FENCE);                                              \
} while (0)


kernel void
tma_serial_pipeline_kernel(__global uint *g2s_desc,
                           __global const uint *input,
                           __global uint *output)
{
  __local uint in0[TILE_WORDS];
  __local uint out0[TILE_WORDS];
  __local uint coords[COORD_WORDS];
  uint lid = get_local_id(0);

  (void)input;
  set_stage_coords(coords, 0u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  TMA_WAIT_ALL();
  compute_tile(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  STORE_TILE_CONST(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
}
