/*
 * Temporary marker overhead microbench.
 *
 * It keeps the same single-tile manual load/compute/store body as the serial
 * TMA movement test and changes only the boundary marker implementation.
 */

#ifndef TILE_ROWS
#define TILE_ROWS 16u
#endif
#ifndef TILE_COLS
#define TILE_COLS 16u
#endif
#ifndef MARK_MODE
#define MARK_MODE 0
#endif

#define MARK_MODE_NONE 0
#define MARK_MODE_CSR_ONLY 1
#define MARK_MODE_BARRIER_ONLY 2
#define MARK_MODE_CSR_BARRIER 3

#define TILE_WORDS (TILE_ROWS * TILE_COLS)
#define WG_SIZE 32u
#define DESC_WORDS 32u
#define COORD_WORDS 32u

#define MARK_BEGIN 0x51u
#define MARK_AFTER_MOVE 0x52u
#define MARK_AFTER_COMPUTE 0x53u
#define MARK_END 0x54u

kernel void
setup_desc_kernel(__global uint *g2s_desc,
                  __global const uint *input)
{
  if (get_global_id(0) == 0) {
    g2s_desc[2] = (uint)input;
  }
}

static void
mark_boundary(uint marker, uint lid)
{
#if MARK_MODE == MARK_MODE_CSR_ONLY || MARK_MODE == MARK_MODE_CSR_BARRIER
  if (lid == 0u) {
    __asm__ volatile("csrw 0x80b, %[v]\n\t" :: [v] "r"(marker) : "memory");
  }
#else
  (void)marker;
  (void)lid;
#endif
#if MARK_MODE == MARK_MODE_BARRIER_ONLY || MARK_MODE == MARK_MODE_CSR_BARRIER
  barrier(CLK_LOCAL_MEM_FENCE);
#endif
}

static void
mark_end(uint marker, uint lid)
{
#if MARK_MODE == MARK_MODE_CSR_ONLY || MARK_MODE == MARK_MODE_CSR_BARRIER
  if (lid == 0u) {
    __asm__ volatile("csrw 0x80b, %[v]\n\t" :: [v] "r"(marker) : "memory");
  }
#else
  (void)marker;
  (void)lid;
#endif
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

#define LOAD_TILE_CONST(DST, INPUT, STAGE, LID) do {                         \
  uint load_addr = (uint)(INPUT) + ((uint)(STAGE) * TILE_WORDS * 4u);         \
  __asm__ volatile("mv %[addr], %[addr]\n\t"                               \
                   : [addr] "+r"(load_addr) :: "memory");                  \
  volatile __global const uint *load_ptr =                                    \
      (volatile __global const uint *)load_addr;                              \
  for (uint idx = (LID); idx < TILE_WORDS; idx += WG_SIZE) {                  \
    (DST)[idx] = load_ptr[idx];                                               \
  }                                                                           \
} while (0)

#define STORE_TILE_CONST(SRC, OUTPUT, STAGE, LID) do {                       \
  uint out_addr = (uint)(OUTPUT) + ((uint)(STAGE) * TILE_WORDS * 4u);         \
  __asm__ volatile("mv %[addr], %[addr]\n\t"                               \
                   : [addr] "+r"(out_addr) :: "memory");                   \
  volatile __global uint *vout = (volatile __global uint *)out_addr;          \
  for (uint idx = (LID); idx < TILE_WORDS; idx += WG_SIZE) {                  \
    vout[idx] = (SRC)[idx];                                                   \
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

#define TMA_WAIT_ALL() do {                                                   \
  __asm__ volatile(".word 0x00006042\n\t" ::: "memory");                  \
  barrier(CLK_LOCAL_MEM_FENCE);                                               \
} while (0)

kernel void
marker_overhead_manual_kernel(__global const uint *input,
                              __global uint *output)
{
  __local uint in0[TILE_WORDS];
  __local uint out0[TILE_WORDS];
  volatile __global const uint *vin = (volatile __global const uint *)input;
  uint lid = get_local_id(0);

  mark_boundary(MARK_BEGIN, lid);
  LOAD_TILE_CONST(in0, vin, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  mark_boundary(MARK_AFTER_MOVE, lid);
  compute_tile(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  mark_boundary(MARK_AFTER_COMPUTE, lid);
  STORE_TILE_CONST(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  mark_end(MARK_END, lid);
}


kernel void
marker_overhead_tma_kernel(__global uint *g2s_desc,
                           __global const uint *input,
                           __global uint *output)
{
  __local uint in0[TILE_WORDS];
  __local uint out0[TILE_WORDS];
  __local uint coords[COORD_WORDS];
  uint lid = get_local_id(0);

  (void)input;
  mark_boundary(MARK_BEGIN, lid);
  set_stage_coords(coords, 0u, lid);
  ISSUE_TMA_G2S_BUF(in0, g2s_desc, coords);
  TMA_WAIT_ALL();
  mark_boundary(MARK_AFTER_MOVE, lid);
  compute_tile(in0, out0, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  mark_boundary(MARK_AFTER_COMPUTE, lid);
  STORE_TILE_CONST(out0, output, 0u, lid);
  barrier(CLK_LOCAL_MEM_FENCE);
  mark_end(MARK_END, lid);
}
