/*
 * TMA GEMM performance microbench.
 *
 * Both paths compute the same tiled SGEMM:
 *   C[(m_tiles*16)x(n_tiles*16)] =
 *     A[(m_tiles*16)x(k_tiles*16)] * B[(k_tiles*16)x(n_tiles*16)]
 *
 * The TMA kernel consumes one K tile per launch and writes each tile partial
 * into a separate global partial buffer. A plain device reduce kernel then sums
 * partials into the completed C tile. The default host sweep uses one
 * output-tile work-group per launch.
 * GVM/RTL currently treats two or more parallel TMA work-groups as an
 * unstable debug case, so performance reporting stays on the serial 1WG path.
 */

#define DESC_WORDS 32u
#define COORD_WORDS 32u
#define TILE_M 16u
#define TILE_N 16u
#define TILE_K 16u
#define C_TILE_WORDS (TILE_M * TILE_N)
#define TILE_WORDS (TILE_M * TILE_K)
#define WG_SIZE 32u
#define MAX_OUTPUTS_PER_WORKITEM (C_TILE_WORDS / WG_SIZE)

kernel void
setup_desc_kernel(__global uint *desc_pair,
                  __global const uint *A,
                  __global const uint *B)
{
  if (get_global_id(0) == 0) {
    desc_pair[2] = (uint)A;
    desc_pair[DESC_WORDS + 2u] = (uint)B;
  }
}

static void
accum_tile(__local uint *As, __local uint *Bs, float *acc, uint lid)
{
  uint slot = 0;
  for (uint idx = lid; idx < C_TILE_WORDS; idx += WG_SIZE) {
    uint row = idx / TILE_N;
    uint col = idx - row * TILE_N;
    float sum = acc[slot];

    for (uint kk = 0; kk < TILE_K; kk++) {
      sum += as_float(As[row * TILE_K + kk]) * as_float(Bs[kk * TILE_N + col]);
    }

    acc[slot] = sum;
    slot++;
  }
}

kernel void
manual_gemm_kernel(__global const uint *A,
                   __global const uint *B,
                   __global uint *C,
                   uint m_tiles,
                   uint n_tiles,
                   uint k_tiles,
                   uint tile_offset)
{
  __local uint As[TILE_WORDS];
  __local uint Bs[TILE_WORDS];
  volatile __global const uint *vA = (volatile __global const uint *)A;
  volatile __global const uint *vB = (volatile __global const uint *)B;
  volatile __local uint *vAs = (volatile __local uint *)As;
  volatile __local uint *vBs = (volatile __local uint *)Bs;

  uint lid = get_local_id(0);
  uint tile_id = tile_offset + get_group_id(0);
  uint tile_m = tile_id / n_tiles;
  uint tile_n = tile_id - tile_m * n_tiles;
  uint n_cols = n_tiles * TILE_N;
  uint k_cols = k_tiles * TILE_K;
  uint m_base = tile_m * TILE_M;
  uint n_base = tile_n * TILE_N;
  float acc[MAX_OUTPUTS_PER_WORKITEM];

  for (uint i = 0; i < MAX_OUTPUTS_PER_WORKITEM; i++) {
    acc[i] = 0.0f;
  }

  for (uint kt = 0; kt < k_tiles; kt++) {
    uint k_base = kt * TILE_K;

    for (uint i = lid; i < TILE_WORDS; i += WG_SIZE) {
      uint row = i / TILE_K;
      uint col = i - row * TILE_K;
      vAs[i] = vA[(m_base + row) * k_cols + k_base + col];

      row = i / TILE_N;
      col = i - row * TILE_N;
      vBs[i] = vB[(k_base + row) * n_cols + n_base + col];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    accum_tile(As, Bs, acc, lid);
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  uint slot = 0;
  for (uint idx = lid; idx < C_TILE_WORDS; idx += WG_SIZE) {
    uint row = idx / TILE_N;
    uint col = idx - row * TILE_N;
    C[(m_base + row) * n_cols + n_base + col] = as_uint(acc[slot]);
    slot++;
  }
}

kernel void
tma_gemm_kernel(__global uint *desc_pair,
                __global const uint *coords_pair,
                __global const uint *A,
                __global const uint *B,
                __global uint *partials,
                uint m_tiles,
                uint n_tiles,
                uint k_tiles,
                uint kt_offset,
                uint tile_offset)
{
  __local uint As[TILE_WORDS];
  __local uint Bs[TILE_WORDS];
  __local uint tma_coords_a[COORD_WORDS];
  __local uint tma_coords_b[COORD_WORDS];

  uint lid = get_local_id(0);
  uint tile_id = tile_offset + get_group_id(0);
  uint tile_m = tile_id / n_tiles;
  uint tile_n = tile_id - tile_m * n_tiles;
  uint kt = kt_offset;
  float acc[MAX_OUTPUTS_PER_WORKITEM];

  (void)A;
  (void)B;
  (void)m_tiles;
  (void)n_tiles;

  for (uint i = 0; i < MAX_OUTPUTS_PER_WORKITEM; i++) {
    acc[i] = 0.0f;
  }

  uint coord_base = (tile_id * k_tiles + kt) * 2u * COORD_WORDS;
  for (uint i = lid; i < COORD_WORDS; i += WG_SIZE) {
    tma_coords_a[i] = coords_pair[coord_base + i];
    tma_coords_b[i] = coords_pair[coord_base + COORD_WORDS + i];
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  uint smem_a = (uint)As;
  uint smem_b = (uint)Bs;
  uint coords_a_ptr = (uint)tma_coords_a;
  uint coords_b_ptr = (uint)tma_coords_b;
  uint desc_a_ptr = (uint)desc_pair;
  uint desc_b_ptr = (uint)(desc_pair + DESC_WORDS);

  __asm__ volatile(
    "vid.v v12\n\t"
    "vsll.vi v12, v12, 2\n\t"
    "vadd.vx v12, v12, %[coords_a]\n\t"
    "vlw12.v v12, 0(v12)\n\t"
    "mv x10, %[smem_a]\n\t"
    "mv x11, %[desc_a]\n\t"
    /* CP_ASYNC_TENSOR rd=x10, rs1=x11, rs2=v12 */
    ".word 0x00C5A542\n\t"
    /* CP_ASYNC_FENCE */
    ".word 0x00006042\n\t"
    :
    : [smem_a] "r"(smem_a), [desc_a] "r"(desc_a_ptr),
      [coords_a] "r"(coords_a_ptr)
    : "x10", "x11", "memory"
  );

  __asm__ volatile(
    "vid.v v12\n\t"
    "vsll.vi v12, v12, 2\n\t"
    "vadd.vx v12, v12, %[coords_b]\n\t"
    "vlw12.v v12, 0(v12)\n\t"
    "mv x10, %[smem_b]\n\t"
    "mv x11, %[desc_b]\n\t"
    /* CP_ASYNC_TENSOR rd=x10, rs1=x11, rs2=v12 */
    ".word 0x00C5A542\n\t"
    /* CP_ASYNC_FENCE */
    ".word 0x00006042\n\t"
    :
    : [smem_b] "r"(smem_b), [desc_b] "r"(desc_b_ptr),
      [coords_b] "r"(coords_b_ptr)
    : "x10", "x11", "memory"
  );
  barrier(CLK_LOCAL_MEM_FENCE);

  accum_tile(As, Bs, acc, lid);
  barrier(CLK_LOCAL_MEM_FENCE);

  uint partial_base = (tile_id * k_tiles + kt) * C_TILE_WORDS;
  uint slot = 0;
  for (uint idx = lid; idx < C_TILE_WORDS; idx += WG_SIZE) {
    partials[partial_base + idx] = as_uint(acc[slot]);
    slot++;
  }
}

kernel void
sum_partials_kernel(__global const uint *partials,
                    __global uint *C,
                    uint m_tiles,
                    uint n_tiles,
                    uint k_tiles,
                    uint tile_offset)
{
  uint lid = get_local_id(0);
  uint tile_id = tile_offset + get_group_id(0);
  uint tile_m = tile_id / n_tiles;
  uint tile_n = tile_id - tile_m * n_tiles;
  uint n_cols = n_tiles * TILE_N;
  uint m_base = tile_m * TILE_M;
  uint n_base = tile_n * TILE_N;

  (void)m_tiles;

  for (uint idx = lid; idx < C_TILE_WORDS; idx += WG_SIZE) {
    uint row = idx / TILE_N;
    uint col = idx - row * TILE_N;
    float sum = 0.0f;

    for (uint kt = 0; kt < k_tiles; kt++) {
      uint partial_base = (tile_id * k_tiles + kt) * C_TILE_WORDS;
      sum += as_float(partials[partial_base + idx]);
    }

    C[(m_base + row) * n_cols + n_base + col] = as_uint(sum);
  }
}
