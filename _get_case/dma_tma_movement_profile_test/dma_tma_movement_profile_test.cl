/*
 * DMA/TMA movement-window profile microbench.
 *
 * One common measured kernel keeps setup/timing/post-copy structure identical
 * and selects only the movement method with runtime path/pair arguments:
 *   manual G2S: global load -> register -> shared, until all lanes finish
 *   TMA/Bulk G2S: DMA issue -> wait_group completion
 *   manual S2G: shared load -> register -> global store, until all lanes finish
 *   TMA/Bulk S2G: DMA issue -> wait_group completion
 *   Tensor roundtrip: tensor G2S -> shared, then tensor/manual S2G -> global
 *
 * Setup, local source initialization, tensor coordinate loading, and validation
 * copies are outside the timed window.
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
#define TILE_BYTES (TILE_WORDS * 4u)
#define WG_SIZE 32u

#define PAIR_BULK_G2S 0u
#define PAIR_TENSOR_G2S 1u
#define PAIR_BULK_S2G 2u
#define PAIR_TENSOR_S2G 3u
#define PAIR_TENSOR_G2S_S2G 4u

#define PATH_MANUAL_G2S 0u
#define PATH_BULK_G2S 1u
#define PATH_TENSOR_G2S 2u
#define PATH_MANUAL_S2G 3u
#define PATH_BULK_S2G 4u
#define PATH_TENSOR_S2G 5u
#define PATH_MANUAL_G2S_S2G 6u
#define PATH_TENSOR_G2S_S2G 7u

static uint
read_cycle_lo(void)
{
  uint v;
  __asm__ volatile("csrr %0, 0xB00\n\t" : "=r"(v) :: "memory");
  return v;
}

static uint
pattern_word(uint idx)
{
  uint base = idx << 2;
  uint b0 = (base * 7u + 0x23u) & 0xffu;
  uint b1 = ((base + 1u) * 7u + 0x23u) & 0xffu;
  uint b2 = ((base + 2u) * 7u + 0x23u) & 0xffu;
  uint b3 = ((base + 3u) * 7u + 0x23u) & 0xffu;
  return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

kernel void
setup_desc_kernel(__global uint *desc,
                  __global const uint *input,
                  __global uint *output,
                  uint desc_base_select)
{
  if (get_global_id(0) == 0) {
    desc[2] = desc_base_select ? (uint)output : (uint)input;
    desc[DESC_WORDS + 2u] = (uint)output;
  }
}

static void
fill_shared_pattern(__local uint *shared, uint lid)
{
  for (uint idx = lid; idx < TILE_WORDS; idx += WG_SIZE) {
    shared[idx] = pattern_word(idx);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
}

static void
copy_shared_to_output(__local uint *shared, __global uint *output, uint lid)
{
  for (uint idx = lid; idx < TILE_WORDS; idx += WG_SIZE) {
    output[idx] = shared[idx];
  }
}

#define DMA_COMMIT_GROUP() do {                                              \
  __asm__ volatile(".word 0x00086042\n\t" ::: "memory");                  \
} while (0)

#define DMA_WAIT_GROUP0_RAW() do {                                           \
  __asm__ volatile(".word 0x000c6042\n\t" ::: "memory");                  \
} while (0)

#define LOAD_TENSOR_COORDS(COORDS) do {                                      \
  uint coords_ptr = (uint)(COORDS);                                          \
  __asm__ volatile(                                                          \
    "vid.v v12\n\t"                                                        \
    "vsll.vi v12, v12, 2\n\t"                                              \
    "vadd.vx v12, v12, %[coords]\n\t"                                     \
    "vlw12.v v12, 0(v12)\n\t"                                             \
    :                                                                        \
    : [coords] "r"(coords_ptr)                                              \
    : "memory"                                                             \
  );                                                                         \
} while (0)

#define ISSUE_BULK_G2S(DST, SRC, SIZE) do {                                  \
  __asm__ volatile(                                                          \
    ".insn r 0x42, 1, 0, %[dst], %[src], %[size]\n\t"                     \
    :                                                                        \
    : [dst] "r"(DST), [src] "r"(SRC), [size] "r"(SIZE)                 \
    : "memory"                                                             \
  );                                                                         \
} while (0)

#define ISSUE_BULK_S2G(DST, SRC, SIZE) do {                                  \
  __asm__ volatile(                                                          \
    ".insn r 0x42, 3, 0, %[dst], %[src], %[size]\n\t"                     \
    :                                                                        \
    : [dst] "r"(DST), [src] "r"(SRC), [size] "r"(SIZE)                 \
    : "memory"                                                             \
  );                                                                         \
} while (0)

#define ISSUE_TENSOR_G2S(DST, DESC) do {                                     \
  uint smem = (uint)(DST);                                                   \
  uint desc_ptr = (uint)(DESC);                                              \
  __asm__ volatile(                                                          \
    "mv x10, %[smem]\n\t"                                                  \
    "mv x11, %[desc]\n\t"                                                  \
    ".word 0x00C5A542\n\t"                                                 \
    :                                                                        \
    : [smem] "r"(smem), [desc] "r"(desc_ptr)                              \
    : "x10", "x11", "memory"                                            \
  );                                                                         \
} while (0)

#define ISSUE_TENSOR_S2G(SRC, DESC) do {                                     \
  uint smem = (uint)(SRC);                                                   \
  uint desc_ptr = (uint)(DESC);                                              \
  __asm__ volatile(                                                          \
    "mv x10, %[smem]\n\t"                                                  \
    "mv x11, %[desc]\n\t"                                                  \
    ".word 0x00C5C542\n\t"                                                 \
    :                                                                        \
    : [smem] "r"(smem), [desc] "r"(desc_ptr)                              \
    : "x10", "x11", "memory"                                            \
  );                                                                         \
} while (0)

kernel void
movement_profile_kernel(__global uint *desc,
                        __global const uint *coords,
                        __global const uint *input,
                        __global uint *output,
                        __global uint *cycles,
                        uint path_kind,
                        uint pair_kind,
                        uint use_dma)
{
  __local uint shared[TILE_WORDS];
  __local uint t0;
  __local uint t1;
  uint lid = get_local_id(0);
  uint is_s2g = (pair_kind == PAIR_BULK_S2G) ||
                (pair_kind == PAIR_TENSOR_S2G);
  uint is_g2s_only = (pair_kind == PAIR_BULK_G2S) ||
                     (pair_kind == PAIR_TENSOR_G2S);
  uint is_tensor_pair = (pair_kind == PAIR_TENSOR_G2S) ||
                        (pair_kind == PAIR_TENSOR_S2G) ||
                        (pair_kind == PAIR_TENSOR_G2S_S2G);
  (void)path_kind;

  if (is_s2g) {
    fill_shared_pattern(shared, lid);
  }
  if (is_tensor_pair) {
    LOAD_TENSOR_COORDS(coords);
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0u) t0 = read_cycle_lo();
  barrier(CLK_LOCAL_MEM_FENCE);

  if (pair_kind == PAIR_BULK_G2S) {
    if (use_dma) {
      if (lid == 0u) {
        ISSUE_BULK_G2S((uint)shared, (uint)input, TILE_BYTES);
        DMA_COMMIT_GROUP();
        DMA_WAIT_GROUP0_RAW();
      }
    } else {
      volatile __global const uint *vin = (volatile __global const uint *)input;
      for (uint idx = lid; idx < TILE_WORDS; idx += WG_SIZE) {
        shared[idx] = vin[idx];
      }
    }
  } else if (pair_kind == PAIR_TENSOR_G2S) {
    if (use_dma) {
      if (lid == 0u) {
        ISSUE_TENSOR_G2S(shared, desc);
        DMA_COMMIT_GROUP();
        DMA_WAIT_GROUP0_RAW();
      }
    } else {
      volatile __global const uint *vin = (volatile __global const uint *)input;
      for (uint idx = lid; idx < TILE_WORDS; idx += WG_SIZE) {
        shared[idx] = vin[idx];
      }
    }
  } else if (pair_kind == PAIR_BULK_S2G) {
    if (use_dma) {
      if (lid == 0u) {
        ISSUE_BULK_S2G((uint)output, (uint)shared, TILE_BYTES);
        DMA_COMMIT_GROUP();
        DMA_WAIT_GROUP0_RAW();
      }
    } else {
      volatile __global uint *vout = (volatile __global uint *)output;
      for (uint idx = lid; idx < TILE_WORDS; idx += WG_SIZE) {
        vout[idx] = shared[idx];
      }
    }
  } else if (pair_kind == PAIR_TENSOR_S2G) {
    if (use_dma) {
      if (lid == 0u) {
        ISSUE_TENSOR_S2G(shared, desc);
        DMA_COMMIT_GROUP();
        DMA_WAIT_GROUP0_RAW();
      }
    } else {
      volatile __global uint *vout = (volatile __global uint *)output;
      for (uint idx = lid; idx < TILE_WORDS; idx += WG_SIZE) {
        vout[idx] = shared[idx];
      }
    }
  } else if (pair_kind == PAIR_TENSOR_G2S_S2G) {
    if (use_dma) {
      if (lid == 0u) {
        ISSUE_TENSOR_G2S(shared, desc);
        DMA_COMMIT_GROUP();
        DMA_WAIT_GROUP0_RAW();
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if (lid == 0u) t1 = read_cycle_lo();
      barrier(CLK_LOCAL_MEM_FENCE);
      if (lid == 0u) {
        ISSUE_TENSOR_S2G(shared, desc + DESC_WORDS);
        DMA_COMMIT_GROUP();
        DMA_WAIT_GROUP0_RAW();
      }
    } else {
      volatile __global const uint *vin = (volatile __global const uint *)input;
      volatile __global uint *vout = (volatile __global uint *)output;
      for (uint idx = lid; idx < TILE_WORDS; idx += WG_SIZE) {
        shared[idx] = vin[idx];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if (lid == 0u) t1 = read_cycle_lo();
      barrier(CLK_LOCAL_MEM_FENCE);
      for (uint idx = lid; idx < TILE_WORDS; idx += WG_SIZE) {
        vout[idx] = shared[idx];
      }
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  if (lid == 0u) {
    uint t2 = read_cycle_lo();
    cycles[0] = t2 - t0;
    if (pair_kind == PAIR_TENSOR_G2S_S2G) {
      cycles[1] = t1 - t0;
      cycles[2] = t2 - t1;
      cycles[3] = 0u;
    }
  }

  if (is_g2s_only) {
    barrier(CLK_LOCAL_MEM_FENCE);
    copy_shared_to_output(shared, output, lid);
  }
}
