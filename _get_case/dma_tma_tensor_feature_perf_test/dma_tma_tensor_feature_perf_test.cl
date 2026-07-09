/*
 * Tensor TMA feature-path diagnostic performance kernel.
 *
 * The host varies the tensor descriptor and runs the same shape in both
 * directions.  direction=0 issues CP_ASYNC_TENSOR_G2S, then copies shared
 * memory to readback for validation.  direction=1 issues CP_ASYNC_TENSOR_S2G.
 */

#define FEATURE_SHARED_ELEMS 256
#define FEATURE_SHARED_BYTES (FEATURE_SHARED_ELEMS * 4u)
#define FEATURE_COORD_WORDS 32
#ifndef FEATURE_ITERATIONS
#define FEATURE_ITERATIONS 8
#endif

#define FEATURE_DIR_G2S 0u
#define FEATURE_DIR_S2G 1u

static uint
feature_read_cycle_lo(void)
{
  uint v;
  __asm__ volatile("csrr %0, 0xB00\n\t" : "=r"(v) :: "memory");
  return v;
}

static uint
feature_pattern_word(uint idx)
{
  uint base = idx << 2;
  uint b0 = (base * 7u + 3u) & 0xffu;
  uint b1 = ((base + 1u) * 7u + 3u) & 0xffu;
  uint b2 = ((base + 2u) * 7u + 3u) & 0xffu;
  uint b3 = ((base + 3u) * 7u + 3u) & 0xffu;
  return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

#define TENSOR_G2S_WAIT_ALL() do {                                          \
  __asm__ volatile(".word 0x00006042\n\t" ::: "memory");                  \
} while (0)

#define DMA_COMMIT_GROUP() do {                                             \
  __asm__ volatile(".word 0x00086042\n\t" ::: "memory");                  \
} while (0)

#define DMA_WAIT_GROUP0() do {                                               \
  __asm__ volatile(".word 0x000c6042\n\t" ::: "memory");                  \
} while (0)

kernel void
dma_tma_tensor_feature_setup_desc_kernel(__global uint *desc,
                                         __global uchar *global_buf)
{
  if (get_global_id(0) == 0) {
    desc[2] = (uint)global_buf;
  }
}

kernel void
dma_tma_tensor_feature_perf_kernel(__global uint *desc,
                                   __global const uint *coords,
                                   __global uchar *global_buf,
                                   __global uint *readback,
                                   uint direction)
{
  __local uint shared_words[FEATURE_SHARED_ELEMS];
  __local uchar *shared_buf = (__local uchar *)shared_words;
  uint lid = get_local_id(0);
  uint lsize = get_local_size(0);

  for (uint i = lid; i < FEATURE_SHARED_ELEMS; i += lsize) {
    shared_words[i] =
      direction == FEATURE_DIR_S2G ? feature_pattern_word(i) : 0u;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  uint coords_ptr = (uint)coords;
  __asm__ volatile(
    "vid.v v12\n\t"
    "vsll.vi v12, v12, 2\n\t"
    "vadd.vx v12, v12, %[coords]\n\t"
    "vlw12.v v12, 0(v12)\n\t"
    :
    : [coords] "r"(coords_ptr)
    : "memory"
  );

  uint shared_addr = (uint)shared_buf;
  uint desc_ptr = (uint)desc;

  uint start = 0;
  if (lid == 0) {
    start = feature_read_cycle_lo();
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (direction == FEATURE_DIR_G2S) {
    __asm__ volatile(
      "mv x11, %[desc]\n\t"
      ".word 0x0005D042\n\t"
      :
      : [desc] "r"(desc_ptr)
      : "x11", "memory"
    );
    __asm__ volatile(
      "mv x10, %[shared]\n\t"
      "mv x11, %[desc]\n\t"
      ".word 0x00C5A542\n\t"
      :
      : [shared] "r"(shared_addr), [desc] "r"(desc_ptr)
      : "x10", "x11", "memory"
    );
    TENSOR_G2S_WAIT_ALL();
  } else {
    if (lid == 0) {
      for (uint iter = 0; iter < FEATURE_ITERATIONS; iter++) {
        __asm__ volatile(
          "mv x10, %[shared]\n\t"
          "mv x11, %[desc]\n\t"
          ".word 0x00C5C542\n\t"
          :
          : [shared] "r"(shared_addr), [desc] "r"(desc_ptr)
          : "x10", "x11", "memory"
        );
        DMA_COMMIT_GROUP();
        DMA_WAIT_GROUP0();
      }
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  if (lid == 0) {
    desc[31] = feature_read_cycle_lo() - start;
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  if (direction == FEATURE_DIR_G2S) {
    for (uint i = lid; i < FEATURE_SHARED_ELEMS; i += lsize) {
      readback[i] = shared_words[i];
    }
  }
}
