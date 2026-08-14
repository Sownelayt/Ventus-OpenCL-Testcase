/*
 * Tensor TMA feature-path diagnostic performance kernel.
 *
 * The host varies the tensor descriptor and runs the same shape in both
 * directions.  direction=0 issues CP_ASYNC_TENSOR_G2S, then copies shared
 * memory to readback for validation.  direction=1 issues CP_ASYNC_TENSOR_S2G.
 */

#include "ventus_tma_v2_opencl.h"

#define FEATURE_SHARED_ELEMS 512
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

static __local uint *
feature_align_local_8(__local uint *base)
{
  return (__local uint *)(((uint)base + 7u) & ~7u);
}

static __local uint *
feature_align_local_1024(__local uchar *base)
{
  /*
   * The functional model below compares a canonical phase-zero swizzled
   * layout.  A 1024B base alignment makes (sharedBase / 128) % {2,4,8}
   * zero for every standard 32/64/128B swizzle, while dedicated Chisel
   * tests exercise all non-zero shared-base phases.
   */
  return (__local uint *)(((uint)base + 1023u) & ~1023u);
}

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
                                   __global uint *metrics,
                                   uint direction,
                                   uint transaction_bytes)
{
  __local uchar shared_raw[FEATURE_SHARED_BYTES + 1024u];
  __local uint *shared_words = feature_align_local_1024(shared_raw);
  __local uint barrier_raw[2];
  __local uint *mbarrier = feature_align_local_8(barrier_raw);
  __local uchar *shared_buf = (__local uchar *)shared_words;
  uint lid = get_local_id(0);
  uint lsize = get_local_size(0);

  for (uint i = lid; i < FEATURE_SHARED_ELEMS; i += lsize) {
    shared_words[i] =
      direction == FEATURE_DIR_S2G ? feature_pattern_word(i) : 0u;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  VENTUS_TMA_LOAD_COORDS_V12(coords);
  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();
    if (direction == FEATURE_DIR_G2S) {
      VENTUS_TMA_MBARRIER_INIT(mbarrier, 1u);
      VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(mbarrier, transaction_bytes);
    } else {
      VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
    }
  }

  uint start = 0;
  if (lid == 0) {
    start = feature_read_cycle_lo();
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (direction == FEATURE_DIR_G2S) {
    if (lid == 0u) {
      VENTUS_TMA_PREFETCH_TENSORMAP(desc);
      VENTUS_TMA_TENSOR_G2S(shared_buf, desc);
      VENTUS_TMA_MBARRIER_WAIT(mbarrier, 0u);
      VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
    }
  } else {
    if (lid == 0) {
      for (uint iter = 0; iter < FEATURE_ITERATIONS; iter++) {
        VENTUS_TMA_TENSOR_S2G(shared_buf, desc);
        VENTUS_TMA_S2G_COMMIT_GROUP();
        VENTUS_TMA_S2G_WAIT_GROUP0();
      }
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  if (lid == 0) {
    metrics[1] = feature_read_cycle_lo() - start;
    VENTUS_TMA_STATUS_READ(metrics[0]);
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  if (direction == FEATURE_DIR_G2S) {
    for (uint i = lid; i < FEATURE_SHARED_ELEMS; i += lsize) {
      readback[i] = shared_words[i];
    }
  }
}
