#include "../common/ventus_tma_v2_opencl.h"

#define WG_SIZE 32u
#define TENSOR_SHARED_CAPACITY 8192u

/*
 * A 1024-byte base alignment makes the CUDA swizzle base phase zero for
 * every supported 32/64/128-byte span.  The RTL has separate non-zero phase
 * tests; keeping this end-to-end kernel phase-neutral makes its host reference
 * independent of the compiler's ordering of local allocations.
 */
static __local uchar *align_local_1024(__local uchar *base)
{
  return (__local uchar *)(((uint)base + 1023u) & ~1023u);
}

static __local uint *align_local_8(__local uint *base)
{
  return (__local uint *)(((uint)base + 7u) & ~7u);
}

kernel void patch_descriptor_base(__global uint *descriptor,
                                  __global uchar *base)
{
  if (get_global_id(0) == 0u) descriptor[2] = (uint)base;
}

kernel void bulk_roundtrip(__global const uchar *input,
                           __global uchar *output,
                           __global uint *status,
                           uint bytes)
{
  __local uchar raw[TENSOR_SHARED_CAPACITY + 1024];
  __local uint barrier_raw[2];
  __local uchar *shared = align_local_1024(raw);
  __local uint *mbarrier = align_local_8(barrier_raw);
  uint lid = get_local_id(0);

  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_MBARRIER_INIT(mbarrier, 1u);
    VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(mbarrier, bytes);
    VENTUS_TMA_BULK_G2S(shared, input, bytes);
    VENTUS_TMA_MBARRIER_WAIT(mbarrier, 0u);
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0u) {
    VENTUS_TMA_BULK_S2G(output, shared, bytes);
    VENTUS_TMA_S2G_COMMIT_GROUP();
    VENTUS_TMA_S2G_WAIT_GROUP0();
    VENTUS_TMA_STATUS_READ(status[0]);
  }
}

kernel void invalid_bulk(__global const uchar *input,
                         __global uchar *output,
                         __global uchar *ordering,
                         __global uint *status,
                         uint direction,
                         uint mode)
{
  __local uchar raw[1056];
  __local uchar *shared = align_local_1024(raw);
  __local uint *shared_words = (__local uint *)shared;
  uint lid = get_local_id(0);
  if (lid < 8u)
    shared_words[lid] = 0xa3a2a1a0u + 0x04040404u * lid;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0u) {
    uint bytes = mode == 2u ? 20u : (mode == 3u ? 0u : 16u);
    VENTUS_TMA_STATUS_CLEAR();
    if (direction == 0u) {
      __global const uchar *source = input + (mode == 0u ? 4u : 0u);
      __local uchar *destination = shared + (mode == 1u ? 4u : 0u);
      VENTUS_TMA_BULK_G2S(destination, source, bytes);
      /* A following legal S2G completion orders the rejected command and
       * makes any accidental shared-memory write observable by the host. */
      VENTUS_TMA_BULK_S2G(output, shared, 16u);
    } else {
      __local uchar *source = shared + (mode == 0u ? 4u : 0u);
      __global uchar *destination = output + (mode == 1u ? 4u : 0u);
      VENTUS_TMA_BULK_S2G(destination, source, bytes);
      VENTUS_TMA_BULK_S2G(ordering, shared, 16u);
    }
    VENTUS_TMA_S2G_COMMIT_GROUP();
    VENTUS_TMA_S2G_WAIT_GROUP0();
    VENTUS_TMA_STATUS_READ(status[0]);
  }
}

kernel void tensor_g2s(__global uint *descriptor,
                       __global const int *coordinates,
                       __global uchar *readback,
                       __global uint *status,
                       uint shared_bytes,
                       uint transaction_bytes,
                       uint prefetch)
{
  __local uchar raw[TENSOR_SHARED_CAPACITY + 1024];
  __local int local_coords[32];
  __local uint barrier_raw[2];
  __local uchar *shared = align_local_1024(raw);
  __local uint *shared_words = (__local uint *)shared;
  __global uint *readback_words = (__global uint *)readback;
  __local uint *mbarrier = align_local_8(barrier_raw);
  uint lid = get_local_id(0);
  for (uint i = lid; i < shared_bytes; i += WG_SIZE) shared[i] = 0;
  local_coords[lid] = lid < 5u ? coordinates[lid] : 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  VENTUS_TMA_LOAD_COORDS_V12(local_coords);

  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();
    if (prefetch) VENTUS_TMA_PREFETCH_TENSORMAP(descriptor);
    VENTUS_TMA_MBARRIER_INIT(mbarrier, 1u);
    VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(mbarrier, transaction_bytes);
    VENTUS_TMA_TENSOR_G2S(shared, descriptor);
    VENTUS_TMA_MBARRIER_WAIT(mbarrier, 0u);
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint i = lid; i < shared_bytes / 4u; i += WG_SIZE)
    readback_words[i] = shared_words[i];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0u) VENTUS_TMA_STATUS_READ(status[0]);
}

kernel void tensor_s2g(__global uint *descriptor,
                       __global const int *coordinates,
                       __global const uchar *shared_seed,
                       __global uint *status,
                       uint shared_bytes)
{
  __local uchar raw[TENSOR_SHARED_CAPACITY + 1024];
  __local int local_coords[32];
  __local uchar *shared = align_local_1024(raw);
  __local uint *shared_words = (__local uint *)shared;
  __global const uint *seed_words = (__global const uint *)shared_seed;
  uint lid = get_local_id(0);
  for (uint i = lid; i < shared_bytes / 4u; i += WG_SIZE)
    shared_words[i] = seed_words[i];
  local_coords[lid] = lid < 5u ? coordinates[lid] : 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  VENTUS_TMA_LOAD_COORDS_V12(local_coords);
  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
    VENTUS_TMA_TENSOR_S2G(shared, descriptor);
    VENTUS_TMA_S2G_COMMIT_GROUP();
    VENTUS_TMA_S2G_WAIT_GROUP0();
    VENTUS_TMA_STATUS_READ(status[0]);
  }
}

kernel void tensor_s2g_reduce(__global uint *descriptor,
                              __global const int *coordinates,
                              __global const uint *shared_seed,
                              __global uint *status,
                              uint shared_words_count,
                              uint reduce_mode)
{
  __local uchar raw[TENSOR_SHARED_CAPACITY + 1024];
  __local int local_coords[32];
  __local uchar *shared = align_local_1024(raw);
  __local uint *shared_words = (__local uint *)shared;
  uint lid = get_local_id(0);
  for (uint i = lid; i < shared_words_count; i += WG_SIZE)
    shared_words[i] = shared_seed[i];
  local_coords[lid] = lid < 5u ? coordinates[lid] : 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  VENTUS_TMA_LOAD_COORDS_V12(local_coords);
  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
    switch (reduce_mode) {
      case 1u: VENTUS_TMA_TENSOR_REDUCE_ADD(shared, descriptor); break;
      case 2u: VENTUS_TMA_TENSOR_REDUCE_MIN(shared, descriptor); break;
      case 3u: VENTUS_TMA_TENSOR_REDUCE_MAX(shared, descriptor); break;
      case 4u: VENTUS_TMA_TENSOR_REDUCE_AND(shared, descriptor); break;
      case 5u: VENTUS_TMA_TENSOR_REDUCE_OR(shared, descriptor); break;
      case 6u: VENTUS_TMA_TENSOR_REDUCE_XOR(shared, descriptor); break;
      default: break;
    }
    VENTUS_TMA_S2G_COMMIT_GROUP();
    VENTUS_TMA_S2G_WAIT_GROUP0();
    VENTUS_TMA_STATUS_READ(status[0]);
  }
}

kernel void bulk_s2g_reduce(__global uint *output,
                            __global const uint *shared_seed,
                            __global uint *status,
                            uint word_count,
                            uint reduce_mode,
                            uint reduce_type)
{
  __local uchar raw[TENSOR_SHARED_CAPACITY + 1024];
  __local uchar *shared = align_local_1024(raw);
  __local uint *shared_words = (__local uint *)shared;
  uint lid = get_local_id(0);
  for (uint i = lid; i < word_count; i += WG_SIZE)
    shared_words[i] = shared_seed[i];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0u) {
    uint bytes = word_count * 4u;
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
    if (reduce_mode == 1u && reduce_type == 0u)
      VENTUS_TMA_BULK_REDUCE_ADD_U32(output, shared, bytes);
    else if (reduce_mode == 1u && reduce_type == 1u)
      VENTUS_TMA_BULK_REDUCE_ADD_S32(output, shared, bytes);
    else if (reduce_mode == 2u && reduce_type == 0u)
      VENTUS_TMA_BULK_REDUCE_MIN_U32(output, shared, bytes);
    else if (reduce_mode == 2u && reduce_type == 1u)
      VENTUS_TMA_BULK_REDUCE_MIN_S32(output, shared, bytes);
    else if (reduce_mode == 3u && reduce_type == 0u)
      VENTUS_TMA_BULK_REDUCE_MAX_U32(output, shared, bytes);
    else if (reduce_mode == 3u && reduce_type == 1u)
      VENTUS_TMA_BULK_REDUCE_MAX_S32(output, shared, bytes);
    else if (reduce_mode == 4u && reduce_type == 2u)
      VENTUS_TMA_BULK_REDUCE_AND_B32(output, shared, bytes);
    else if (reduce_mode == 5u && reduce_type == 2u)
      VENTUS_TMA_BULK_REDUCE_OR_B32(output, shared, bytes);
    else if (reduce_mode == 6u && reduce_type == 2u)
      VENTUS_TMA_BULK_REDUCE_XOR_B32(output, shared, bytes);
    VENTUS_TMA_S2G_COMMIT_GROUP();
    VENTUS_TMA_S2G_WAIT_GROUP0();
    VENTUS_TMA_STATUS_READ(status[0]);
  }
}

kernel void invalid_bulk_reduce_encoding(__global uint *output,
                                         __global const uint *shared_seed,
                                         __global uint *status,
                                         uint variant)
{
  __local uchar raw[TENSOR_SHARED_CAPACITY + 1024];
  __local uchar *shared = align_local_1024(raw);
  __local uint *shared_words = (__local uint *)shared;
  uint lid = get_local_id(0);
  if (lid < 8u) shared_words[lid] = shared_seed[lid];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
    if (variant == 0u) {
      /* ADD+B32 is not a CUDA-legal operation/type pair. */
      VENTUS_TMA_BULK_REDUCE_WORD(output, shared, 32u, "0x18");
    } else {
      /* reduce mode 7 is reserved. */
      VENTUS_TMA_BULK_REDUCE_WORD(output, shared, 32u, "0x70");
    }
    VENTUS_TMA_S2G_COMMIT_GROUP();
    VENTUS_TMA_S2G_WAIT_GROUP0();
    VENTUS_TMA_STATUS_READ(status[0]);
  }
}

kernel void tensormap_invalidate_reload(__global uint *descriptor,
                                        __global const int *coordinates,
                                        __global const uchar *first,
                                        __global const uchar *second,
                                        __global uchar *readback,
                                        __global uint *status)
{
  __local uchar raw[1280];
  __local int local_coords[32];
  __local uint barrier_raw[2];
  __local uchar *shared = align_local_1024(raw);
  __local uint *mbarrier = align_local_8(barrier_raw);
  uint lid = get_local_id(0);
  local_coords[lid] = lid < 5u ? coordinates[lid] : 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  VENTUS_TMA_LOAD_COORDS_V12(local_coords);

  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();
    descriptor[2] = (uint)first;
    __asm__ volatile("fence rw, rw" ::: "memory");
    VENTUS_TMA_PREFETCH_TENSORMAP(descriptor);
    VENTUS_TMA_MBARRIER_INIT(mbarrier, 1u);
    VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(mbarrier, 128u);
    VENTUS_TMA_TENSOR_G2S(shared, descriptor);
    VENTUS_TMA_MBARRIER_WAIT(mbarrier, 0u);
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
    for (uint i = 0u; i < 128u; ++i) readback[i] = shared[i];

    descriptor[2] = (uint)second;
    __asm__ volatile("fence rw, rw" ::: "memory");
    VENTUS_TMA_INVALIDATE_TENSORMAP(descriptor);
    VENTUS_TMA_MBARRIER_INIT(mbarrier, 1u);
    VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(mbarrier, 128u);
    VENTUS_TMA_TENSOR_G2S(shared, descriptor);
    VENTUS_TMA_MBARRIER_WAIT(mbarrier, 0u);
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
    /*
     * Keep both snapshots in the same single-lane control region.  GVM has a
     * pre-existing reconvergence defect after this long lane-0-only sequence:
     * a work-group-parallel copy here can inherit a sparse execution mask and
     * make an otherwise complete 128-byte TMA write look partial.
     */
    for (uint i = 0u; i < 128u; ++i) readback[128u + i] = shared[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0u) VENTUS_TMA_STATUS_READ(status[0]);
}

kernel void invalid_tensor(__global uint *descriptor,
                           __global const int *coordinates,
                           __global uint *status,
                           uint direction)
{
  __local uchar raw[1152];
  __local int local_coords[32];
  __local uint barrier_raw[2];
  __local uchar *shared = align_local_1024(raw);
  __local uint *mbarrier = align_local_8(barrier_raw);
  uint lid = get_local_id(0);
  local_coords[lid] = lid < 5u ? coordinates[lid] : 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  VENTUS_TMA_LOAD_COORDS_V12(local_coords);
  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();
    if (direction == 0u) {
      VENTUS_TMA_MBARRIER_INIT(mbarrier, 1u);
      VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(mbarrier, 16u);
      VENTUS_TMA_TENSOR_G2S(shared, descriptor);
      /* The rejected tensor must not consume binding credit.  A legal bulk
       * command retires that credit and also orders the sticky status read. */
      VENTUS_TMA_BULK_G2S(
          shared, (__global const uchar *)(descriptor[2]), 16u);
      VENTUS_TMA_MBARRIER_WAIT(mbarrier, 0u);
      VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
    } else {
      VENTUS_TMA_TENSOR_S2G(shared, descriptor);
      VENTUS_TMA_S2G_COMMIT_GROUP();
      VENTUS_TMA_S2G_WAIT_GROUP0();
    }
    VENTUS_TMA_STATUS_READ(status[0]);
  }
}

kernel void mixed_bidirectional(__global const uchar *g2s_input,
                                __global uchar *g2s_readback,
                                __global uchar *s2g_output,
                                __global uint *status)
{
  __local uchar raw[1408];
  __local uint barrier_raw[2];
  __local uchar *shared = align_local_1024(raw);
  __local uint *shared_words = (__local uint *)shared;
  __global uint *g2s_words = (__global uint *)g2s_readback;
  __local uint *mbarrier = align_local_8(barrier_raw);
  uint lid = get_local_id(0);
  for (uint i = lid; i < 8u; i += WG_SIZE)
    shared_words[16u + i] = 0xa3a2a1a0u + 0x04040404u * i;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
    VENTUS_TMA_MBARRIER_INIT(mbarrier, 1u);
    VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(mbarrier, 32u);
    VENTUS_TMA_BULK_G2S(shared, g2s_input, 32u);
    VENTUS_TMA_BULK_S2G(s2g_output, shared + 64u, 32u);
    VENTUS_TMA_S2G_COMMIT_GROUP();
    VENTUS_TMA_MBARRIER_WAIT(mbarrier, 0u);
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint i = lid; i < 8u; i += WG_SIZE) g2s_words[i] = shared_words[i];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0u) {
    VENTUS_TMA_S2G_WAIT_GROUP0();
    VENTUS_TMA_STATUS_READ(status[0]);
  }
}

kernel void mixed_tensor_g2s_bulk_s2g(__global uint *descriptor,
                                      __global const int *coordinates,
                                      __global uchar *tensor_readback,
                                      __global uchar *bulk_output,
                                      __global uint *status)
{
  __local uchar raw[1408];
  __local int local_coords[32];
  __local uint barrier_raw[2];
  __local uchar *shared = align_local_1024(raw);
  __local uint *shared_words = (__local uint *)shared;
  __global uint *readback_words = (__global uint *)tensor_readback;
  __local uint *mbarrier = align_local_8(barrier_raw);
  uint lid = get_local_id(0);
  local_coords[lid] = lid < 5u ? coordinates[lid] : 0;
  for (uint i = lid; i < 8u; i += WG_SIZE)
    shared_words[64u + i] = 0x63626160u + 0x04040404u * i;
  barrier(CLK_LOCAL_MEM_FENCE);
  VENTUS_TMA_LOAD_COORDS_V12(local_coords);
  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
    VENTUS_TMA_MBARRIER_INIT(mbarrier, 1u);
    VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(mbarrier, 32u);
    VENTUS_TMA_TENSOR_G2S(shared, descriptor);
    VENTUS_TMA_BULK_S2G(bulk_output, shared + 256u, 32u);
    VENTUS_TMA_S2G_COMMIT_GROUP();
    VENTUS_TMA_MBARRIER_WAIT(mbarrier, 0u);
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint i = lid; i < 8u; i += WG_SIZE)
    readback_words[i] = shared_words[i];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0u) {
    VENTUS_TMA_S2G_WAIT_GROUP0();
    VENTUS_TMA_STATUS_READ(status[0]);
  }
}

kernel void mixed_bulk_g2s_tensor_s2g(__global uint *descriptor,
                                      __global const int *coordinates,
                                      __global const uchar *bulk_input,
                                      __global uchar *bulk_readback,
                                      __global uint *status)
{
  __local uchar raw[1408];
  __local int local_coords[32];
  __local uint barrier_raw[2];
  __local uchar *shared = align_local_1024(raw);
  __local uint *shared_words = (__local uint *)shared;
  __global uint *readback_words = (__global uint *)bulk_readback;
  __local uint *mbarrier = align_local_8(barrier_raw);
  uint lid = get_local_id(0);
  local_coords[lid] = lid < 5u ? coordinates[lid] : 0;
  for (uint i = lid; i < 8u; i += WG_SIZE)
    shared_words[64u + i] = 0x93929190u + 0x04040404u * i;
  barrier(CLK_LOCAL_MEM_FENCE);
  VENTUS_TMA_LOAD_COORDS_V12(local_coords);
  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
    VENTUS_TMA_MBARRIER_INIT(mbarrier, 1u);
    VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(mbarrier, 32u);
    VENTUS_TMA_BULK_G2S(shared, bulk_input, 32u);
    VENTUS_TMA_TENSOR_S2G(shared + 256u, descriptor);
    VENTUS_TMA_S2G_COMMIT_GROUP();
    VENTUS_TMA_MBARRIER_WAIT(mbarrier, 0u);
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint i = lid; i < 8u; i += WG_SIZE)
    readback_words[i] = shared_words[i];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0u) {
    VENTUS_TMA_S2G_WAIT_GROUP0();
    VENTUS_TMA_STATUS_READ(status[0]);
  }
}

kernel void group_ring_wrap(__global uchar *output, __global uint *status)
{
  __local uchar raw[1152];
  __local uchar *shared = align_local_1024(raw);
  __local uint *shared_words = (__local uint *)shared;
  uint lid = get_local_id(0);
  for (uint i = lid; i < 16u; i += WG_SIZE)
    shared_words[i] = 0x43424140u + 0x04040404u * i;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
    VENTUS_TMA_BULK_S2G(output + 0u, shared + 0u, 16u);
    VENTUS_TMA_S2G_COMMIT_GROUP();
    VENTUS_TMA_S2G_WAIT_GROUP3();
    VENTUS_TMA_BULK_S2G(output + 16u, shared + 16u, 16u);
    VENTUS_TMA_S2G_COMMIT_GROUP();
    VENTUS_TMA_S2G_WAIT_GROUP2();
    VENTUS_TMA_BULK_S2G(output + 32u, shared + 32u, 16u);
    VENTUS_TMA_S2G_COMMIT_GROUP();
    VENTUS_TMA_S2G_WAIT_GROUP1();
    VENTUS_TMA_BULK_S2G(output + 48u, shared + 48u, 16u);
    VENTUS_TMA_S2G_COMMIT_GROUP();
    VENTUS_TMA_S2G_WAIT_GROUP0();
    VENTUS_TMA_STATUS_READ(status[0]);
  }
}

kernel void multiwarp_mbarrier(__global const uchar *input,
                               __global uchar *readback,
                               __global uint *status)
{
  __local uchar raw[1408];
  __local uint barrier_raw[8];
  __local uchar *shared = align_local_1024(raw);
  __local uint *shared_words = (__local uint *)shared;
  __global uint *readback_words = (__global uint *)readback;
  uint lid = get_local_id(0);
  __local uint *mbarrier = align_local_8(barrier_raw);
  if (lid == 0u || lid == 32u) VENTUS_TMA_STATUS_CLEAR();
  if (lid == 0u) {
    VENTUS_TMA_MBARRIER_INIT(mbarrier, 2u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0u) {
    VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(mbarrier, 32u);
    VENTUS_TMA_BULK_G2S(shared, input, 32u);
    VENTUS_TMA_MBARRIER_WAIT(mbarrier, 0u);
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
  } else if (lid == 32u) {
    VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(mbarrier, 32u);
    VENTUS_TMA_BULK_G2S(shared + 128u, input + 32u, 32u);
    VENTUS_TMA_MBARRIER_WAIT(mbarrier, 0u);
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < 8u) readback_words[lid] = shared_words[lid];
  else if (lid >= 32u && lid < 40u)
    readback_words[lid - 24u] = shared_words[32u + lid - 32u];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0u) VENTUS_TMA_STATUS_READ(status[0]);
  else if (lid == 32u) VENTUS_TMA_STATUS_READ(status[1]);
}

kernel void mbarrier_phase_reuse(__global const uchar *input,
                                 __global uchar *readback,
                                 __global uint *status)
{
  __local uchar raw[1152];
  __local uint barrier_raw[2];
  __local uchar *shared = align_local_1024(raw);
  __local uint *shared_words = (__local uint *)shared;
  __global uint *readback_words = (__global uint *)readback;
  __local uint *mbarrier = align_local_8(barrier_raw);
  uint lid = get_local_id(0);

  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_MBARRIER_INIT(mbarrier, 1u);

    VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(mbarrier, 32u);
    VENTUS_TMA_BULK_G2S(shared, input, 32u);
    VENTUS_TMA_MBARRIER_WAIT(mbarrier, 0u);

    VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(mbarrier, 32u);
    VENTUS_TMA_BULK_G2S(shared + 32u, input + 32u, 32u);
    VENTUS_TMA_MBARRIER_WAIT(mbarrier, 1u);
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint i = lid; i < 16u; i += WG_SIZE)
    readback_words[i] = shared_words[i];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0u) {
    VENTUS_TMA_STATUS_READ(status[0]);
  }
}

kernel void status_sticky_clear(__global uint *observed)
{
  __local uint barrier_raw[2];
  __local uint *mbarrier = align_local_8(barrier_raw);
  uint lid = get_local_id(0);
  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();

    /* funct6 with an invalid zimm=15: detail=15, code=invalid-group-op. */
    __asm__ volatile(".word 0x0007e042\n\t" ::: "memory");
    VENTUS_TMA_STATUS_READ(observed[0]);

    /* The later mbarrier error must not replace the first sticky error. */
    VENTUS_TMA_MBARRIER_INIT(
        (__local uint *)((__local uchar *)mbarrier + 4u), 1u);
    VENTUS_TMA_STATUS_READ(observed[1]);

    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_MBARRIER_INIT(
        (__local uint *)((__local uchar *)mbarrier + 4u), 1u);
    VENTUS_TMA_STATUS_READ(observed[2]);
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_STATUS_READ(observed[3]);
  }
}

kernel void multi_wg_roundtrip(__global const uchar *input,
                               __global uchar *output,
                               __global uint *status)
{
  __local uchar raw[1152];
  __local uint barrier_raw[2];
  __local uchar *shared = align_local_1024(raw);
  __local uint *mbarrier = align_local_8(barrier_raw);
  uint lid = get_local_id(0);
  if (lid == 0u) {
    uint args_addr;
    uint input_addr;
    uint wg;
    uint data_offset;
    __asm__ volatile(
      "csrr %[args], 0x803\n\t"
      "lw %[args], 4(%[args])\n\t"
      "lw %[input], 0(%[args])\n\t"
      "csrr %[wg], 0x808\n\t"
      "slli %[offset], %[wg], 6\n\t"
      "add %[input], %[input], %[offset]\n\t"
      : [args] "=&r"(args_addr), [input] "=&r"(input_addr),
        [wg] "=&r"(wg), [offset] "=&r"(data_offset)
      : : "memory");
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_MBARRIER_INIT(mbarrier, 1u);
    VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(mbarrier, 64u);
    VENTUS_TMA_BULK_G2S(shared, input_addr, 64u);
    VENTUS_TMA_MBARRIER_WAIT(mbarrier, 0u);
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0u) {
    uint args_addr;
    uint output_addr;
    uint status_addr;
    uint wg;
    uint data_offset;
    __asm__ volatile(
      "csrr %[args], 0x803\n\t"
      "lw %[args], 4(%[args])\n\t"
      "lw %[output], 4(%[args])\n\t"
      "lw %[status], 8(%[args])\n\t"
      "csrr %[wg], 0x808\n\t"
      "slli %[offset], %[wg], 6\n\t"
      "add %[output], %[output], %[offset]\n\t"
      "slli %[wg], %[wg], 2\n\t"
      "add %[status], %[status], %[wg]\n\t"
      : [args] "=&r"(args_addr), [output] "=&r"(output_addr),
        [status] "=&r"(status_addr), [wg] "=&r"(wg),
        [offset] "=&r"(data_offset)
      : : "memory");
    VENTUS_TMA_BULK_S2G(output_addr, shared, 64u);
    VENTUS_TMA_S2G_COMMIT_GROUP();
    VENTUS_TMA_S2G_WAIT_GROUP0();
    uint status_value;
    VENTUS_TMA_STATUS_READ(status_value);
    __asm__ volatile("sw %[value], 0(%[address])\n\t"
                     : : [value] "r"(status_value),
                         [address] "r"(status_addr) : "memory");
  }
}
