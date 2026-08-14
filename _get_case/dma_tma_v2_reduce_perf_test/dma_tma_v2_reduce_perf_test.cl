/*
 * TMA tensor S2G reduce performance comparison.
 *
 * The TMA path lets each workgroup contribute one shared-memory tile directly
 * to the same global tile.  The baseline writes one private partial tile per
 * workgroup and launches a second kernel to reduce those partials.  Shared
 * initialization, descriptor setup, validation copies, and host dispatch are
 * outside the measured mcycle windows.
 */

#include "../common/ventus_tma_v2_opencl.h"

#define WG_SIZE 32u
#define MAX_TILE_WORDS 128u

static uint
read_cycle_lo(void)
{
  uint value;
  __asm__ volatile("csrr %0, 0xB00\n\t" : "=r"(value) :: "memory");
  return value;
}

static __local uint *
align_local_128(__local uchar *base)
{
  return (__local uint *)(((uint)base + 127u) & ~127u);
}

kernel void
patch_descriptor_base(__global uint *descriptor, __global uint *output)
{
  if (get_global_id(0) == 0u) descriptor[2] = (uint)output;
}

kernel void
tma_reduce_add_perf(__global uint *descriptor,
                    __global uint *status,
                    __global uint *times,
                    uint words)
{
  __local uchar raw[MAX_TILE_WORDS * 4u + 128u];
  __local int local_coords[WG_SIZE];
  __local uint *shared = align_local_128(raw);
  uint lid = get_local_id(0);
  uint group = get_group_id(0);

  for (uint index = lid; index < words; index += WG_SIZE)
    shared[index] = group + 1u;
  local_coords[lid] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  VENTUS_TMA_LOAD_COORDS_V12(local_coords);

  if (lid == 0u) times[group * 2u] = read_cycle_lo();
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
    VENTUS_TMA_TENSOR_REDUCE_ADD(shared, descriptor);
    VENTUS_TMA_S2G_COMMIT_GROUP();
    VENTUS_TMA_S2G_WAIT_GROUP0();
    times[group * 2u + 1u] = read_cycle_lo();
    VENTUS_TMA_STATUS_READ(status[group]);
  }
}

kernel void
write_partial_perf(__global uint *partials,
                   __global uint *times,
                   uint words)
{
  __local uint shared[MAX_TILE_WORDS];
  uint lid = get_local_id(0);
  uint group = get_group_id(0);

  for (uint index = lid; index < words; index += WG_SIZE)
    shared[index] = group + 1u;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0u) times[group * 2u] = read_cycle_lo();
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint index = lid; index < words; index += WG_SIZE)
    partials[group * words + index] = shared[index];
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (lid == 0u) times[group * 2u + 1u] = read_cycle_lo();
}

kernel void
finalize_partial_perf(__global const uint *partials,
                      __global uint *output,
                      __global uint *times,
                      uint words,
                      uint splits)
{
  uint lid = get_local_id(0);
  if (lid == 0u) times[0] = read_cycle_lo();
  barrier(CLK_GLOBAL_MEM_FENCE);

  for (uint index = lid; index < words; index += WG_SIZE) {
    uint sum = 0u;
    for (uint split = 0u; split < splits; ++split)
      sum += partials[split * words + index];
    output[index] += sum;
  }

  barrier(CLK_GLOBAL_MEM_FENCE);
  if (lid == 0u) times[1] = read_cycle_lo();
}
