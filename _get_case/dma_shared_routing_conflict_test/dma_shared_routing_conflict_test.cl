/*
 * Shared bank-conflict + DMA response routing kernel.
 *
 * A DMA writes into dma_shared while all lanes repeatedly access conflict
 * slots at lid * 32 words, which maps the active lanes to the same shared
 * bank. The host verifies both the DMA payload and the conflict values.
 */

#define DMA_SHARED_BYTES 512
#define DMA_SHARED_WORDS (DMA_SHARED_BYTES / 4)
#define CONFLICT_WORDS   4096
#define SCRATCH_WORDS    (DMA_SHARED_WORDS + CONFLICT_WORDS)

kernel void
dma_shared_routing_conflict_kernel(__global const uchar *src,
                                   __global uchar *dma_out,
                                   __global uint *conflict_out,
                                   uint copy_bytes,
                                   uint rounds,
                                   uint src_offset)
{
  __local uint scratch[SCRATCH_WORDS];
  __local uchar *dma_shared = (__local uchar *)scratch;
  volatile __local uint *conflict = scratch + DMA_SHARED_WORDS;
  uint lid = get_local_id(0);
  uint lsize = get_local_size(0);

  for (uint i = lid; i < SCRATCH_WORDS; i += lsize) {
    scratch[i] = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    uint src_addr;
    uint dst_addr = (uint)dma_shared;
    __asm__ volatile(
      "csrr %[src], 0x803\n\t"
      "lw   %[src], 4(%[src])\n\t"
      "lw   %[src], 0(%[src])\n\t"
      "add  %[src], %[src], %[off]\n\t"
      "mv   x10, %[src]\n\t"
      "mv   x11, %[dst]\n\t"
      "mv   x12, %[size]\n\t"
      ".word 0x00c515c2\n\t"
      : [src] "=&r"(src_addr)
      : [off] "r"(src_offset), [dst] "r"(dst_addr),
        [size] "r"(copy_bytes)
      : "memory", "x10", "x11", "x12"
    );
  }

  uint slot = lid * 32;
  uint value = 0xA5000000u + lid;
  conflict[slot] = value;
  for (uint r = 0; r < rounds; r++) {
    uint cur = conflict[slot];
    cur = cur + ((r + 1) * 17u) + lid;
    conflict[slot] = cur;
    value = cur;
  }

  if (lid == 0) {
    __asm__ volatile(".word 0x00004042\n\t" ::: "memory");
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  uint words = copy_bytes / 4;
  for (uint i = lid; i < words; i += lsize) {
    ((__global uint *)dma_out)[i] = ((__local uint *)dma_shared)[i];
  }
  conflict_out[lid] = value;
}
