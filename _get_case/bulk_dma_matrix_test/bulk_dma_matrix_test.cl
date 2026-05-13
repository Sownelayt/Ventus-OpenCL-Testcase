/*
 * Bulk DMA matrix kernel.
 *
 * Thread 0 issues CP_ASYNC_BULK from src + src_offset into a shared-memory
 * byte buffer at dst_offset, fences, then the workgroup writes the copied
 * bytes back to dst for host-side byte comparison.
 */

#define SHARED_BUF_BYTES 512

kernel void
bulk_dma_matrix_kernel(__global const uchar *src,
                       __global uchar *dst,
                       uint src_offset,
                       uint copy_bytes,
                       uint dst_offset)
{
  __local uchar shared_buf[SHARED_BUF_BYTES];
  int lid = get_local_id(0);
  int lsize = get_local_size(0);

  for (uint i = lid; i < SHARED_BUF_BYTES / 4; i += lsize) {
    ((__local uint *)shared_buf)[i] = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  uint dst_addr = (uint)shared_buf + dst_offset;

  if (lid == 0) {
    uint src_addr;
    __asm__ volatile(
      "csrr %[src], 0x803\n\t"
      "lw   %[src], 4(%[src])\n\t"
      "lw   %[src], 0(%[src])\n\t"
      "add  %[src], %[src], %[off]\n\t"
      "mv   x10, %[src]\n\t"
      "mv   x11, %[dst]\n\t"
      "mv   x12, %[size]\n\t"
      ".word 0x00c515c2\n\t"
      ".word 0x00004042\n\t"
      : [src] "=&r"(src_addr)
      : [off] "r"(src_offset), [dst] "r"(dst_addr),
        [size] "r"(copy_bytes)
      : "memory", "x10", "x11", "x12"
    );
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  uint words = copy_bytes / 4;
  __local uint *shared_words = (__local uint *)(shared_buf + dst_offset);
  __global uint *dst_words = (__global uint *)dst;
  for (uint i = lid; i < words; i += lsize) {
    dst_words[i] = shared_words[i];
  }
}
