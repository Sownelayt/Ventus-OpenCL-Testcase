/*
 * Same-workgroup multi-warp DMA + fence kernel.
 *
 * Each warp leader issues CP_ASYNC_BULK operations into disjoint shared-memory
 * segments. A single CP_ASYNC_FENCE follows the per-warp DMA burst, then that
 * warp writes only its own segments back to global output.
 */

#define MAX_SHARED_BYTES 1024

kernel void
multi_warp_dma_fence_kernel(__global const uchar *src,
                            __global uchar *dst,
                            uint copy_bytes,
                            uint dmas_per_warp,
                            uint src_base_offset,
                            uint src_stride)
{
  __local uchar shared_buf[MAX_SHARED_BYTES];
  uint lid = get_local_id(0);
  uint warp_id = lid >> 5;
  uint lane_id = lid & 31;

  if (lane_id == 0) {
    uint shared_base = (uint)shared_buf;
    uint src_addr;
    uint seg;
    uint dst_addr;

    __asm__ volatile(
      "csrr %[src], 0x803\n\t"
      "lw   %[src], 4(%[src])\n\t"
      "lw   %[src], 0(%[src])\n\t"
      "csrr %[seg], 0x805\n\t"
      "mul  %[seg], %[seg], %[dpw]\n\t"
      "mul  %[dst], %[seg], %[size]\n\t"
      "add  %[dst], %[dst], %[shared]\n\t"
      "mul  %[seg], %[seg], %[stride]\n\t"
      "add  %[src], %[src], %[base]\n\t"
      "add  %[src], %[src], %[seg]\n\t"
      ".insn r 0x42, 1, 0, %[dst], %[src], %[size]\n\t"
      : [src] "=&r"(src_addr), [seg] "=&r"(seg),
        [dst] "=&r"(dst_addr)
      : [shared] "r"(shared_base), [size] "r"(copy_bytes),
        [dpw] "r"(dmas_per_warp), [base] "r"(src_base_offset),
        [stride] "r"(src_stride)
      : "memory"
    );

    __asm__ volatile(".word 0x00006042\n\t" ::: "memory");
  }

  uint warp_words = (dmas_per_warp * copy_bytes) / 4;
  uint warp_base_word = warp_id * warp_words;
  for (uint i = lane_id; i < warp_words; i += 32) {
    ((__global uint *)dst)[warp_base_word + i] =
      ((__local uint *)shared_buf)[warp_base_word + i];
  }
}

kernel void
multi_warp_dma_fence2_kernel(__global const uchar *src,
                             __global uchar *dst,
                             uint copy_bytes,
                             uint dmas_per_warp,
                             uint src_base_offset,
                             uint src_stride)
{
  __local uchar shared_buf[MAX_SHARED_BYTES];
  uint lid = get_local_id(0);
  uint warp_id = lid >> 5;
  uint lane_id = lid & 31;

  if (lane_id == 0) {
    uint shared_base = (uint)shared_buf;
    uint src_addr;
    uint seg;
    uint dst_addr;

    __asm__ volatile(
      "csrr %[src], 0x803\n\t"
      "lw   %[src], 4(%[src])\n\t"
      "lw   %[src], 0(%[src])\n\t"
      "csrr %[seg], 0x805\n\t"
      "mul  %[seg], %[seg], %[dpw]\n\t"
      "mul  %[dst], %[seg], %[size]\n\t"
      "add  %[dst], %[dst], %[shared]\n\t"
      "mul  %[seg], %[seg], %[stride]\n\t"
      "add  %[src], %[src], %[base]\n\t"
      "add  %[src], %[src], %[seg]\n\t"
      ".insn r 0x42, 1, 0, %[dst], %[src], %[size]\n\t"
      : [src] "=&r"(src_addr), [seg] "=&r"(seg),
        [dst] "=&r"(dst_addr)
      : [shared] "r"(shared_base), [size] "r"(copy_bytes),
        [dpw] "r"(dmas_per_warp), [base] "r"(src_base_offset),
        [stride] "r"(src_stride)
      : "memory"
    );

    __asm__ volatile(
      "csrr %[src], 0x803\n\t"
      "lw   %[src], 4(%[src])\n\t"
      "lw   %[src], 0(%[src])\n\t"
      "csrr %[seg], 0x805\n\t"
      "mul  %[seg], %[seg], %[dpw]\n\t"
      "addi %[seg], %[seg], 1\n\t"
      "mul  %[dst], %[seg], %[size]\n\t"
      "add  %[dst], %[dst], %[shared]\n\t"
      "mul  %[seg], %[seg], %[stride]\n\t"
      "add  %[src], %[src], %[base]\n\t"
      "add  %[src], %[src], %[seg]\n\t"
      ".insn r 0x42, 1, 0, %[dst], %[src], %[size]\n\t"
      : [src] "=&r"(src_addr), [seg] "=&r"(seg),
        [dst] "=&r"(dst_addr)
      : [shared] "r"(shared_base), [size] "r"(copy_bytes),
        [dpw] "r"(dmas_per_warp), [base] "r"(src_base_offset),
        [stride] "r"(src_stride)
      : "memory"
    );

    __asm__ volatile(".word 0x00006042\n\t" ::: "memory");
  }

  uint warp_words = (dmas_per_warp * copy_bytes) / 4;
  uint warp_base_word = warp_id * warp_words;
  for (uint i = lane_id; i < warp_words; i += 32) {
    ((__global uint *)dst)[warp_base_word + i] =
      ((__local uint *)shared_buf)[warp_base_word + i];
  }
}
