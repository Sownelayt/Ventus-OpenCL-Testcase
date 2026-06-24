/*
 * CP_ASYNC_BULK_S2G directed kernels.
 *
 * funct3=3 ABI:
 *   rd  = global destination pointer
 *   rs1 = shared source pointer
 *   rs2 = size_bytes
 */

#define SHARED_BUF_BYTES 512

static uint
s2g_pattern_word(uint word_index)
{
  uint base = word_index << 2;
  uint b0 = (base * 7u + 0x23u) & 0xffu;
  uint b1 = ((base + 1u) * 7u + 0x23u) & 0xffu;
  uint b2 = ((base + 2u) * 7u + 0x23u) & 0xffu;
  uint b3 = ((base + 3u) * 7u + 0x23u) & 0xffu;
  return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

kernel void
shared_to_global_dma_kernel(__global uchar *dst,
                            uint src_offset,
                            uint copy_bytes,
                            uint dst_offset)
{
  __local uint shared_words[SHARED_BUF_BYTES / 4];
  int lid = get_local_id(0);
  int lsize = get_local_size(0);

  for (uint i = lid; i < SHARED_BUF_BYTES / 4; i += lsize) {
    shared_words[i] = s2g_pattern_word(i);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    uint src_addr = (uint)((__local uchar *)shared_words) + src_offset;
    uint dst_addr;
    __asm__ volatile(
      "csrr %[dst], 0x803\n\t"
      "lw   %[dst], 4(%[dst])\n\t"
      "lw   %[dst], 0(%[dst])\n\t"
      "add  %[dst], %[dst], %[dst_off]\n\t"
      "mv   x10, %[src]\n\t"
      "mv   x11, %[dst]\n\t"
      "mv   x12, %[size]\n\t"
      ".word 0x00c535c2\n\t"
      ".word 0x00006042\n\t"
      : [dst] "=&r"(dst_addr)
      : [src] "r"(src_addr), [dst_off] "r"(dst_offset),
        [size] "r"(copy_bytes)
      : "memory"
    );
  }
}

kernel void
g2s_s2g_roundtrip_kernel(__global const uchar *src,
                         __global uchar *dst,
                         uint copy_bytes)
{
  __local uchar shared_buf[SHARED_BUF_BYTES];
  int lid = get_local_id(0);
  int lsize = get_local_size(0);

  for (uint i = lid; i < SHARED_BUF_BYTES / 4; i += lsize) {
    ((__local uint *)shared_buf)[i] = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    uint src_addr;
    uint dst_addr;
    uint shared_addr = (uint)shared_buf;
    __asm__ volatile(
      "csrr %[src], 0x803\n\t"
      "lw   %[src], 4(%[src])\n\t"
      "lw   %[dst], 4(%[src])\n\t"
      "lw   %[src], 0(%[src])\n\t"
      "mv   x10, %[src]\n\t"
      "mv   x11, %[shared]\n\t"
      "mv   x12, %[size]\n\t"
      ".word 0x00c515c2\n\t"
      ".word 0x00006042\n\t"
      "mv   x10, %[shared]\n\t"
      "mv   x11, %[dst]\n\t"
      "mv   x12, %[size]\n\t"
      ".word 0x00c535c2\n\t"
      ".word 0x00006042\n\t"
      : [src] "=&r"(src_addr), [dst] "=&r"(dst_addr)
      : [shared] "r"(shared_addr), [size] "r"(copy_bytes)
      : "memory"
    );
  }
}


kernel void
shared_to_global_dual_small_kernel(__global uchar *dst,
                                   uint copy_a,
                                   uint copy_b,
                                   uint dst_offset)
{
  __local uint shared_words[SHARED_BUF_BYTES / 4];
  int lid = get_local_id(0);
  int lsize = get_local_size(0);

  for (uint i = lid; i < SHARED_BUF_BYTES / 4; i += lsize) {
    shared_words[i] = s2g_pattern_word(i);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    uint src_a = (uint)((__local uchar *)shared_words);
    uint src_b = src_a + 64u;
    uint dst_a = (uint)dst + dst_offset;
    uint dst_b = dst_a + copy_a;
    __asm__ volatile(
      ".insn r 0x42, 3, 0, %[dst], %[src], %[size]\n\t"
      :
      : [dst] "r"(dst_a), [src] "r"(src_a), [size] "r"(copy_a)
      : "memory");
    __asm__ volatile(
      ".insn r 0x42, 3, 0, %[dst], %[src], %[size]\n\t"
      :
      : [dst] "r"(dst_b), [src] "r"(src_b), [size] "r"(copy_b)
      : "memory");
    __asm__ volatile(".word 0x00006042\n\t" ::: "memory");
  }
}
