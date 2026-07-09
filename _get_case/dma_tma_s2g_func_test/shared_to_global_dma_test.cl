/*
 * CP_ASYNC_BULK_S2G directed kernels.
 *
 * funct3=3 ABI:
 *   rd  = global destination pointer
 *   rs1 = shared source pointer
 *   rs2 = size_bytes
 */

#define SHARED_BUF_BYTES 1024

#define S2G_COMMIT_GROUP() do {                                             \
  __asm__ volatile(".word 0x00086042\n\t" ::: "memory");                  \
} while (0)

#define S2G_WAIT_GROUP0() do {                                               \
  __asm__ volatile(".word 0x000c6042\n\t" ::: "memory");                  \
} while (0)

#define S2G_WAIT_GROUP1() do {                                               \
  __asm__ volatile(".word 0x000ce042\n\t" ::: "memory");                  \
} while (0)

#define S2G_WAIT_GROUP2() do {                                               \
  __asm__ volatile(".word 0x000d6042\n\t" ::: "memory");                  \
} while (0)

#define S2G_WAIT_GROUP3() do {                                               \
  __asm__ volatile(".word 0x000de042\n\t" ::: "memory");                  \
} while (0)

#define S2G_ISSUE(dst_addr, src_addr, size_bytes) do {                       \
  __asm__ volatile(                                                           \
    ".insn r 0x42, 3, 0, %[dst], %[src], %[size]\n\t"                     \
    :                                                                         \
    : [dst] "r"(dst_addr), [src] "r"(src_addr), [size] "r"(size_bytes)   \
    : "memory");                                                            \
} while (0)

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

kernel void
shared_to_global_four_issue_kernel(__global uchar *dst)
{
  __local uint shared_words[SHARED_BUF_BYTES / 4];
  int lid = get_local_id(0);
  int lsize = get_local_size(0);

  for (uint i = lid; i < SHARED_BUF_BYTES / 4; i += lsize) {
    shared_words[i] = s2g_pattern_word(i);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    uint src_base = (uint)((__local uchar *)shared_words);
    uint dst_base = (uint)dst;
    uint src0 = src_base;
    uint src1 = src_base + 128u;
    uint src2 = src_base + 256u;
    uint src3 = src_base + 384u;
    uint dst0 = dst_base;
    uint dst1 = dst_base + 128u;
    uint dst2 = dst_base + 256u;
    uint dst3 = dst_base + 384u;
    uint size = 128u;
    __asm__ volatile(
      ".insn r 0x42, 3, 0, %[dst], %[src], %[size]\n\t"
      :
      : [dst] "r"(dst0), [src] "r"(src0), [size] "r"(size)
      : "memory");
    __asm__ volatile(
      ".insn r 0x42, 3, 0, %[dst], %[src], %[size]\n\t"
      :
      : [dst] "r"(dst1), [src] "r"(src1), [size] "r"(size)
      : "memory");
    __asm__ volatile(
      ".insn r 0x42, 3, 0, %[dst], %[src], %[size]\n\t"
      :
      : [dst] "r"(dst2), [src] "r"(src2), [size] "r"(size)
      : "memory");
    __asm__ volatile(
      ".insn r 0x42, 3, 0, %[dst], %[src], %[size]\n\t"
      :
      : [dst] "r"(dst3), [src] "r"(src3), [size] "r"(size)
      : "memory");
    __asm__ volatile(".word 0x00006042\n\t" ::: "memory");
  }
}

kernel void
shared_to_global_six_issue_kernel(__global uchar *dst)
{
  __local uint shared_words[SHARED_BUF_BYTES / 4];
  int lid = get_local_id(0);
  int lsize = get_local_size(0);

  for (uint i = lid; i < SHARED_BUF_BYTES / 4; i += lsize) {
    shared_words[i] = s2g_pattern_word(i);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    uint src_base = (uint)((__local uchar *)shared_words);
    uint dst_base = (uint)dst;
    uint size = 128u;
    S2G_ISSUE(dst_base + 0u, src_base + 0u, size);
    S2G_ISSUE(dst_base + 128u, src_base + 128u, size);
    S2G_ISSUE(dst_base + 256u, src_base + 256u, size);
    S2G_ISSUE(dst_base + 384u, src_base + 384u, size);
    S2G_ISSUE(dst_base + 512u, src_base + 512u, size);
    S2G_ISSUE(dst_base + 640u, src_base + 640u, size);
    __asm__ volatile(".word 0x00006042\n\t" ::: "memory");
  }
}

kernel void
shared_to_global_eight_issue_kernel(__global uchar *dst)
{
  __local uint shared_words[SHARED_BUF_BYTES / 4];
  int lid = get_local_id(0);
  int lsize = get_local_size(0);

  for (uint i = lid; i < SHARED_BUF_BYTES / 4; i += lsize) {
    shared_words[i] = s2g_pattern_word(i);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    uint src_base = (uint)((__local uchar *)shared_words);
    uint dst_base = (uint)dst;
    uint size = 128u;
    S2G_ISSUE(dst_base + 0u, src_base + 0u, size);
    S2G_ISSUE(dst_base + 128u, src_base + 128u, size);
    S2G_ISSUE(dst_base + 256u, src_base + 256u, size);
    S2G_ISSUE(dst_base + 384u, src_base + 384u, size);
    S2G_ISSUE(dst_base + 512u, src_base + 512u, size);
    S2G_ISSUE(dst_base + 640u, src_base + 640u, size);
    S2G_ISSUE(dst_base + 768u, src_base + 768u, size);
    S2G_ISSUE(dst_base + 896u, src_base + 896u, size);
    __asm__ volatile(".word 0x00006042\n\t" ::: "memory");
  }
}

kernel void
shared_to_global_empty_commit_kernel(__global uchar *dst)
{
  int lid = get_local_id(0);
  if (lid == 0) {
    S2G_COMMIT_GROUP();
    S2G_WAIT_GROUP0();
  }
}

kernel void
shared_to_global_wait_oldest_kernel(__global uchar *dst)
{
  __local uint shared_words[SHARED_BUF_BYTES / 4];
  int lid = get_local_id(0);
  int lsize = get_local_size(0);

  for (uint i = lid; i < SHARED_BUF_BYTES / 4; i += lsize) {
    shared_words[i] = s2g_pattern_word(i);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    uint src_base = (uint)((__local uchar *)shared_words);
    uint dst_base = (uint)dst;
    uint src0 = src_base;
    uint src1 = src_base + 128u;
    uint dst0 = dst_base;
    uint dst1 = dst_base + 128u;
    uint size = 128u;
    __asm__ volatile(
      ".insn r 0x42, 3, 0, %[dst], %[src], %[size]\n\t"
      :
      : [dst] "r"(dst0), [src] "r"(src0), [size] "r"(size)
      : "memory");
    __asm__ volatile(
      ".insn r 0x42, 3, 0, %[dst], %[src], %[size]\n\t"
      :
      : [dst] "r"(dst1), [src] "r"(src1), [size] "r"(size)
      : "memory");
    __asm__ volatile(".word 0x0000e042\n\t" ::: "memory");
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  for (uint i = lid; i < 32u; i += lsize) {
    shared_words[i] = s2g_pattern_word(i + 64u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    uint src2 = (uint)((__local uchar *)shared_words);
    uint dst2 = (uint)dst + 256u;
    uint size = 128u;
    __asm__ volatile(
      ".insn r 0x42, 3, 0, %[dst], %[src], %[size]\n\t"
      :
      : [dst] "r"(dst2), [src] "r"(src2), [size] "r"(size)
      : "memory");
    __asm__ volatile(".word 0x00006042\n\t" ::: "memory");
  }
}

kernel void
shared_to_global_wait_group_kernel(__global uchar *dst)
{
  __local uint shared_words[SHARED_BUF_BYTES / 4];
  int lid = get_local_id(0);
  int lsize = get_local_size(0);

  for (uint i = lid; i < SHARED_BUF_BYTES / 4; i += lsize) {
    shared_words[i] = s2g_pattern_word(i);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    uint src_base = (uint)((__local uchar *)shared_words);
    uint dst_base = (uint)dst;
    uint size = 128u;
    __asm__ volatile(
      ".insn r 0x42, 3, 0, %[dst], %[src], %[size]\n\t"
      :
      : [dst] "r"(dst_base), [src] "r"(src_base), [size] "r"(size)
      : "memory");
    S2G_COMMIT_GROUP();
    __asm__ volatile(
      ".insn r 0x42, 3, 0, %[dst], %[src], %[size]\n\t"
      :
      : [dst] "r"(dst_base + 128u), [src] "r"(src_base + 128u),
        [size] "r"(size)
      : "memory");
    S2G_COMMIT_GROUP();
    __asm__ volatile(
      ".insn r 0x42, 3, 0, %[dst], %[src], %[size]\n\t"
      :
      : [dst] "r"(dst_base + 256u), [src] "r"(src_base + 256u),
        [size] "r"(size)
      : "memory");
    S2G_COMMIT_GROUP();
    S2G_WAIT_GROUP2();
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  for (uint i = lid; i < 32u; i += lsize) {
    shared_words[i] = s2g_pattern_word(i + 96u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    uint src3 = (uint)((__local uchar *)shared_words);
    uint dst3 = (uint)dst + 384u;
    uint size = 128u;
    __asm__ volatile(
      ".insn r 0x42, 3, 0, %[dst], %[src], %[size]\n\t"
      :
      : [dst] "r"(dst3), [src] "r"(src3), [size] "r"(size)
      : "memory");
    S2G_COMMIT_GROUP();
    S2G_WAIT_GROUP0();
  }
}

kernel void
shared_to_global_group_wrap_wait0123_kernel(__global uchar *dst)
{
  __local uint shared_words[SHARED_BUF_BYTES / 4];
  uint lid = get_local_id(0);
  uint lsize = get_local_size(0);

  for (uint i = lid; i < SHARED_BUF_BYTES / 4; i += lsize) {
    shared_words[i] = s2g_pattern_word(i);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    uint src_base = (uint)((__local uchar *)shared_words);
    uint dst_base = (uint)dst;
    uint size = 128u;
    S2G_ISSUE(dst_base + 0u, src_base + 0u, size);
    S2G_COMMIT_GROUP();
    S2G_ISSUE(dst_base + 128u, src_base + 128u, size);
    S2G_COMMIT_GROUP();
    S2G_ISSUE(dst_base + 256u, src_base + 256u, size);
    S2G_COMMIT_GROUP();
    S2G_ISSUE(dst_base + 384u, src_base + 384u, size);
    S2G_COMMIT_GROUP();
    S2G_WAIT_GROUP3();
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  for (uint i = lid; i < 32u; i += lsize) {
    shared_words[i] = s2g_pattern_word(i + 128u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    uint src_base = (uint)((__local uchar *)shared_words);
    uint dst_base = (uint)dst;
    uint size = 128u;
    S2G_ISSUE(dst_base + 512u, src_base + 0u, size);
    S2G_COMMIT_GROUP();
    S2G_WAIT_GROUP2();
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  for (uint i = lid; i < 32u; i += lsize) {
    shared_words[32u + i] = s2g_pattern_word(i + 160u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    uint src_base = (uint)((__local uchar *)shared_words);
    uint dst_base = (uint)dst;
    uint size = 128u;
    S2G_ISSUE(dst_base + 640u, src_base + 128u, size);
    S2G_COMMIT_GROUP();
    S2G_WAIT_GROUP1();
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  for (uint i = lid; i < 32u; i += lsize) {
    shared_words[64u + i] = s2g_pattern_word(i + 192u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    uint src_base = (uint)((__local uchar *)shared_words);
    uint dst_base = (uint)dst;
    uint size = 128u;
    S2G_ISSUE(dst_base + 768u, src_base + 256u, size);
    S2G_COMMIT_GROUP();
    S2G_WAIT_GROUP0();
  }
}

kernel void
shared_to_global_group_multi_issue_wrap_kernel(__global uchar *dst)
{
  __local uint shared_words[SHARED_BUF_BYTES / 4];
  uint lid = get_local_id(0);
  uint lsize = get_local_size(0);

  for (uint i = lid; i < SHARED_BUF_BYTES / 4; i += lsize) {
    shared_words[i] = s2g_pattern_word(i);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    uint src_base = (uint)((__local uchar *)shared_words);
    uint dst_base = (uint)dst;
    uint size = 128u;

    S2G_ISSUE(dst_base + 0u, src_base + 0u, size);
    S2G_ISSUE(dst_base + 128u, src_base + 128u, size);
    S2G_ISSUE(dst_base + 256u, src_base + 256u, size);
    S2G_COMMIT_GROUP();

    S2G_ISSUE(dst_base + 384u, src_base + 384u, size);
    S2G_COMMIT_GROUP();
    S2G_ISSUE(dst_base + 512u, src_base + 512u, size);
    S2G_COMMIT_GROUP();
    S2G_ISSUE(dst_base + 640u, src_base + 640u, size);
    S2G_COMMIT_GROUP();

    S2G_WAIT_GROUP3();

    S2G_ISSUE(dst_base + 768u, src_base + 768u, size);
    S2G_COMMIT_GROUP();
    S2G_WAIT_GROUP0();
  }
}

kernel void
shared_to_global_cross_page_tlb_abc_kernel(__global uchar *dst)
{
  __local uint shared_words[SHARED_BUF_BYTES / 4];
  uint lid = get_local_id(0);
  uint lsize = get_local_size(0);

  for (uint i = lid; i < SHARED_BUF_BYTES / 4; i += lsize) {
    shared_words[i] = s2g_pattern_word(i);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    uint src_base = (uint)((__local uchar *)shared_words);
    uint dst_base = (uint)dst;
    uint size = 128u;
    S2G_ISSUE(dst_base + 0u, src_base + 0u, size);
    S2G_COMMIT_GROUP();
    S2G_ISSUE(dst_base + 4096u, src_base + 128u, size);
    S2G_COMMIT_GROUP();
    S2G_ISSUE(dst_base + 8192u, src_base + 256u, size);
    S2G_COMMIT_GROUP();
    S2G_ISSUE(dst_base + 0u, src_base + 384u, size);
    S2G_COMMIT_GROUP();
    S2G_WAIT_GROUP0();
  }
}

kernel void
shared_to_global_multi_warp_kernel(__global uchar *dst)
{
  __local uint shared_words[SHARED_BUF_BYTES / 4];
  uint lid = get_local_id(0);
  uint lsize = get_local_size(0);
  uint lane = lid & 31u;

  for (uint i = lid; i < SHARED_BUF_BYTES / 4; i += lsize) {
    shared_words[i] = s2g_pattern_word(i);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lane == 0u) {
    uint shared_base = (uint)((__local uchar *)shared_words);
    uint dst_base = (uint)dst;
    uint src;
    uint out;
    uint warp;
    uint size = 128u;
    __asm__ volatile(
      "csrr %[warp], 0x805\n\t"
      "slli %[src], %[warp], 7\n\t"
      "add  %[src], %[src], %[shared]\n\t"
      "slli %[out], %[warp], 7\n\t"
      "add  %[out], %[out], %[dst]\n\t"
      ".insn r 0x42, 3, 0, %[out], %[src], %[size]\n\t"
      : [src] "=&r"(src), [out] "=&r"(out), [warp] "=&r"(warp)
      : [shared] "r"(shared_base), [dst] "r"(dst_base),
        [size] "r"(size)
      : "memory");
    __asm__ volatile(".word 0x00006042\n\t" ::: "memory");
  }
}
