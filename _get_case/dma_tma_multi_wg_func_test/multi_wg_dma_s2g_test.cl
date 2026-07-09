/* Multi-workgroup CP_ASYNC_BULK S2G kernel. */

#define S2G_COMMIT_GROUP() do {                                             \
  __asm__ volatile(".word 0x00086042\n\t" ::: "memory");                  \
} while (0)

#define S2G_WAIT_GROUP0() do {                                               \
  __asm__ volatile(".word 0x000c6042\n\t" ::: "memory");                  \
} while (0)

#define DMA_WAIT_ALL() do {                                                  \
  __asm__ volatile(".word 0x00006042\n\t" ::: "memory");                  \
} while (0)

#define ISSUE_BULK_G2S(dst_addr, src_addr, size_bytes) do {                  \
  __asm__ volatile(                                                           \
    ".insn r 0x42, 1, 0, %[dst], %[src], %[size]\n\t"                     \
    :                                                                         \
    : [dst] "r"(dst_addr), [src] "r"(src_addr), [size] "r"(size_bytes)   \
    : "memory");                                                            \
} while (0)

#define S2G_ISSUE(dst_addr, src_addr, size_bytes) do {                       \
  __asm__ volatile(                                                           \
    ".insn r 0x42, 3, 0, %[dst], %[src], %[size]\n\t"                     \
    :                                                                         \
    : [dst] "r"(dst_addr), [src] "r"(src_addr), [size] "r"(size_bytes)   \
    : "memory");                                                            \
} while (0)

kernel void
multi_wg_dma_s2g_copy(__global int *dst, int count)
{
  __local int shared_buf[64];
  int lid = get_local_id(0);

  unsigned int gid;
  __asm__ volatile("csrr %0, 0x808" : "=r"(gid));

  if (lid < count) {
    shared_buf[lid] = (int)(0xB7000000u + gid * 0x1000u + (unsigned)lid);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    unsigned int src_addr = (unsigned int)shared_buf;
    unsigned int dst_addr;
    unsigned int tmp;
    unsigned int byte_count = (unsigned int)count * 4u;
    __asm__ volatile(
      "csrr %[dst], 0x803\n\t"
      "lw   %[dst], 4(%[dst])\n\t"
      "lw   %[dst], 0(%[dst])\n\t"
      "csrr %[tmp], 0x808\n\t"
      "mul  %[tmp], %[tmp], %[size]\n\t"
      "add  %[dst], %[dst], %[tmp]\n\t"
      "mv   x10, %[src]\n\t"
      "mv   x11, %[dst]\n\t"
      "mv   x12, %[size]\n\t"
      ".word 0x00c535c2\n\t"
      ".word 0x00006042\n\t"
      : [dst] "=&r"(dst_addr), [tmp] "=&r"(tmp)
      : [src] "r"(src_addr), [size] "r"(byte_count)
      : "memory");
  }
}

kernel void
multi_wg_dma_s2g_group_wait(__global int *dst, int count)
{
  __local int shared_buf[64];
  int lid = get_local_id(0);

  unsigned int gid;
  __asm__ volatile("csrr %0, 0x808" : "=r"(gid));

  if (lid < count) {
    shared_buf[lid] =
      (int)(0xB7100000u + gid * 0x1000u + (unsigned)lid);
    shared_buf[count + lid] =
      (int)(0xB7200000u + gid * 0x1000u + (unsigned)lid);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    unsigned int dst_addr;
    unsigned int tmp;
    unsigned int byte_count = (unsigned int)count * 4u;
    unsigned int block_bytes = byte_count * 2u;
    unsigned int src_addr = (unsigned int)shared_buf;
    __asm__ volatile(
      "csrr %[dst], 0x803\n\t"
      "lw   %[dst], 4(%[dst])\n\t"
      "lw   %[dst], 0(%[dst])\n\t"
      "csrr %[tmp], 0x808\n\t"
      "mul  %[tmp], %[tmp], %[block]\n\t"
      "add  %[dst], %[dst], %[tmp]\n\t"
      : [dst] "=&r"(dst_addr), [tmp] "=&r"(tmp)
      : [block] "r"(block_bytes)
      : "memory");
    S2G_ISSUE(dst_addr, src_addr, byte_count);
    S2G_COMMIT_GROUP();
    S2G_ISSUE(dst_addr + byte_count, src_addr + byte_count, byte_count);
    S2G_COMMIT_GROUP();
    S2G_WAIT_GROUP0();
  }
}

kernel void
multi_wg_dma_s2g_4wg_group_page(__global int *dst, int count)
{
  __local int shared_buf[128];
  int lid = get_local_id(0);

  unsigned int gid;
  __asm__ volatile("csrr %0, 0x808" : "=r"(gid));

  if (lid < count) {
    shared_buf[lid] =
      (int)(0xB7600000u + gid * 0x1000u + (unsigned)lid);
    shared_buf[count + lid] =
      (int)(0xB7700000u + gid * 0x1000u + (unsigned)lid);
    shared_buf[2 * count + lid] =
      (int)(0xB7800000u + gid * 0x1000u + (unsigned)lid);
    shared_buf[3 * count + lid] =
      (int)(0xB7900000u + gid * 0x1000u + (unsigned)lid);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    uint byte_count = (uint)count * 4u;
    uint src_base = (uint)shared_buf;
    uint dst_base;
    uint tmp0;
    uint tmp1;
    uint tmp2;

    __asm__ volatile(
      "csrr %[dst], 0x803\n\t"
      "lw   %[dst], 4(%[dst])\n\t"
      "lw   %[dst], 0(%[dst])\n\t"
      "csrr %[tmp0], 0x808\n\t"
      "slli %[tmp1], %[tmp0], 13\n\t"
      "slli %[tmp2], %[tmp0], 12\n\t"
      "add  %[tmp1], %[tmp1], %[tmp2]\n\t"
      "slli %[tmp2], %[tmp0], 6\n\t"
      "add  %[tmp1], %[tmp1], %[tmp2]\n\t"
      "add  %[dst], %[dst], %[tmp1]\n\t"
      : [dst] "=&r"(dst_base), [tmp0] "=&r"(tmp0),
        [tmp1] "=&r"(tmp1), [tmp2] "=&r"(tmp2)
      :
      : "memory");

    S2G_ISSUE(dst_base, src_base, byte_count);
    S2G_COMMIT_GROUP();
    S2G_ISSUE(dst_base + 4096u, src_base + byte_count, byte_count);
    S2G_COMMIT_GROUP();
    S2G_ISSUE(dst_base + 8192u, src_base + 2u * byte_count, byte_count);
    S2G_COMMIT_GROUP();
    S2G_ISSUE(dst_base + 12288u, src_base + 3u * byte_count, byte_count);
    S2G_COMMIT_GROUP();
    S2G_WAIT_GROUP0();
  }
}

kernel void
multi_wg_dma_mixed_g2s_s2g(__global const int *src,
                           __global int *dst,
                           int count)
{
  __local int shared_buf[128];
  int lid = get_local_id(0);

  unsigned int gid;
  __asm__ volatile("csrr %0, 0x808" : "=r"(gid));

  if (lid < count) {
    shared_buf[lid] = 0;
    shared_buf[count + lid] =
      (int)(0xB7A00000u + gid * 0x1000u + (unsigned)lid);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    uint byte_count = (uint)count * 4u;
    uint g2s_shared = (uint)shared_buf;
    uint s2g_shared = (uint)(shared_buf + count);
    uint g2s_src;
    uint dst_base;
    uint arg_base;
    uint tmp0;
    uint tmp1;

    __asm__ volatile(
      "csrr %[arg], 0x803\n\t"
      "lw   %[arg], 4(%[arg])\n\t"
      "lw   %[src], 0(%[arg])\n\t"
      "lw   %[dst], 4(%[arg])\n\t"
      "csrr %[tmp0], 0x808\n\t"
      "slli %[tmp1], %[tmp0], 6\n\t"
      "add  %[src], %[src], %[tmp1]\n\t"
      "slli %[tmp1], %[tmp0], 7\n\t"
      "add  %[dst], %[dst], %[tmp1]\n\t"
      : [arg] "=&r"(arg_base), [src] "=&r"(g2s_src),
        [dst] "=&r"(dst_base), [tmp0] "=&r"(tmp0),
        [tmp1] "=&r"(tmp1)
      :
      : "memory");

    ISSUE_BULK_G2S(g2s_shared, g2s_src, byte_count);
    S2G_ISSUE(dst_base, s2g_shared, byte_count);
    S2G_COMMIT_GROUP();
    S2G_WAIT_GROUP0();
    DMA_WAIT_ALL();
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid < count) {
    uint out_base = gid << 5;  /* count is fixed to 16 in the host suite. */
    dst[out_base + 16u + (uint)lid] = shared_buf[lid];
  }
}

kernel void
multi_wg_dma_s2g_cross_page(__global int *dst, int count)
{
  __local int shared_buf[64];
  int lid = get_local_id(0);

  unsigned int gid;
  __asm__ volatile("csrr %0, 0x808" : "=r"(gid));

  if (lid < count) {
    shared_buf[lid] =
      (int)(0xB7300000u + gid * 0x1000u + (unsigned)lid);
    shared_buf[count + lid] =
      (int)(0xB7400000u + gid * 0x1000u + (unsigned)lid);
    shared_buf[2 * count + lid] =
      (int)(0xB7500000u + gid * 0x1000u + (unsigned)lid);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    unsigned int dst_addr;
    unsigned int tmp;
    unsigned int byte_count = (unsigned int)count * 4u;
    unsigned int block_bytes = 3u * 4096u + byte_count;
    unsigned int src_addr = (unsigned int)shared_buf;
    __asm__ volatile(
      "csrr %[dst], 0x803\n\t"
      "lw   %[dst], 4(%[dst])\n\t"
      "lw   %[dst], 0(%[dst])\n\t"
      "csrr %[tmp], 0x808\n\t"
      "mul  %[tmp], %[tmp], %[block]\n\t"
      "add  %[dst], %[dst], %[tmp]\n\t"
      : [dst] "=&r"(dst_addr), [tmp] "=&r"(tmp)
      : [block] "r"(block_bytes)
      : "memory");
    S2G_ISSUE(dst_addr, src_addr, byte_count);
    S2G_COMMIT_GROUP();
    S2G_ISSUE(dst_addr + 4096u, src_addr + byte_count, byte_count);
    S2G_COMMIT_GROUP();
    S2G_ISSUE(dst_addr + 8192u, src_addr + 2u * byte_count, byte_count);
    S2G_COMMIT_GROUP();
    S2G_WAIT_GROUP0();
  }
}
