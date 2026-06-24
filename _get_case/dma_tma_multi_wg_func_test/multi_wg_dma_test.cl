/*
 * Multi-workgroup DMA test kernel.
 * 2 workgroups, each WG's thread 0 uses CP_ASYNC_BULK to copy
 * its segment from global memory to shared memory, then all threads
 * read shared memory back to global output for host verification.
 *
 * WG0 copies src[0..count-1] → shared_buf → dst[0..count-1]
 * WG1 copies src[count..2*count-1] → shared_buf → dst[count..2*count-1]
 */
kernel void
multi_wg_dma_copy(__global const int *src,
                  __global int *dst,
                  int count)
{
  __local int shared_buf[64];
  int lid = get_local_id(0);

  /* Read workgroup ID via CSR 0x808 (scalar) to avoid
     compiler vector-scalar mismatch with get_group_id() */
  unsigned int gid;
  __asm__ volatile("csrr %0, 0x808" : "=r"(gid));

  int byte_count = count * 4;
  unsigned int dst_addr = (unsigned int)shared_buf;

  /* Only thread 0 of each workgroup issues DMA */
  if (lid == 0) {
    /*
     * Load src pointer from kernel args, then add group offset.
     * CSR 0x803 → metadata_base → *(+4) = args_base → *(+0) = src
     * CSR 0x808 = workgroup_id_x (scalar, read directly to avoid
     *             compiler vector-to-scalar register mismatch)
     * group_offset = wg_id * byte_count
     */
    unsigned int src_addr;
    unsigned int tmp;
    __asm__ volatile(
      /* Load kernel args base → src pointer */
      "csrr %[src], 0x803\n\t"
      "lw   %[src], 4(%[src])\n\t"
      "lw   %[src], 0(%[src])\n\t"
      /* Compute group offset: CSR 0x808 = wg_id_x */
      "csrr %[tmp], 0x808\n\t"
      "mul  %[tmp], %[tmp], %[size]\n\t"
      "add  %[src], %[src], %[tmp]\n\t"
      /* CP_ASYNC_BULK: rd=dst_shared, rs1=src_global, rs2=byte_count */
      ".insn r 0x42, 1, 0, %[dst], %[src], %[size]\n\t"
      /* CP_ASYNC_FENCE */
      ".word 0x00006042\n\t"
      : [src] "=&r"(src_addr), [tmp] "=&r"(tmp)
      : [dst] "r"(dst_addr), [size] "r"(byte_count)
      : "memory"
    );
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  /* Each thread copies its element from shared to global dst */
  if (lid < count) {
    int group_offset = gid * count;
    dst[group_offset + lid] = shared_buf[lid];
  }
}

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
    unsigned int dst_addr = (unsigned int)(dst + gid * (unsigned)count);
    unsigned int src_addr = (unsigned int)shared_buf;
    unsigned int byte_count = (unsigned int)count * 4u;
    __asm__ volatile(
      ".insn r 0x42, 3, 0, %[dst], %[src], %[size]\n\t"
      ".word 0x00006042\n\t"
      :
      : [dst] "r"(dst_addr), [src] "r"(src_addr), [size] "r"(byte_count)
      : "memory");
  }
}

kernel void
multi_wg_tma_g2s_copy(__global uint *desc,
                      __global const uint *coords,
                      __global const uint *src,
                      __global uint *dst)
{
  __local uint shared_buf[16];
  int lid = get_local_id(0);

  unsigned int gid;
  __asm__ volatile("csrr %0, 0x808" : "=r"(gid));

  if (lid == 0) {
    desc[gid * 32u + 2u] = (uint)(src + gid * 16u);
    for (int i = 0; i < 16; i++) shared_buf[i] = 0u;
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  uint smem = (uint)shared_buf;
  uint desc_ptr = (uint)(desc + gid * 32u);
  uint coords_ptr = (uint)(coords + gid * 32u);

  __asm__ volatile(
    "vid.v v12\n\t"
    "vsll.vi v12, v12, 2\n\t"
    "vadd.vx v12, v12, %[coords]\n\t"
    "vlw12.v v12, 0(v12)\n\t"
    "mv x10, %[smem]\n\t"
    "mv x11, %[desc]\n\t"
    ".word 0x00C5A542\n\t"
    ".word 0x00006042\n\t"
    :
    : [smem] "r"(smem), [desc] "r"(desc_ptr), [coords] "r"(coords_ptr)
    : "memory");
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid < 16) {
    dst[gid * 16u + (unsigned)lid] = shared_buf[lid];
  }
}

kernel void
multi_wg_tma_s2g_copy(__global uint *desc,
                      __global const uint *coords,
                      __global uint *dst)
{
  __local uint shared_buf[16];
  int lid = get_local_id(0);

  unsigned int gid;
  __asm__ volatile("csrr %0, 0x808" : "=r"(gid));

  if (lid < 16) {
    shared_buf[lid] = 0xC2000000u + gid * 0x1000u + (unsigned)lid;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  uint coords_ptr = (uint)(coords + gid * 32u);
  __asm__ volatile(
    "vid.v v12\n\t"
    "vsll.vi v12, v12, 2\n\t"
    "vadd.vx v12, v12, %[coords]\n\t"
    "vlw12.v v12, 0(v12)\n\t"
    :
    : [coords] "r"(coords_ptr)
    : "memory");

  if (lid == 0) {
    desc[gid * 32u + 2u] = (uint)(dst + gid * 16u);
    uint shared_addr = (uint)shared_buf;
    uint desc_ptr = (uint)(desc + gid * 32u);
    __asm__ volatile(
      "mv x10, %[shared]\n\t"
      "mv x11, %[desc]\n\t"
      ".word 0x00C5C542\n\t"
      ".word 0x00006042\n\t"
      :
      : [shared] "r"(shared_addr), [desc] "r"(desc_ptr)
      : "memory");
  }
}
