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
      /* Place operands */
      "mv   x10, %[src]\n\t"
      "mv   x11, %[dst]\n\t"
      "mv   x12, %[size]\n\t"
      /* CP_ASYNC_BULK: rs1=x10, rs2=x12, rd=x11 */
      ".word 0x00c515c2\n\t"
      /* CP_ASYNC_FENCE */
      ".word 0x00006042\n\t"
      : [src] "=&r"(dst_addr), [tmp] "=&r"(tmp)
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
