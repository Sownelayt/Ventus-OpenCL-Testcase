/* Multi-workgroup CP_ASYNC_BULK S2G kernel. */

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
