/* Multi-workgroup descriptor-form TMA S2G kernel. */

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

  uint coords_ptr;
  uint tmp;
  __asm__ volatile(
    "csrr %[tmp], 0x803\n\t"
    "lw   %[tmp], 4(%[tmp])\n\t"
    "lw   %[coords], 4(%[tmp])\n\t"
    "csrr %[tmp], 0x808\n\t"
    "slli %[tmp], %[tmp], 7\n\t"
    "add  %[coords], %[coords], %[tmp]\n\t"
    "vid.v v12\n\t"
    "vsll.vi v12, v12, 2\n\t"
    "vadd.vx v12, v12, %[coords]\n\t"
    "vlw12.v v12, 0(v12)\n\t"
    : [coords] "=&r"(coords_ptr), [tmp] "=&r"(tmp)
    :
    : "memory");

  if (lid == 0) {
    uint shared_addr = (uint)shared_buf;
    uint desc_ptr;
    uint dst_ptr;
    uint offset;
    __asm__ volatile(
      "csrr %[offset], 0x803\n\t"
      "lw   %[offset], 4(%[offset])\n\t"
      "lw   %[desc], 0(%[offset])\n\t"
      "lw   %[dst], 8(%[offset])\n\t"
      "csrr %[offset], 0x808\n\t"
      "slli %[offset], %[offset], 7\n\t"
      "add  %[desc], %[desc], %[offset]\n\t"
      "srli %[offset], %[offset], 1\n\t"
      "add  %[dst], %[dst], %[offset]\n\t"
      "sw   %[dst], 8(%[desc])\n\t"
      "mv   x10, %[shared]\n\t"
      "mv   x11, %[desc]\n\t"
      ".word 0x00C5C542\n\t"
      ".word 0x00006042\n\t"
      : [desc] "=&r"(desc_ptr), [dst] "=&r"(dst_ptr),
        [offset] "=&r"(offset)
      : [shared] "r"(shared_addr)
      : "memory");
  }
}
