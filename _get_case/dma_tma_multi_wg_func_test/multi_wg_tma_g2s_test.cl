/* Multi-workgroup descriptor-form TMA G2S kernel. */

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
  uint desc_ptr;
  uint coords_ptr;
  uint tmp;

  __asm__ volatile(
    "csrr %[tmp], 0x803\n\t"
    "lw   %[tmp], 4(%[tmp])\n\t"
    "lw   %[desc], 0(%[tmp])\n\t"
    "lw   %[coords], 4(%[tmp])\n\t"
    "csrr %[tmp], 0x808\n\t"
    "slli %[tmp], %[tmp], 7\n\t"
    "add  %[desc], %[desc], %[tmp]\n\t"
    "add  %[coords], %[coords], %[tmp]\n\t"
    "vid.v v12\n\t"
    "vsll.vi v12, v12, 2\n\t"
    "vadd.vx v12, v12, %[coords]\n\t"
    "vlw12.v v12, 0(v12)\n\t"
    "mv   x10, %[smem]\n\t"
    "mv   x11, %[desc]\n\t"
    ".word 0x00C5A542\n\t"
    ".word 0x00006042\n\t"
    : [desc] "=&r"(desc_ptr), [coords] "=&r"(coords_ptr),
      [tmp] "=&r"(tmp)
    : [smem] "r"(smem)
    : "memory");
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid < 16) {
    dst[gid * 16u + (unsigned)lid] = shared_buf[lid];
  }
}
