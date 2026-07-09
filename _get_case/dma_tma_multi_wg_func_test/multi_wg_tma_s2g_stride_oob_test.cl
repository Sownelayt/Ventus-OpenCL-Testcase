/* Multi-workgroup tensor S2G stride/OOB kernel. */

#define SHARED_ELEMS 64
#define DESC_BASE_BYTES 3072u

kernel void
multi_wg_tma_s2g_stride_oob(__global const uint *desc_guard,
                            __global uint *desc,
                            __global const uint *coords,
                            __global uchar *dst)
{
  __local uint shared_buf[SHARED_ELEMS];
  int lid = get_local_id(0);

  uint gid;
  __asm__ volatile("csrr %0, 0x808" : "=r"(gid));

  for (uint i = (uint)lid; i < SHARED_ELEMS; i += get_local_size(0)) {
    shared_buf[i] = 0xC2200000u + gid * 0x1000u + i;
  }
  if (lid == 0 && desc_guard[0] == 0x13579bdfu) {
    shared_buf[0] = desc_guard[0];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  uint coords_ptr;
  uint tmp;
  __asm__ volatile(
    "csrr %[tmp], 0x803\n\t"
    "lw   %[tmp], 4(%[tmp])\n\t"
    "lw   %[coords], 8(%[tmp])\n\t"
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
    uint desc_ptr;
    uint dst_ptr;
    uint gid_reg;
    uint desc_offset;
    uint dst_offset;
    uint shared_addr = (uint)shared_buf;
    __asm__ volatile(
      "csrr %[desc_off], 0x803\n\t"
      "lw   %[desc_off], 4(%[desc_off])\n\t"
      "lw   %[desc], 4(%[desc_off])\n\t"
      "lw   %[dst], 12(%[desc_off])\n\t"
      "add  %[desc], %[desc], %[base]\n\t"
      "csrr %[gid], 0x808\n\t"
      "slli %[desc_off], %[gid], 7\n\t"
      "slli %[dst_off], %[gid], 8\n\t"
      "add  %[desc], %[desc], %[desc_off]\n\t"
      "add  %[dst], %[dst], %[dst_off]\n\t"
      "sw   %[dst], 8(%[desc])\n\t"
      "mv   x10, %[shared]\n\t"
      "mv   x11, %[desc]\n\t"
      ".word 0x00C5C542\n\t"
      ".word 0x00006042\n\t"
      : [desc] "=&r"(desc_ptr), [dst] "=&r"(dst_ptr),
        [gid] "=&r"(gid_reg), [desc_off] "=&r"(desc_offset),
        [dst_off] "=&r"(dst_offset)
      : [shared] "r"(shared_addr), [base] "r"(DESC_BASE_BYTES)
      : "memory");
  }
}
