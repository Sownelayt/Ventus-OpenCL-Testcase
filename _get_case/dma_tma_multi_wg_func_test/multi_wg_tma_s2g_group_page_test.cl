/* Multi-workgroup tensor S2G group/page kernel. */

#define SHARED_ELEMS 64
#define DESC_BASE_BYTES 2048u

#define S2G_COMMIT_GROUP() do {                                             \
  __asm__ volatile(".word 0x00086042\n\t" ::: "memory");                  \
} while (0)

#define S2G_WAIT_GROUP0() do {                                               \
  __asm__ volatile(".word 0x000c6042\n\t" ::: "memory");                  \
} while (0)

kernel void
multi_wg_tma_s2g_group_page(__global const uint *desc_guard,
                            __global uint *desc,
                            __global const uint *coords,
                            __global uchar *dst)
{
  __local uint shared_buf[SHARED_ELEMS];
  int lid = get_local_id(0);

  uint gid;
  __asm__ volatile("csrr %0, 0x808" : "=r"(gid));

  for (uint i = (uint)lid; i < 16u; i += get_local_size(0)) {
    shared_buf[i] = 0xC2400000u + gid * 0x1000u + i;
  }
  if (lid == 0 && desc_guard[0] == 0x55aa55aau) {
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
    uint tmp0;
    uint tmp1;
    uint tmp2;
    uint shared_addr = (uint)shared_buf;
    __asm__ volatile(
      "csrr %[tmp0], 0x803\n\t"
      "lw   %[tmp0], 4(%[tmp0])\n\t"
      "lw   %[desc], 4(%[tmp0])\n\t"
      "lw   %[dst], 12(%[tmp0])\n\t"
      "add  %[desc], %[desc], %[base]\n\t"
      "csrr %[gid], 0x808\n\t"
      "slli %[tmp0], %[gid], 7\n\t"
      "add  %[desc], %[desc], %[tmp0]\n\t"
      "slli %[tmp0], %[gid], 13\n\t"
      "slli %[tmp1], %[gid], 12\n\t"
      "add  %[tmp0], %[tmp0], %[tmp1]\n\t"
      "slli %[tmp2], %[gid], 6\n\t"
      "add  %[tmp0], %[tmp0], %[tmp2]\n\t"
      "add  %[tmp0], %[tmp0], %[tmp1]\n\t"
      "add  %[dst], %[dst], %[tmp0]\n\t"
      "sw   %[dst], 8(%[desc])\n\t"
      "mv   x10, %[shared]\n\t"
      "mv   x11, %[desc]\n\t"
      ".word 0x00C5C542\n\t"
      : [desc] "=&r"(desc_ptr), [dst] "=&r"(dst_ptr),
        [gid] "=&r"(gid_reg), [tmp0] "=&r"(tmp0),
        [tmp1] "=&r"(tmp1), [tmp2] "=&r"(tmp2)
      : [shared] "r"(shared_addr), [base] "r"(DESC_BASE_BYTES)
      : "memory");
    S2G_COMMIT_GROUP();
    S2G_WAIT_GROUP0();
  }
}
