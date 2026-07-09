
/*
 * Tensor shared-to-global smoke kernel.
 *
 * The kernel fills a small FP32 tensor in shared memory, then issues
 * descriptor-form CP_ASYNC_TENSOR_S2G (funct3=4) to copy that tensor into a
 * global destination selected by the descriptor and the runtime coords.
 */

#define SHARED_ELEMS 256
#define SHARED_BYTES  (SHARED_ELEMS * 4)

static uint
pattern_word(unsigned idx)
{
  unsigned base = idx << 2;
  unsigned b0 = (base * 7u + 3u) & 0xffu;
  unsigned b1 = ((base + 1u) * 7u + 3u) & 0xffu;
  unsigned b2 = ((base + 2u) * 7u + 3u) & 0xffu;
  unsigned b3 = ((base + 3u) * 7u + 3u) & 0xffu;
  return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

kernel void
tensor_shared_to_global_kernel(__global uint *desc,
                               __global const uint *coords,
                               __global uchar *dst)
{
  __local uint shared_buf[SHARED_ELEMS];
  int lid = get_local_id(0);
  int lsize = get_local_size(0);

  for (uint i = lid; i < SHARED_ELEMS; i += lsize) {
    shared_buf[i] = pattern_word(i);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  uint coords_ptr = (uint)coords;
  __asm__ volatile(
    "vid.v v12\n\t"
    "vsll.vi v12, v12, 2\n\t"
    "vadd.vx v12, v12, %[coords]\n\t"
    "vlw12.v v12, 0(v12)\n\t"
    :
    : [coords] "r"(coords_ptr)
    : "memory"
  );

  if (lid == 0) {
    uint shared_addr = (uint)shared_buf;
    uint desc_ptr = (uint)desc;
    desc[2] = (uint)dst;
    __asm__ volatile(
      "mv x10, %[shared]\n\t"
      "mv x11, %[desc]\n\t"
      /* CP_ASYNC_TENSOR_S2G rd=x10, rs1=x11, rs2=v12 */
      ".word 0x00C5C542\n\t"
      /* CP_ASYNC_FENCE / wait-all */
      ".word 0x00006042\n\t"
      :
      : [shared] "r"(shared_addr), [desc] "r"(desc_ptr)
      : "memory"
    );
  }
}
