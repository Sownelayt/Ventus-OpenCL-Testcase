/*
 * Generic TMA (tensor DMA) driver kernel for the matrix test.
 *
 * Protocol:
 *   - Host passes desc_src[0..31] as a 128B tensor-map descriptor and
 *     desc_src[32..63] as dynamic coords[0..4].
 *   - Kernel patches descriptor.globalAddress with the runtime src pointer,
 *     loads coords into VRS2, then issues descriptor-form CP_ASYNC_TENSOR
 *     (funct3=2) plus CP_ASYNC_FENCE.
 *
 * Size limit: SHARED_BUF_BYTES must be >= largest box we ever test.
 */

#define SHARED_BUF_BYTES 1024  /* 256 x FP32 max */

kernel void
tma_matrix_kernel(__global uint *desc_src,
                  __global const uchar *src,
                  __global uchar *dst,
                  uint dst_bytes)
{
  __local uchar shared_buf[SHARED_BUF_BYTES];

  int lid = get_local_id(0);
  int lsize = get_local_size(0);

  /* ---- 1. Thread 0 patches descriptor.globalAddress and clears shared ---- */
  if (lid == 0) {
    desc_src[2] = (uint)src;
    for (int i = 0; i < SHARED_BUF_BYTES / 4; i++) {
      ((__local uint *)shared_buf)[i] = 0;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  /* ---- 2. Load dynamic coords into VGPRs and issue descriptor TMA ---- */
  uint smem = (uint)shared_buf;
  uint desc_ptr = (uint)desc_src;
  uint coords_ptr = (uint)(desc_src + 32);
  __asm__ volatile(
    "vid.v v12\n\t"
    "vsll.vi v12, v12, 2\n\t"
    "vadd.vx v12, v12, %[coords]\n\t"
    "vlw12.v v12, 0(v12)\n\t"
    "mv x10, %[smem]\n\t"
    "mv x11, %[desc]\n\t"
    /* CP_ASYNC_TENSOR rd=x10, rs1=x11, rs2=v12 */
    ".word 0x00C5A542\n\t"
    /* CP_ASYNC_FENCE */
    ".word 0x00006042\n\t"
    :
    : [smem] "r"(smem), [desc] "r"(desc_ptr), [coords] "r"(coords_ptr)
    : "memory"
  );
  barrier(CLK_LOCAL_MEM_FENCE);

  /* ---- 3. Byte-wise copy shared_buf[0..dst_bytes) back to dst ---- */
  uint n_words = dst_bytes / 4;
  for (uint i = lid; i < n_words; i += lsize) {
    ((__global uint *)dst)[i] = ((__local uint *)shared_buf)[i];
  }
  uint tail_base = n_words * 4;
  for (uint i = lid; tail_base + i < dst_bytes; i += lsize) {
    dst[tail_base + i] = shared_buf[tail_base + i];
  }
}
