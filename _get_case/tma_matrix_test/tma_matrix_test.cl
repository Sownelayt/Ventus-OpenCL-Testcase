/*
 * Generic TMA (tensor DMA) driver kernel for the matrix test.
 *
 * Protocol:
 *   - Host passes a 96-word descriptor template in `desc_src`, with
 *     `desc_src[2]`  = byte offset of globalAddress from `src`
 *     `desc_src[32]` = byte offset of BoxAddress from `src`
 *     `desc_src[64]` = 0 (kernel patches with the shared_buf pointer)
 *     All other fields are as described in spike/riscv/insns/cp_async_tensor.h,
 *     including desc_src[44] swizzleMode.
 *   - Kernel copies the descriptor into __local param_buf[96], patches the
 *     three runtime addresses, loads VRS1/VRS2/VRS3 into VGPRs, issues
 *     CP_ASYNC_TENSOR + CP_ASYNC_FENCE, then writes the first `dst_bytes`
 *     of shared_buf back to `dst`.
 *
 * Size limit: SHARED_BUF_BYTES — must be >= largest box we ever test.
 */

#define SHARED_BUF_BYTES 1024  /* 256 x FP32 max */

kernel void
tma_matrix_kernel(__global const uint *desc_src,
                  __global const uchar *src,
                  __global uchar *dst,
                  uint dst_bytes)
{
  __local uint param_buf[96];
  __local uchar shared_buf[SHARED_BUF_BYTES];

  int lid = get_local_id(0);
  int lsize = get_local_size(0);

  /* ---- 1. Thread 0 copies descriptor and patches runtime addresses ---- */
  if (lid == 0) {
    for (int i = 0; i < 96; i++) param_buf[i] = desc_src[i];
    param_buf[2]  = (uint)src + desc_src[2];   /* globalAddress */
    param_buf[32] = (uint)src + desc_src[32];  /* BoxAddress */
    param_buf[64] = (uint)shared_buf;          /* dst shared address */

    /* Also zero shared_buf so that mis-behaving impls show all-zeros rather
     * than stale data from a previous kernel invocation. */
    for (int i = 0; i < SHARED_BUF_BYTES / 4; i++) {
      ((__local uint *)shared_buf)[i] = 0;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* ---- 2. Load descriptor into VGPRs and issue DMA ---- */
  uint pb = (uint)param_buf;
  __asm__ volatile(
    "vid.v v10\n\t"
    "vsll.vi v10, v10, 2\n\t"
    "vadd.vx v10, v10, %[pb]\n\t"
    "vlw12.v v11, 256(v10)\n\t"
    "vlw12.v v12, 128(v10)\n\t"
    "vlw12.v v10, 0(v10)\n\t"
    /* CP_ASYNC_TENSOR v11 <- v10, v12, v11 (rs1=v10, rs2=v12, rd=v11) */
    ".word 0x00C535C2\n\t"
    /* CP_ASYNC_FENCE */
    ".word 0x00004042\n\t"
    :
    : [pb] "r"(pb)
    : "memory"
  );
  barrier(CLK_LOCAL_MEM_FENCE);

  /* ---- 3. Byte-wise copy shared_buf[0..dst_bytes) back to dst ----
   *
   * Use 4-byte chunks when both src and dst are aligned + dst_bytes % 4 == 0;
   * the test harness sizes its buffers so this is always the case. Keep a
   * byte tail path for safety if dst_bytes is odd.
   */
  uint n_words = dst_bytes / 4;
  for (uint i = lid; i < n_words; i += lsize) {
    ((__global uint *)dst)[i] = ((__local uint *)shared_buf)[i];
  }
  uint tail_base = n_words * 4;
  for (uint i = lid; tail_base + i < dst_bytes; i += lsize) {
    dst[tail_base + i] = shared_buf[tail_base + i];
  }
}
