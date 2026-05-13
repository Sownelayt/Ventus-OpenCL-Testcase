/*
 * Descriptor-addressed TMA smoke kernel.
 *
 * The host fills descriptor v0 fields that are independent of OpenCL device
 * addresses. Thread 0 patches descriptor.globalAddress with the runtime src
 * pointer, then all lanes issue optional PREFETCH_TENSORMAP and
 * CP_ASYNC_TENSOR_G2S using scalar pointer operands.
 */

#define SHARED_BUF_BYTES 128

kernel void
tma_descriptor_kernel(__global uint *desc,
                      __global const uint *coords,
                      __global const uchar *src,
                      __global uchar *dst,
                      uint dst_bytes,
                      uint use_prefetch)
{
  __local uchar shared_buf[SHARED_BUF_BYTES];

  int lid = get_local_id(0);
  int lsize = get_local_size(0);

  if (lid == 0) {
    desc[2] = (uint)src;
    for (int i = 0; i < SHARED_BUF_BYTES / 4; i++) {
      ((__local uint *)shared_buf)[i] = 0;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  uint smem = (uint)shared_buf;
  uint desc_ptr = (uint)desc;
  uint coords_ptr = (uint)coords;

  if (use_prefetch) {
    __asm__ volatile(
      "mv x11, %[desc]\n\t"
      /* PREFETCH_TENSORMAP rs1=x11 */
      ".word 0x0005D042\n\t"
      :
      : [desc] "r"(desc_ptr)
      : "x11", "memory"
    );
  }

  __asm__ volatile(
    "mv x10, %[smem]\n\t"
    "mv x11, %[desc]\n\t"
    "mv x12, %[coords]\n\t"
    /* CP_ASYNC_TENSOR_G2S rd=x10, rs1=x11, rs2=x12 */
    ".word 0x00C5E542\n\t"
    /* CP_ASYNC_FENCE */
    ".word 0x00004042\n\t"
    :
    : [smem] "r"(smem), [desc] "r"(desc_ptr), [coords] "r"(coords_ptr)
    : "x10", "x11", "x12", "memory"
  );
  barrier(CLK_LOCAL_MEM_FENCE);

  uint n_words = dst_bytes / 4;
  for (uint i = lid; i < n_words; i += lsize) {
    ((__global uint *)dst)[i] = ((__local uint *)shared_buf)[i];
  }
}
