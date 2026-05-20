/*
 * Tensor DMA smoke kernel: copy a 4x4 FP32 tensor from global memory to shared
 * memory using descriptor-form CP_ASYNC_TENSOR (funct3=2), then write shared
 * memory back to global output for host verification.
 */

#define DESC_WORDS 32
#define COORD_WORDS 32

kernel void
tensor_dma_copy(__global uint *desc,
                __global const uint *coords,
                __global const float *src,
                __global float *dst)
{
  __local float shared_buf[16];

  int lid = get_local_id(0);

  if (lid == 0) {
    desc[2] = (uint)src;
    for (int i = 0; i < 16; i++) shared_buf[i] = 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  uint smem = (uint)shared_buf;
  uint desc_ptr = (uint)desc;
  uint coords_ptr = (uint)coords;

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

  if (lid < 16) {
    dst[lid] = shared_buf[lid];
  }
}
