/*
 * Descriptor-addressed TMA smoke kernels.
 *
 * The host fills descriptor v0 fields that are independent of OpenCL device
 * addresses. Thread 0 patches descriptor.globalAddress with the runtime src
 * pointer, then all lanes issue optional PREFETCH_TENSORMAP and
 * CP_ASYNC_TENSOR with scalar descriptor/dst operands plus a VGPR dynamic
 * parameter block. VRS2 word 0..4 carries coords[0..4].
 */

#define SHARED_BUF_BYTES 256
#define COPY_BYTES 64

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
      : "memory"
    );
  }

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

  uint n_words = dst_bytes / 4;
  for (uint i = lid; i < n_words; i += lsize) {
    ((__global uint *)dst)[i] = ((__local uint *)shared_buf)[i];
  }
}

kernel void
tma_descriptor_prefetch_other_kernel(__global uint *prefetch_desc,
                                     __global const uint *prefetch_coords,
                                     __global const uchar *prefetch_src,
                                     __global uint *copy_desc,
                                     __global const uint *copy_coords,
                                     __global const uchar *copy_src,
                                     __global uchar *dst,
                                     uint dst_bytes)
{
  __local uchar shared_buf[SHARED_BUF_BYTES];

  int lid = get_local_id(0);
  int lsize = get_local_size(0);

  if (lid == 0) {
    prefetch_desc[2] = (uint)prefetch_src;
    copy_desc[2] = (uint)copy_src;
    for (int i = 0; i < SHARED_BUF_BYTES / 4; i++) {
      ((__local uint *)shared_buf)[i] = 0;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  uint prefetch_desc_ptr = (uint)prefetch_desc;
  uint copy_desc_ptr = (uint)copy_desc;
  uint copy_coords_ptr = (uint)copy_coords;
  uint smem = (uint)shared_buf;

  (void)prefetch_coords;
  __asm__ volatile(
    "mv x11, %[desc]\n\t"
    /* PREFETCH_TENSORMAP rs1=x11 */
    ".word 0x0005D042\n\t"
    :
    : [desc] "r"(prefetch_desc_ptr)
    : "memory"
  );

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
    : [smem] "r"(smem), [desc] "r"(copy_desc_ptr),
      [coords] "r"(copy_coords_ptr)
    : "memory"
  );
  barrier(CLK_LOCAL_MEM_FENCE);

  uint n_words = dst_bytes / 4;
  for (uint i = lid; i < n_words; i += lsize) {
    ((__global uint *)dst)[i] = ((__local uint *)shared_buf)[i];
  }
}

kernel void
tma_descriptor_dual_single_fence_kernel(__global uint *desc_a,
                                        __global const uint *coords_a,
                                        __global const uchar *src_a,
                                        __global uint *desc_b,
                                        __global const uint *coords_b,
                                        __global const uchar *src_b,
                                        __global uchar *dst,
                                        uint dst_bytes)
{
  __local uchar shared_buf[SHARED_BUF_BYTES];

  int lid = get_local_id(0);
  int lsize = get_local_size(0);

  if (lid == 0) {
    desc_a[2] = (uint)src_a;
    desc_b[2] = (uint)src_b;
    for (int i = 0; i < SHARED_BUF_BYTES / 4; i++) {
      ((__local uint *)shared_buf)[i] = 0;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  uint smem_a = (uint)shared_buf;
  uint smem_b = (uint)(shared_buf + COPY_BYTES);
  uint desc_a_ptr = (uint)desc_a;
  uint desc_b_ptr = (uint)desc_b;
  uint coords_a_ptr = (uint)coords_a;
  uint coords_b_ptr = (uint)coords_b;

  __asm__ volatile(
    "vid.v v12\n\t"
    "vsll.vi v12, v12, 2\n\t"
    "vadd.vx v12, v12, %[coords_a]\n\t"
    "vlw12.v v12, 0(v12)\n\t"
    "mv x10, %[smem_a]\n\t"
    "mv x11, %[desc_a]\n\t"
    /* CP_ASYNC_TENSOR rd=x10, rs1=x11, rs2=v12 */
    ".word 0x00C5A542\n\t"
    "vid.v v12\n\t"
    "vsll.vi v12, v12, 2\n\t"
    "vadd.vx v12, v12, %[coords_b]\n\t"
    "vlw12.v v12, 0(v12)\n\t"
    "mv x10, %[smem_b]\n\t"
    "mv x11, %[desc_b]\n\t"
    /* CP_ASYNC_TENSOR rd=x10, rs1=x11, rs2=v12 */
    ".word 0x00C5A542\n\t"
    /* One CP_ASYNC_FENCE must drain both TMA copies. */
    ".word 0x00006042\n\t"
    :
    : [smem_a] "r"(smem_a), [smem_b] "r"(smem_b),
      [desc_a] "r"(desc_a_ptr), [desc_b] "r"(desc_b_ptr),
      [coords_a] "r"(coords_a_ptr), [coords_b] "r"(coords_b_ptr)
    : "memory"
  );
  barrier(CLK_LOCAL_MEM_FENCE);

  uint n_words = dst_bytes / 4;
  for (uint i = lid; i < n_words; i += lsize) {
    ((__global uint *)dst)[i] = ((__local uint *)shared_buf)[i];
  }
}
