/*
 * Unified DMA/TMA G2S functional kernels.
 *
 * This file is generated from the former per-case kernel sources and is
 * now the single OpenCL source built by dma_tma_g2s_func_test. Keep new
 * G2S functional kernels here instead of adding standalone .cl projects.
 */


/* --------------------------------------------------------------------------
 * Section: dma_test.cl
 * -------------------------------------------------------------------------- */
#line 1 "dma_test.cl"
/*
 * DMA test kernel: copy data from global memory to shared memory
 * using CP_ASYNC_BULK + CP_ASYNC_FENCE, then write shared memory
 * back to global output for host verification.
 *
 * DMA instruction encoding (R-type, opcode 0x42):
 *   CP_ASYNC_BULK:  funct3=1, rs1=src_global_addr, rs2=byte_count, rd=dst_shared_addr
 *   CP_ASYNC_FENCE: funct3=6, all regs=x0
 *
 * All three operands use scalar registers:
 *   A1 = A1_RS1  (scalar rs1) → in1 (global source address)
 *   A2 = A2_RS2  (scalar rs2) → in2 (byte count)
 *   A3 = A3_FRS3 (scalar rd)  → in3 (shared memory destination address)
 *
 * Since opcode 0x42 is custom, we use .word with calculated encoding.
 */
kernel void
dma_copy(__global const int *src,
         __global int *dst,
         int count)
{
  /* Each workgroup copies 'count' ints from global src to shared memory,
     then reads shared memory back to global dst. */
  __local int shared_buf[64];
  int lid = get_local_id(0);
  int gid = get_global_id(0);

  /* Precompute shared_buf address and byte_count BEFORE divergence.
     src_addr is loaded directly from kernel args inside asm to avoid
     compiler generating unsupported FP instructions for s-register moves. */
  int byte_count = count * 4;
  unsigned int dst_addr = (unsigned int)shared_buf;

  /* Only thread 0 of each workgroup issues DMA. */
  if (lid == 0) {
    /*
     * Load src pointer directly from kernel args buffer inside asm.
     * Layout: CSR 0x803 = metadata_base, *(metadata_base+4) = args_base,
     *         args[0]=src, args[1]=dst, args[2]=count
     * group_offset = 0 for single-workgroup test, so src_addr = args[0].
     * For dst_addr and byte_count, use C-level values (which the compiler
     * can handle without generating problematic instructions).
     */
    __asm__ volatile(
      /* Load kernel args base: read CSR 0x803 → metadata → *(metadata+4) */
      "csrr %[src], 0x803\n\t"      /* src = metadata_base */
      "lw   %[src], 4(%[src])\n\t"  /* src = args_buffer_base */
      "lw   %[src], 0(%[src])\n\t"  /* src = src pointer (args[0]) */
      /* Now: src=global addr, dst=shared addr, size=byte count
       * We need them in x10(rs1), x11(rd), x12(rs2) for the .word encoding.
       * Use explicit register moves to place them. */
      "mv   x10, %[src]\n\t"        /* x10 = global source address */
      "mv   x11, %[dst]\n\t"        /* x11 = shared mem dest addr */
      "mv   x12, %[size]\n\t"       /* x12 = byte count */
      /* CP_ASYNC_BULK: rs1=x10(src), rs2=x12(size), rd=x11(dst_shared) */
      ".word 0x00c515c2\n\t"
      /* CP_ASYNC_FENCE: wait for DMA completion */
      ".word 0x00006042\n\t"
      : [src] "=&r"(dst_addr)
      : [dst] "r"(dst_addr), [size] "r"(byte_count)
      : "memory"
    );
  }

  /* barrier to ensure all threads see the DMA result */
  barrier(CLK_LOCAL_MEM_FENCE);

  /* Each thread copies its element from shared to global dst */
  if (lid < count) {
    int group_offset = get_group_id(0) * count;
    dst[group_offset + lid] = shared_buf[lid];
  }
}

/* --------------------------------------------------------------------------
 * Section: copysize_test.cl
 * -------------------------------------------------------------------------- */
#line 1 "copysize_test.cl"
/*
 * CP_ASYNC_COPYSIZE test kernel.
 * Tests the CP_ASYNC_COPYSIZE instruction (funct3=0) which copies
 * a fixed-size block: 4 << copysize_field bytes.
 *
 * Encoding: R-type, opcode=0x42, funct3=000
 *   rs1 = global source address
 *   rd  = shared memory destination address
 *   inst[26:25] = copysize field (0→4B, 1→8B, 2→16B, 3→32B)
 *
 * This test uses copysize=2 (16 bytes = 4 ints).
 *
 * Machine code calculation for copysize=2:
 *   bits[31:25] = 0000010 (funct7, with bit26=1 for copysize=2)
 *   bits[24:20] = 01100 (rs2=x12, unused but encoded)
 *   bits[19:15] = 01010 (rs1=x10, global src addr)
 *   bits[14:12] = 000   (funct3=0 for COPYSIZE)
 *   bits[11:7]  = 01011 (rd=x11, shared dst addr)
 *   bits[6:0]   = 1000010 (opcode=0x42)
 *   .word = 0x04C505C2
 */
kernel void
dma_copysize(__global const int *src,
             __global int *dst)
{
  /* 16 bytes = 4 ints for copysize=2 */
  __local int shared_buf[64];
  int lid = get_local_id(0);

  unsigned int dst_addr = (unsigned int)shared_buf;

  if (lid == 0) {
    __asm__ volatile(
      /* Load src pointer from kernel args */
      "csrr %[src], 0x803\n\t"
      "lw   %[src], 4(%[src])\n\t"
      "lw   %[src], 0(%[src])\n\t"
      /* Place operands */
      "mv   x10, %[src]\n\t"
      "mv   x11, %[dst]\n\t"
      "mv   x12, x0\n\t"
      /* CP_ASYNC_COPYSIZE: copysize=2, 16 bytes */
      ".word 0x04c505c2\n\t"
      /* CP_ASYNC_FENCE */
      ".word 0x00006042\n\t"
      : [src] "=&r"(dst_addr)
      : [dst] "r"(dst_addr)
      : "memory"
    );
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  /* Each of the first 4 threads copies 1 int from shared to global dst */
  if (lid < 4) {
    dst[lid] = shared_buf[lid];
  }
}


kernel void
dma_copysize_4b(__global const uchar *src,
                __global uchar *dst,
                uint src_offset)
{
  __local uint shared_word[8];
  int lid = get_local_id(0);
  if (lid < 8) shared_word[lid] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    uint src_addr;
    uint dst_addr = (uint)shared_word;
    __asm__ volatile(
      "csrr %[src], 0x803\n\t"
      "lw   %[src], 4(%[src])\n\t"
      "lw   %[src], 0(%[src])\n\t"
      "add  %[src], %[src], %[off]\n\t"
      "mv   x10, %[src]\n\t"
      "mv   x11, %[dst]\n\t"
      "mv   x12, x0\n\t"
      ".word 0x00c505c2\n\t"
      ".word 0x00006042\n\t"
      : [src] "=&r"(src_addr)
      : [dst] "r"(dst_addr), [off] "r"(src_offset)
      : "memory");
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0) ((__global uint *)dst)[0] = shared_word[0];
}

kernel void
dma_copysize_8b(__global const uchar *src,
                __global uchar *dst,
                uint src_offset)
{
  __local uint shared_word[8];
  int lid = get_local_id(0);
  if (lid < 8) shared_word[lid] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    uint src_addr;
    uint dst_addr = (uint)shared_word;
    __asm__ volatile(
      "csrr %[src], 0x803\n\t"
      "lw   %[src], 4(%[src])\n\t"
      "lw   %[src], 0(%[src])\n\t"
      "add  %[src], %[src], %[off]\n\t"
      "mv   x10, %[src]\n\t"
      "mv   x11, %[dst]\n\t"
      "mv   x12, x0\n\t"
      ".word 0x02c505c2\n\t"
      ".word 0x00006042\n\t"
      : [src] "=&r"(src_addr)
      : [dst] "r"(dst_addr), [off] "r"(src_offset)
      : "memory");
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint i = lid; i < 2; i += get_local_size(0)) ((__global uint *)dst)[i] = shared_word[i];
}

kernel void
dma_copysize_16b(__global const uchar *src,
                 __global uchar *dst,
                 uint src_offset)
{
  __local uint shared_word[8];
  int lid = get_local_id(0);
  if (lid < 8) shared_word[lid] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    uint src_addr;
    uint dst_addr = (uint)shared_word;
    __asm__ volatile(
      "csrr %[src], 0x803\n\t"
      "lw   %[src], 4(%[src])\n\t"
      "lw   %[src], 0(%[src])\n\t"
      "add  %[src], %[src], %[off]\n\t"
      "mv   x10, %[src]\n\t"
      "mv   x11, %[dst]\n\t"
      "mv   x12, x0\n\t"
      ".word 0x04c505c2\n\t"
      ".word 0x00006042\n\t"
      : [src] "=&r"(src_addr)
      : [dst] "r"(dst_addr), [off] "r"(src_offset)
      : "memory");
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint i = lid; i < 4; i += get_local_size(0)) ((__global uint *)dst)[i] = shared_word[i];
}

kernel void
dma_copysize_32b(__global const uchar *src,
                 __global uchar *dst,
                 uint src_offset)
{
  __local uint shared_word[8];
  int lid = get_local_id(0);
  if (lid < 8) shared_word[lid] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    uint src_addr;
    uint dst_addr = (uint)shared_word;
    __asm__ volatile(
      "csrr %[src], 0x803\n\t"
      "lw   %[src], 4(%[src])\n\t"
      "lw   %[src], 0(%[src])\n\t"
      "add  %[src], %[src], %[off]\n\t"
      "mv   x10, %[src]\n\t"
      "mv   x11, %[dst]\n\t"
      "mv   x12, x0\n\t"
      ".word 0x06c505c2\n\t"
      ".word 0x00006042\n\t"
      : [src] "=&r"(src_addr)
      : [dst] "r"(dst_addr), [off] "r"(src_offset)
      : "memory");
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint i = lid; i < 8; i += get_local_size(0)) ((__global uint *)dst)[i] = shared_word[i];
}

/* --------------------------------------------------------------------------
 * Section: bulk_dma_matrix_test.cl
 * -------------------------------------------------------------------------- */
#line 1 "bulk_dma_matrix_test.cl"
/*
 * Bulk DMA matrix kernel.
 *
 * Thread 0 issues CP_ASYNC_BULK from src + src_offset into a shared-memory
 * byte buffer at dst_offset, fences, then the workgroup writes the copied
 * bytes back to dst for host-side byte comparison.
 */

#define SHARED_BUF_BYTES 512

kernel void
bulk_dma_matrix_kernel(__global const uchar *src,
                       __global uchar *dst,
                       uint src_offset,
                       uint copy_bytes,
                       uint dst_offset)
{
  __local uchar shared_buf[SHARED_BUF_BYTES];
  int lid = get_local_id(0);
  int lsize = get_local_size(0);

  for (uint i = lid; i < SHARED_BUF_BYTES / 4; i += lsize) {
    ((__local uint *)shared_buf)[i] = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  uint dst_addr = (uint)shared_buf + dst_offset;

  if (lid == 0) {
    uint src_addr;
    __asm__ volatile(
      "csrr %[src], 0x803\n\t"
      "lw   %[src], 4(%[src])\n\t"
      "lw   %[src], 0(%[src])\n\t"
      "add  %[src], %[src], %[off]\n\t"
      "mv   x10, %[src]\n\t"
      "mv   x11, %[dst]\n\t"
      "mv   x12, %[size]\n\t"
      ".word 0x00c515c2\n\t"
      ".word 0x00006042\n\t"
      : [src] "=&r"(src_addr)
      : [off] "r"(src_offset), [dst] "r"(dst_addr),
        [size] "r"(copy_bytes)
      : "memory"
    );
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  uint words = copy_bytes / 4;
  __local uint *shared_words = (__local uint *)(shared_buf + dst_offset);
  __global uint *dst_words = (__global uint *)dst;
  for (uint i = lid; i < words; i += lsize) {
    dst_words[i] = shared_words[i];
  }
}


kernel void
bulk_dual_g2s_single_fence_kernel(__global const uchar *src_a,
                                  __global const uchar *src_b,
                                  __global uchar *dst,
                                  uint copy_bytes,
                                  uint src_a_offset,
                                  uint src_b_offset)
{
  __local uchar shared_buf[SHARED_BUF_BYTES];
  int lid = get_local_id(0);
  int lsize = get_local_size(0);

  for (uint i = lid; i < SHARED_BUF_BYTES / 4; i += lsize) {
    ((__local uint *)shared_buf)[i] = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    uint shared_a = (uint)shared_buf;
    uint shared_b = shared_a + copy_bytes;
    uint src_a_addr = (uint)src_a + src_a_offset;
    uint src_b_addr = (uint)src_b + src_b_offset;
    __asm__ volatile(
      ".insn r 0x42, 1, 0, %[dst], %[src], %[size]\n\t"
      :
      : [dst] "r"(shared_a), [src] "r"(src_a_addr), [size] "r"(copy_bytes)
      : "memory");
    __asm__ volatile(
      ".insn r 0x42, 1, 0, %[dst], %[src], %[size]\n\t"
      :
      : [dst] "r"(shared_b), [src] "r"(src_b_addr), [size] "r"(copy_bytes)
      : "memory");
    __asm__ volatile(".word 0x00006042\n\t" ::: "memory");
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  uint words = (copy_bytes * 2u) / 4u;
  for (uint i = lid; i < words; i += lsize) {
    ((__global uint *)dst)[i] = ((__local uint *)shared_buf)[i];
  }
}

/* --------------------------------------------------------------------------
 * Section: tensor_dma_test.cl
 * -------------------------------------------------------------------------- */
#line 1 "tensor_dma_test.cl"
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

/* --------------------------------------------------------------------------
 * Section: tma_descriptor_test.cl
 * -------------------------------------------------------------------------- */
#line 1 "tma_descriptor_test.cl"
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

/* --------------------------------------------------------------------------
 * Section: tma_matrix_test.cl
 * -------------------------------------------------------------------------- */
#line 1 "tma_matrix_test.cl"
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

/* --------------------------------------------------------------------------
 * Section: mixed_async_fence_test.cl
 * -------------------------------------------------------------------------- */
#line 1 "mixed_async_fence_test.cl"
/*
 * Mixed async DMA/TMA fence directed kernels.
 */

#define BULK_COPY_BYTES 128
#define BULK_SHARED_BYTES (BULK_COPY_BYTES * 2)
#define BULK_WORDS (BULK_COPY_BYTES / 4)

#define TENSOR_SRC_BYTES (8 * 8 * 4)
#define TENSOR_COPY_BYTES 64
#define TENSOR_SHARED_BYTES (TENSOR_COPY_BYTES * 2)
#define TENSOR_GLOBAL_BYTES TENSOR_SRC_BYTES

#define BULK_STRESS_SHARED_BYTES (BULK_COPY_BYTES * 4)
#define COMPLEX_SHARED_BYTES 512
#define COMPLEX_BULK_G2S_OFF 0
#define COMPLEX_BULK_S2G_OFF 128
#define COMPLEX_TENSOR_G2S_OFF 256
#define COMPLEX_TENSOR_S2G_OFF 320
#define COMPLEX_OUT_BULK_G2S_OFF 0
#define COMPLEX_OUT_BULK_S2G_OFF 128
#define COMPLEX_OUT_TENSOR_G2S_OFF 256
#define COMPLEX_OUT_TENSOR_S2G_OFF 320

static uint
bulk_pattern_word(uint word_index)
{
  uint base = word_index << 2;
  uint b0 = (base * 7u + 0x23u) & 0xffu;
  uint b1 = ((base + 1u) * 7u + 0x23u) & 0xffu;
  uint b2 = ((base + 2u) * 7u + 0x23u) & 0xffu;
  uint b3 = ((base + 3u) * 7u + 0x23u) & 0xffu;
  return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

static uint
stress_pattern_word(uint word_index, uint seed)
{
  uint base = word_index << 2;
  uint b0 = (base * 13u + seed) & 0xffu;
  uint b1 = ((base + 1u) * 13u + seed) & 0xffu;
  uint b2 = ((base + 2u) * 13u + seed) & 0xffu;
  uint b3 = ((base + 3u) * 13u + seed) & 0xffu;
  return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

static uint
tensor_s2g_pattern_word(uint word_index)
{
  uint base = word_index << 2;
  uint b0 = (base * 7u + 3u) & 0xffu;
  uint b1 = ((base + 1u) * 7u + 3u) & 0xffu;
  uint b2 = ((base + 2u) * 7u + 3u) & 0xffu;
  uint b3 = ((base + 3u) * 7u + 3u) & 0xffu;
  return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

kernel void
bulk_async_fence_mixed_kernel(__global const uchar *src,
                              __global uchar *verify_dst,
                              __global uchar *s2g_dst,
                              uint copy_bytes)
{
  __local uint shared_words[BULK_SHARED_BYTES / 4];

  int lid = get_local_id(0);
  int lsize = get_local_size(0);

  for (uint i = lid; i < BULK_WORDS; i += lsize) {
    shared_words[i] = 0;
    shared_words[i + BULK_WORDS] = bulk_pattern_word(i);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    uint g2s_shared_addr = (uint)((__local uchar *)shared_words);
    uint src_addr = (uint)src;
    uint s2g_shared_addr = g2s_shared_addr + BULK_COPY_BYTES;
    uint s2g_dst_addr = (uint)s2g_dst;
    uint noise = g2s_shared_addr ^ src_addr ^ copy_bytes;

    __asm__ volatile(
      ".insn r 0x42, 1, 0, %[dst], %[src], %[size]\n\t"
      :
      : [dst] "r"(g2s_shared_addr), [src] "r"(src_addr), [size] "r"(copy_bytes)
      : "memory"
    );

    noise = noise * 17u + 23u;
    __asm__ volatile("" : "+r"(noise) :: "memory");

    __asm__ volatile(
      ".insn r 0x42, 3, 0, %[dst], %[src], %[size]\n\t"
      :
      : [dst] "r"(s2g_dst_addr), [src] "r"(s2g_shared_addr), [size] "r"(copy_bytes)
      : "memory"
    );

    noise ^= noise >> 3;
    __asm__ volatile("" : "+r"(noise) :: "memory");
    __asm__ volatile(".word 0x00006042\n\t" ::: "memory");
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint i = lid; i < BULK_WORDS; i += lsize) {
    ((__global uint *)verify_dst)[i] = shared_words[i];
  }
}

kernel void
tensor_async_fence_mixed_kernel(__global uint *desc_a,
                                __global const uint *coords_a,
                                __global const uchar *src_a,
                                __global uint *desc_b,
                                __global const uint *coords_b,
                                __global const uchar *src_b,
                                __global uchar *dst,
                                uint dst_bytes)
{
  __local uchar shared_buf[TENSOR_SHARED_BYTES];

  int lid = get_local_id(0);
  int lsize = get_local_size(0);

  if (lid == 0) {
    desc_a[2] = (uint)src_a;
    desc_b[2] = (uint)src_b;
    for (int i = 0; i < TENSOR_SHARED_BYTES / 4; i++) {
      ((__local uint *)shared_buf)[i] = 0;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  uint smem_a = (uint)shared_buf;
  uint smem_b = (uint)(shared_buf + TENSOR_COPY_BYTES);
  uint desc_a_ptr = (uint)desc_a;
  uint desc_b_ptr = (uint)desc_b;
  uint coords_a_ptr = (uint)coords_a;
  uint coords_b_ptr = (uint)coords_b;

  __asm__ volatile(
    "mv x11, %[desc]\n\t"
    ".word 0x0005D042\n\t"
    :
    : [desc] "r"(desc_a_ptr)
    : "memory"
  );

  __asm__ volatile(
    "vid.v v12\n\t"
    "vsll.vi v12, v12, 2\n\t"
    "vadd.vx v12, v12, %[coords]\n\t"
    "vlw12.v v12, 0(v12)\n\t"
    "mv x10, %[smem]\n\t"
    "mv x11, %[desc]\n\t"
    ".word 0x00C5A542\n\t"
    :
    : [smem] "r"(smem_a), [desc] "r"(desc_a_ptr), [coords] "r"(coords_a_ptr)
    : "memory"
  );

  uint noise = smem_a ^ desc_b_ptr ^ coords_a_ptr;
  noise = noise * 19u + 7u;
  __asm__ volatile("" : "+r"(noise) :: "memory");

  __asm__ volatile(
    "mv x11, %[desc]\n\t"
    ".word 0x0005D042\n\t"
    :
    : [desc] "r"(desc_b_ptr)
    : "memory"
  );

  __asm__ volatile(
    "vid.v v12\n\t"
    "vsll.vi v12, v12, 2\n\t"
    "vadd.vx v12, v12, %[coords]\n\t"
    "vlw12.v v12, 0(v12)\n\t"
    "mv x10, %[smem]\n\t"
    "mv x11, %[desc]\n\t"
    ".word 0x00C5A542\n\t"
    :
    : [smem] "r"(smem_b), [desc] "r"(desc_b_ptr), [coords] "r"(coords_b_ptr)
    : "memory"
  );

  noise ^= noise >> 5;
  __asm__ volatile("" : "+r"(noise) :: "memory");
  __asm__ volatile(".word 0x00006042\n\t" ::: "memory");
  barrier(CLK_LOCAL_MEM_FENCE);

  uint n_words = dst_bytes / 4;
  for (uint i = lid; i < n_words; i += lsize) {
    ((__global uint *)dst)[i] = ((__local uint *)shared_buf)[i];
  }
}

kernel void
bulk_multi_issue_fence_kernel(__global const uchar *src_a,
                              __global const uchar *src_b,
                              __global uchar *verify_a,
                              __global uchar *verify_b,
                              __global uchar *s2g_a,
                              __global uchar *s2g_b,
                              __global uint *marker,
                              uint copy_bytes)
{
  __local uint shared_words[BULK_STRESS_SHARED_BYTES / 4];

  int lid = get_local_id(0);
  int lsize = get_local_size(0);

  for (uint i = lid; i < BULK_COPY_BYTES / 4; i += lsize) {
    shared_words[i] = 0;
    shared_words[i + BULK_COPY_BYTES / 4] = 0;
    shared_words[i + (BULK_COPY_BYTES / 4) * 2] =
      stress_pattern_word(i, 0x41u);
    shared_words[i + (BULK_COPY_BYTES / 4) * 3] =
      stress_pattern_word(i, 0x9du);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    uint base = (uint)((__local uchar *)shared_words);
    uint g2s_a = base;
    uint g2s_b = base + BULK_COPY_BYTES;
    uint s2g_src_a = base + BULK_COPY_BYTES * 2;
    uint s2g_src_b = base + BULK_COPY_BYTES * 3;
    uint noise = copy_bytes ^ (uint)src_a ^ (uint)s2g_b ^ 0x13572468u;

    marker[0] = noise;

    __asm__ volatile(
      ".insn r 0x42, 1, 0, %[dst], %[src], %[size]\n\t"
      :
      : [dst] "r"(g2s_a), [src] "r"((uint)src_a), [size] "r"(copy_bytes)
      : "memory"
    );

    noise = noise * 33u + 17u;
    __asm__ volatile("" : "+r"(noise) :: "memory");

    __asm__ volatile(
      ".insn r 0x42, 3, 0, %[dst], %[src], %[size]\n\t"
      :
      : [dst] "r"((uint)s2g_a), [src] "r"(s2g_src_a), [size] "r"(copy_bytes)
      : "memory"
    );

    marker[1] = noise ^ 0x2468ace0u;

    __asm__ volatile(
      ".insn r 0x42, 1, 0, %[dst], %[src], %[size]\n\t"
      :
      : [dst] "r"(g2s_b), [src] "r"((uint)src_b), [size] "r"(copy_bytes)
      : "memory"
    );

    noise ^= noise >> 7;
    noise += 0x10203040u;
    __asm__ volatile("" : "+r"(noise) :: "memory");

    __asm__ volatile(
      ".insn r 0x42, 3, 0, %[dst], %[src], %[size]\n\t"
      :
      : [dst] "r"((uint)s2g_b), [src] "r"(s2g_src_b), [size] "r"(copy_bytes)
      : "memory"
    );

    marker[2] = noise;
    __asm__ volatile(".word 0x00006042\n\t" ::: "memory");
    marker[3] = noise ^ marker[0] ^ marker[1];
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint i = lid; i < BULK_COPY_BYTES / 4; i += lsize) {
    ((__global uint *)verify_a)[i] = shared_words[i];
    ((__global uint *)verify_b)[i] = shared_words[i + BULK_COPY_BYTES / 4];
  }
}

kernel void
tensor_bulk_bidirectional_fence_kernel(__global uint *tensor_g2s_desc,
                                       __global const uint *tensor_g2s_coords,
                                       __global const uchar *tensor_src,
                                       __global uint *tensor_s2g_desc,
                                       __global const uint *tensor_s2g_coords,
                                       __global const uchar *bulk_src,
                                       __global uchar *out,
                                       __global uint *marker,
                                       uint bulk_bytes)
{
  __local uint shared_words[COMPLEX_SHARED_BYTES / 4];

  int lid = get_local_id(0);
  int lsize = get_local_size(0);

  for (uint i = lid; i < COMPLEX_SHARED_BYTES / 4; i += lsize) {
    shared_words[i] = 0;
  }
  for (uint i = lid; i < BULK_COPY_BYTES / 4; i += lsize) {
    shared_words[(COMPLEX_BULK_S2G_OFF / 4) + i] =
      stress_pattern_word(i, 0x55u);
  }
  for (uint i = lid; i < TENSOR_COPY_BYTES / 4; i += lsize) {
    shared_words[(COMPLEX_TENSOR_S2G_OFF / 4) + i] =
      tensor_s2g_pattern_word(i);
  }
  if (lid == 0) {
    tensor_g2s_desc[2] = (uint)tensor_src;
    tensor_s2g_desc[2] = (uint)(out + COMPLEX_OUT_TENSOR_S2G_OFF);
    marker[0] = 0xfeed0001u;
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  uint base = (uint)((__local uchar *)shared_words);
  uint bulk_g2s_shared = base + COMPLEX_BULK_G2S_OFF;
  uint bulk_s2g_shared = base + COMPLEX_BULK_S2G_OFF;
  uint tensor_g2s_shared = base + COMPLEX_TENSOR_G2S_OFF;
  uint tensor_s2g_shared = base + COMPLEX_TENSOR_S2G_OFF;
  uint tensor_g2s_desc_ptr = (uint)tensor_g2s_desc;
  uint tensor_s2g_desc_ptr = (uint)tensor_s2g_desc;
  uint tensor_g2s_coords_ptr = (uint)tensor_g2s_coords;
  uint tensor_s2g_coords_ptr = (uint)tensor_s2g_coords;

  if (lid == 0) {
    uint bulk_g2s_dst = bulk_g2s_shared;
    uint bulk_g2s_src = (uint)bulk_src;
    __asm__ volatile(
      ".insn r 0x42, 1, 0, %[dst], %[src], %[size]\n\t"
      :
      : [dst] "r"(bulk_g2s_dst), [src] "r"(bulk_g2s_src),
        [size] "r"(bulk_bytes)
      : "memory"
    );
    marker[1] = 0xfeed0002u;
  }

  __asm__ volatile(
    "mv x11, %[desc]\n\t"
    ".word 0x0005D042\n\t"
    :
    : [desc] "r"(tensor_g2s_desc_ptr)
    : "x11", "memory"
  );

  __asm__ volatile(
    "vid.v v12\n\t"
    "vsll.vi v12, v12, 2\n\t"
    "vadd.vx v12, v12, %[coords]\n\t"
    "vlw12.v v12, 0(v12)\n\t"
    "mv x10, %[smem]\n\t"
    "mv x11, %[desc]\n\t"
    ".word 0x00C5A542\n\t"
    :
    : [smem] "r"(tensor_g2s_shared), [desc] "r"(tensor_g2s_desc_ptr),
      [coords] "r"(tensor_g2s_coords_ptr)
    : "x10", "x11", "memory"
  );

  if (lid == 0) {
    uint bulk_s2g_dst = (uint)(out + COMPLEX_OUT_BULK_S2G_OFF);
    uint bulk_s2g_src = bulk_s2g_shared;
    __asm__ volatile(
      ".insn r 0x42, 3, 0, %[dst], %[src], %[size]\n\t"
      :
      : [dst] "r"(bulk_s2g_dst), [src] "r"(bulk_s2g_src),
        [size] "r"(bulk_bytes)
      : "memory"
    );
    marker[2] = 0xfeed0003u;
  }

  __asm__ volatile(
    "vid.v v12\n\t"
    "vsll.vi v12, v12, 2\n\t"
    "vadd.vx v12, v12, %[coords]\n\t"
    "vlw12.v v12, 0(v12)\n\t"
    :
    : [coords] "r"(tensor_s2g_coords_ptr)
    : "memory"
  );

  if (lid == 0) {
    __asm__ volatile(
      "mv x10, %[smem]\n\t"
      "mv x11, %[desc]\n\t"
      ".word 0x00C5C542\n\t"
      :
      : [smem] "r"(tensor_s2g_shared), [desc] "r"(tensor_s2g_desc_ptr)
      : "x10", "x11", "memory"
    );
    marker[3] = 0xfeed0004u;
  }

  __asm__ volatile(".word 0x00006042\n\t" ::: "memory");
  if (lid == 0) {
    __asm__ volatile(".word 0x00006042\n\t" ::: "memory");
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint i = lid; i < BULK_COPY_BYTES / 4; i += lsize) {
    ((__global uint *)(out + COMPLEX_OUT_BULK_G2S_OFF))[i] =
      shared_words[(COMPLEX_BULK_G2S_OFF / 4) + i];
  }
  for (uint i = lid; i < TENSOR_COPY_BYTES / 4; i += lsize) {
    ((__global uint *)(out + COMPLEX_OUT_TENSOR_G2S_OFF))[i] =
      shared_words[(COMPLEX_TENSOR_G2S_OFF / 4) + i];
  }
}


/* --------------------------------------------------------------------------
 * Section: dma_shared_routing_conflict_test.cl
 * -------------------------------------------------------------------------- */
#line 1 "dma_shared_routing_conflict_test.cl"
/*
 * Shared bank-conflict + DMA response routing kernel.
 *
 * A DMA writes into dma_shared while all lanes repeatedly access conflict
 * slots at lid * 32 words, which maps the active lanes to the same shared
 * bank. The host verifies both the DMA payload and the conflict values.
 */

#define DMA_SHARED_BYTES 512
#define DMA_SHARED_WORDS (DMA_SHARED_BYTES / 4)
#define CONFLICT_WORDS   4096
#define SCRATCH_WORDS    (DMA_SHARED_WORDS + CONFLICT_WORDS)

kernel void
dma_shared_routing_conflict_kernel(__global const uchar *src,
                                   __global uchar *dma_out,
                                   __global uint *conflict_out,
                                   uint copy_bytes,
                                   uint rounds,
                                   uint src_offset)
{
  __local uint scratch[SCRATCH_WORDS];
  __local uchar *dma_shared = (__local uchar *)scratch;
  volatile __local uint *conflict = scratch + DMA_SHARED_WORDS;
  uint lid = get_local_id(0);
  uint lsize = get_local_size(0);

  for (uint i = lid; i < SCRATCH_WORDS; i += lsize) {
    scratch[i] = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    uint src_addr;
    uint dst_addr = (uint)dma_shared;
    __asm__ volatile(
      "csrr %[src], 0x803\n\t"
      "lw   %[src], 4(%[src])\n\t"
      "lw   %[src], 0(%[src])\n\t"
      "add  %[src], %[src], %[off]\n\t"
      "mv   x10, %[src]\n\t"
      "mv   x11, %[dst]\n\t"
      "mv   x12, %[size]\n\t"
      ".word 0x00c515c2\n\t"
      : [src] "=&r"(src_addr)
      : [off] "r"(src_offset), [dst] "r"(dst_addr),
        [size] "r"(copy_bytes)
      : "memory"
    );
  }

  uint slot = lid * 32;
  uint value = 0xA5000000u + lid;
  conflict[slot] = value;
  for (uint r = 0; r < rounds; r++) {
    uint cur = conflict[slot];
    cur = cur + ((r + 1) * 17u) + lid;
    conflict[slot] = cur;
    value = cur;
  }

  if (lid == 0) {
    __asm__ volatile(".word 0x00006042\n\t" ::: "memory");
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  uint words = copy_bytes / 4;
  for (uint i = lid; i < words; i += lsize) {
    ((__global uint *)dma_out)[i] = ((__local uint *)dma_shared)[i];
  }
  conflict_out[lid] = value;
}

/* --------------------------------------------------------------------------
 * Section: multi_warp_dma_fence_test.cl
 * -------------------------------------------------------------------------- */
#line 1 "multi_warp_dma_fence_test.cl"
/*
 * Same-workgroup multi-warp DMA + fence kernel.
 *
 * Each warp leader issues CP_ASYNC_BULK operations into disjoint shared-memory
 * segments. A single CP_ASYNC_FENCE follows the per-warp DMA burst, then that
 * warp writes only its own segments back to global output.
 */

#define MAX_SHARED_BYTES 1024

kernel void
multi_warp_dma_fence_kernel(__global const uchar *src,
                            __global uchar *dst,
                            uint copy_bytes,
                            uint dmas_per_warp,
                            uint src_base_offset,
                            uint src_stride)
{
  __local uchar shared_buf[MAX_SHARED_BYTES];
  uint lid = get_local_id(0);
  uint warp_id = lid >> 5;
  uint lane_id = lid & 31;

  if (lane_id == 0) {
    uint shared_base = (uint)shared_buf;
    uint src_addr;
    uint seg;
    uint dst_addr;

    __asm__ volatile(
      "csrr %[src], 0x803\n\t"
      "lw   %[src], 4(%[src])\n\t"
      "lw   %[src], 0(%[src])\n\t"
      "csrr %[seg], 0x805\n\t"
      "mul  %[seg], %[seg], %[dpw]\n\t"
      "mul  %[dst], %[seg], %[size]\n\t"
      "add  %[dst], %[dst], %[shared]\n\t"
      "mul  %[seg], %[seg], %[stride]\n\t"
      "add  %[src], %[src], %[base]\n\t"
      "add  %[src], %[src], %[seg]\n\t"
      ".insn r 0x42, 1, 0, %[dst], %[src], %[size]\n\t"
      : [src] "=&r"(src_addr), [seg] "=&r"(seg),
        [dst] "=&r"(dst_addr)
      : [shared] "r"(shared_base), [size] "r"(copy_bytes),
        [dpw] "r"(dmas_per_warp), [base] "r"(src_base_offset),
        [stride] "r"(src_stride)
      : "memory"
    );

    __asm__ volatile(".word 0x00006042\n\t" ::: "memory");
  }

  uint warp_words = (dmas_per_warp * copy_bytes) / 4;
  uint warp_base_word = warp_id * warp_words;
  if (lane_id == 0) {
    for (uint i = 0; i < warp_words; i++) {
      ((__global uint *)dst)[warp_base_word + i] =
        ((__local uint *)shared_buf)[warp_base_word + i];
    }
  }
}

kernel void
multi_warp_dma_fence2_kernel(__global const uchar *src,
                             __global uchar *dst,
                             uint copy_bytes,
                             uint dmas_per_warp,
                             uint src_base_offset,
                             uint src_stride)
{
  __local uchar shared_buf[MAX_SHARED_BYTES];
  uint lid = get_local_id(0);
  uint warp_id = lid >> 5;
  uint lane_id = lid & 31;

  if (lane_id == 0) {
    uint shared_base = (uint)shared_buf;
    uint src_addr;
    uint seg;
    uint dst_addr;

    __asm__ volatile(
      "csrr %[src], 0x803\n\t"
      "lw   %[src], 4(%[src])\n\t"
      "lw   %[src], 0(%[src])\n\t"
      "csrr %[seg], 0x805\n\t"
      "mul  %[seg], %[seg], %[dpw]\n\t"
      "mul  %[dst], %[seg], %[size]\n\t"
      "add  %[dst], %[dst], %[shared]\n\t"
      "mul  %[seg], %[seg], %[stride]\n\t"
      "add  %[src], %[src], %[base]\n\t"
      "add  %[src], %[src], %[seg]\n\t"
      ".insn r 0x42, 1, 0, %[dst], %[src], %[size]\n\t"
      : [src] "=&r"(src_addr), [seg] "=&r"(seg),
        [dst] "=&r"(dst_addr)
      : [shared] "r"(shared_base), [size] "r"(copy_bytes),
        [dpw] "r"(dmas_per_warp), [base] "r"(src_base_offset),
        [stride] "r"(src_stride)
      : "memory"
    );

    __asm__ volatile(
      "csrr %[src], 0x803\n\t"
      "lw   %[src], 4(%[src])\n\t"
      "lw   %[src], 0(%[src])\n\t"
      "csrr %[seg], 0x805\n\t"
      "mul  %[seg], %[seg], %[dpw]\n\t"
      "addi %[seg], %[seg], 1\n\t"
      "mul  %[dst], %[seg], %[size]\n\t"
      "add  %[dst], %[dst], %[shared]\n\t"
      "mul  %[seg], %[seg], %[stride]\n\t"
      "add  %[src], %[src], %[base]\n\t"
      "add  %[src], %[src], %[seg]\n\t"
      ".insn r 0x42, 1, 0, %[dst], %[src], %[size]\n\t"
      : [src] "=&r"(src_addr), [seg] "=&r"(seg),
        [dst] "=&r"(dst_addr)
      : [shared] "r"(shared_base), [size] "r"(copy_bytes),
        [dpw] "r"(dmas_per_warp), [base] "r"(src_base_offset),
        [stride] "r"(src_stride)
      : "memory"
    );

    __asm__ volatile(".word 0x00006042\n\t" ::: "memory");
  }

  uint warp_words = (dmas_per_warp * copy_bytes) / 4;
  uint warp_base_word = warp_id * warp_words;
  if (lane_id == 0) {
    for (uint i = 0; i < warp_words; i++) {
      ((__global uint *)dst)[warp_base_word + i] =
        ((__local uint *)shared_buf)[warp_base_word + i];
    }
  }
}
