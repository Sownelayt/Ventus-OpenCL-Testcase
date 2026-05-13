/*
 * Tensor DMA test kernel: copy a 2D tensor from global memory to shared memory
 * using CP_ASYNC_TENSOR (funct3=3) + CP_ASYNC_FENCE, then write shared memory
 * back to global output for host verification.
 *
 * CP_ASYNC_TENSOR reads from VGPR (vector registers) in RTL hardware.
 * DecodeUnit: A1=VRS1, A2=VRS2, A3=VRS3 — reads from VGPR bank.
 * SGPR ("mv x10, val") and VGPR ("v10") are SEPARATE register banks.
 * Must use vector instructions (vid.v, vsll.vi, vadd.vx, vlw12.v) to load
 * per-lane parameter values into VGPR v10/v12/v11.
 *
 * Encoding: CP_ASYNC_TENSOR with rs1=v10, rs2=v12, rd=v11
 *   0000000 01100 01010 011 01011 1000010 = 0x00C535C2
 * CP_ASYNC_FENCE: 0x00004042
 */
kernel void
tensor_dma_copy(__global const float *src,
                __global float *dst)
{
  __local float shared_buf[16]; /* 4x4 FP32 = 64 bytes */

  /* Parameter buffer: 3 arrays of 32 uint32_t for VRS1/VRS2/VRS3
   * Total: 96 * 4 = 384 bytes in shared memory */
  __local unsigned int param_buf[96];

  int lid = get_local_id(0);

  /* ---- 1. All threads read src global address from kernel args ---- */
  unsigned int src_addr;
  __asm__ volatile(
    "csrr %[addr], 0x803\n\t"      /* metadata_base */
    "lw   %[addr], 4(%[addr])\n\t" /* args_buffer_base */
    "lw   %[addr], 0(%[addr])\n\t" /* src pointer (args[0]) */
    : [addr] "=&r"(src_addr)
    :
    : "memory"
  );

  unsigned int shared_addr = (unsigned int)shared_buf;

  /* ---- 2. Thread 0 fills parameter buffer ---- */
  if (lid == 0) {
    /* Zero-init all 96 entries */
    for (int i = 0; i < 96; i++) param_buf[i] = 0;

    /* VRS1 params [0..31]: global tensor descriptor */
    param_buf[0]  = 6;           /* dataType = FLOAT32 */
    param_buf[1]  = 2;           /* tensorRank = 2 */
    param_buf[2]  = src_addr;    /* globalAddress */
    param_buf[3]  = 4;           /* globalDim[0] = 4 */
    param_buf[4]  = 4;           /* globalDim[1] = 4 */
    param_buf[5]  = 1;           /* globalDim[2] = 1 */
    param_buf[6]  = 1;           /* globalDim[3] = 1 */
    param_buf[7]  = 1;           /* globalDim[4] = 1 */
    param_buf[8]  = 16;          /* globalStrides[0] = 4*4 = 16 bytes per row */

    /* VRS2 params [32..63]: box descriptor */
    param_buf[32] = src_addr;    /* BoxAddress (same as tensor base for full copy) */
    param_buf[33] = 4;           /* boxDim[0] = 4 */
    param_buf[34] = 4;           /* boxDim[1] = 4 */
    param_buf[35] = 1;           /* boxDim[2] = 1 */
    param_buf[36] = 1;           /* boxDim[3] = 1 */
    param_buf[37] = 1;           /* boxDim[4] = 1 */
    param_buf[38] = 1;           /* elementStrides[0] = 1 */
    param_buf[39] = 1;           /* elementStrides[1] = 1 */
    param_buf[40] = 1;           /* elementStrides[2] = 1 */
    param_buf[41] = 1;           /* elementStrides[3] = 1 */
    param_buf[42] = 1;           /* elementStrides[4] = 1 */
    /* param_buf[43..45] = interleave/swizzle/L2/oobfill = 0 */

    /* VRS3 params [64..95]: destination */
    param_buf[64] = shared_addr; /* dst shared memory address */
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  /* ---- 3. Load per-lane values into VGPR using vector instructions ---- */
  unsigned int pb = (unsigned int)param_buf;

  __asm__ volatile(
    /* Build per-thread byte offsets in VGPR v10:
     * vid.v  v10        → v10[i] = i  (thread index)
     * vsll.vi v10,v10,2 → v10[i] = i*4  (byte offset)
     * vadd.vx v10,v10,pb → v10[i] = param_buf + i*4  (per-thread address)
     */
    "vid.v v10\n\t"
    "vsll.vi v10, v10, 2\n\t"
    "vadd.vx v10, v10, %[pb]\n\t"

    /* Load VRS3 (param_buf[64+i]) into VGPR v11 */
    "vlw12.v v11, 256(v10)\n\t"
    /* Load VRS2 (param_buf[32+i]) into VGPR v12 */
    "vlw12.v v12, 128(v10)\n\t"
    /* Load VRS1 (param_buf[i]) into VGPR v10 — must be last (overwrites base) */
    "vlw12.v v10, 0(v10)\n\t"

    /* CP_ASYNC_TENSOR: rs1=v10(VRS1), rs2=v12(VRS2), rd=v11(VRS3) */
    ".word 0x00C535C2\n\t"
    /* CP_ASYNC_FENCE */
    ".word 0x00004042\n\t"
    :
    : [pb] "r"(pb)
    : "memory"
  );

  /* ---- 4. Wait for DMA completion ---- */
  barrier(CLK_LOCAL_MEM_FENCE);

  /* ---- 5. Copy from shared memory to global dst ---- */
  if (lid < 16) {
    dst[lid] = shared_buf[lid];
  }
}
