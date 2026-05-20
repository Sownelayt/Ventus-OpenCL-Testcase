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
