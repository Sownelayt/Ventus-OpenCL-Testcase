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
