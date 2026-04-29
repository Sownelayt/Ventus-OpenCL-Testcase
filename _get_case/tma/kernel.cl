__kernel void dma_3(__global const float *input, __global float *output) {
  __local float shared[16];
  int lid = get_local_id(0);

  if (lid == 0) {
    unsigned int src_addr;
    unsigned int dst_addr = (unsigned int)shared;
    unsigned int byte_count = 16 * sizeof(float);

    __asm__ __volatile__(
      "csrr %[src], 0x803\n\t"
      "lw   %[src], 4(%[src])\n\t"
      "lw   %[src], 0(%[src])\n\t"
      "mv   x10, %[src]\n\t"
      "mv   x11, %[dst]\n\t"
      "mv   x12, %[size]\n\t"
      ".word 0x00c515c2\n\t"
      ".word 0x00004042\n\t"
      : [src] "=&r"(src_addr)
      : [dst] "r"(dst_addr), [size] "r"(byte_count)
      : "memory", "x10", "x11", "x12"
    );
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid < 16) {
    output[lid] = shared[lid];
  }
}
