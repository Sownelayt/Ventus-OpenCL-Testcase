/*
 * V1 bidirectional contention microbenchmark.
 *
 * The timed region contains DMA issue and completion only. G2S and S2G use
 * independent shared/global regions so no data dependency serializes them.
 */

#include "../common/ventus_tma_v2_opencl.h"

#ifndef MAX_REGION_BYTES
#define MAX_REGION_BYTES 4096u
#endif

#define DESC_WORDS 32u
#define RESULT_WORDS 16u
#define SCENARIO_G2S_S2G 0u
#define SCENARIO_S2G_G2S 1u

#define LOAD_WG_COORDS_V12(desc_arg) do {                                  \
  uint _desc = (uint)(desc_arg);                                             \
  __asm__ volatile(                                                          \
    "csrr t0, 0x808\n\t"                                                 \
    "slli t0, t0, 7\n\t"                                                \
    "add t0, %[desc], t0\n\t"                                            \
    "addi t0, t0, 256\n\t"                                              \
    "vid.v v12\n\t"                                                      \
    "vsll.vi v12, v12, 2\n\t"                                            \
    "vadd.vx v12, v12, t0\n\t"                                          \
    "vlw12.v v12, 0(v12)\n\t"                                             \
    : : [desc] "r"(_desc) : "t0", "memory");                           \
} while (0)

#define TIMED_BULK_G2S_S2G(input_arg, output_arg, shared_g2s_arg,             \
                           shared_s2g_arg, mbarrier_arg, results_arg,          \
                           bytes_arg, stride_arg) do {                         \
  uint _input = (uint)(input_arg);                                             \
  uint _output = (uint)(output_arg);                                           \
  uint _shared_g2s = (uint)(shared_g2s_arg);                                   \
  uint _shared_s2g = (uint)(shared_s2g_arg);                                   \
  uint _mbarrier = (uint)(mbarrier_arg);                                       \
  uint _results = (uint)(results_arg);                                         \
  uint _bytes = (uint)(bytes_arg);                                             \
  uint _stride = (uint)(stride_arg);                                           \
  __asm__ volatile(                                                            \
    "csrr t4, 0x808\n\t"                                                   \
    "mul t4, t4, %[stride]\n\t"                                            \
    "add t5, %[input], t4\n\t"                                              \
    "add t6, %[output], t4\n\t"                                             \
    "csrr t0, 0xB00\n\t"                                                   \
    ".insn r 0x42, 1, 0, %[shared_g2s], t5, %[bytes]\n\t"             \
    "csrr t1, 0xB00\n\t"                                                   \
    ".insn r 0x42, 3, 0, t6, %[shared_s2g], %[bytes]\n\t"             \
    "csrr t2, 0xB00\n\t"                                                   \
    ".word 0x00086042\n\t"                                                 \
    "mv x10, %[mbarrier]\n\t"                                               \
    "li x11, 0\n\t"                                                        \
    ".word 0x04b57042\n\t"                                                 \
    ".word 0x06007042\n\t"                                                 \
    ".word 0x000c6042\n\t"                                                 \
    "csrr t3, 0xB00\n\t"                                                   \
    "csrr t4, 0x814\n\t"                                                   \
    "csrr t5, 0x808\n\t"                                                   \
    "slli t5, t5, 6\n\t"                                                   \
    "add t5, %[results], t5\n\t"                                            \
    "sub t6, t3, t0\n\t"                                                   \
    "sw t6, 0(t5)\n\t"                                                     \
    "sub t6, t1, t0\n\t"                                                   \
    "sw t6, 4(t5)\n\t"                                                     \
    "sub t6, t2, t1\n\t"                                                   \
    "sw t6, 8(t5)\n\t"                                                     \
    "sub t6, t3, t2\n\t"                                                   \
    "sw t6, 12(t5)\n\t"                                                    \
    "sw t4, 16(t5)\n\t"                                                    \
    : : [input] "r"(_input), [output] "r"(_output),                        \
        [shared_g2s] "r"(_shared_g2s), [shared_s2g] "r"(_shared_s2g),     \
        [mbarrier] "r"(_mbarrier), [results] "r"(_results),               \
        [bytes] "r"(_bytes), [stride] "r"(_stride)                          \
    : "t0", "t1", "t2", "t3", "t4", "t5", "t6",                 \
      "x10", "x11", "memory");                                          \
} while (0)

#define TIMED_BULK_S2G_G2S(input_arg, output_arg, shared_g2s_arg,             \
                           shared_s2g_arg, mbarrier_arg, results_arg,          \
                           bytes_arg, stride_arg) do {                         \
  uint _input = (uint)(input_arg);                                             \
  uint _output = (uint)(output_arg);                                           \
  uint _shared_g2s = (uint)(shared_g2s_arg);                                   \
  uint _shared_s2g = (uint)(shared_s2g_arg);                                   \
  uint _mbarrier = (uint)(mbarrier_arg);                                       \
  uint _results = (uint)(results_arg);                                         \
  uint _bytes = (uint)(bytes_arg);                                             \
  uint _stride = (uint)(stride_arg);                                           \
  __asm__ volatile(                                                            \
    "csrr t4, 0x808\n\t"                                                   \
    "mul t4, t4, %[stride]\n\t"                                            \
    "add t5, %[input], t4\n\t"                                              \
    "add t6, %[output], t4\n\t"                                             \
    "csrr t0, 0xB00\n\t"                                                   \
    ".insn r 0x42, 3, 0, t6, %[shared_s2g], %[bytes]\n\t"             \
    "csrr t1, 0xB00\n\t"                                                   \
    ".insn r 0x42, 1, 0, %[shared_g2s], t5, %[bytes]\n\t"             \
    "csrr t2, 0xB00\n\t"                                                   \
    ".word 0x00086042\n\t"                                                 \
    "mv x10, %[mbarrier]\n\t"                                               \
    "li x11, 0\n\t"                                                        \
    ".word 0x04b57042\n\t"                                                 \
    ".word 0x06007042\n\t"                                                 \
    ".word 0x000c6042\n\t"                                                 \
    "csrr t3, 0xB00\n\t"                                                   \
    "csrr t4, 0x814\n\t"                                                   \
    "csrr t5, 0x808\n\t"                                                   \
    "slli t5, t5, 6\n\t"                                                   \
    "add t5, %[results], t5\n\t"                                            \
    "sub t6, t3, t0\n\t"                                                   \
    "sw t6, 0(t5)\n\t"                                                     \
    "sub t6, t1, t0\n\t"                                                   \
    "sw t6, 4(t5)\n\t"                                                     \
    "sub t6, t2, t1\n\t"                                                   \
    "sw t6, 8(t5)\n\t"                                                     \
    "sub t6, t3, t2\n\t"                                                   \
    "sw t6, 12(t5)\n\t"                                                    \
    "sw t4, 16(t5)\n\t"                                                    \
    : : [input] "r"(_input), [output] "r"(_output),                        \
        [shared_g2s] "r"(_shared_g2s), [shared_s2g] "r"(_shared_s2g),     \
        [mbarrier] "r"(_mbarrier), [results] "r"(_results),               \
        [bytes] "r"(_bytes), [stride] "r"(_stride)                          \
    : "t0", "t1", "t2", "t3", "t4", "t5", "t6",                 \
      "x10", "x11", "memory");                                          \
} while (0)

#define TIMED_TENSOR_G2S_S2G(desc_arg, shared_g2s_arg,                       \
                             shared_s2g_arg, mbarrier_arg, results_arg) do {    \
  uint _desc = (uint)(desc_arg);                                               \
  uint _shared_g2s = (uint)(shared_g2s_arg);                                   \
  uint _shared_s2g = (uint)(shared_s2g_arg);                                   \
  uint _mbarrier = (uint)(mbarrier_arg);                                       \
  uint _results = (uint)(results_arg);                                         \
  __asm__ volatile(                                                            \
    "csrr t0, 0xB00\n\t"                                                   \
    "mv x10, %[shared_g2s]\n\t"                                             \
    "mv x11, %[desc]\n\t"                                                   \
    ".word 0x00c5a542\n\t"                                                 \
    "csrr t1, 0xB00\n\t"                                                   \
    "mv x10, %[shared_s2g]\n\t"                                             \
    "addi x11, %[desc], 128\n\t"                                               \
    ".word 0x00c5c542\n\t"                                                 \
    "csrr t2, 0xB00\n\t"                                                   \
    ".word 0x00086042\n\t"                                                 \
    "mv x10, %[mbarrier]\n\t"                                               \
    "li x11, 0\n\t"                                                        \
    ".word 0x04b57042\n\t"                                                 \
    ".word 0x06007042\n\t"                                                 \
    ".word 0x000c6042\n\t"                                                 \
    "csrr t3, 0xB00\n\t"                                                   \
    "csrr t4, 0x814\n\t"                                                   \
    "csrr t5, 0x808\n\t"                                                   \
    "slli t5, t5, 6\n\t"                                                   \
    "add t5, %[results], t5\n\t"                                            \
    "sub t6, t3, t0\n\t"                                                   \
    "sw t6, 0(t5)\n\t"                                                     \
    "sub t6, t1, t0\n\t"                                                   \
    "sw t6, 4(t5)\n\t"                                                     \
    "sub t6, t2, t1\n\t"                                                   \
    "sw t6, 8(t5)\n\t"                                                     \
    "sub t6, t3, t2\n\t"                                                   \
    "sw t6, 12(t5)\n\t"                                                    \
    "sw t4, 16(t5)\n\t"                                                    \
    : : [desc] "r"(_desc),                                                   \
        [shared_g2s] "r"(_shared_g2s), [shared_s2g] "r"(_shared_s2g),     \
        [mbarrier] "r"(_mbarrier), [results] "r"(_results)                \
    : "t0", "t1", "t2", "t3", "t4", "t5", "t6",                 \
      "x10", "x11", "memory");                                          \
} while (0)

#define TIMED_TENSOR_S2G_G2S(desc_arg, shared_g2s_arg,                       \
                             shared_s2g_arg, mbarrier_arg, results_arg) do {    \
  uint _desc = (uint)(desc_arg);                                               \
  uint _shared_g2s = (uint)(shared_g2s_arg);                                   \
  uint _shared_s2g = (uint)(shared_s2g_arg);                                   \
  uint _mbarrier = (uint)(mbarrier_arg);                                       \
  uint _results = (uint)(results_arg);                                         \
  __asm__ volatile(                                                            \
    "csrr t0, 0xB00\n\t"                                                   \
    "mv x10, %[shared_s2g]\n\t"                                             \
    "addi x11, %[desc], 128\n\t"                                               \
    ".word 0x00c5c542\n\t"                                                 \
    "csrr t1, 0xB00\n\t"                                                   \
    "mv x10, %[shared_g2s]\n\t"                                             \
    "mv x11, %[desc]\n\t"                                                   \
    ".word 0x00c5a542\n\t"                                                 \
    "csrr t2, 0xB00\n\t"                                                   \
    ".word 0x00086042\n\t"                                                 \
    "mv x10, %[mbarrier]\n\t"                                               \
    "li x11, 0\n\t"                                                        \
    ".word 0x04b57042\n\t"                                                 \
    ".word 0x06007042\n\t"                                                 \
    ".word 0x000c6042\n\t"                                                 \
    "csrr t3, 0xB00\n\t"                                                   \
    "csrr t4, 0x814\n\t"                                                   \
    "csrr t5, 0x808\n\t"                                                   \
    "slli t5, t5, 6\n\t"                                                   \
    "add t5, %[results], t5\n\t"                                            \
    "sub t6, t3, t0\n\t"                                                   \
    "sw t6, 0(t5)\n\t"                                                     \
    "sub t6, t1, t0\n\t"                                                   \
    "sw t6, 4(t5)\n\t"                                                     \
    "sub t6, t2, t1\n\t"                                                   \
    "sw t6, 8(t5)\n\t"                                                     \
    "sub t6, t3, t2\n\t"                                                   \
    "sw t6, 12(t5)\n\t"                                                    \
    "sw t4, 16(t5)\n\t"                                                    \
    : : [desc] "r"(_desc),                                                   \
        [shared_g2s] "r"(_shared_g2s), [shared_s2g] "r"(_shared_s2g),     \
        [mbarrier] "r"(_mbarrier), [results] "r"(_results)                \
    : "t0", "t1", "t2", "t3", "t4", "t5", "t6",                 \
      "x10", "x11", "memory");                                          \
} while (0)

#define TIMED_BULK_G2S_ONLY(input_arg, shared_arg, mbarrier_arg, results_arg,  \
                            bytes_arg, stride_arg) do {                        \
  uint _input = (uint)(input_arg);                                             \
  uint _shared = (uint)(shared_arg);                                           \
  uint _mbarrier = (uint)(mbarrier_arg);                                       \
  uint _results = (uint)(results_arg);                                         \
  uint _bytes = (uint)(bytes_arg);                                             \
  uint _stride = (uint)(stride_arg);                                           \
  __asm__ volatile(                                                            \
    "csrr t4, 0x808\n\t"                                                   \
    "mul t4, t4, %[stride]\n\t"                                            \
    "add t5, %[input], t4\n\t"                                              \
    "csrr t0, 0xB00\n\t"                                                   \
    ".insn r 0x42, 1, 0, %[shared], t5, %[bytes]\n\t"                 \
    "csrr t1, 0xB00\n\t"                                                   \
    "mv x10, %[mbarrier]\n\t"                                               \
    "li x11, 0\n\t"                                                        \
    ".word 0x04b57042\n\t"                                                 \
    ".word 0x06007042\n\t"                                                 \
    "csrr t2, 0xB00\n\t"                                                   \
    "csrr t3, 0x814\n\t"                                                   \
    "csrr t4, 0x808\n\t"                                                   \
    "slli t4, t4, 6\n\t"                                                   \
    "add t4, %[results], t4\n\t"                                            \
    "sub t5, t2, t0\n\t"                                                   \
    "sw t5, 0(t4)\n\t"                                                     \
    "sub t5, t1, t0\n\t"                                                   \
    "sw t5, 4(t4)\n\t"                                                     \
    "sub t5, t2, t1\n\t"                                                   \
    "sw t5, 8(t4)\n\t"                                                     \
    "sw t3, 12(t4)\n\t"                                                    \
    : : [input] "r"(_input), [shared] "r"(_shared),                        \
        [mbarrier] "r"(_mbarrier), [results] "r"(_results),               \
        [bytes] "r"(_bytes), [stride] "r"(_stride)                          \
    : "t0", "t1", "t2", "t3", "t4", "t5", "x10", "x11",       \
      "memory");                                                             \
} while (0)

#define TIMED_BULK_S2G_ONLY(output_arg, shared_arg, results_arg, bytes_arg,    \
                            stride_arg) do {                                   \
  uint _output = (uint)(output_arg);                                           \
  uint _shared = (uint)(shared_arg);                                           \
  uint _results = (uint)(results_arg);                                         \
  uint _bytes = (uint)(bytes_arg);                                             \
  uint _stride = (uint)(stride_arg);                                           \
  __asm__ volatile(                                                            \
    "csrr t4, 0x808\n\t"                                                   \
    "mul t4, t4, %[stride]\n\t"                                            \
    "add t5, %[output], t4\n\t"                                             \
    "csrr t0, 0xB00\n\t"                                                   \
    ".insn r 0x42, 3, 0, t5, %[shared], %[bytes]\n\t"                 \
    "csrr t1, 0xB00\n\t"                                                   \
    ".word 0x00086042\n\t"                                                 \
    ".word 0x000c6042\n\t"                                                 \
    "csrr t2, 0xB00\n\t"                                                   \
    "csrr t3, 0x814\n\t"                                                   \
    "csrr t4, 0x808\n\t"                                                   \
    "slli t4, t4, 6\n\t"                                                   \
    "add t4, %[results], t4\n\t"                                            \
    "sub t5, t2, t0\n\t"                                                   \
    "sw t5, 16(t4)\n\t"                                                    \
    "sub t5, t1, t0\n\t"                                                   \
    "sw t5, 20(t4)\n\t"                                                    \
    "sub t5, t2, t1\n\t"                                                   \
    "sw t5, 24(t4)\n\t"                                                    \
    "sw t3, 28(t4)\n\t"                                                    \
    : : [output] "r"(_output), [shared] "r"(_shared),                      \
        [results] "r"(_results), [bytes] "r"(_bytes),                     \
        [stride] "r"(_stride)                                                \
    : "t0", "t1", "t2", "t3", "t4", "t5", "memory");            \
} while (0)

/* Keep the correctness copyback outside the measured DMA window.  A scalar
 * loop avoids the long compiler-generated vector loop that can desynchronize
 * the GVM lockstep checker in 16 KiB, multi-WG cases. */
#define SCALAR_G2S_READBACK(shared_arg) do {                                  \
  uint _shared = (uint)(shared_arg);                                          \
  uint _args;                                                                \
  uint _readback;                                                            \
  uint _bytes;                                                               \
  uint _stride;                                                              \
  uint _wg;                                                                  \
  uint _limit;                                                               \
  uint _tmp;                                                                 \
  __asm__ volatile(                                                           \
    "csrr %[args], 0x803\n\t"                                              \
    "lw %[args], 4(%[args])\n\t"                                           \
    "lw %[readback], 4(%[args])\n\t"                                       \
    "lw %[bytes], 12(%[args])\n\t"                                         \
    "lw %[stride], 16(%[args])\n\t"                                        \
    "csrr %[wg], 0x808\n\t"                                                \
    "mul %[stride], %[wg], %[stride]\n\t"                                   \
    "add %[readback], %[readback], %[stride]\n\t"                           \
    "add %[limit], %[shared], %[bytes]\n\t"                                  \
    "1:\n\t"                                                               \
    "lw %[tmp], 0(%[shared])\n\t"                                          \
    "sw %[tmp], 0(%[readback])\n\t"                                        \
    "addi %[readback], %[readback], 4\n\t"                                 \
    "addi %[shared], %[shared], 4\n\t"                                     \
    "bltu %[shared], %[limit], 1b\n\t"                                     \
    : [args] "=&r"(_args), [readback] "=&r"(_readback),                  \
      [bytes] "=&r"(_bytes), [stride] "=&r"(_stride),                    \
      [wg] "=&r"(_wg), [limit] "=&r"(_limit), [tmp] "=&r"(_tmp),      \
      [shared] "+&r"(_shared)                                             \
    : : "memory");                                                         \
} while (0)

#define TIMED_BULK_BURST(input_arg, output_arg, shared_g2s_arg,              \
                         shared_s2g_arg, mbarrier_arg, results_arg,          \
                         bytes_arg, depth_arg) do {                           \
  uint _input = (uint)(input_arg);                                           \
  uint _output = (uint)(output_arg);                                         \
  uint _shared_g2s = (uint)(shared_g2s_arg);                                 \
  uint _shared_s2g = (uint)(shared_s2g_arg);                                 \
  uint _mbarrier = (uint)(mbarrier_arg);                                     \
  uint _results = (uint)(results_arg);                                       \
  uint _bytes = (uint)(bytes_arg);                                           \
  uint _depth = (uint)(depth_arg);                                           \
  __asm__ volatile(                                                          \
    "csrw 0x814, x0\n\t"                                                 \
    "mv x10, %[mbarrier]\n\t"                                            \
    "mv x11, %[depth]\n\t"                                               \
    ".word 0x00b57042\n\t"                                               \
    ".word 0x06007042\n\t"                                               \
    "mv t1, %[input]\n\t"                                                \
    "mv t2, %[output]\n\t"                                               \
    "mv t3, %[shared_g2s]\n\t"                                           \
    "mv t4, %[shared_s2g]\n\t"                                           \
    "mv t5, %[depth]\n\t"                                               \
    "mv x12, %[results]\n\t"                                             \
    "mv x13, %[bytes]\n\t"                                               \
    "li x14, 0\n\t"                                                      \
    "li x15, 0\n\t"                                                      \
    "li x16, 0\n\t"                                                      \
    "li x17, 0\n\t"                                                      \
    "csrr t0, 0xB00\n\t"                                                \
    "1:\n\t"                                                            \
    "csrr x18, 0xB00\n\t"                                               \
    "mv x10, %[mbarrier]\n\t"                                            \
    "mv x11, x13\n\t"                                                   \
    ".word 0x02b57042\n\t"                                               \
    "csrr x19, 0xB00\n\t"                                               \
    "sub x19, x19, x18\n\t"                                             \
    "lw x10, 36(x12)\n\t"                                               \
    "add x10, x10, x19\n\t"                                             \
    "sw x10, 36(x12)\n\t"                                               \
    "csrr x18, 0xB00\n\t"                                               \
    ".insn r 0x42, 1, 0, t3, t1, x13\n\t"                               \
    "csrr x19, 0xB00\n\t"                                               \
    "sub x19, x19, x18\n\t"                                             \
    "add x14, x14, x19\n\t"                                             \
    "bgeu x16, x19, 2f\n\t"                                             \
    "mv x16, x19\n\t"                                                   \
    "2:\n\t"                                                            \
    "csrr x18, 0xB00\n\t"                                               \
    ".insn r 0x42, 3, 0, t2, t4, x13\n\t"                               \
    "csrr x19, 0xB00\n\t"                                               \
    "sub x19, x19, x18\n\t"                                             \
    "add x15, x15, x19\n\t"                                             \
    "bgeu x17, x19, 3f\n\t"                                             \
    "mv x17, x19\n\t"                                                   \
    "3:\n\t"                                                            \
    "csrr x18, 0xB00\n\t"                                               \
    ".word 0x00086042\n\t"                                               \
    "csrr x19, 0xB00\n\t"                                               \
    "sub x19, x19, x18\n\t"                                             \
    "lw x10, 32(x12)\n\t"                                               \
    "add x10, x10, x19\n\t"                                             \
    "sw x10, 32(x12)\n\t"                                               \
    "add t1, t1, x13\n\t"                                               \
    "add t2, t2, x13\n\t"                                               \
    "add t3, t3, x13\n\t"                                               \
    "add t4, t4, x13\n\t"                                               \
    "addi t5, t5, -1\n\t"                                               \
    "bnez t5, 1b\n\t"                                                   \
    "csrr t6, 0xB00\n\t"                                                \
    "mv x10, %[mbarrier]\n\t"                                            \
    "li x11, 0\n\t"                                                      \
    ".word 0x04b57042\n\t"                                               \
    ".word 0x06007042\n\t"                                               \
    "csrr x18, 0xB00\n\t"                                               \
    ".word 0x000c6042\n\t"                                               \
    "csrr x19, 0xB00\n\t"                                               \
    "sw x14, 4(x12)\n\t"                                                \
    "sw x15, 8(x12)\n\t"                                                \
    "sw x16, 12(x12)\n\t"                                               \
    "sw x17, 16(x12)\n\t"                                               \
    "sub x10, x18, t6\n\t"                                              \
    "sw x10, 20(x12)\n\t"                                               \
    "add x14, x14, x15\n\t"                                             \
    "lw x11, 32(x12)\n\t"                                               \
    "add x14, x14, x11\n\t"                                             \
    "add x14, x14, x10\n\t"                                             \
    "sub x10, x19, x18\n\t"                                             \
    "sw x10, 24(x12)\n\t"                                               \
    "add x14, x14, x10\n\t"                                             \
    "sw x14, 0(x12)\n\t"                                                \
    "csrr x10, 0x814\n\t"                                               \
    "sw x10, 28(x12)\n\t"                                               \
    : : [input] "r"(_input), [output] "r"(_output),                      \
        [shared_g2s] "r"(_shared_g2s), [shared_s2g] "r"(_shared_s2g),   \
        [mbarrier] "r"(_mbarrier), [results] "r"(_results),             \
        [bytes] "r"(_bytes), [depth] "r"(_depth)                         \
    : "t0", "t1", "t2", "t3", "t4", "t5", "t6",                  \
      "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", \
      "x18", "x19", "memory");                                         \
} while (0)

#define LOAD_COORD_BLOCK_V12(desc_arg, offset_arg) do {                     \
  uint _desc = (uint)(desc_arg);                                             \
  __asm__ volatile(                                                          \
    "addi t0, %[desc], %[offset]\n\t"                                     \
    "vid.v v12\n\t"                                                      \
    "vsll.vi v12, v12, 2\n\t"                                            \
    "vadd.vx v12, v12, t0\n\t"                                          \
    "vlw12.v v12, 0(v12)\n\t"                                             \
    : : [desc] "r"(_desc), [offset] "i"(offset_arg)                       \
    : "t0", "memory");                                                   \
} while (0)

#define TIMED_TENSOR_BURST_INIT(mbarrier_arg, results_arg, depth_arg) do {   \
  uint _mbarrier = (uint)(mbarrier_arg);                                     \
  uint _results = (uint)(results_arg);                                       \
  uint _depth = (uint)(depth_arg);                                           \
  __asm__ volatile(                                                          \
    "csrw 0x814, x0\n\t"                                                 \
    "mv x10, %[mbarrier]\n\t"                                            \
    "mv x11, %[depth]\n\t"                                               \
    ".word 0x00b57042\n\t"                                               \
    ".word 0x06007042\n\t"                                               \
    "mv t0, %[results]\n\t"                                              \
    "sw zero, 0(t0)\n\t"                                                 \
    "sw zero, 4(t0)\n\t"                                                 \
    "sw zero, 8(t0)\n\t"                                                 \
    "sw zero, 12(t0)\n\t"                                                \
    "sw zero, 16(t0)\n\t"                                                \
    "sw zero, 20(t0)\n\t"                                                \
    "sw zero, 24(t0)\n\t"                                                \
    "sw zero, 28(t0)\n\t"                                                \
    "sw zero, 32(t0)\n\t"                                                \
    "sw zero, 36(t0)\n\t"                                                \
    : : [mbarrier] "r"(_mbarrier), [results] "r"(_results),             \
        [depth] "r"(_depth)                                                \
    : "t0", "x10", "x11", "memory");                                  \
} while (0)

#define TIMED_TENSOR_BURST_COMMAND(desc_arg, shared_g2s_arg,                \
                                   shared_s2g_arg, mbarrier_arg,            \
                                   results_arg, bytes_arg, index_arg) do {   \
  uint _desc = (uint)(desc_arg);                                             \
  uint _shared_g2s = (uint)(shared_g2s_arg);                                 \
  uint _shared_s2g = (uint)(shared_s2g_arg);                                 \
  uint _mbarrier = (uint)(mbarrier_arg);                                     \
  uint _results = (uint)(results_arg);                                       \
  uint _bytes = (uint)(bytes_arg);                                           \
  __asm__ volatile(                                                          \
    "li t0, %[index]\n\t"                                                \
    "mul t0, t0, %[bytes]\n\t"                                          \
    "add t1, %[shared_g2s], t0\n\t"                                     \
    "add t2, %[shared_s2g], t0\n\t"                                     \
    "mv x12, %[results]\n\t"                                             \
    "csrr t3, 0xB00\n\t"                                                \
    "mv x10, %[mbarrier]\n\t"                                            \
    "mv x11, %[bytes]\n\t"                                               \
    ".word 0x02b57042\n\t"                                               \
    "csrr t4, 0xB00\n\t"                                                \
    "sub t4, t4, t3\n\t"                                               \
    "lw x13, 36(x12)\n\t"                                               \
    "add x13, x13, t4\n\t"                                              \
    "sw x13, 36(x12)\n\t"                                               \
    "csrr t3, 0xB00\n\t"                                                \
    "mv x10, t1\n\t"                                                    \
    "mv x11, %[desc]\n\t"                                               \
    ".word 0x00c5a542\n\t"                                               \
    "csrr t4, 0xB00\n\t"                                                \
    "mv x10, t2\n\t"                                                    \
    "addi x11, %[desc], 128\n\t"                                        \
    ".word 0x00c5c542\n\t"                                               \
    "csrr t5, 0xB00\n\t"                                                \
    ".word 0x00086042\n\t"                                               \
    "csrr t6, 0xB00\n\t"                                                \
    "sub t4, t4, t3\n\t"                                               \
    "sub t5, t5, t3\n\t"                                               \
    "sub t5, t5, t4\n\t"                                               \
    "sub t6, t6, t3\n\t"                                               \
    "sub t6, t6, t4\n\t"                                               \
    "sub t6, t6, t5\n\t"                                               \
    "lw x13, 4(x12)\n\t"                                                \
    "add x13, x13, t4\n\t"                                              \
    "sw x13, 4(x12)\n\t"                                                \
    "lw x13, 8(x12)\n\t"                                                \
    "add x13, x13, t5\n\t"                                              \
    "sw x13, 8(x12)\n\t"                                                \
    "lw x13, 12(x12)\n\t"                                               \
    "bgeu x13, t4, 1f\n\t"                                              \
    "sw t4, 12(x12)\n\t"                                                \
    "1:\n\t"                                                            \
    "lw x13, 16(x12)\n\t"                                               \
    "bgeu x13, t5, 2f\n\t"                                              \
    "sw t5, 16(x12)\n\t"                                                \
    "2:\n\t"                                                            \
    "lw x13, 32(x12)\n\t"                                               \
    "add x13, x13, t6\n\t"                                              \
    "sw x13, 32(x12)\n\t"                                               \
    : : [desc] "r"(_desc), [shared_g2s] "r"(_shared_g2s),               \
        [shared_s2g] "r"(_shared_s2g), [mbarrier] "r"(_mbarrier),       \
        [results] "r"(_results), [bytes] "r"(_bytes),                   \
        [index] "i"(index_arg)                                             \
    : "t0", "t1", "t2", "t3", "t4", "t5", "t6",                  \
      "x10", "x11", "x12", "x13", "memory");                        \
} while (0)

#define TIMED_TENSOR_BURST_FINISH(mbarrier_arg, results_arg) do {           \
  uint _mbarrier = (uint)(mbarrier_arg);                                     \
  uint _results = (uint)(results_arg);                                       \
  __asm__ volatile(                                                          \
    "csrr t0, 0xB00\n\t"                                                \
    "mv x10, %[mbarrier]\n\t"                                            \
    "li x11, 0\n\t"                                                      \
    ".word 0x04b57042\n\t"                                               \
    ".word 0x06007042\n\t"                                               \
    "csrr t1, 0xB00\n\t"                                                \
    ".word 0x000c6042\n\t"                                               \
    "csrr t2, 0xB00\n\t"                                                \
    "mv t3, %[results]\n\t"                                              \
    "sub t4, t1, t0\n\t"                                               \
    "sw t4, 20(t3)\n\t"                                                 \
    "sub t5, t2, t1\n\t"                                               \
    "sw t5, 24(t3)\n\t"                                                 \
    "lw t6, 4(t3)\n\t"                                                  \
    "lw x12, 8(t3)\n\t"                                                 \
    "add t6, t6, x12\n\t"                                               \
    "lw x12, 32(t3)\n\t"                                                \
    "add t6, t6, x12\n\t"                                               \
    "add t6, t6, t4\n\t"                                                \
    "add t6, t6, t5\n\t"                                                \
    "sw t6, 0(t3)\n\t"                                                  \
    "csrr x12, 0x814\n\t"                                               \
    "sw x12, 28(t3)\n\t"                                                \
    : : [mbarrier] "r"(_mbarrier), [results] "r"(_results)              \
    : "t0", "t1", "t2", "t3", "t4", "t5", "t6",                  \
      "x10", "x11", "x12", "memory");                                \
} while (0)

#define TIMED_TENSOR_G2S_ONLY(desc_arg, shared_arg, mbarrier_arg,             \
                              results_arg) do {                                \
  uint _desc = (uint)(desc_arg);                                               \
  uint _shared = (uint)(shared_arg);                                           \
  uint _mbarrier = (uint)(mbarrier_arg);                                       \
  uint _results = (uint)(results_arg);                                         \
  __asm__ volatile(                                                            \
    "csrr t0, 0xB00\n\t"                                                   \
    "mv x10, %[shared]\n\t"                                                \
    "mv x11, %[desc]\n\t"                                                  \
    ".word 0x00c5a542\n\t"                                                 \
    "csrr t1, 0xB00\n\t"                                                   \
    "mv x10, %[mbarrier]\n\t"                                               \
    "li x11, 0\n\t"                                                        \
    ".word 0x04b57042\n\t"                                                 \
    ".word 0x06007042\n\t"                                                 \
    "csrr t2, 0xB00\n\t"                                                   \
    "csrr t3, 0x814\n\t"                                                   \
    "csrr t4, 0x808\n\t"                                                   \
    "slli t4, t4, 6\n\t"                                                   \
    "add t4, %[results], t4\n\t"                                            \
    "sub t5, t2, t0\n\t"                                                   \
    "sw t5, 0(t4)\n\t"                                                     \
    "sub t5, t1, t0\n\t"                                                   \
    "sw t5, 4(t4)\n\t"                                                     \
    "sub t5, t2, t1\n\t"                                                   \
    "sw t5, 8(t4)\n\t"                                                     \
    "sw t3, 12(t4)\n\t"                                                    \
    : : [desc] "r"(_desc), [shared] "r"(_shared),                          \
        [mbarrier] "r"(_mbarrier), [results] "r"(_results)                \
    : "t0", "t1", "t2", "t3", "t4", "t5", "x10", "x11",       \
      "memory");                                                             \
} while (0)

#define TIMED_TENSOR_S2G_ONLY(desc_arg, shared_arg, results_arg) do {          \
  uint _desc = (uint)(desc_arg);                                               \
  uint _shared = (uint)(shared_arg);                                           \
  uint _results = (uint)(results_arg);                                         \
  __asm__ volatile(                                                            \
    "csrr t0, 0xB00\n\t"                                                   \
    "mv x10, %[shared]\n\t"                                                \
    "addi x11, %[desc], 128\n\t"                                               \
    ".word 0x00c5c542\n\t"                                                 \
    "csrr t1, 0xB00\n\t"                                                   \
    ".word 0x00086042\n\t"                                                 \
    ".word 0x000c6042\n\t"                                                 \
    "csrr t2, 0xB00\n\t"                                                   \
    "csrr t3, 0x814\n\t"                                                   \
    "csrr t4, 0x808\n\t"                                                   \
    "slli t4, t4, 6\n\t"                                                   \
    "add t4, %[results], t4\n\t"                                            \
    "sub t5, t2, t0\n\t"                                                   \
    "sw t5, 16(t4)\n\t"                                                    \
    "sub t5, t1, t0\n\t"                                                   \
    "sw t5, 20(t4)\n\t"                                                    \
    "sub t5, t2, t1\n\t"                                                   \
    "sw t5, 24(t4)\n\t"                                                    \
    "sw t3, 28(t4)\n\t"                                                    \
    : : [desc] "r"(_desc), [shared] "r"(_shared),                          \
        [results] "r"(_results)                                              \
    : "t0", "t1", "t2", "t3", "t4", "t5", "x10", "x11",       \
      "memory");                                                             \
} while (0)

static uint read_cycle_lo(void)
{
  uint value;
  __asm__ volatile("csrr %0, 0xB00\n\t" : "=r"(value) :: "memory");
  return value;
}

static uint pattern_word(uint idx)
{
  uint base = idx << 2;
  uint b0 = (base * 13u + 0x31u) & 0xffu;
  uint b1 = ((base + 1u) * 13u + 0x31u) & 0xffu;
  uint b2 = ((base + 2u) * 13u + 0x31u) & 0xffu;
  uint b3 = ((base + 3u) * 13u + 0x31u) & 0xffu;
  return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

static void init_shared(__local uint *shared_g2s,
                        __local uint *shared_s2g,
                        uint words, uint global_base, uint lid)
{
  for (uint idx = lid; idx < words; idx += get_local_size(0)) {
    shared_g2s[idx] = 0u;
    shared_s2g[idx] = pattern_word(global_base + idx);
  }
}

static void copy_readback(__local uint *shared_g2s,
                          __global uint *readback,
                          uint words, uint global_base, uint lid)
{
  for (uint idx = lid; idx < words; idx += get_local_size(0)) {
    readback[global_base + idx] = shared_g2s[idx];
  }
}

kernel void setup_desc_kernel(__global uint *desc,
                              __global const uint *input,
                              __global uint *output)
{
  if (get_global_id(0) == 0u) {
    desc[2] = (uint)input;
    desc[DESC_WORDS + 2u] = (uint)output;
  }
}

kernel void contention_g2s_s2g_bulk_kernel(
    __global const uint *input, __global uint *bundle,
    __global uint *results, uint bytes, uint region_stride)
{
  __local uchar shared_raw[(2u * MAX_REGION_BYTES) + 128u];
  __local uint *shared_g2s = (__local uint *)
      (((uint)shared_raw + 127u) & ~127u);
  __local uint *shared_s2g = (__local uint *)
      ((__local uchar *)shared_g2s + MAX_REGION_BYTES);
  __local uint mbarrier[2] __attribute__((aligned(8)));
  uint lid = get_local_id(0);
  uint words = bytes >> 2;
  uint base_word = get_group_id(0) * words;
  init_shared(shared_g2s, shared_s2g, words, base_word, lid);
  if (lid < RESULT_WORDS) {
    results[get_group_id(0) * RESULT_WORDS + lid] = 0u;
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_MBARRIER_INIT(mbarrier, 1u);
    VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(mbarrier, bytes);
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
    TIMED_BULK_G2S_S2G(input, bundle, shared_g2s, shared_s2g, mbarrier,
                       results, bytes, region_stride);
  }
}

kernel void contention_s2g_g2s_bulk_kernel(
    __global const uint *input, __global uint *bundle,
    __global uint *results, uint bytes, uint region_stride)
{
  __local uchar shared_raw[(2u * MAX_REGION_BYTES) + 128u];
  __local uint *shared_g2s = (__local uint *)
      (((uint)shared_raw + 127u) & ~127u);
  __local uint *shared_s2g = (__local uint *)
      ((__local uchar *)shared_g2s + MAX_REGION_BYTES);
  __local uint mbarrier[2] __attribute__((aligned(8)));
  uint lid = get_local_id(0);
  uint words = bytes >> 2;
  uint base_word = get_group_id(0) * words;
  init_shared(shared_g2s, shared_s2g, words, base_word, lid);
  if (lid < RESULT_WORDS) {
    results[get_group_id(0) * RESULT_WORDS + lid] = 0u;
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_MBARRIER_INIT(mbarrier, 1u);
    VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(mbarrier, bytes);
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
    TIMED_BULK_S2G_G2S(input, bundle, shared_g2s, shared_s2g, mbarrier,
                       results, bytes, region_stride);
  }
}

kernel void contention_g2s_s2g_tensor_kernel(
    __global uint *desc, __global uint *results, uint bytes)
{
  __local uchar shared_raw[(2u * MAX_REGION_BYTES) + 128u];
  __local uint *shared_g2s = (__local uint *)
      (((uint)shared_raw + 127u) & ~127u);
  __local uint *shared_s2g = (__local uint *)
      ((__local uchar *)shared_g2s + MAX_REGION_BYTES);
  __local uint mbarrier[2] __attribute__((aligned(8)));
  uint lid = get_local_id(0);
  uint words = bytes >> 2;
  uint base_word = get_group_id(0) * words;
  init_shared(shared_g2s, shared_s2g, words, base_word, lid);
  if (lid < RESULT_WORDS) {
    results[get_group_id(0) * RESULT_WORDS + lid] = 0u;
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  LOAD_WG_COORDS_V12(desc);
  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_MBARRIER_INIT(mbarrier, 1u);
    VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(mbarrier, bytes);
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
    TIMED_TENSOR_G2S_S2G(desc, shared_g2s, shared_s2g, mbarrier,
                         results);
  }
}

kernel void contention_s2g_g2s_tensor_kernel(
    __global uint *desc, __global uint *results, uint bytes)
{
  __local uchar shared_raw[(2u * MAX_REGION_BYTES) + 128u];
  __local uint *shared_g2s = (__local uint *)
      (((uint)shared_raw + 127u) & ~127u);
  __local uint *shared_s2g = (__local uint *)
      ((__local uchar *)shared_g2s + MAX_REGION_BYTES);
  __local uint mbarrier[2] __attribute__((aligned(8)));
  uint lid = get_local_id(0);
  uint words = bytes >> 2;
  uint base_word = get_group_id(0) * words;
  init_shared(shared_g2s, shared_s2g, words, base_word, lid);
  if (lid < RESULT_WORDS) {
    results[get_group_id(0) * RESULT_WORDS + lid] = 0u;
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  LOAD_WG_COORDS_V12(desc);
  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_MBARRIER_INIT(mbarrier, 1u);
    VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(mbarrier, bytes);
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
    TIMED_TENSOR_S2G_G2S(desc, shared_g2s, shared_s2g, mbarrier,
                         results);
  }
}

kernel void contention_g2s_only_bulk_kernel(
    __global const uint *input, __global uint *readback,
    __global uint *results, uint bytes, uint region_stride)
{
  __local uchar shared_raw[MAX_REGION_BYTES + 128u];
  __local uint *shared = (__local uint *)
      (((uint)shared_raw + 127u) & ~127u);
  __local uint mbarrier[2] __attribute__((aligned(8)));
  uint lid = get_local_id(0);
  uint words = bytes >> 2;
  uint logical_base = get_group_id(0) * words;
  uint physical_base = (get_group_id(0) * region_stride) >> 2;
  for (uint idx = lid; idx < words; idx += get_local_size(0)) shared[idx] = 0u;
  if (lid < RESULT_WORDS) {
    results[get_group_id(0) * RESULT_WORDS + lid] = 0u;
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_MBARRIER_INIT(mbarrier, 1u);
    VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(mbarrier, bytes);
    TIMED_BULK_G2S_ONLY(input, shared, mbarrier, results, bytes,
                        region_stride);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0u) {
    SCALAR_G2S_READBACK(shared);
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  (void)logical_base;
}

kernel void contention_s2g_only_bulk_kernel(
    __global uint *output, __global uint *results,
    uint bytes, uint region_stride)
{
  __local uchar shared_raw[MAX_REGION_BYTES + 128u];
  __local uint *shared = (__local uint *)
      (((uint)shared_raw + 127u) & ~127u);
  uint lid = get_local_id(0);
  uint words = bytes >> 2;
  uint logical_base = get_group_id(0) * words;
  for (uint idx = lid; idx < words; idx += get_local_size(0)) {
    shared[idx] = pattern_word(logical_base + idx);
  }
  if (lid < RESULT_WORDS) {
    results[get_group_id(0) * RESULT_WORDS + lid] = 0u;
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
    TIMED_BULK_S2G_ONLY(output, shared, results, bytes, region_stride);
  }
}

kernel void contention_g2s_only_tensor_kernel(
    __global uint *desc, __global uint *readback,
    __global uint *results, uint bytes, uint region_stride)
{
  __local uchar shared_raw[MAX_REGION_BYTES + 128u];
  __local uint *shared = (__local uint *)
      (((uint)shared_raw + 127u) & ~127u);
  __local uint mbarrier[2] __attribute__((aligned(8)));
  uint lid = get_local_id(0);
  uint words = bytes >> 2;
  uint physical_base = (get_group_id(0) * region_stride) >> 2;
  for (uint idx = lid; idx < words; idx += get_local_size(0)) shared[idx] = 0u;
  if (lid < RESULT_WORDS) {
    results[get_group_id(0) * RESULT_WORDS + lid] = 0u;
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  LOAD_WG_COORDS_V12(desc);
  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_MBARRIER_INIT(mbarrier, 1u);
    VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(mbarrier, bytes);
    TIMED_TENSOR_G2S_ONLY(desc, shared, mbarrier, results);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0u) {
    SCALAR_G2S_READBACK(shared);
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

kernel void contention_s2g_only_tensor_kernel(
    __global uint *desc, __global uint *results, uint bytes)
{
  __local uchar shared_raw[MAX_REGION_BYTES + 128u];
  __local uint *shared = (__local uint *)
      (((uint)shared_raw + 127u) & ~127u);
  uint lid = get_local_id(0);
  uint words = bytes >> 2;
  uint logical_base = get_group_id(0) * words;
  for (uint idx = lid; idx < words; idx += get_local_size(0)) {
    shared[idx] = pattern_word(logical_base + idx);
  }
  if (lid < RESULT_WORDS) {
    results[get_group_id(0) * RESULT_WORDS + lid] = 0u;
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  LOAD_WG_COORDS_V12(desc);
  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
    TIMED_TENSOR_S2G_ONLY(desc, shared, results);
  }
}

kernel void contention_dualwarp_bulk_kernel(
    __global const uint *input, __global uint *bundle,
    __global uint *results, uint bytes, uint region_stride)
{
  __local uchar shared_raw[(2u * MAX_REGION_BYTES) + 128u];
  __local uint *shared_g2s = (__local uint *)
      (((uint)shared_raw + 127u) & ~127u);
  __local uint *shared_s2g = (__local uint *)
      ((__local uchar *)shared_g2s + MAX_REGION_BYTES);
  __local uint mbarrier[2] __attribute__((aligned(8)));
  uint lid = get_local_id(0);
  uint words = bytes >> 2;
  uint base_word = get_group_id(0) * words;
  init_shared(shared_g2s, shared_s2g, words, base_word, lid);
  if (lid < RESULT_WORDS) {
    results[get_group_id(0) * RESULT_WORDS + lid] = 0u;
  }
  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_MBARRIER_INIT(mbarrier, 1u);
    VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(mbarrier, bytes);
  } else if (lid == 32u) {
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0u) {
    TIMED_BULK_G2S_ONLY(input, shared_g2s, mbarrier, results, bytes,
                        region_stride);
  } else if (lid == 32u) {
    TIMED_BULK_S2G_ONLY(bundle, shared_s2g, results, bytes, region_stride);
  }
}

kernel void contention_dualwarp_tensor_kernel(
    __global uint *desc, __global uint *results, uint bytes)
{
  __local uchar shared_raw[(2u * MAX_REGION_BYTES) + 128u];
  __local uint *shared_g2s = (__local uint *)
      (((uint)shared_raw + 127u) & ~127u);
  __local uint *shared_s2g = (__local uint *)
      ((__local uchar *)shared_g2s + MAX_REGION_BYTES);
  __local uint mbarrier[2] __attribute__((aligned(8)));
  uint lid = get_local_id(0);
  uint words = bytes >> 2;
  uint base_word = get_group_id(0) * words;
  init_shared(shared_g2s, shared_s2g, words, base_word, lid);
  if (lid < RESULT_WORDS) {
    results[get_group_id(0) * RESULT_WORDS + lid] = 0u;
  }
  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_MBARRIER_INIT(mbarrier, 1u);
    VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(mbarrier, bytes);
  } else if (lid == 32u) {
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  LOAD_WG_COORDS_V12(desc);
  if (lid == 0u) {
    TIMED_TENSOR_G2S_ONLY(desc, shared_g2s, mbarrier, results);
  } else if (lid == 32u) {
    TIMED_TENSOR_S2G_ONLY(desc, shared_s2g, results);
  }
}

kernel void contention_burst_bulk_kernel(
    __global const uint *input, __global uint *bundle,
    __global uint *results, uint bytes, uint depth)
{
  __local uchar shared_raw[(2u * MAX_REGION_BYTES) + 128u];
  __local uint *shared_g2s = (__local uint *)
      (((uint)shared_raw + 127u) & ~127u);
  __local uint *shared_s2g = (__local uint *)
      ((__local uchar *)shared_g2s + MAX_REGION_BYTES);
  __local uint mbarrier[2] __attribute__((aligned(8)));
  uint lid = get_local_id(0);
  uint words = bytes >> 2;
  uint total_words = depth * words;
  init_shared(shared_g2s, shared_s2g, total_words, 0u, lid);
  if (lid < RESULT_WORDS) results[lid] = 0u;
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  if (lid == 0u) {
    TIMED_BULK_BURST(input, bundle, shared_g2s, shared_s2g, mbarrier,
                     results, bytes, depth);
  }
}

kernel void contention_burst_tensor_kernel(
    __global uint *desc, __global uint *results, uint bytes, uint depth)
{
  __local uchar shared_raw[(2u * MAX_REGION_BYTES) + 128u];
  __local uint *shared_g2s = (__local uint *)
      (((uint)shared_raw + 127u) & ~127u);
  __local uint *shared_s2g = (__local uint *)
      ((__local uchar *)shared_g2s + MAX_REGION_BYTES);
  __local uint mbarrier[2] __attribute__((aligned(8)));
  uint lid = get_local_id(0);
  uint words = bytes >> 2;
  uint total_words = depth * words;
  init_shared(shared_g2s, shared_s2g, total_words, 0u, lid);
  if (lid == 0u) {
    TIMED_TENSOR_BURST_INIT(mbarrier, results, depth);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  LOAD_COORD_BLOCK_V12(desc, 256);
  if (lid == 0u) {
    TIMED_TENSOR_BURST_COMMAND(desc, shared_g2s, shared_s2g, mbarrier,
                               results, bytes, 0);
  }
  LOAD_COORD_BLOCK_V12(desc, 384);
  if (lid == 0u) {
    TIMED_TENSOR_BURST_COMMAND(desc, shared_g2s, shared_s2g, mbarrier,
                               results, bytes, 1);
  }
  if (depth > 2u) {
    LOAD_COORD_BLOCK_V12(desc, 512);
    if (lid == 0u) {
      TIMED_TENSOR_BURST_COMMAND(desc, shared_g2s, shared_s2g, mbarrier,
                                 results, bytes, 2);
    }
    LOAD_COORD_BLOCK_V12(desc, 640);
    if (lid == 0u) {
      TIMED_TENSOR_BURST_COMMAND(desc, shared_g2s, shared_s2g, mbarrier,
                                 results, bytes, 3);
    }
  }
  if (lid == 0u) {
    TIMED_TENSOR_BURST_FINISH(mbarrier, results);
  }
}
