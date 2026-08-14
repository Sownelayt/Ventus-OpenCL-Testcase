#ifndef VENTUS_TMA_V2_OPENCL_H
#define VENTUS_TMA_V2_OPENCL_H

/*
 * Ventus TMA v2 OpenCL instruction wrappers.
 *
 * Tensor instructions use x10=shared, x11=descriptor, v12=coordinates.
 * mbarrier instructions use x10=64-bit shared object and x11=value.
 * The raw words are kept here so kernel sources do not duplicate ABI magic.
 */

typedef struct __attribute__((aligned(8))) {
  ulong opaque;
} ventus_tma_mbarrier_t;

#define VENTUS_TMA_LOAD_COORDS_V12(coords_ptr_arg) do {                     \
  uint _tma_coords = (uint)(coords_ptr_arg);                                \
  __asm__ volatile(                                                         \
    "vid.v v12\n\t"                                                       \
    "vsll.vi v12, v12, 2\n\t"                                             \
    "vadd.vx v12, v12, %[coords]\n\t"                                     \
    "vlw12.v v12, 0(v12)\n\t"                                             \
    : : [coords] "r"(_tma_coords) : "memory");                            \
} while (0)

#define VENTUS_TMA_BULK_G2S(shared_dst_arg, global_src_arg, bytes_arg) do {  \
  uint _tma_dst = (uint)(shared_dst_arg);                                   \
  uint _tma_src = (uint)(global_src_arg);                                   \
  uint _tma_bytes = (uint)(bytes_arg);                                      \
  __asm__ volatile(                                                         \
    ".insn r 0x42, 1, 0, %[dst], %[src], %[bytes]\n\t"                  \
    : : [dst] "r"(_tma_dst), [src] "r"(_tma_src),                         \
        [bytes] "r"(_tma_bytes) : "memory");                              \
} while (0)

#define VENTUS_TMA_TENSOR_G2S(shared_dst_arg, descriptor_arg) do {          \
  uint _tma_shared = (uint)(shared_dst_arg);                                \
  uint _tma_desc = (uint)(descriptor_arg);                                  \
  __asm__ volatile(                                                         \
    "mv x10, %[shared]\n\t"                                                \
    "mv x11, %[desc]\n\t"                                                  \
    ".word 0x00c5a542\n\t"                                                \
    : : [shared] "r"(_tma_shared), [desc] "r"(_tma_desc)                  \
    : "x10", "x11", "memory");                                          \
} while (0)

#define VENTUS_TMA_BULK_S2G(global_dst_arg, shared_src_arg, bytes_arg) do {  \
  uint _tma_dst = (uint)(global_dst_arg);                                   \
  uint _tma_src = (uint)(shared_src_arg);                                   \
  uint _tma_bytes = (uint)(bytes_arg);                                      \
  __asm__ volatile(                                                         \
    ".insn r 0x42, 3, 0, %[dst], %[src], %[bytes]\n\t"                  \
    : : [dst] "r"(_tma_dst), [src] "r"(_tma_src),                         \
        [bytes] "r"(_tma_bytes) : "memory");                              \
} while (0)

/*
 * CUDA-compatible cp.reduce.async.bulk subset implemented by Ventus.
 * Every operation is element-wise atomic and uses S2G bulk-group completion.
 * Supported types are u32/s32 for arithmetic and b32 for bitwise operations.
 */
#define VENTUS_TMA_BULK_REDUCE_WORD(global_dst_arg, shared_src_arg,          \
                                    bytes_arg, funct7_arg) do {              \
  uint _tma_dst = (uint)(global_dst_arg);                                    \
  uint _tma_src = (uint)(shared_src_arg);                                    \
  uint _tma_bytes = (uint)(bytes_arg);                                       \
  __asm__ volatile(                                                          \
    ".insn r 0x42, 3, " funct7_arg ", %[dst], %[src], %[bytes]\n\t"        \
    : : [dst] "r"(_tma_dst), [src] "r"(_tma_src),                          \
        [bytes] "r"(_tma_bytes) : "memory");                               \
} while (0)

#define VENTUS_TMA_BULK_REDUCE_ADD_U32(d, s, n) \
  VENTUS_TMA_BULK_REDUCE_WORD((d), (s), (n), "0x10")
#define VENTUS_TMA_BULK_REDUCE_ADD_S32(d, s, n) \
  VENTUS_TMA_BULK_REDUCE_WORD((d), (s), (n), "0x14")
#define VENTUS_TMA_BULK_REDUCE_MIN_U32(d, s, n) \
  VENTUS_TMA_BULK_REDUCE_WORD((d), (s), (n), "0x20")
#define VENTUS_TMA_BULK_REDUCE_MIN_S32(d, s, n) \
  VENTUS_TMA_BULK_REDUCE_WORD((d), (s), (n), "0x24")
#define VENTUS_TMA_BULK_REDUCE_MAX_U32(d, s, n) \
  VENTUS_TMA_BULK_REDUCE_WORD((d), (s), (n), "0x30")
#define VENTUS_TMA_BULK_REDUCE_MAX_S32(d, s, n) \
  VENTUS_TMA_BULK_REDUCE_WORD((d), (s), (n), "0x34")
#define VENTUS_TMA_BULK_REDUCE_AND_B32(d, s, n) \
  VENTUS_TMA_BULK_REDUCE_WORD((d), (s), (n), "0x48")
#define VENTUS_TMA_BULK_REDUCE_OR_B32(d, s, n) \
  VENTUS_TMA_BULK_REDUCE_WORD((d), (s), (n), "0x58")
#define VENTUS_TMA_BULK_REDUCE_XOR_B32(d, s, n) \
  VENTUS_TMA_BULK_REDUCE_WORD((d), (s), (n), "0x68")

#define VENTUS_TMA_TENSOR_S2G(shared_src_arg, descriptor_arg) do {          \
  uint _tma_shared = (uint)(shared_src_arg);                                \
  uint _tma_desc = (uint)(descriptor_arg);                                  \
  __asm__ volatile(                                                         \
    "mv x10, %[shared]\n\t"                                                \
    "mv x11, %[desc]\n\t"                                                  \
    ".word 0x00c5c542\n\t"                                                \
    : : [shared] "r"(_tma_shared), [desc] "r"(_tma_desc)                  \
    : "x10", "x11", "memory");                                          \
} while (0)

/*
 * Tensor reduce submits one 32-bit AMO per in-bounds element to the shared
 * L2 atomic endpoint.  S2G group completion waits for every final AMO ack.
 */
#define VENTUS_TMA_TENSOR_REDUCE_WORD(shared_src_arg, descriptor_arg, word) do { \
  uint _tma_shared = (uint)(shared_src_arg);                                \
  uint _tma_desc = (uint)(descriptor_arg);                                  \
  __asm__ volatile(                                                         \
    "mv x10, %[shared]\n\t"                                                \
    "mv x11, %[desc]\n\t"                                                  \
    ".word " word "\n\t"                                                   \
    : : [shared] "r"(_tma_shared), [desc] "r"(_tma_desc)                  \
    : "x10", "x11", "memory");                                          \
} while (0)

#define VENTUS_TMA_TENSOR_REDUCE_ADD(s, d) \
  VENTUS_TMA_TENSOR_REDUCE_WORD((s), (d), "0x20c5c542")
#define VENTUS_TMA_TENSOR_REDUCE_MIN(s, d) \
  VENTUS_TMA_TENSOR_REDUCE_WORD((s), (d), "0x40c5c542")
#define VENTUS_TMA_TENSOR_REDUCE_MAX(s, d) \
  VENTUS_TMA_TENSOR_REDUCE_WORD((s), (d), "0x60c5c542")
#define VENTUS_TMA_TENSOR_REDUCE_AND(s, d) \
  VENTUS_TMA_TENSOR_REDUCE_WORD((s), (d), "0x80c5c542")
#define VENTUS_TMA_TENSOR_REDUCE_OR(s, d) \
  VENTUS_TMA_TENSOR_REDUCE_WORD((s), (d), "0xa0c5c542")
#define VENTUS_TMA_TENSOR_REDUCE_XOR(s, d) \
  VENTUS_TMA_TENSOR_REDUCE_WORD((s), (d), "0xc0c5c542")

#define VENTUS_TMA_PREFETCH_TENSORMAP(descriptor_arg) do {                  \
  uint _tma_desc = (uint)(descriptor_arg);                                  \
  __asm__ volatile(                                                         \
    "mv x11, %[desc]\n\t"                                                  \
    ".word 0x0005d042\n\t"                                                \
    : : [desc] "r"(_tma_desc) : "x11", "memory");                        \
} while (0)

#define VENTUS_TMA_INVALIDATE_TENSORMAP(descriptor_arg) do {                \
  uint _tma_desc = (uint)(descriptor_arg);                                  \
  __asm__ volatile(                                                         \
    "mv x11, %[desc]\n\t"                                                  \
    ".word 0x0805d042\n\t"                                                \
    : : [desc] "r"(_tma_desc) : "x11", "memory");                        \
} while (0)

#define VENTUS_TMA_MBARRIER_INIT(address_arg, arrivals_arg) do {            \
  uint _tma_address = (uint)(address_arg);                                  \
  uint _tma_value = (uint)(arrivals_arg);                                   \
  __asm__ volatile(                                                         \
    "mv x10, %[address]\n\t"                                               \
    "mv x11, %[value]\n\t"                                                 \
    ".word 0x00b57042\n\t"                                                \
    : : [address] "r"(_tma_address), [value] "r"(_tma_value)              \
    : "x10", "x11", "memory");                                          \
} while (0)

#define VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(address_arg, bytes_arg) do {    \
  uint _tma_address = (uint)(address_arg);                                  \
  uint _tma_value = (uint)(bytes_arg);                                      \
  __asm__ volatile(                                                         \
    "mv x10, %[address]\n\t"                                               \
    "mv x11, %[value]\n\t"                                                 \
    ".word 0x02b57042\n\t"                                                \
    : : [address] "r"(_tma_address), [value] "r"(_tma_value)              \
    : "x10", "x11", "memory");                                          \
} while (0)

#define VENTUS_TMA_MBARRIER_WAIT(address_arg, old_phase_arg) do {           \
  uint _tma_address = (uint)(address_arg);                                  \
  uint _tma_value = (uint)(old_phase_arg);                                  \
  __asm__ volatile(                                                         \
    "mv x10, %[address]\n\t"                                               \
    "mv x11, %[value]\n\t"                                                 \
    ".word 0x04b57042\n\t"                                                \
    : : [address] "r"(_tma_address), [value] "r"(_tma_value)              \
    : "x10", "x11", "memory");                                          \
} while (0)

#define VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED() do {                          \
  __asm__ volatile(".word 0x06007042\n\t" ::: "memory");                 \
} while (0)

#define VENTUS_TMA_S2G_COMMIT_GROUP() do {                                  \
  __asm__ volatile(".word 0x00086042\n\t" ::: "memory");                 \
} while (0)
#define VENTUS_TMA_S2G_WAIT_GROUP0() do {                                   \
  __asm__ volatile(".word 0x000c6042\n\t" ::: "memory");                 \
} while (0)
#define VENTUS_TMA_S2G_WAIT_GROUP1() do {                                   \
  __asm__ volatile(".word 0x000ce042\n\t" ::: "memory");                 \
} while (0)
#define VENTUS_TMA_S2G_WAIT_GROUP2() do {                                   \
  __asm__ volatile(".word 0x000d6042\n\t" ::: "memory");                 \
} while (0)
#define VENTUS_TMA_S2G_WAIT_GROUP3() do {                                   \
  __asm__ volatile(".word 0x000de042\n\t" ::: "memory");                 \
} while (0)

#define VENTUS_TMA_STATUS_CLEAR() do {                                      \
  __asm__ volatile("csrw 0x814, x0\n\t" ::: "memory");                   \
} while (0)
#define VENTUS_TMA_STATUS_READ(value_arg) do {                              \
  uint _tma_status;                                                         \
  __asm__ volatile("csrr %[status], 0x814\n\t"                            \
                   : [status] "=r"(_tma_status) :: "memory");             \
  (value_arg) = _tma_status;                                                \
} while (0)

#endif
