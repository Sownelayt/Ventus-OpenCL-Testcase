/*
 * Mixed completion and routing directed kernels for S2G coverage.
 *
 * These kernels live in the S2G functional suite because they stress the
 * S2G-side shared read route, L2 ack route, group completion domain, and
 * tensor/bulk S2G interaction.  They intentionally keep the data shapes small
 * and checkable; performance is not the goal here.
 */

#define MIXED_BULK_BYTES 128
#define MIXED_BULK_WORDS (MIXED_BULK_BYTES / 4)
#define MIXED_TENSOR_SRC_BYTES (8 * 8 * 4)
#define MIXED_TENSOR_COPY_BYTES 64
#define MIXED_TENSOR_GLOBAL_BYTES MIXED_TENSOR_SRC_BYTES
#define MIXED_COMPLEX_SHARED_BYTES 512

#define MIXED_COMPLEX_BULK_G2S_OFF 0
#define MIXED_COMPLEX_BULK_S2G_OFF 128
#define MIXED_COMPLEX_TENSOR_G2S_OFF 256
#define MIXED_COMPLEX_TENSOR_S2G_OFF 320
#define MIXED_COMPLEX_OUT_BULK_G2S_OFF 0
#define MIXED_COMPLEX_OUT_BULK_S2G_OFF 128
#define MIXED_COMPLEX_OUT_TENSOR_G2S_OFF 256
#define MIXED_COMPLEX_OUT_TENSOR_S2G_OFF 320

#define MIXED_LONG_SHARED_BYTES 768
#define MIXED_LONG_BULK_G2S_OFF 0
#define MIXED_LONG_BULK_S2G0_OFF 128
#define MIXED_LONG_TENSOR_G2S_OFF 256
#define MIXED_LONG_TENSOR_S2G0_OFF 320
#define MIXED_LONG_BULK_S2G1_OFF 384
#define MIXED_LONG_TENSOR_S2G1_OFF 512
#define MIXED_LONG_OUT_BULK_G2S_OFF 0
#define MIXED_LONG_OUT_BULK_S2G0_OFF 128
#define MIXED_LONG_OUT_TENSOR_G2S_OFF 256
#define MIXED_LONG_OUT_TENSOR_S2G0_OFF 384
#define MIXED_LONG_OUT_BULK_S2G1_OFF 640
#define MIXED_LONG_OUT_TENSOR_S2G1_OFF 768

#define S2G_MIXED_ROUTE_DMA_SHARED_BYTES 512
#define S2G_MIXED_ROUTE_DMA_WORDS (S2G_MIXED_ROUTE_DMA_SHARED_BYTES / 4)
#define S2G_MIXED_ROUTE_CONFLICT_WORDS 4096
#define S2G_MIXED_ROUTE_SCRATCH_WORDS \
  (S2G_MIXED_ROUTE_DMA_WORDS + S2G_MIXED_ROUTE_CONFLICT_WORDS)

#define DMA_WAIT_ALL() do {                                                  \
  __asm__ volatile(".word 0x00006042\n\t" ::: "memory");                  \
} while (0)

#define S2G_COMMIT_GROUP_RAW() do {                                          \
  __asm__ volatile(".word 0x00086042\n\t" ::: "memory");                  \
} while (0)

#define S2G_WAIT_GROUP0_RAW() do {                                           \
  __asm__ volatile(".word 0x000c6042\n\t" ::: "memory");                  \
} while (0)

#define DMA_WAIT_GROUP1_RAW() do {                                           \
  __asm__ volatile(".word 0x000ce042\n\t" ::: "memory");                  \
} while (0)

#define ISSUE_BULK_G2S(dst_addr, src_addr, size_bytes) do {                  \
  __asm__ volatile(                                                           \
    ".insn r 0x42, 1, 0, %[dst], %[src], %[size]\n\t"                     \
    :                                                                         \
    : [dst] "r"(dst_addr), [src] "r"(src_addr), [size] "r"(size_bytes)   \
    : "memory");                                                            \
} while (0)

#define ISSUE_BULK_S2G(dst_addr, src_addr, size_bytes) do {                  \
  __asm__ volatile(                                                           \
    ".insn r 0x42, 3, 0, %[dst], %[src], %[size]\n\t"                     \
    :                                                                         \
    : [dst] "r"(dst_addr), [src] "r"(src_addr), [size] "r"(size_bytes)   \
    : "memory");                                                            \
} while (0)

static uint
mixed_bulk_pattern_word(uint word_index)
{
  uint base = word_index << 2;
  uint b0 = (base * 7u + 0x23u) & 0xffu;
  uint b1 = ((base + 1u) * 7u + 0x23u) & 0xffu;
  uint b2 = ((base + 2u) * 7u + 0x23u) & 0xffu;
  uint b3 = ((base + 3u) * 7u + 0x23u) & 0xffu;
  return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

static uint
mixed_stress_pattern_word(uint word_index, uint seed)
{
  uint base = word_index << 2;
  uint b0 = (base * 13u + seed) & 0xffu;
  uint b1 = ((base + 1u) * 13u + seed) & 0xffu;
  uint b2 = ((base + 2u) * 13u + seed) & 0xffu;
  uint b3 = ((base + 3u) * 13u + seed) & 0xffu;
  return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

static uint
mixed_tensor_s2g_pattern_word(uint word_index)
{
  uint base = word_index << 2;
  uint b0 = (base * 7u + 3u) & 0xffu;
  uint b1 = ((base + 1u) * 7u + 3u) & 0xffu;
  uint b2 = ((base + 2u) * 7u + 3u) & 0xffu;
  uint b3 = ((base + 3u) * 7u + 3u) & 0xffu;
  return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

kernel void
s2g_routing_conflict_kernel(__global uchar *dst,
                            __global uint *conflict_out,
                            uint copy_bytes,
                            uint rounds,
                            uint src_offset)
{
  __local uint scratch[S2G_MIXED_ROUTE_SCRATCH_WORDS];
  __local uchar *dma_shared = (__local uchar *)scratch;
  volatile __local uint *conflict = scratch + S2G_MIXED_ROUTE_DMA_WORDS;
  uint lid = get_local_id(0);
  uint lsize = get_local_size(0);

  for (uint i = lid; i < S2G_MIXED_ROUTE_DMA_WORDS; i += lsize) {
    scratch[i] = mixed_bulk_pattern_word(i);
  }
  for (uint i = S2G_MIXED_ROUTE_DMA_WORDS + lid;
       i < S2G_MIXED_ROUTE_SCRATCH_WORDS; i += lsize) {
    scratch[i] = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    uint src = (uint)(dma_shared + src_offset);
    uint out = (uint)dst;
    ISSUE_BULK_S2G(out, src, copy_bytes);
  }

  uint slot = lid * 32u;
  uint value = 0xA5000000u + lid;
  conflict[slot] = value;
  for (uint r = 0; r < rounds; r++) {
    uint cur = conflict[slot];
    cur = cur + ((r + 1u) * 17u) + lid;
    conflict[slot] = cur;
    value = cur;
  }

  if (lid == 0) {
    DMA_WAIT_ALL();
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  conflict_out[lid] = value;
}

kernel void
dma_group_keep1_preserves_newer_s2g_kernel(__global const uchar *g2s_src,
                                           __global uchar *g2s_out,
                                           __global uchar *s2g_dst,
                                           __global uint *marker,
                                           uint copy_bytes)
{
  __local uint shared_words[(MIXED_BULK_BYTES * 2) / 4];
  uint lid = get_local_id(0);
  uint lsize = get_local_size(0);

  for (uint i = lid; i < MIXED_BULK_WORDS; i += lsize) {
    shared_words[i] = 0;
    shared_words[i + MIXED_BULK_WORDS] = mixed_stress_pattern_word(i, 0x61u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    uint base = (uint)((__local uchar *)shared_words);
    uint g2s_shared = base;
    uint s2g_shared = base + MIXED_BULK_BYTES;
    ISSUE_BULK_G2S(g2s_shared, (uint)g2s_src, copy_bytes);
    S2G_COMMIT_GROUP_RAW();
    ISSUE_BULK_S2G((uint)s2g_dst, s2g_shared, copy_bytes);
    S2G_COMMIT_GROUP_RAW();
    DMA_WAIT_GROUP1_RAW();
    marker[0] = 0x67010001u;
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  for (uint i = lid; i < MIXED_BULK_WORDS; i += lsize) {
    ((__global uint *)g2s_out)[i] = shared_words[i];
  }

  if (lid == 0) {
    S2G_WAIT_GROUP0_RAW();
    marker[1] = 0x67010002u;
  }
}

kernel void
dma_group_wait0_drains_g2s_s2g_kernel(__global const uchar *g2s_src,
                                      __global uchar *g2s_out,
                                      __global uchar *s2g_dst,
                                      __global uint *marker,
                                      uint copy_bytes)
{
  __local uint shared_words[(MIXED_BULK_BYTES * 2) / 4];
  uint lid = get_local_id(0);
  uint lsize = get_local_size(0);

  for (uint i = lid; i < MIXED_BULK_WORDS; i += lsize) {
    shared_words[i] = 0;
    shared_words[i + MIXED_BULK_WORDS] = mixed_stress_pattern_word(i, 0x73u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {
    uint base = (uint)((__local uchar *)shared_words);
    uint g2s_shared = base;
    uint s2g_shared = base + MIXED_BULK_BYTES;
    ISSUE_BULK_G2S(g2s_shared, (uint)g2s_src, copy_bytes);
    S2G_COMMIT_GROUP_RAW();
    ISSUE_BULK_S2G((uint)s2g_dst, s2g_shared, copy_bytes);
    S2G_COMMIT_GROUP_RAW();
    S2G_WAIT_GROUP0_RAW();
    marker[0] = 0x67020001u;
    marker[1] = 0x67020002u;
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  for (uint i = lid; i < MIXED_BULK_WORDS; i += lsize) {
    ((__global uint *)g2s_out)[i] = shared_words[i];
  }
}

kernel void
bulk_tensor_s2g_same_fence_kernel(__global uint *tensor_desc,
                                  __global const uint *tensor_coords,
                                  __global uchar *bulk_dst,
                                  __global uchar *tensor_dst)
{
  __local uint shared_words[(MIXED_BULK_BYTES +
                             MIXED_TENSOR_COPY_BYTES) / 4];
  uint lid = get_local_id(0);
  uint lsize = get_local_size(0);

  for (uint i = lid; i < MIXED_BULK_WORDS; i += lsize) {
    shared_words[i] = mixed_stress_pattern_word(i, 0x7au);
  }
  for (uint i = lid; i < MIXED_TENSOR_COPY_BYTES / 4; i += lsize) {
    shared_words[MIXED_BULK_WORDS + i] =
      mixed_tensor_s2g_pattern_word(i);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  uint coords_ptr = (uint)tensor_coords;
  __asm__ volatile(
    "vid.v v12\n\t"
    "vsll.vi v12, v12, 2\n\t"
    "vadd.vx v12, v12, %[coords]\n\t"
    "vlw12.v v12, 0(v12)\n\t"
    :
    : [coords] "r"(coords_ptr)
    : "memory"
  );

  if (lid == 0) {
    uint base = (uint)((__local uchar *)shared_words);
    uint tensor_shared = base + MIXED_BULK_BYTES;
    tensor_desc[2] = (uint)tensor_dst;
    ISSUE_BULK_S2G((uint)bulk_dst, base, MIXED_BULK_BYTES);
    __asm__ volatile(
      "mv x10, %[smem]\n\t"
      "mv x11, %[desc]\n\t"
      ".word 0x00C5C542\n\t"
      :
      : [smem] "r"(tensor_shared), [desc] "r"((uint)tensor_desc)
      : "x10", "x11", "memory"
    );
    DMA_WAIT_ALL();
  }
}

kernel void
bidirectional_descriptor_mix_kernel(__global uint *tensor_g2s_desc,
                                    __global const uint *tensor_g2s_coords,
                                    __global const uchar *tensor_src,
                                    __global uint *tensor_s2g_desc,
                                    __global const uint *tensor_s2g_coords,
                                    __global const uchar *bulk_src,
                                    __global uchar *out,
                                    __global uint *marker,
                                    uint bulk_bytes)
{
  __local uint shared_words[MIXED_COMPLEX_SHARED_BYTES / 4];

  uint lid = get_local_id(0);
  uint lsize = get_local_size(0);

  for (uint i = lid; i < MIXED_COMPLEX_SHARED_BYTES / 4; i += lsize) {
    shared_words[i] = 0;
  }
  for (uint i = lid; i < MIXED_BULK_BYTES / 4; i += lsize) {
    shared_words[(MIXED_COMPLEX_BULK_S2G_OFF / 4) + i] =
      mixed_stress_pattern_word(i, 0x55u);
  }
  for (uint i = lid; i < MIXED_TENSOR_COPY_BYTES / 4; i += lsize) {
    shared_words[(MIXED_COMPLEX_TENSOR_S2G_OFF / 4) + i] =
      mixed_tensor_s2g_pattern_word(i);
  }
  if (lid == 0) {
    tensor_g2s_desc[2] = (uint)tensor_src;
    tensor_s2g_desc[2] = (uint)(out + MIXED_COMPLEX_OUT_TENSOR_S2G_OFF);
    marker[0] = 0xfeed0001u;
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  uint base = (uint)((__local uchar *)shared_words);
  uint bulk_g2s_shared = base + MIXED_COMPLEX_BULK_G2S_OFF;
  uint bulk_s2g_shared = base + MIXED_COMPLEX_BULK_S2G_OFF;
  uint tensor_g2s_shared = base + MIXED_COMPLEX_TENSOR_G2S_OFF;
  uint tensor_s2g_shared = base + MIXED_COMPLEX_TENSOR_S2G_OFF;
  uint tensor_g2s_desc_ptr = (uint)tensor_g2s_desc;
  uint tensor_s2g_desc_ptr = (uint)tensor_s2g_desc;
  uint tensor_g2s_coords_ptr = (uint)tensor_g2s_coords;
  uint tensor_s2g_coords_ptr = (uint)tensor_s2g_coords;

  if (lid == 0) {
    ISSUE_BULK_G2S(bulk_g2s_shared, (uint)bulk_src, bulk_bytes);
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
    ISSUE_BULK_S2G((uint)(out + MIXED_COMPLEX_OUT_BULK_S2G_OFF),
                   bulk_s2g_shared, bulk_bytes);
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

  DMA_WAIT_ALL();
  if (lid == 0) {
    DMA_WAIT_ALL();
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint i = lid; i < MIXED_BULK_BYTES / 4; i += lsize) {
    ((__global uint *)(out + MIXED_COMPLEX_OUT_BULK_G2S_OFF))[i] =
      shared_words[(MIXED_COMPLEX_BULK_G2S_OFF / 4) + i];
  }
  for (uint i = lid; i < MIXED_TENSOR_COPY_BYTES / 4; i += lsize) {
    ((__global uint *)(out + MIXED_COMPLEX_OUT_TENSOR_G2S_OFF))[i] =
      shared_words[(MIXED_COMPLEX_TENSOR_G2S_OFF / 4) + i];
  }
}

kernel void
mixed_tensor_outstanding_long_kernel(__global uint *tensor_g2s_desc,
                                     __global const uint *tensor_g2s_coords,
                                     __global const uchar *tensor_src,
                                     __global uint *tensor_s2g0_desc,
                                     __global const uint *tensor_s2g0_coords,
                                     __global uint *tensor_s2g1_desc,
                                     __global const uint *tensor_s2g1_coords,
                                     __global const uchar *bulk_src,
                                     __global uchar *out,
                                     __global uint *marker,
                                     uint bulk_bytes)
{
  __local uint shared_words[MIXED_LONG_SHARED_BYTES / 4];

  uint lid = get_local_id(0);
  uint lsize = get_local_size(0);

  for (uint i = lid; i < MIXED_LONG_SHARED_BYTES / 4; i += lsize) {
    shared_words[i] = 0;
  }
  for (uint i = lid; i < MIXED_BULK_BYTES / 4; i += lsize) {
    shared_words[(MIXED_LONG_BULK_S2G0_OFF / 4) + i] =
      mixed_stress_pattern_word(i, 0x83u);
    shared_words[(MIXED_LONG_BULK_S2G1_OFF / 4) + i] =
      mixed_stress_pattern_word(i, 0x91u);
  }
  for (uint i = lid; i < MIXED_TENSOR_COPY_BYTES / 4; i += lsize) {
    shared_words[(MIXED_LONG_TENSOR_S2G0_OFF / 4) + i] =
      mixed_tensor_s2g_pattern_word(i);
    shared_words[(MIXED_LONG_TENSOR_S2G1_OFF / 4) + i] =
      mixed_tensor_s2g_pattern_word(i);
  }

  if (lid == 0) {
    tensor_g2s_desc[2] = (uint)tensor_src;
    tensor_s2g0_desc[2] = (uint)(out + MIXED_LONG_OUT_TENSOR_S2G0_OFF);
    tensor_s2g1_desc[2] = (uint)(out + MIXED_LONG_OUT_TENSOR_S2G1_OFF);
    marker[0] = 0xfeed1001u;
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  uint base = (uint)((__local uchar *)shared_words);
  uint bulk_g2s_shared = base + MIXED_LONG_BULK_G2S_OFF;
  uint bulk_s2g0_shared = base + MIXED_LONG_BULK_S2G0_OFF;
  uint tensor_g2s_shared = base + MIXED_LONG_TENSOR_G2S_OFF;
  uint tensor_s2g0_shared = base + MIXED_LONG_TENSOR_S2G0_OFF;
  uint bulk_s2g1_shared = base + MIXED_LONG_BULK_S2G1_OFF;
  uint tensor_s2g1_shared = base + MIXED_LONG_TENSOR_S2G1_OFF;

  if (lid == 0) {
    ISSUE_BULK_G2S(bulk_g2s_shared, (uint)bulk_src, bulk_bytes);
    marker[1] = 0xfeed1002u;
  }

  __asm__ volatile(
    "mv x11, %[desc]\n\t"
    ".word 0x0005D042\n\t"
    :
    : [desc] "r"((uint)tensor_g2s_desc)
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
    : [smem] "r"(tensor_g2s_shared), [desc] "r"((uint)tensor_g2s_desc),
      [coords] "r"((uint)tensor_g2s_coords)
    : "x10", "x11", "memory"
  );

  if (lid == 0) {
    ISSUE_BULK_S2G((uint)(out + MIXED_LONG_OUT_BULK_S2G0_OFF),
                   bulk_s2g0_shared, bulk_bytes);
    S2G_COMMIT_GROUP_RAW();
    marker[2] = 0xfeed1003u;
  }

  __asm__ volatile(
    "vid.v v12\n\t"
    "vsll.vi v12, v12, 2\n\t"
    "vadd.vx v12, v12, %[coords]\n\t"
    "vlw12.v v12, 0(v12)\n\t"
    :
    : [coords] "r"((uint)tensor_s2g0_coords)
    : "memory"
  );

  if (lid == 0) {
    __asm__ volatile(
      "mv x10, %[smem]\n\t"
      "mv x11, %[desc]\n\t"
      ".word 0x00C5C542\n\t"
      :
      : [smem] "r"(tensor_s2g0_shared), [desc] "r"((uint)tensor_s2g0_desc)
      : "x10", "x11", "memory"
    );
    marker[3] = 0xfeed1004u;

    ISSUE_BULK_S2G((uint)(out + MIXED_LONG_OUT_BULK_S2G1_OFF),
                   bulk_s2g1_shared, bulk_bytes);
    S2G_COMMIT_GROUP_RAW();
    marker[4] = 0xfeed1005u;
  }

  __asm__ volatile(
    "vid.v v12\n\t"
    "vsll.vi v12, v12, 2\n\t"
    "vadd.vx v12, v12, %[coords]\n\t"
    "vlw12.v v12, 0(v12)\n\t"
    :
    : [coords] "r"((uint)tensor_s2g1_coords)
    : "memory"
  );

  if (lid == 0) {
    __asm__ volatile(
      "mv x10, %[smem]\n\t"
      "mv x11, %[desc]\n\t"
      ".word 0x00C5C542\n\t"
      :
      : [smem] "r"(tensor_s2g1_shared), [desc] "r"((uint)tensor_s2g1_desc)
      : "x10", "x11", "memory"
    );
    marker[5] = 0xfeed1006u;
  }

  DMA_WAIT_ALL();
  if (lid == 0) {
    DMA_WAIT_ALL();
    marker[6] = 0xfeed1007u;
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint i = lid; i < MIXED_BULK_BYTES / 4; i += lsize) {
    ((__global uint *)(out + MIXED_LONG_OUT_BULK_G2S_OFF))[i] =
      shared_words[(MIXED_LONG_BULK_G2S_OFF / 4) + i];
  }
  for (uint i = lid; i < MIXED_TENSOR_COPY_BYTES / 4; i += lsize) {
    ((__global uint *)(out + MIXED_LONG_OUT_TENSOR_G2S_OFF))[i] =
      shared_words[(MIXED_LONG_TENSOR_G2S_OFF / 4) + i];
  }
}
