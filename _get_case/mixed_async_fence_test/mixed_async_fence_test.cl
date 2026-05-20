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

