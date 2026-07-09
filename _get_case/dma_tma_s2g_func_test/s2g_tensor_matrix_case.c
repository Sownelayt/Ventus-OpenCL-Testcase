#define pattern_word s2g_tensor_matrix_pattern_word
#define swizzle_offset s2g_tensor_matrix_swizzle_offset
#define tensor_physical_offset s2g_tensor_matrix_physical_offset
#define tensor_shared_offset s2g_tensor_matrix_shared_offset
#define build_desc s2g_tensor_matrix_build_desc
#define fill_expected s2g_tensor_matrix_fill_expected
#define first_mismatch s2g_tensor_matrix_first_mismatch
#define check_case s2g_tensor_matrix_check_case

/*
 * Application-level CP_ASYNC_TENSOR_S2G directed test.
 *
 * The kernel fills shared memory with a small FP32 pattern and copies it into
 * a global destination tensor selected by descriptor coords. The host validates
 * both the tensor write and the untouched guard bytes around it.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/ventus_opencl_test.h"

#define DESC_WORDS 32
#define COORD_WORDS 32
#define MAX_RANK 5
#define GLOBAL_ELEMS_X 32
#define GLOBAL_ELEMS_Y 32
#define GLOBAL_BYTES (GLOBAL_ELEMS_X * GLOBAL_ELEMS_Y * 4)
#define TENSOR_S2G_CASE_COUNT 37
#define TENSOR_S2G_DESC_SLOTS 40

typedef struct {
  const char *name;
  unsigned dataType;
  unsigned rank;
  unsigned interleaveMode;
  unsigned swizzleMode;
  unsigned globalDim[MAX_RANK];
  unsigned globalStrides[MAX_RANK];
  unsigned boxDim[MAX_RANK];
  unsigned elementStrides[MAX_RANK];
  unsigned coord[MAX_RANK];
  unsigned desc_slot;
} tensor_s2g_case_t;

static uint32_t
pattern_word(unsigned idx)
{
  unsigned base = idx << 2;
  unsigned b0 = (base * 7u + 3u) & 0xffu;
  unsigned b1 = ((base + 1u) * 7u + 3u) & 0xffu;
  unsigned b2 = ((base + 2u) * 7u + 3u) & 0xffu;
  unsigned b3 = ((base + 3u) * 7u + 3u) & 0xffu;
  return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

static size_t
swizzle_offset(size_t logical_off, size_t row, unsigned mode)
{
  if (mode == 0) return logical_off;
  unsigned chunk_bits = mode;  /* 1/2/3 for 32B/64B/128B */
  size_t span = (size_t)16 << chunk_bits;
  size_t chunk_mask = ((size_t)1 << chunk_bits) - 1;
  size_t low = logical_off & 0xf;
  size_t chunk = (logical_off >> 4) & chunk_mask;
  size_t row_low = row & chunk_mask;
  return (logical_off & ~(span - 1)) | ((chunk ^ row_low) << 4) | low;
}

static size_t
tensor_physical_offset(const tensor_s2g_case_t *c, const unsigned coord[MAX_RANK])
{
  const unsigned es = 4;
  if (c->interleaveMode == 0 || c->rank < 3) {
    size_t off = (size_t)coord[0] * es;
    for (unsigned d = 1; d < c->rank; d++) {
      off += (size_t)coord[d] * c->globalStrides[d - 1];
    }
    return off;
  }

  size_t slice_bytes = c->interleaveMode == 1 ? 16u : 32u;
  size_t channels_per_slice = slice_bytes / es;
  if (channels_per_slice == 0) channels_per_slice = 1;
  size_t c_slice = coord[0] / channels_per_slice;
  size_t c_in_slice = coord[0] % channels_per_slice;
  size_t c_slice_stride =
    (size_t)c->globalStrides[c->rank - 3] * c->globalDim[c->rank - 2];

  size_t off = c_in_slice * es + c_slice * c_slice_stride;
  for (unsigned d = 1; d < c->rank; d++) {
    off += (size_t)coord[d] * c->globalStrides[d - 1];
  }
  return off;
}

static size_t
tensor_shared_offset(const tensor_s2g_case_t *c,
                     const unsigned idx[MAX_RANK],
                     const unsigned out_dim[MAX_RANK])
{
  const unsigned es = 4;
  size_t off = 0;
  size_t mul = es;
  for (unsigned d = 0; d < c->rank; d++) {
    off += (size_t)idx[d] * mul;
    mul *= out_dim[d];
  }

  size_t row = 0;
  size_t row_mul = 1;
  for (unsigned d = 1; d < c->rank; d++) {
    row += (size_t)idx[d] * row_mul;
    row_mul *= out_dim[d];
  }
  return swizzle_offset(off, row, c->swizzleMode);
}

static unsigned
tensor_out_dim(const tensor_s2g_case_t *c, unsigned dim)
{
  unsigned box = c->boxDim[dim] ? c->boxDim[dim] : 1;
  unsigned stride = c->elementStrides[dim] ? c->elementStrides[dim] : 1;
  if (dim == 0 || stride <= 1) return box;
  return (box + stride - 1u) / stride;
}

static void
build_desc(uint32_t *desc, const tensor_s2g_case_t *c)
{
  memset(desc, 0, DESC_WORDS * sizeof(uint32_t));
  desc[0] = 0x56544d41u;
  unsigned dataType = c->dataType ? c->dataType : 6u;
  desc[1] = (dataType & 0xfu) | ((c->rank & 0xfu) << 4) |
            ((c->interleaveMode & 0x3u) << 8) |
            ((c->swizzleMode & 0x3u) << 10);
  desc[2] = 0;                        /* patched to dst at runtime */
  desc[3] = 128;
  for (unsigned i = 0; i < MAX_RANK; i++) {
    desc[4 + i] = c->globalDim[i] ? c->globalDim[i] : 1;
    desc[9 + i] = i == 0 ? 4 : c->globalStrides[i - 1];
    desc[14 + i] = c->boxDim[i] ? c->boxDim[i] : 1;
    desc[19 + i] = c->elementStrides[i] ? c->elementStrides[i] : 1;
  }
}

static void
fill_expected(uint8_t *expected, const tensor_s2g_case_t *c)
{
  memset(expected, 0xcd, GLOBAL_BYTES);
  unsigned out_dim[MAX_RANK] = {1, 1, 1, 1, 1};
  size_t total = 1;
  for (unsigned d = 0; d < c->rank; d++) {
    out_dim[d] = tensor_out_dim(c, d);
    total *= out_dim[d];
  }

  for (size_t lin = 0; lin < total; lin++) {
    unsigned idx[MAX_RANK] = {0, 0, 0, 0, 0};
    size_t rem = lin;
    for (unsigned d = 0; d < c->rank; d++) {
      idx[d] = (unsigned)(rem % out_dim[d]);
      rem /= out_dim[d];
    }

    unsigned global_coord[MAX_RANK] = {0, 0, 0, 0, 0};
    int oob = 0;
    for (unsigned d = 0; d < c->rank; d++) {
      unsigned stride = c->elementStrides[d] ? c->elementStrides[d] : 1;
      global_coord[d] = c->coord[d] + idx[d] * stride;
      if (global_coord[d] >= c->globalDim[d]) oob = 1;
    }
    if (oob) continue;

    size_t shared_off = tensor_shared_offset(c, idx, out_dim);
    uint32_t word = pattern_word((unsigned)(shared_off / 4u));
    size_t dst_off = tensor_physical_offset(c, global_coord);
    memcpy(expected + dst_off, &word, sizeof(word));
  }
}

static int
first_mismatch(const uint8_t *a, const uint8_t *b, size_t n)
{
  for (size_t i = 0; i < n; i++) {
    if (a[i] != b[i]) return (int)i;
  }
  return -1;
}

static int
check_case(const char *name, const uint8_t *got, const uint8_t *expected)
{
  int mismatch = first_mismatch(got, expected, GLOBAL_BYTES);
  if (mismatch >= 0) {
    fprintf(stderr, "FAIL %s at byte %d: got=%02x exp=%02x\n",
            name, mismatch, got[mismatch], expected[mismatch]);
    return 1;
  }
  printf("PASS %s bytes=%u\n", name, GLOBAL_BYTES);
  return 0;
}

int
s2g_tensor_matrix_case_main(void)
{
  cl_int err = CL_SUCCESS;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_mem desc_bufs[TENSOR_S2G_DESC_SLOTS] = {NULL};
  cl_mem coords_buf = NULL;
  cl_mem dst_buf = NULL;
  int exit_code = 1;

  uint32_t desc[DESC_WORDS];
  uint32_t coords[COORD_WORDS] = {0};
  uint8_t got[GLOBAL_BYTES];
  uint8_t expected[GLOBAL_BYTES];

  err = ventus_get_default_device(&context, &device, &queue, NULL);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device");
  err = ventus_build_program_from_source(context, device,
                                         "tensor_shared_to_global_test.cl", &program);
  CHECK_OPENCL_ERROR_IN("ventus_build_program_from_source");
  kernel = clCreateKernel(program, "tensor_shared_to_global_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel");

  coords_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(coords), coords, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(coords)");
  dst_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(got), NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(dst)");

  const tensor_s2g_case_t cases[TENSOR_S2G_CASE_COUNT] = {
    {
      .name = "rank1_32_fp32_full_line",
      .rank = 1,
      .globalDim = {32, 1, 1, 1, 1},
      .globalStrides = {0, 0, 0, 0, 0},
      .boxDim = {32, 1, 1, 1, 1},
      .coord = {0, 0, 0, 0, 0},
      .desc_slot = 0,
    },
    {
      .name = "rank1_64_fp32_two_lines",
      .rank = 1,
      .globalDim = {64, 1, 1, 1, 1},
      .globalStrides = {0, 0, 0, 0, 0},
      .boxDim = {64, 1, 1, 1, 1},
      .coord = {0, 0, 0, 0, 0},
      .desc_slot = 1,
    },
    {
      .name = "rank1_24_fp32_partial_tail",
      .rank = 1,
      .globalDim = {64, 1, 1, 1, 1},
      .globalStrides = {0, 0, 0, 0, 0},
      .boxDim = {24, 1, 1, 1, 1},
      .coord = {4, 0, 0, 0, 0},
      .desc_slot = 2,
    },
    {
      .name = "rank2_4x4_origin",
      .rank = 2,
      .globalDim = {GLOBAL_ELEMS_X, GLOBAL_ELEMS_Y, 1, 1, 1},
      .globalStrides = {GLOBAL_ELEMS_X * 4, 0, 0, 0, 0},
      .boxDim = {4, 4, 1, 1, 1},
      .coord = {0, 0, 0, 0, 0},
      .desc_slot = 16,
    },
    {
      .name = "rank2_16x16_contiguous",
      .rank = 2,
      .globalDim = {16, 16, 1, 1, 1},
      .globalStrides = {64, 0, 0, 0, 0},
      .boxDim = {16, 16, 1, 1, 1},
      .coord = {0, 0, 0, 0, 0},
      .desc_slot = 17,
    },
    {
      .name = "rank2_subbox_8x8_at_2_2",
      .rank = 2,
      .globalDim = {GLOBAL_ELEMS_X, GLOBAL_ELEMS_Y, 1, 1, 1},
      .globalStrides = {GLOBAL_ELEMS_X * 4, 0, 0, 0, 0},
      .boxDim = {8, 8, 1, 1, 1},
      .coord = {2, 2, 0, 0, 0},
      .desc_slot = 5,
    },
    {
      .name = "rank2_padded_rows_stride64",
      .rank = 2,
      .globalDim = {8, 4, 1, 1, 1},
      .globalStrides = {64, 0, 0, 0, 0},
      .boxDim = {4, 4, 1, 1, 1},
      .coord = {0, 0, 0, 0, 0},
      .desc_slot = 6,
    },
    {
      .name = "rank2_element_stride2_cols",
      .rank = 2,
      .globalDim = {16, 4, 1, 1, 1},
      .globalStrides = {64, 0, 0, 0, 0},
      .boxDim = {8, 4, 1, 1, 1},
      .elementStrides = {2, 1, 1, 1, 1},
      .coord = {0, 0, 0, 0, 0},
      .desc_slot = 7,
    },
    {
      .name = "rank2_swizzle32_rows",
      .rank = 2,
      .swizzleMode = 1,
      .globalDim = {8, 4, 1, 1, 1},
      .globalStrides = {32, 0, 0, 0, 0},
      .boxDim = {8, 4, 1, 1, 1},
      .coord = {0, 0, 0, 0, 0},
      .desc_slot = 20,
    },
    {
      .name = "rank2_swizzle64_rows",
      .rank = 2,
      .swizzleMode = 2,
      .globalDim = {16, 2, 1, 1, 1},
      .globalStrides = {64, 0, 0, 0, 0},
      .boxDim = {16, 2, 1, 1, 1},
      .coord = {0, 0, 0, 0, 0},
      .desc_slot = 21,
    },
    {
      .name = "rank2_swizzle128_subbox_row1",
      .rank = 2,
      .swizzleMode = 3,
      .globalDim = {32, 3, 1, 1, 1},
      .globalStrides = {128, 0, 0, 0, 0},
      .boxDim = {32, 2, 1, 1, 1},
      .coord = {0, 1, 0, 0, 0},
      .desc_slot = 22,
    },
    {
      .name = "rank3_plain_2x2x2",
      .rank = 3,
      .globalDim = {2, 2, 2, 1, 1},
      .globalStrides = {8, 16, 0, 0, 0},
      .boxDim = {2, 2, 2, 1, 1},
      .coord = {0, 0, 0, 0, 0},
      .desc_slot = 8,
    },
    {
      .name = "rank3_subbox_2x2x2_at_1_1_1",
      .rank = 3,
      .globalDim = {4, 4, 4, 1, 1},
      .globalStrides = {16, 64, 0, 0, 0},
      .boxDim = {2, 2, 2, 1, 1},
      .coord = {1, 1, 1, 0, 0},
      .desc_slot = 18,
    },
    {
      .name = "rank3_oob_dim2_suppress",
      .rank = 3,
      .globalDim = {4, 4, 3, 1, 1},
      .globalStrides = {16, 64, 0, 0, 0},
      .boxDim = {4, 4, 3, 1, 1},
      .coord = {0, 0, 2, 0, 0},
      .desc_slot = 19,
    },
    {
      .name = "rank4_plain_small",
      .rank = 4,
      .globalDim = {2, 2, 2, 2, 1},
      .globalStrides = {8, 16, 32, 0, 0},
      .boxDim = {2, 2, 2, 2, 1},
      .coord = {0, 0, 0, 0, 0},
      .desc_slot = 9,
    },
    {
      .name = "rank5_plain_small",
      .rank = 5,
      .globalDim = {2, 2, 2, 2, 2},
      .globalStrides = {8, 16, 32, 64, 0},
      .boxDim = {2, 2, 2, 2, 2},
      .coord = {0, 0, 0, 0, 0},
      .desc_slot = 10,
    },
    {
      .name = "interleave16_rank3",
      .rank = 3,
      .interleaveMode = 1,
      .globalDim = {6, 3, 2, 1, 1},
      .globalStrides = {16, 96, 0, 0, 0},
      .boxDim = {6, 2, 1, 1, 1},
      .coord = {0, 1, 1, 0, 0},
      .desc_slot = 11,
    },
    {
      .name = "interleave32_rank3",
      .rank = 3,
      .interleaveMode = 2,
      .globalDim = {10, 2, 1, 1, 1},
      .globalStrides = {32, 128, 0, 0, 0},
      .boxDim = {10, 2, 1, 1, 1},
      .coord = {0, 0, 0, 0, 0},
      .desc_slot = 12,
    },
    {
      .name = "rank2_oob_dim0",
      .rank = 2,
      .globalDim = {8, 4, 1, 1, 1},
      .globalStrides = {32, 0, 0, 0, 0},
      .boxDim = {12, 4, 1, 1, 1},
      .coord = {0, 0, 0, 0, 0},
      .desc_slot = 13,
    },
    {
      .name = "rank2_oob_dim1",
      .rank = 2,
      .globalDim = {8, 4, 1, 1, 1},
      .globalStrides = {32, 0, 0, 0, 0},
      .boxDim = {8, 4, 1, 1, 1},
      .coord = {0, 2, 0, 0, 0},
      .desc_slot = 14,
    },
    {
      .name = "rank2_oob_subbox_partial",
      .rank = 2,
      .globalDim = {8, 4, 1, 1, 1},
      .globalStrides = {32, 0, 0, 0, 0},
      .boxDim = {6, 4, 1, 1, 1},
      .coord = {5, 2, 0, 0, 0},
      .desc_slot = 23,
    },
    {
      .name = "rank3_long_4x4x4",
      .rank = 3,
      .globalDim = {4, 4, 4, 1, 1},
      .globalStrides = {16, 64, 0, 0, 0},
      .boxDim = {4, 4, 4, 1, 1},
      .coord = {0, 0, 0, 0, 0},
      .desc_slot = 24,
    },
    {
      .name = "rank3_stride_oob_long",
      .rank = 3,
      .globalDim = {8, 4, 4, 1, 1},
      .globalStrides = {64, 256, 0, 0, 0},
      .boxDim = {6, 4, 3, 1, 1},
      .elementStrides = {2, 1, 1, 1, 1},
      .coord = {0, 1, 2, 0, 0},
      .desc_slot = 25,
    },
    {
      .name = "rank4_long_4x4x2x2",
      .rank = 4,
      .globalDim = {4, 4, 2, 2, 1},
      .globalStrides = {16, 64, 128, 0, 0},
      .boxDim = {4, 4, 2, 2, 1},
      .coord = {0, 0, 0, 0, 0},
      .desc_slot = 26,
    },
    {
      .name = "rank5_long_4x2x2x2x2",
      .rank = 5,
      .globalDim = {4, 2, 2, 2, 2},
      .globalStrides = {16, 32, 64, 128, 0},
      .boxDim = {4, 2, 2, 2, 2},
      .coord = {0, 0, 0, 0, 0},
      .desc_slot = 27,
    },
    {
      .name = "interleave32_long_rank3",
      .rank = 3,
      .interleaveMode = 2,
      .globalDim = {16, 4, 2, 1, 1},
      .globalStrides = {32, 128, 0, 0, 0},
      .boxDim = {16, 4, 2, 1, 1},
      .coord = {0, 0, 0, 0, 0},
      .desc_slot = 28,
    },
    {
      .name = "swizzle128_oob_stride",
      .rank = 2,
      .swizzleMode = 3,
      .globalDim = {32, 4, 1, 1, 1},
      .globalStrides = {128, 0, 0, 0, 0},
      .boxDim = {32, 4, 1, 1, 1},
      .coord = {0, 2, 0, 0, 0},
      .desc_slot = 29,
    },
    {
      .name = "u32_rank2_4x4_alias",
      .dataType = 2,
      .rank = 2,
      .globalDim = {8, 8, 1, 1, 1},
      .globalStrides = {32, 0, 0, 0, 0},
      .boxDim = {4, 4, 1, 1, 1},
      .coord = {2, 2, 0, 0, 0},
      .desc_slot = 30,
    },
    {
      .name = "i32_rank1_32_alias",
      .dataType = 5,
      .rank = 1,
      .globalDim = {32, 1, 1, 1, 1},
      .globalStrides = {0, 0, 0, 0, 0},
      .boxDim = {32, 1, 1, 1, 1},
      .coord = {0, 0, 0, 0, 0},
      .desc_slot = 31,
    },
    {
      .name = "fuzz_tensor_rank2_stride_oob_swizzle32",
      .rank = 2,
      .swizzleMode = 1,
      .globalDim = {16, 4, 1, 1, 1},
      .globalStrides = {64, 0, 0, 0, 0},
      .boxDim = {10, 4, 1, 1, 1},
      .elementStrides = {2, 1, 1, 1, 1},
      .coord = {1, 1, 0, 0, 0},
      .desc_slot = 32,
    },
    {
      .name = "fuzz_tensor_rank3_stride_subbox",
      .rank = 3,
      .globalDim = {8, 5, 4, 1, 1},
      .globalStrides = {32, 160, 0, 0, 0},
      .boxDim = {5, 3, 2, 1, 1},
      .elementStrides = {1, 2, 1, 1, 1},
      .coord = {2, 0, 2, 0, 0},
      .desc_slot = 33,
    },
    {
      .name = "fuzz_tensor_rank3_swizzle64_oob",
      .rank = 3,
      .swizzleMode = 2,
      .globalDim = {16, 4, 3, 1, 1},
      .globalStrides = {64, 256, 0, 0, 0},
      .boxDim = {16, 3, 2, 1, 1},
      .coord = {0, 2, 2, 0, 0},
      .desc_slot = 34,
    },
    {
      .name = "fuzz_tensor_rank5_sparse_stride",
      .rank = 5,
      .globalDim = {5, 3, 3, 2, 2},
      .globalStrides = {20, 60, 180, 360, 0},
      .boxDim = {3, 2, 2, 2, 1},
      .elementStrides = {2, 1, 1, 1, 1},
      .coord = {0, 1, 1, 0, 1},
      .desc_slot = 35,
    },
    {
      .name = "descriptor_cache_A_fill",
      .rank = 1,
      .globalDim = {32, 1, 1, 1, 1},
      .globalStrides = {0, 0, 0, 0, 0},
      .boxDim = {32, 1, 1, 1, 1},
      .coord = {0, 0, 0, 0, 0},
      .desc_slot = 15,
    },
    {
      .name = "descriptor_cache_B_fill",
      .rank = 2,
      .globalDim = {8, 8, 1, 1, 1},
      .globalStrides = {32, 0, 0, 0, 0},
      .boxDim = {4, 4, 1, 1, 1},
      .coord = {1, 1, 0, 0, 0},
      .desc_slot = 3,
    },
    {
      .name = "descriptor_cache_C_replace",
      .rank = 2,
      .globalDim = {8, 8, 1, 1, 1},
      .globalStrides = {32, 0, 0, 0, 0},
      .boxDim = {4, 4, 1, 1, 1},
      .coord = {3, 2, 0, 0, 0},
      .desc_slot = 4,
    },
    {
      .name = "descriptor_cache_A_reuse",
      .rank = 1,
      .globalDim = {32, 1, 1, 1, 1},
      .globalStrides = {0, 0, 0, 0, 0},
      .boxDim = {32, 1, 1, 1, 1},
      .coord = {0, 0, 0, 0, 0},
      .desc_slot = 15,
    },
  };
  const char *case_filter = getenv("S2G_TENSOR_CASE_FILTER");

  size_t global = 32, local = 32;
  for (size_t c = 0; c < TENSOR_S2G_DESC_SLOTS; c++) {
    desc_bufs[c] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                  DESC_WORDS * sizeof(uint32_t), NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer(desc)");
  }

  size_t selected_count = 0;
  for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
    if (case_filter && case_filter[0] &&
        !strstr(cases[c].name, case_filter)) {
      continue;
    }
    selected_count++;
    unsigned slot = cases[c].desc_slot;
    if (slot >= TENSOR_S2G_DESC_SLOTS) {
      fprintf(stderr, "FAIL %s invalid desc slot %u\n", cases[c].name, slot);
      goto FINISH;
    }
    build_desc(desc, &cases[c]);
    memset(coords, 0, sizeof(coords));
    for (unsigned i = 0; i < MAX_RANK; i++) coords[i] = cases[c].coord[i];
    memset(got, 0xcd, sizeof(got));
    fill_expected(expected, &cases[c]);

    err = clEnqueueWriteBuffer(queue, desc_bufs[slot], CL_TRUE, 0, sizeof(desc), desc,
                               0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueWriteBuffer(desc)");
    err = clEnqueueWriteBuffer(queue, coords_buf, CL_TRUE, 0, sizeof(coords),
                               coords, 0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueWriteBuffer(coords)");
    err = clEnqueueFillBuffer(queue, dst_buf, &(uint8_t){0xcd}, sizeof(uint8_t),
                              0, sizeof(got), 0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueFillBuffer(dst)");

    err  = clSetKernelArg(kernel, 0, sizeof(desc_bufs[slot]), &desc_bufs[slot]);
    err |= clSetKernelArg(kernel, 1, sizeof(coords_buf), &coords_buf);
    err |= clSetKernelArg(kernel, 2, sizeof(dst_buf), &dst_buf);
    CHECK_OPENCL_ERROR_IN("clSetKernelArg");

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                                 0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel");
    err = clFinish(queue);
    CHECK_OPENCL_ERROR_IN("clFinish");
    err = clEnqueueReadBuffer(queue, dst_buf, CL_TRUE, 0, sizeof(got), got,
                              0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(dst)");

    if (check_case(cases[c].name, got, expected) != 0) goto FINISH;
  }

  if (selected_count == 0) {
    fprintf(stderr, "FAIL no tensor S2G case matched filter '%s'\n",
            case_filter);
    goto FINISH;
  }

  exit_code = 0;

FINISH:
  if (dst_buf) clReleaseMemObject(dst_buf);
  if (coords_buf) clReleaseMemObject(coords_buf);
  for (size_t c = 0; c < sizeof(desc_bufs) / sizeof(desc_bufs[0]); c++) {
    if (desc_bufs[c]) clReleaseMemObject(desc_bufs[c]);
  }
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
  return exit_code;
}
