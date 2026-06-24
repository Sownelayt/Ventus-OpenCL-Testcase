#define pattern_word s2g_tensor_matrix_pattern_word
#define tensor_physical_offset s2g_tensor_matrix_physical_offset
#define build_desc s2g_tensor_matrix_build_desc
#define fill_expected s2g_tensor_matrix_fill_expected
#define first_mismatch s2g_tensor_matrix_first_mismatch
#define check_case s2g_tensor_matrix_check_case

/*
 * Application-level CP_ASYNC_TENSOR_S2G directed test.
 *
 * The kernel fills shared memory with a 4x4 FP32 pattern and copies it into a
 * global destination tensor selected by descriptor coords. The host validates
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
#define GLOBAL_ELEMS_X 8
#define GLOBAL_ELEMS_Y 8
#define GLOBAL_BYTES (GLOBAL_ELEMS_X * GLOBAL_ELEMS_Y * 4)
#define TENSOR_BYTES (4 * 4 * 4)

typedef struct {
  const char *name;
  unsigned rank;
  unsigned interleaveMode;
  unsigned swizzleMode;
  unsigned globalDim[MAX_RANK];
  unsigned globalStrides[MAX_RANK];
  unsigned boxDim[MAX_RANK];
  unsigned coord[MAX_RANK];
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

static void
build_desc(uint32_t *desc, const tensor_s2g_case_t *c)
{
  memset(desc, 0, DESC_WORDS * sizeof(uint32_t));
  desc[0] = 0x56544d41u;
  desc[1] = (6u & 0xfu) | ((c->rank & 0xfu) << 4) |
            ((c->interleaveMode & 0x3u) << 8) |
            ((c->swizzleMode & 0x3u) << 10);
  desc[2] = 0;                        /* patched to dst at runtime */
  desc[3] = 128;
  for (unsigned i = 0; i < MAX_RANK; i++) {
    desc[4 + i] = c->globalDim[i] ? c->globalDim[i] : 1;
    desc[9 + i] = i == 0 ? 4 : c->globalStrides[i - 1];
    desc[14 + i] = c->boxDim[i] ? c->boxDim[i] : 1;
    desc[19 + i] = 1;
  }
}

static void
fill_expected(uint8_t *expected, const tensor_s2g_case_t *c)
{
  memset(expected, 0xcd, GLOBAL_BYTES);
  unsigned out_dim[MAX_RANK] = {1, 1, 1, 1, 1};
  size_t total = 1;
  for (unsigned d = 0; d < c->rank; d++) {
    out_dim[d] = c->boxDim[d] ? c->boxDim[d] : 1;
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
      global_coord[d] = c->coord[d] + idx[d];
      if (global_coord[d] >= c->globalDim[d]) oob = 1;
    }
    if (oob) continue;

    uint32_t word = pattern_word((unsigned)lin);
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
  cl_mem desc_buf = NULL;
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

  desc_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            DESC_WORDS * sizeof(uint32_t), NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(desc)");
  coords_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(coords), coords, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(coords)");
  dst_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(got), NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(dst)");

  const tensor_s2g_case_t cases[] = {
    {
      .name = "origin",
      .rank = 2,
      .globalDim = {GLOBAL_ELEMS_X, GLOBAL_ELEMS_Y, 1, 1, 1},
      .globalStrides = {GLOBAL_ELEMS_X * 4, 0, 0, 0, 0},
      .boxDim = {4, 4, 1, 1, 1},
      .coord = {0, 0, 0, 0, 0},
    },
    {
      .name = "subbox",
      .rank = 2,
      .globalDim = {GLOBAL_ELEMS_X, GLOBAL_ELEMS_Y, 1, 1, 1},
      .globalStrides = {GLOBAL_ELEMS_X * 4, 0, 0, 0, 0},
      .boxDim = {4, 4, 1, 1, 1},
      .coord = {2, 1, 0, 0, 0},
    },
    {
      .name = "interleave16_rank3",
      .rank = 3,
      .interleaveMode = 1,
      .globalDim = {6, 3, 2, 1, 1},
      .globalStrides = {16, 96, 0, 0, 0},
      .boxDim = {6, 2, 1, 1, 1},
      .coord = {0, 1, 1, 0, 0},
    },
  };

  size_t global = 32, local = 32;
  for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
    build_desc(desc, &cases[c]);
    memset(coords, 0, sizeof(coords));
    for (unsigned i = 0; i < MAX_RANK; i++) coords[i] = cases[c].coord[i];
    memset(got, 0xcd, sizeof(got));
    fill_expected(expected, &cases[c]);

    err = clEnqueueWriteBuffer(queue, desc_buf, CL_TRUE, 0, sizeof(desc), desc,
                               0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueWriteBuffer(desc)");
    err = clEnqueueWriteBuffer(queue, coords_buf, CL_TRUE, 0, sizeof(coords),
                               coords, 0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueWriteBuffer(coords)");
    err = clEnqueueFillBuffer(queue, dst_buf, &(uint8_t){0xcd}, sizeof(uint8_t),
                              0, sizeof(got), 0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueFillBuffer(dst)");

    err  = clSetKernelArg(kernel, 0, sizeof(desc_buf), &desc_buf);
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

  exit_code = 0;

FINISH:
  if (dst_buf) clReleaseMemObject(dst_buf);
  if (coords_buf) clReleaseMemObject(coords_buf);
  if (desc_buf) clReleaseMemObject(desc_buf);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
  return exit_code;
}
