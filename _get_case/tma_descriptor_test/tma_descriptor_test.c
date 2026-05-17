/*
 * Descriptor-addressed TMA host smoke test.
 *
 * Runs a 2D FP32 4x4 subbox copy twice: without prefetch and with
 * PREFETCH_TENSORMAP. The kernel loads coords into the CP_ASYNC_TENSOR_G2S
 * VRS2 dynamic parameter block, so both paths must produce identical bytes.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/ventus_opencl_test.h"

#define DESC_WORDS 32
#define COORD_WORDS 32
#define SRC_BYTES (8 * 8 * 4)
#define DST_BYTES (4 * 4 * 4)

static uint32_t
desc_control(unsigned data_type, unsigned rank)
{
  return (data_type & 0xfu) | ((rank & 0xfu) << 4);
}

static void
fill_src(uint8_t *src, size_t n)
{
  for (size_t i = 0; i < n; i++) src[i] = (uint8_t)((i * 5 + 11) & 0xff);
}

static void
build_desc(uint32_t *desc)
{
  memset(desc, 0, DESC_WORDS * sizeof(uint32_t));
  desc[0] = 0x56544d41u;          /* "VTMA" */
  desc[1] = desc_control(6, 2);   /* FP32, rank=2 */
  desc[2] = 0;                    /* kernel patches runtime src pointer */
  desc[3] = 128;
  desc[4] = 8;                    /* globalDim[0] */
  desc[5] = 8;                    /* globalDim[1] */
  desc[6] = desc[7] = desc[8] = 1;
  desc[9] = 4;                    /* byteStride[0] */
  desc[10] = 32;                  /* byteStride[1] */
  desc[14] = 4;                   /* boxDim[0] */
  desc[15] = 4;                   /* boxDim[1] */
  desc[16] = desc[17] = desc[18] = 1;
  for (int i = 0; i < 5; i++) desc[19 + i] = 1;
}

static void
compute_expected(const uint8_t *src, uint8_t *expected)
{
  const unsigned coord0 = 2;
  const unsigned coord1 = 2;
  for (unsigned y = 0; y < 4; y++) {
    for (unsigned x = 0; x < 4; x++) {
      size_t src_off = ((coord1 + y) * 8 + (coord0 + x)) * 4;
      size_t dst_off = (y * 4 + x) * 4;
      memcpy(expected + dst_off, src + src_off, 4);
    }
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

int
main(void)
{
  cl_int err = CL_SUCCESS;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_mem desc_buf = NULL, coords_buf = NULL, src_buf = NULL, dst_buf = NULL;
  int exit_code = 1;

  uint8_t src[SRC_BYTES];
  uint8_t expected[DST_BYTES];
  uint8_t got[DST_BYTES];
  uint32_t desc[DESC_WORDS];
  uint32_t coords[COORD_WORDS];

  fill_src(src, sizeof(src));
  compute_expected(src, expected);
  memset(coords, 0, sizeof(coords));
  coords[0] = 2;
  coords[1] = 2;

  err = ventus_get_default_device(&context, &device, &queue, NULL);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device");
  err = ventus_build_program_from_source(context, device,
                                         "tma_descriptor_test.cl", &program);
  CHECK_OPENCL_ERROR_IN("ventus_build_program_from_source");
  kernel = clCreateKernel(program, "tma_descriptor_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel");

  desc_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
                            DESC_WORDS * sizeof(uint32_t), NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(desc)");
  coords_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(coords), coords, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(coords)");
  src_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(src), src, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(src)");
  dst_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(got), NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(dst)");

  for (unsigned use_prefetch = 0; use_prefetch <= 1; use_prefetch++) {
    build_desc(desc);
    memset(got, 0, sizeof(got));
    err = clEnqueueWriteBuffer(queue, desc_buf, CL_TRUE, 0, sizeof(desc), desc,
                               0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueWriteBuffer(desc)");
    err = clEnqueueFillBuffer(queue, dst_buf, &(uint32_t){0}, sizeof(uint32_t),
                              0, sizeof(got), 0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueFillBuffer(dst)");

    unsigned dst_bytes = DST_BYTES;
    err  = clSetKernelArg(kernel, 0, sizeof(desc_buf), &desc_buf);
    err |= clSetKernelArg(kernel, 1, sizeof(coords_buf), &coords_buf);
    err |= clSetKernelArg(kernel, 2, sizeof(src_buf), &src_buf);
    err |= clSetKernelArg(kernel, 3, sizeof(dst_buf), &dst_buf);
    err |= clSetKernelArg(kernel, 4, sizeof(dst_bytes), &dst_bytes);
    err |= clSetKernelArg(kernel, 5, sizeof(use_prefetch), &use_prefetch);
    CHECK_OPENCL_ERROR_IN("clSetKernelArg");

    size_t global = 32, local = 32;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                                 0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel");
    err = clFinish(queue);
    CHECK_OPENCL_ERROR_IN("clFinish");
    err = clEnqueueReadBuffer(queue, dst_buf, CL_TRUE, 0, sizeof(got), got,
                              0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(dst)");

    int mismatch = first_mismatch(got, expected, sizeof(got));
    if (mismatch >= 0) {
      fprintf(stderr, "FAIL use_prefetch=%u at byte %d: got=%02x exp=%02x\n",
              use_prefetch, mismatch, got[mismatch], expected[mismatch]);
      goto FINISH;
    }
    printf("PASS descriptor_g2s use_prefetch=%u bytes=%u\n",
           use_prefetch, dst_bytes);
  }

  printf("OK tma_descriptor_test\n");
  exit_code = 0;

FINISH:
  if (dst_buf) clReleaseMemObject(dst_buf);
  if (src_buf) clReleaseMemObject(src_buf);
  if (coords_buf) clReleaseMemObject(coords_buf);
  if (desc_buf) clReleaseMemObject(desc_buf);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
  return exit_code;
}
