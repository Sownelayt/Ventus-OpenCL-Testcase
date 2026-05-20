/*
 * Descriptor-addressed TMA host smoke test.
 *
 * Runs a 2D FP32 4x4 subbox copy with descriptor-form CP_ASYNC_TENSOR
 * (funct3=2), CP_ASYNC_FENCE (funct3=6), and PREFETCH_TENSORMAP
 * (funct3=5).  The extra cases guard against descriptor-cache key mistakes
 * and verify that one fence drains two back-to-back TMA copies.
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
#define DUAL_DST_BYTES (DST_BYTES * 2)

static uint32_t
desc_control(unsigned data_type, unsigned rank)
{
  return (data_type & 0xfu) | ((rank & 0xfu) << 4);
}

static void
fill_src(uint8_t *src, size_t n, unsigned seed)
{
  for (size_t i = 0; i < n; i++) {
    src[i] = (uint8_t)((i * 5 + seed) & 0xff);
  }
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
compute_expected(const uint8_t *src, unsigned coord0, unsigned coord1,
                 uint8_t *expected)
{
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

static int
check_bytes(const char *label, const uint8_t *got, const uint8_t *expected,
            size_t n)
{
  int mismatch = first_mismatch(got, expected, n);
  if (mismatch >= 0) {
    fprintf(stderr, "FAIL %s at byte %d: got=%02x exp=%02x\n",
            label, mismatch, got[mismatch], expected[mismatch]);
    return 1;
  }
  printf("PASS %s bytes=%zu\n", label, n);
  return 0;
}

int
main(void)
{
  cl_int err = CL_SUCCESS;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL, prefetch_other_kernel = NULL, dual_kernel = NULL;
  cl_mem desc_a_buf = NULL, desc_b_buf = NULL;
  cl_mem coords_a_buf = NULL, coords_b_buf = NULL;
  cl_mem src_a_buf = NULL, src_b_buf = NULL, dst_buf = NULL;
  int exit_code = 1;

  uint8_t src_a[SRC_BYTES];
  uint8_t src_b[SRC_BYTES];
  uint8_t expected_a[DST_BYTES];
  uint8_t expected_b[DST_BYTES];
  uint8_t expected_dual[DUAL_DST_BYTES];
  uint8_t got[DUAL_DST_BYTES];
  uint32_t desc_a[DESC_WORDS];
  uint32_t desc_b[DESC_WORDS];
  uint32_t coords_a[COORD_WORDS];
  uint32_t coords_b[COORD_WORDS];

  fill_src(src_a, sizeof(src_a), 11);
  fill_src(src_b, sizeof(src_b), 0x83);
  compute_expected(src_a, 2, 2, expected_a);
  compute_expected(src_b, 1, 3, expected_b);
  memcpy(expected_dual, expected_a, DST_BYTES);
  memcpy(expected_dual + DST_BYTES, expected_b, DST_BYTES);
  memset(coords_a, 0, sizeof(coords_a));
  memset(coords_b, 0, sizeof(coords_b));
  coords_a[0] = 2;
  coords_a[1] = 2;
  coords_b[0] = 1;
  coords_b[1] = 3;

  err = ventus_get_default_device(&context, &device, &queue, NULL);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device");
  err = ventus_build_program_from_source(context, device,
                                         "tma_descriptor_test.cl", &program);
  CHECK_OPENCL_ERROR_IN("ventus_build_program_from_source");
  kernel = clCreateKernel(program, "tma_descriptor_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel");
  prefetch_other_kernel =
    clCreateKernel(program, "tma_descriptor_prefetch_other_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(prefetch_other)");
  dual_kernel =
    clCreateKernel(program, "tma_descriptor_dual_single_fence_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(dual_single_fence)");

  desc_a_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
                              DESC_WORDS * sizeof(uint32_t), NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(desc_a)");
  desc_b_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
                              DESC_WORDS * sizeof(uint32_t), NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(desc_b)");
  coords_a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(coords_a), coords_a, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(coords_a)");
  coords_b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(coords_b), coords_b, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(coords_b)");
  src_a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             sizeof(src_a), src_a, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(src_a)");
  src_b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             sizeof(src_b), src_b, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(src_b)");
  dst_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(got), NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(dst)");

  for (unsigned use_prefetch = 0; use_prefetch <= 1; use_prefetch++) {
    build_desc(desc_a);
    memset(got, 0, sizeof(got));
    err = clEnqueueWriteBuffer(queue, desc_a_buf, CL_TRUE, 0, sizeof(desc_a),
                               desc_a, 0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueWriteBuffer(desc_a)");
    err = clEnqueueFillBuffer(queue, dst_buf, &(uint32_t){0}, sizeof(uint32_t),
                              0, sizeof(got), 0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueFillBuffer(dst)");

    unsigned dst_bytes = DST_BYTES;
    err  = clSetKernelArg(kernel, 0, sizeof(desc_a_buf), &desc_a_buf);
    err |= clSetKernelArg(kernel, 1, sizeof(coords_a_buf), &coords_a_buf);
    err |= clSetKernelArg(kernel, 2, sizeof(src_a_buf), &src_a_buf);
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

    char label[64];
    snprintf(label, sizeof(label), "descriptor_funct3 use_prefetch=%u",
             use_prefetch);
    if (check_bytes(label, got, expected_a, DST_BYTES) != 0) goto FINISH;
  }

  build_desc(desc_a);
  build_desc(desc_b);
  memset(got, 0, sizeof(got));
  err = clEnqueueWriteBuffer(queue, desc_a_buf, CL_TRUE, 0, sizeof(desc_a),
                             desc_a, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueWriteBuffer(desc_a prefetch_other)");
  err = clEnqueueWriteBuffer(queue, desc_b_buf, CL_TRUE, 0, sizeof(desc_b),
                             desc_b, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueWriteBuffer(desc_b prefetch_other)");
  err = clEnqueueFillBuffer(queue, dst_buf, &(uint32_t){0}, sizeof(uint32_t),
                            0, sizeof(got), 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueFillBuffer(dst prefetch_other)");

  unsigned dst_bytes = DST_BYTES;
  err  = clSetKernelArg(prefetch_other_kernel, 0, sizeof(desc_a_buf),
                        &desc_a_buf);
  err |= clSetKernelArg(prefetch_other_kernel, 1, sizeof(coords_a_buf),
                        &coords_a_buf);
  err |= clSetKernelArg(prefetch_other_kernel, 2, sizeof(src_a_buf),
                        &src_a_buf);
  err |= clSetKernelArg(prefetch_other_kernel, 3, sizeof(desc_b_buf),
                        &desc_b_buf);
  err |= clSetKernelArg(prefetch_other_kernel, 4, sizeof(coords_b_buf),
                        &coords_b_buf);
  err |= clSetKernelArg(prefetch_other_kernel, 5, sizeof(src_b_buf),
                        &src_b_buf);
  err |= clSetKernelArg(prefetch_other_kernel, 6, sizeof(dst_buf), &dst_buf);
  err |= clSetKernelArg(prefetch_other_kernel, 7, sizeof(dst_bytes),
                        &dst_bytes);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(prefetch_other)");

  size_t global = 32, local = 32;
  err = clEnqueueNDRangeKernel(queue, prefetch_other_kernel, 1, NULL,
                               &global, &local, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel(prefetch_other)");
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(prefetch_other)");
  err = clEnqueueReadBuffer(queue, dst_buf, CL_TRUE, 0, sizeof(got), got,
                            0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(prefetch_other)");
  if (check_bytes("prefetch_descA_then_tensor_descB",
                  got, expected_b, DST_BYTES) != 0) {
    goto FINISH;
  }

  build_desc(desc_a);
  build_desc(desc_b);
  memset(got, 0, sizeof(got));
  err = clEnqueueWriteBuffer(queue, desc_a_buf, CL_TRUE, 0, sizeof(desc_a),
                             desc_a, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueWriteBuffer(desc_a dual)");
  err = clEnqueueWriteBuffer(queue, desc_b_buf, CL_TRUE, 0, sizeof(desc_b),
                             desc_b, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueWriteBuffer(desc_b dual)");
  err = clEnqueueFillBuffer(queue, dst_buf, &(uint32_t){0}, sizeof(uint32_t),
                            0, sizeof(got), 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueFillBuffer(dst dual)");

  unsigned dual_dst_bytes = DUAL_DST_BYTES;
  err  = clSetKernelArg(dual_kernel, 0, sizeof(desc_a_buf), &desc_a_buf);
  err |= clSetKernelArg(dual_kernel, 1, sizeof(coords_a_buf), &coords_a_buf);
  err |= clSetKernelArg(dual_kernel, 2, sizeof(src_a_buf), &src_a_buf);
  err |= clSetKernelArg(dual_kernel, 3, sizeof(desc_b_buf), &desc_b_buf);
  err |= clSetKernelArg(dual_kernel, 4, sizeof(coords_b_buf), &coords_b_buf);
  err |= clSetKernelArg(dual_kernel, 5, sizeof(src_b_buf), &src_b_buf);
  err |= clSetKernelArg(dual_kernel, 6, sizeof(dst_buf), &dst_buf);
  err |= clSetKernelArg(dual_kernel, 7, sizeof(dual_dst_bytes),
                        &dual_dst_bytes);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(dual)");

  err = clEnqueueNDRangeKernel(queue, dual_kernel, 1, NULL,
                               &global, &local, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel(dual)");
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(dual)");
  err = clEnqueueReadBuffer(queue, dst_buf, CL_TRUE, 0, sizeof(got), got,
                            0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(dual)");
  if (check_bytes("dual_tensor_single_fence", got, expected_dual,
                  DUAL_DST_BYTES) != 0) {
    goto FINISH;
  }

  printf("OK tma_descriptor_test\n");
  exit_code = 0;

FINISH:
  if (dst_buf) clReleaseMemObject(dst_buf);
  if (src_b_buf) clReleaseMemObject(src_b_buf);
  if (src_a_buf) clReleaseMemObject(src_a_buf);
  if (coords_b_buf) clReleaseMemObject(coords_b_buf);
  if (coords_a_buf) clReleaseMemObject(coords_a_buf);
  if (desc_b_buf) clReleaseMemObject(desc_b_buf);
  if (desc_a_buf) clReleaseMemObject(desc_a_buf);
  if (dual_kernel) clReleaseKernel(dual_kernel);
  if (prefetch_other_kernel) clReleaseKernel(prefetch_other_kernel);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
  return exit_code;
}
