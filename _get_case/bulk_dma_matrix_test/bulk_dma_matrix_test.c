/*
 * Application-level CP_ASYNC_BULK matrix test.
 *
 * Covers payloads near and across the 128B L2 cacheline boundary, plus a
 * non-zero shared-memory destination offset. The host compares exact bytes.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/ventus_opencl_test.h"

#define LOCAL_SIZE 32
#define SRC_BYTES  768

typedef struct {
  const char *name;
  unsigned src_offset;
  unsigned copy_bytes;
  unsigned dst_offset;
} bulk_case_t;

static const bulk_case_t g_cases[] = {
  {"bulk_32B_at_120",      120,  32,  0},
  {"bulk_8B_at_124",       124,   8,  0},
  {"bulk_192B_aligned",      0, 192,  0},
  {"bulk_64B_dst_offset",   64,  64, 16},
};

static void
fill_src(uint8_t *src, size_t nbytes)
{
  for (size_t i = 0; i < nbytes; i++) {
    src[i] = (uint8_t)((i * 5 + 0x31) & 0xFF);
  }
}

static int
run_one(cl_context context, cl_command_queue queue, cl_kernel kernel,
        const bulk_case_t *c)
{
  int rc = 1;
  cl_int err;
  cl_mem src_buf = NULL, dst_buf = NULL;
  uint8_t *src = (uint8_t *)malloc(SRC_BYTES);
  uint8_t *dst = (uint8_t *)calloc(c->copy_bytes, 1);

  if (!src || !dst) {
    printf("  FAIL: host alloc\n");
    goto FINISH;
  }
  if (c->src_offset + c->copy_bytes > SRC_BYTES) {
    printf("  FAIL: source window exceeds SRC_BYTES\n");
    goto FINISH;
  }

  fill_src(src, SRC_BYTES);

  src_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           SRC_BYTES, src, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer src");
  dst_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                           c->copy_bytes, NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer dst");

  cl_uint src_offset = c->src_offset;
  cl_uint copy_bytes = c->copy_bytes;
  cl_uint dst_offset = c->dst_offset;
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg 0");
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg 1");
  err = clSetKernelArg(kernel, 2, sizeof(cl_uint), &src_offset);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg 2");
  err = clSetKernelArg(kernel, 3, sizeof(cl_uint), &copy_bytes);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg 3");
  err = clSetKernelArg(kernel, 4, sizeof(cl_uint), &dst_offset);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg 4");

  size_t gsz = LOCAL_SIZE, lsz = LOCAL_SIZE;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gsz, &lsz,
                               0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel");
  err = clEnqueueReadBuffer(queue, dst_buf, CL_TRUE, 0, c->copy_bytes,
                            dst, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer");

  rc = 0;
  for (unsigned i = 0; i < c->copy_bytes; i++) {
    uint8_t exp = src[c->src_offset + i];
    if (dst[i] != exp) {
      printf("  FAIL byte[%u]: expected 0x%02x got 0x%02x\n",
             i, exp, dst[i]);
      rc = 1;
      break;
    }
  }

FINISH:
  if (src_buf) clReleaseMemObject(src_buf);
  if (dst_buf) clReleaseMemObject(dst_buf);
  free(src);
  free(dst);
  return rc;
}

int
main(int argc, char **argv)
{
  cl_int err;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_platform_id platform = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  size_t pass = 0, fail = 0, skip = 0;
  const char *filter = argc > 1 ? argv[1] : NULL;

  err = ventus_get_default_device(&context, &device, &queue, &platform);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device");
  err = ventus_build_program_from_source(context, device,
                                         "bulk_dma_matrix_test.cl", &program);
  CHECK_OPENCL_ERROR_IN("ventus_build_program_from_source");
  kernel = clCreateKernel(program, "bulk_dma_matrix_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel");

  for (size_t i = 0; i < sizeof(g_cases) / sizeof(g_cases[0]); i++) {
    const bulk_case_t *c = &g_cases[i];
    if (filter && !strstr(c->name, filter)) { skip++; continue; }

    printf("[%zu/%zu] %s src_off=%u bytes=%u dst_off=%u\n",
           i + 1, sizeof(g_cases) / sizeof(g_cases[0]), c->name,
           c->src_offset, c->copy_bytes, c->dst_offset);
    if (run_one(context, queue, kernel, c) == 0) {
      printf("      PASS\n");
      pass++;
    } else {
      printf("      FAIL\n");
      fail++;
    }
  }

  printf("\n=== bulk DMA matrix summary ===\n");
  printf("  pass: %zu\n  fail: %zu\n  skip: %zu\n", pass, fail, skip);
  if (fail == 0 && pass > 0) printf("OK\n");
  else printf("FAILED\n");

FINISH:
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
  return (fail == 0 && pass > 0) ? 0 : 1;
}
