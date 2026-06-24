#define g_cases g2s_routing_conflict_cases
#define fill_src g2s_routing_conflict_fill_src
#define expected_conflict_value g2s_routing_conflict_expected_value
#define run_one g2s_routing_conflict_run_one
/*
 * Shared bank-conflict pressure test for DMA response routing.
 *
 * The kernel overlaps a CP_ASYNC_BULK response to shared memory with repeated
 * same-bank normal shared accesses. A failure in sourceTag propagation usually
 * shows up as either a DMA payload mismatch, a conflict value mismatch, or hang.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/ventus_opencl_test.h"

#define SRC_BYTES 768

typedef struct {
  const char *name;
  unsigned local_size;
  unsigned copy_bytes;
  unsigned rounds;
  unsigned src_offset;
} routing_case_t;

static const routing_case_t g_cases[] = {
  {"routing_64B_conflict64",            64,  64, 32,   0},
  {"routing_192B_crossline_conflict64", 64, 192, 64, 120},
};

static void
fill_src(uint8_t *src, size_t nbytes)
{
  for (size_t i = 0; i < nbytes; i++) {
    src[i] = (uint8_t)((i * 11 + 0x43) & 0xFF);
  }
}

static uint32_t
expected_conflict_value(unsigned lid, unsigned rounds)
{
  uint32_t value = 0xA5000000u + lid;
  for (unsigned r = 0; r < rounds; r++) {
    value = value + ((r + 1) * 17u) + lid;
  }
  return value;
}

static int
run_one(cl_context context, cl_command_queue queue, cl_kernel kernel,
        const routing_case_t *c)
{
  int rc = 1;
  cl_int err;
  cl_mem src_buf = NULL, dma_out_buf = NULL, conflict_out_buf = NULL;
  uint8_t *src = (uint8_t *)malloc(SRC_BYTES);
  uint8_t *dma_out = (uint8_t *)calloc(c->copy_bytes, 1);
  uint32_t *conflict_out = (uint32_t *)calloc(c->local_size, sizeof(uint32_t));

  if (!src || !dma_out || !conflict_out) {
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
  dma_out_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                               c->copy_bytes, NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer dma_out");
  conflict_out_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    c->local_size * sizeof(uint32_t),
                                    NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer conflict_out");

  cl_uint copy_bytes = c->copy_bytes;
  cl_uint rounds = c->rounds;
  cl_uint src_offset = c->src_offset;
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg 0");
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dma_out_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg 1");
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &conflict_out_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg 2");
  err = clSetKernelArg(kernel, 3, sizeof(cl_uint), &copy_bytes);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg 3");
  err = clSetKernelArg(kernel, 4, sizeof(cl_uint), &rounds);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg 4");
  err = clSetKernelArg(kernel, 5, sizeof(cl_uint), &src_offset);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg 5");

  size_t gsz = c->local_size, lsz = c->local_size;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gsz, &lsz,
                               0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel");
  err = clEnqueueReadBuffer(queue, dma_out_buf, CL_TRUE, 0, c->copy_bytes,
                            dma_out, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer dma_out");
  err = clEnqueueReadBuffer(queue, conflict_out_buf, CL_TRUE, 0,
                            c->local_size * sizeof(uint32_t), conflict_out,
                            0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer conflict_out");

  rc = 0;
  for (unsigned i = 0; i < c->copy_bytes; i++) {
    uint8_t exp = src[c->src_offset + i];
    if (dma_out[i] != exp) {
      printf("  FAIL dma byte[%u]: expected 0x%02x got 0x%02x\n",
             i, exp, dma_out[i]);
      rc = 1;
      goto FINISH;
    }
  }
  for (unsigned lid = 0; lid < c->local_size; lid++) {
    uint32_t exp = expected_conflict_value(lid, c->rounds);
    if (conflict_out[lid] != exp) {
      printf("  FAIL conflict lid%u: expected 0x%08x got 0x%08x\n",
             lid, exp, conflict_out[lid]);
      rc = 1;
      goto FINISH;
    }
  }

FINISH:
  if (src_buf) clReleaseMemObject(src_buf);
  if (dma_out_buf) clReleaseMemObject(dma_out_buf);
  if (conflict_out_buf) clReleaseMemObject(conflict_out_buf);
  free(src);
  free(dma_out);
  free(conflict_out);
  return rc;
}

int
g2s_routing_conflict_case_main(int argc, char **argv)
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
                                         "dma_tma_g2s_func_test.cl",
                                         &program);
  CHECK_OPENCL_ERROR_IN("ventus_build_program_from_source");
  kernel = clCreateKernel(program, "dma_shared_routing_conflict_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel");

  for (size_t i = 0; i < sizeof(g_cases) / sizeof(g_cases[0]); i++) {
    const routing_case_t *c = &g_cases[i];
    if (filter && !strstr(c->name, filter)) { skip++; continue; }

    printf("[%zu/%zu] %s local=%u bytes=%u rounds=%u\n",
           i + 1, sizeof(g_cases) / sizeof(g_cases[0]), c->name,
           c->local_size, c->copy_bytes, c->rounds);
    if (run_one(context, queue, kernel, c) == 0) {
      printf("      PASS\n");
      pass++;
    } else {
      printf("      FAIL\n");
      fail++;
    }
  }

  printf("\n=== DMA shared routing conflict summary ===\n");
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
