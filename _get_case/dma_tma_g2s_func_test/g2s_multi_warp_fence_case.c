#define g_cases g2s_multi_warp_fence_cases
#define fill_src g2s_multi_warp_fence_fill_src
#define run_one g2s_multi_warp_fence_run_one
/*
 * Same-workgroup multi-warp CP_ASYNC_BULK + CP_ASYNC_FENCE test.
 *
 * The scheduler unit tests already cover per-warp inflight accounting. This
 * testcase drives the OpenCL/runtime/RTL path with several active warps and
 * verifies that each warp's DMA burst reaches its own shared-memory segment.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/ventus_opencl_test.h"

#define SRC_BYTES 2048

typedef struct {
  const char *name;
  unsigned num_warps;
  unsigned dmas_per_warp;
  unsigned copy_bytes;
  unsigned src_base_offset;
  unsigned src_stride;
} mw_case_t;

static const mw_case_t g_cases[] = {
  {"2warp_1dma_each_fence",        2, 1,  64,   0,  64},
  {"2warp_2dma_each_single_fence", 2, 2,  64,   0,  64},
  {"4warp_1dma_each_fence",        4, 1,  64,   0,  64},
  {"2warp_cross_cacheline_each",   2, 1,  32, 120, 256},
};

static void
fill_src(uint8_t *src, size_t nbytes)
{
  for (size_t i = 0; i < nbytes; i++) {
    src[i] = (uint8_t)((i * 7 + 0x19) & 0xFF);
  }
}

static int
run_one(cl_context context, cl_command_queue queue, cl_kernel kernel1,
        cl_kernel kernel2,
        const mw_case_t *c)
{
  int rc = 1;
  cl_int err;
  cl_kernel kernel = c->dmas_per_warp == 2 ? kernel2 : kernel1;
  cl_mem src_buf = NULL, dst_buf = NULL;
  unsigned total_segments = c->num_warps * c->dmas_per_warp;
  unsigned dst_bytes = total_segments * c->copy_bytes;
  unsigned src_end = c->src_base_offset +
                     (total_segments - 1) * c->src_stride + c->copy_bytes;
  uint8_t *src = (uint8_t *)malloc(SRC_BYTES);
  uint8_t *dst = (uint8_t *)calloc(dst_bytes, 1);

  if (!src || !dst) {
    printf("  FAIL: host alloc\n");
    goto FINISH;
  }
  if (src_end > SRC_BYTES || dst_bytes > 1024) {
    printf("  FAIL: case exceeds static buffer bounds\n");
    goto FINISH;
  }

  fill_src(src, SRC_BYTES);
  src_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           SRC_BYTES, src, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer src");
  dst_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                           dst_bytes, NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer dst");

  cl_uint copy_bytes = c->copy_bytes;
  cl_uint dmas_per_warp = c->dmas_per_warp;
  cl_uint src_base_offset = c->src_base_offset;
  cl_uint src_stride = c->src_stride;
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg 0");
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg 1");
  err = clSetKernelArg(kernel, 2, sizeof(cl_uint), &copy_bytes);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg 2");
  err = clSetKernelArg(kernel, 3, sizeof(cl_uint), &dmas_per_warp);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg 3");
  err = clSetKernelArg(kernel, 4, sizeof(cl_uint), &src_base_offset);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg 4");
  err = clSetKernelArg(kernel, 5, sizeof(cl_uint), &src_stride);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg 5");

  size_t lsz = c->num_warps * 32;
  size_t gsz = lsz;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gsz, &lsz,
                               0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel");
  err = clEnqueueReadBuffer(queue, dst_buf, CL_TRUE, 0, dst_bytes,
                            dst, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer");

  rc = 0;
  for (unsigned seg = 0; seg < total_segments; seg++) {
    unsigned src_off = c->src_base_offset + seg * c->src_stride;
    unsigned dst_off = seg * c->copy_bytes;
    for (unsigned i = 0; i < c->copy_bytes; i++) {
      uint8_t exp = src[src_off + i];
      uint8_t got = dst[dst_off + i];
      if (got != exp) {
        printf("  FAIL seg%u byte[%u]: expected 0x%02x got 0x%02x\n",
               seg, i, exp, got);
        rc = 1;
        goto FINISH;
      }
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
g2s_multi_warp_fence_case_main(int argc, char **argv)
{
  cl_int err;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_platform_id platform = NULL;
  cl_program program = NULL;
  cl_kernel kernel1 = NULL;
  cl_kernel kernel2 = NULL;
  size_t pass = 0, fail = 0, skip = 0;
  const char *filter = argc > 1 ? argv[1] : NULL;

  err = ventus_get_default_device(&context, &device, &queue, &platform);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device");
  err = ventus_build_program_from_source(context, device,
                                         "dma_tma_g2s_func_test.cl",
                                         &program);
  CHECK_OPENCL_ERROR_IN("ventus_build_program_from_source");
  kernel1 = clCreateKernel(program, "multi_warp_dma_fence_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel kernel1");
  kernel2 = clCreateKernel(program, "multi_warp_dma_fence2_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel kernel2");

  for (size_t i = 0; i < sizeof(g_cases) / sizeof(g_cases[0]); i++) {
    const mw_case_t *c = &g_cases[i];
    if (filter && !strstr(c->name, filter)) { skip++; continue; }

    printf("[%zu/%zu] %s warps=%u dmas=%u bytes=%u\n",
           i + 1, sizeof(g_cases) / sizeof(g_cases[0]), c->name,
           c->num_warps, c->dmas_per_warp, c->copy_bytes);
    if (run_one(context, queue, kernel1, kernel2, c) == 0) {
      printf("      PASS\n");
      pass++;
    } else {
      printf("      FAIL\n");
      fail++;
    }
  }

  printf("\n=== multi-warp DMA fence summary ===\n");
  printf("  pass: %zu\n  fail: %zu\n  skip: %zu\n", pass, fail, skip);
  if (fail == 0 && pass > 0) printf("OK\n");
  else printf("FAILED\n");

FINISH:
  if (kernel2) clReleaseKernel(kernel2);
  if (kernel1) clReleaseKernel(kernel1);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
  return (fail == 0 && pass > 0) ? 0 : 1;
}
