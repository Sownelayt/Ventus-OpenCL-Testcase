/*
 * CP_ASYNC_COPYSIZE test host program.
 * Tests the funct=0 DMA path which copies 4 << copysize bytes.
 * Uses copysize=2 → 16 bytes = 4 ints.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/ventus_opencl_test.h"

int
g2s_copysize_smoke_case_main(int argc, char **argv)
{
  /* copysize=2 → 16 bytes = 4 ints */
  int count = 4;
  int wg_size = 32;

  printf("CP_ASYNC_COPYSIZE test: copysize=2 (16 bytes, %d ints), wg_size=%d\n",
         count, wg_size);

  cl_int err;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_platform_id platform = NULL;
  cl_program program = NULL;

  err = ventus_get_default_device(&context, &device, &queue, &platform);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device");

  err = ventus_build_program_from_source(context, device, "dma_tma_g2s_func_test.cl",
                                         &program);
  CHECK_OPENCL_ERROR_IN("ventus_build_program_from_source");

  cl_int *src = (cl_int *)malloc(count * sizeof(cl_int));
  cl_int *dst = (cl_int *)calloc(count, sizeof(cl_int));
  for (int i = 0; i < count; i++) {
    src[i] = 0xC500 + i;  /* recognizable pattern */
  }

  cl_mem src_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  count * sizeof(cl_int), src, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer src");

  cl_mem dst_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                  count * sizeof(cl_int), NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer dst");

  cl_kernel kernel = clCreateKernel(program, "dma_copysize", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel");

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg 0");
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg 1");

  size_t global_size = wg_size;
  size_t local_size = wg_size;

  printf("Launching kernel...\n");
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size,
                               &local_size, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel");

  err = clEnqueueReadBuffer(queue, dst_buf, CL_TRUE, 0,
                            count * sizeof(cl_int), dst, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer");

  /* Verify */
  int pass = 1;
  for (int i = 0; i < count; i++) {
    if (dst[i] != src[i]) {
      printf("FAIL at [%d]: expected 0x%08x, got 0x%08x\n", i, src[i], dst[i]);
      pass = 0;
    }
  }

  if (pass)
    printf("OK\n");
  else
    printf("FAILED\n");

FINISH:
  if (src_buf) clReleaseMemObject(src_buf);
  if (dst_buf) clReleaseMemObject(dst_buf);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
  free(src);
  free(dst);

  return pass ? 0 : 1;
}
