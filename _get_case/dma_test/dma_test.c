/*
 * DMA test host program.
 * Launches a kernel that uses DMA (CP_ASYNC_BULK + CP_ASYNC_FENCE)
 * to copy data from global to shared memory, then back to global.
 * Verifies the result on host.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/ventus_opencl_test.h"

int
main(int argc, char **argv)
{
  /* Default: 16 ints, 1 workgroup of 32 threads */
  int count = 16;
  int wg_size = 32;

  if (argc > 1) count = atoi(argv[1]);
  if (argc > 2) wg_size = atoi(argv[2]);

  if (count > 64) {
    fprintf(stderr, "count must be <= 64 (shared_buf size)\n");
    return 1;
  }
  if (count > wg_size) {
    fprintf(stderr, "count must be <= wg_size\n");
    return 1;
  }

  printf("DMA test: count=%d, wg_size=%d\n", count, wg_size);

  cl_int err;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_platform_id platform = NULL;
  cl_program program = NULL;

  err = ventus_get_default_device(&context, &device, &queue, &platform);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device");

  err = ventus_build_program_from_source(context, device, "dma_test.cl",
                                         &program);
  CHECK_OPENCL_ERROR_IN("ventus_build_program_from_source");

  /* Allocate and initialize source data */
  cl_int *src = (cl_int *)malloc(count * sizeof(cl_int));
  cl_int *dst = (cl_int *)malloc(count * sizeof(cl_int));
  for (int i = 0; i < count; i++) {
    src[i] = 0xD0A0 + i;  /* recognizable pattern */
    dst[i] = 0;
  }

  cl_mem src_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  count * sizeof(cl_int), src, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer src");

  cl_mem dst_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                  count * sizeof(cl_int), NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer dst");

  cl_kernel kernel = clCreateKernel(program, "dma_copy", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel");

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg 0");
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg 1");
  err = clSetKernelArg(kernel, 2, sizeof(cl_int), &count);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg 2");

  size_t global_size = wg_size;  /* 1 workgroup */
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
