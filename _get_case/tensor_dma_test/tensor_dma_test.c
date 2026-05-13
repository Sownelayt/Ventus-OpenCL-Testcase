/*
 * Tensor DMA test host program.
 * Launches a kernel that uses CP_ASYNC_TENSOR to copy a 4x4 FP32 tensor
 * from global memory to shared memory, then back to global.
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
  const int rows = 4;
  const int cols = 4;
  const int count = rows * cols; /* 16 elements */
  const int wg_size = 32;       /* one full warp */

  printf("Tensor DMA test: %dx%d FP32, wg_size=%d\n", rows, cols, wg_size);

  cl_int err;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_platform_id platform = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_mem src_buf = NULL, dst_buf = NULL;

  err = ventus_get_default_device(&context, &device, &queue, &platform);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device");

  err = ventus_build_program_from_source(context, device, "tensor_dma_test.cl",
                                         &program);
  CHECK_OPENCL_ERROR_IN("ventus_build_program_from_source");

  /* Allocate and initialize source data: 1.0f, 2.0f, ..., 16.0f */
  float *src = (float *)malloc(count * sizeof(float));
  float *dst = (float *)calloc(count, sizeof(float));
  for (int i = 0; i < count; i++) {
    src[i] = (float)(i + 1);
  }

  src_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           count * sizeof(float), src, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer src");

  dst_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                           count * sizeof(float), NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer dst");

  kernel = clCreateKernel(program, "tensor_dma_copy", &err);
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
                            count * sizeof(float), dst, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer");

  /* Verify element-wise */
  int pass = 1;
  for (int i = 0; i < count; i++) {
    if (dst[i] != src[i]) {
      printf("FAIL at [%d]: expected %.1f, got %.1f\n", i, src[i], dst[i]);
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
