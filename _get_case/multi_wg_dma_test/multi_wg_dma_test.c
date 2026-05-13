/*
 * Multi-workgroup DMA test host program.
 * Launches 2 workgroups, each uses DMA to copy its data segment
 * from global to shared memory and back. Verifies all data.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/ventus_opencl_test.h"

int
main(int argc, char **argv)
{
  int count = 16;    /* elements per workgroup */
  int wg_size = 32;  /* threads per workgroup */
  int num_wg = 2;    /* number of workgroups */

  if (argc > 1) count = atoi(argv[1]);
  if (argc > 2) wg_size = atoi(argv[2]);
  if (argc > 3) num_wg = atoi(argv[3]);

  if (count > 64) {
    fprintf(stderr, "count must be <= 64 (shared_buf size)\n");
    return 1;
  }
  if (count > wg_size) {
    fprintf(stderr, "count must be <= wg_size\n");
    return 1;
  }

  int total = count * num_wg;
  printf("Multi-WG DMA test: count=%d, wg_size=%d, num_wg=%d, total=%d\n",
         count, wg_size, num_wg, total);

  cl_int err;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_platform_id platform = NULL;
  cl_program program = NULL;

  err = ventus_get_default_device(&context, &device, &queue, &platform);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device");

  err = ventus_build_program_from_source(context, device,
                                         "multi_wg_dma_test.cl", &program);
  CHECK_OPENCL_ERROR_IN("ventus_build_program_from_source");

  /* Allocate and initialize source data: total elements across all WGs */
  cl_int *src = (cl_int *)malloc(total * sizeof(cl_int));
  cl_int *dst = (cl_int *)calloc(total, sizeof(cl_int));
  for (int i = 0; i < total; i++) {
    src[i] = 0xDA00 + i;  /* recognizable pattern per WG */
  }

  cl_mem src_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  total * sizeof(cl_int), src, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer src");

  cl_mem dst_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                  total * sizeof(cl_int), NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer dst");

  cl_kernel kernel = clCreateKernel(program, "multi_wg_dma_copy", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel");

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg 0");
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg 1");
  err = clSetKernelArg(kernel, 2, sizeof(cl_int), &count);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg 2");

  size_t global_size = (size_t)wg_size * num_wg;  /* num_wg workgroups */
  size_t local_size = wg_size;

  printf("Launching kernel: global_size=%zu, local_size=%zu\n",
         global_size, local_size);
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size,
                               &local_size, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel");

  err = clEnqueueReadBuffer(queue, dst_buf, CL_TRUE, 0,
                            total * sizeof(cl_int), dst, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer");

  /* Verify all workgroups' data.  Each WG runs in a separate CU with its
   * own independent ShareMem, so there is no per-WG LDS aliasing. */
  int pass = 1;
  for (int wg = 0; wg < num_wg; wg++) {
    for (int i = 0; i < count; i++) {
      int idx = wg * count + i;
      if (dst[idx] != src[idx]) {
        printf("FAIL WG%d [%d]: expected 0x%08x, got 0x%08x\n",
               wg, i, src[idx], dst[idx]);
        pass = 0;
      }
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
