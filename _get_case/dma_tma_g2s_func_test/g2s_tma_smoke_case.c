#define desc_control g2s_tensor_smoke_desc_control
#define build_desc g2s_tensor_smoke_build_desc
/*
 * Tensor DMA smoke host program.
 * Launches a kernel that uses descriptor-form CP_ASYNC_TENSOR (funct3=2)
 * to copy a 4x4 FP32 tensor from global memory to shared memory, then back to
 * global memory for host verification.
 */

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/ventus_opencl_test.h"

#define DESC_WORDS 32
#define COORD_WORDS 32

static uint32_t
desc_control(unsigned data_type, unsigned rank)
{
  return (data_type & 0xfu) | ((rank & 0xfu) << 4);
}

static void
build_desc(uint32_t *desc)
{
  memset(desc, 0, DESC_WORDS * sizeof(uint32_t));
  desc[0] = 0x56544d41u;
  desc[1] = desc_control(6, 2);  /* FP32, rank=2 */
  desc[2] = 0;                   /* kernel patches runtime src pointer */
  desc[3] = 128;
  desc[4] = 4;
  desc[5] = 4;
  desc[6] = desc[7] = desc[8] = 1;
  desc[9] = 4;
  desc[10] = 16;
  desc[14] = 4;
  desc[15] = 4;
  desc[16] = desc[17] = desc[18] = 1;
  for (int i = 0; i < 5; i++) desc[19 + i] = 1;
}

int
g2s_tensor_smoke_case_main(int argc, char **argv)
{
  (void)argc;
  (void)argv;

  const int rows = 4;
  const int cols = 4;
  const int count = rows * cols;
  const int wg_size = 32;

  printf("Tensor DMA test: %dx%d FP32 descriptor ABI, wg_size=%d\n",
         rows, cols, wg_size);

  cl_int err;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_platform_id platform = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_mem desc_buf = NULL, coords_buf = NULL, src_buf = NULL, dst_buf = NULL;
  float *src = NULL;
  float *dst = NULL;
  int pass = 1;

  err = ventus_get_default_device(&context, &device, &queue, &platform);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device");

  err = ventus_build_program_from_source(context, device, "dma_tma_g2s_func_test.cl",
                                         &program);
  CHECK_OPENCL_ERROR_IN("ventus_build_program_from_source");

  src = (float *)malloc(count * sizeof(float));
  dst = (float *)calloc(count, sizeof(float));
  uint32_t desc[DESC_WORDS];
  uint32_t coords[COORD_WORDS];
  build_desc(desc);
  memset(coords, 0, sizeof(coords));
  for (int i = 0; i < count; i++) {
    src[i] = (float)(i + 1);
  }

  desc_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            sizeof(desc), desc, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer desc");
  coords_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(coords), coords, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer coords");
  src_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           count * sizeof(float), src, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer src");
  dst_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                           count * sizeof(float), NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer dst");

  kernel = clCreateKernel(program, "tensor_dma_copy", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel");

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &desc_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg 0");
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &coords_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg 1");
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &src_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg 2");
  err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &dst_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg 3");

  size_t global_size = wg_size;
  size_t local_size = wg_size;

  printf("Launching kernel...\n");
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size,
                               &local_size, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel");

  err = clEnqueueReadBuffer(queue, dst_buf, CL_TRUE, 0,
                            count * sizeof(float), dst, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer");

  for (int i = 0; i < count; i++) {
    if (dst[i] != src[i]) {
      printf("FAIL at [%d]: expected %.1f, got %.1f\n", i, src[i], dst[i]);
      pass = 0;
    }
  }

  if (pass) printf("OK\n");
  else printf("FAILED\n");

FINISH:
  if (desc_buf) clReleaseMemObject(desc_buf);
  if (coords_buf) clReleaseMemObject(coords_buf);
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
