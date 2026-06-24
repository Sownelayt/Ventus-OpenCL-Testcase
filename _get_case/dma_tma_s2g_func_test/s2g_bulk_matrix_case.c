#define g_cases s2g_bulk_matrix_cases
#define pattern_byte s2g_bulk_matrix_pattern_byte
#define fill_seed s2g_bulk_matrix_fill_seed
#define check_region s2g_bulk_matrix_check_region
#define run_one s2g_bulk_matrix_run_one
/*
 * Application-level CP_ASYNC_BULK_S2G directed test.
 *
 * Covers the funct3=3 shared -> global ABI, partial cacheline writes,
 * destination cacheline crossing, shared source offsets, and a G2S/S2G
 * roundtrip separated by CP_ASYNC_FENCE/WAIT_ALL.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/ventus_opencl_test.h"

#define LOCAL_SIZE 32
#define SHARED_BUF_BYTES 512
#define DST_GUARD 64
#define MODE_S2G 0
#define MODE_ROUNDTRIP 1

typedef struct {
  const char *name;
  unsigned mode;
  unsigned src_offset;
  unsigned copy_bytes;
  unsigned dst_offset;
} s2g_case_t;

static const s2g_case_t g_cases[] = {
  {"basic_128B_aligned",       MODE_S2G,       0, 128,   0},
  {"partial_32B",              MODE_S2G,      16,  32,   0},
  {"partial_tail_96B",         MODE_S2G,      64,  96,   0},
  {"dst_offset_cross_line",    MODE_S2G,       0,  64, 112},
  {"src_shared_offset_128B",   MODE_S2G,      48, 128,  32},
  {"g2s_wait_s2g_roundtrip",   MODE_ROUNDTRIP, 0, 128,   0},
};

static uint8_t
pattern_byte(unsigned idx)
{
  return (uint8_t)((idx * 7u + 0x23u) & 0xffu);
}

static void
fill_seed(uint8_t *src, size_t nbytes)
{
  for (size_t i = 0; i < nbytes; i++) {
    src[i] = (uint8_t)((i * 11u + 0x5du) & 0xffu);
  }
}

static int
check_region(const s2g_case_t *c, const uint8_t *dst, const uint8_t *seed)
{
  unsigned dst_bytes = c->dst_offset + c->copy_bytes + DST_GUARD;
  for (unsigned i = 0; i < dst_bytes; i++) {
    uint8_t exp = 0xcd;
    if (i >= c->dst_offset && i < c->dst_offset + c->copy_bytes) {
      unsigned rel = i - c->dst_offset;
      exp = (c->mode == MODE_ROUNDTRIP) ? seed[rel] : pattern_byte(c->src_offset + rel);
    }
    if (dst[i] != exp) {
      printf("  FAIL byte[%u]: expected 0x%02x got 0x%02x\n", i, exp, dst[i]);
      return 1;
    }
  }
  return 0;
}

static int
run_one(cl_context context, cl_command_queue queue,
        cl_kernel s2g_kernel, cl_kernel roundtrip_kernel,
        const s2g_case_t *c)
{
  int rc = 1;
  cl_int err;
  cl_mem dst_buf = NULL, seed_buf = NULL;
  unsigned dst_bytes = c->dst_offset + c->copy_bytes + DST_GUARD;
  uint8_t *dst_init = (uint8_t *)malloc(dst_bytes);
  uint8_t *dst = (uint8_t *)malloc(dst_bytes);
  uint8_t *seed = (uint8_t *)malloc(c->copy_bytes ? c->copy_bytes : 1);

  if (!dst_init || !dst || !seed) {
    printf("  FAIL: host alloc\n");
    goto FINISH;
  }
  if (c->src_offset + c->copy_bytes > SHARED_BUF_BYTES) {
    printf("  FAIL: shared source window exceeds SHARED_BUF_BYTES\n");
    goto FINISH;
  }

  memset(dst_init, 0xcd, dst_bytes);
  fill_seed(seed, c->copy_bytes ? c->copy_bytes : 1);

  dst_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           dst_bytes, dst_init, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer dst");

  size_t gsz = LOCAL_SIZE, lsz = LOCAL_SIZE;
  if (c->mode == MODE_ROUNDTRIP) {
    seed_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              c->copy_bytes, seed, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer seed");
    cl_uint copy_bytes = c->copy_bytes;
    err = clSetKernelArg(roundtrip_kernel, 0, sizeof(cl_mem), &seed_buf);
    CHECK_OPENCL_ERROR_IN("clSetKernelArg roundtrip 0");
    err = clSetKernelArg(roundtrip_kernel, 1, sizeof(cl_mem), &dst_buf);
    CHECK_OPENCL_ERROR_IN("clSetKernelArg roundtrip 1");
    err = clSetKernelArg(roundtrip_kernel, 2, sizeof(cl_uint), &copy_bytes);
    CHECK_OPENCL_ERROR_IN("clSetKernelArg roundtrip 2");
    err = clEnqueueNDRangeKernel(queue, roundtrip_kernel, 1, NULL, &gsz, &lsz,
                                 0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel roundtrip");
  } else {
    cl_uint src_offset = c->src_offset;
    cl_uint copy_bytes = c->copy_bytes;
    cl_uint dst_offset = c->dst_offset;
    err = clSetKernelArg(s2g_kernel, 0, sizeof(cl_mem), &dst_buf);
    CHECK_OPENCL_ERROR_IN("clSetKernelArg s2g 0");
    err = clSetKernelArg(s2g_kernel, 1, sizeof(cl_uint), &src_offset);
    CHECK_OPENCL_ERROR_IN("clSetKernelArg s2g 1");
    err = clSetKernelArg(s2g_kernel, 2, sizeof(cl_uint), &copy_bytes);
    CHECK_OPENCL_ERROR_IN("clSetKernelArg s2g 2");
    err = clSetKernelArg(s2g_kernel, 3, sizeof(cl_uint), &dst_offset);
    CHECK_OPENCL_ERROR_IN("clSetKernelArg s2g 3");
    err = clEnqueueNDRangeKernel(queue, s2g_kernel, 1, NULL, &gsz, &lsz,
                                 0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel s2g");
  }

  err = clEnqueueReadBuffer(queue, dst_buf, CL_TRUE, 0, dst_bytes,
                            dst, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer dst");
  rc = check_region(c, dst, seed);

FINISH:
  if (seed_buf) clReleaseMemObject(seed_buf);
  if (dst_buf) clReleaseMemObject(dst_buf);
  free(dst_init);
  free(dst);
  free(seed);
  return rc;
}

int
s2g_bulk_matrix_case_main(int argc, char **argv)
{
  cl_int err;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_platform_id platform = NULL;
  cl_program program = NULL;
  cl_kernel s2g_kernel = NULL, roundtrip_kernel = NULL;
  size_t pass = 0, fail = 0, skip = 0;
  const char *filter = argc > 1 ? argv[1] : NULL;

  err = ventus_get_default_device(&context, &device, &queue, &platform);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device");
  err = ventus_build_program_from_source(context, device,
                                         "shared_to_global_dma_test.cl", &program);
  CHECK_OPENCL_ERROR_IN("ventus_build_program_from_source");
  s2g_kernel = clCreateKernel(program, "shared_to_global_dma_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel s2g");
  roundtrip_kernel = clCreateKernel(program, "g2s_s2g_roundtrip_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel roundtrip");

  for (size_t i = 0; i < sizeof(g_cases) / sizeof(g_cases[0]); i++) {
    const s2g_case_t *c = &g_cases[i];
    if (filter && !strstr(c->name, filter)) { skip++; continue; }

    printf("[%zu/%zu] %s src_off=%u bytes=%u dst_off=%u\n",
           i + 1, sizeof(g_cases) / sizeof(g_cases[0]), c->name,
           c->src_offset, c->copy_bytes, c->dst_offset);
    if (run_one(context, queue, s2g_kernel, roundtrip_kernel, c) == 0) {
      printf("      PASS\n");
      pass++;
    } else {
      printf("      FAIL\n");
      fail++;
    }
  }

  printf("\n=== shared->global DMA summary ===\n");
  printf("  pass: %zu\n  fail: %zu\n  skip: %zu\n", pass, fail, skip);
  if (fail == 0 && pass > 0) printf("OK\n");
  else printf("FAILED\n");

FINISH:
  if (roundtrip_kernel) clReleaseKernel(roundtrip_kernel);
  if (s2g_kernel) clReleaseKernel(s2g_kernel);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
  return (fail == 0 && pass > 0) ? 0 : 1;
}
