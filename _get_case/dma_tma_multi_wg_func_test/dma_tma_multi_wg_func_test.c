/*
 * Multi-workgroup DMA/TMA functional test.
 *
 * Multi-WG coverage is isolated from single-WG functional suites because GVM
 * checker and CTA/WG scheduling failures have a different debugging surface.
 * Bulk G2S host coverage lives in a named case module, while S2G/TMA variants
 * are native suite cases in this same executable.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dma_tma_multi_wg_cases.h"
#include "../common/ventus_opencl_test.h"

#define SUITE_SKIP 77

typedef int (*suite_fn_t)(void);

typedef struct {
  const char *name;
  const char *tags;
  const char *description;
  int default_run;
  suite_fn_t run;
} suite_t;

#define MULTI_WG_TMA_DESC_WORDS 32
#define MULTI_WG_TMA_COORD_WORDS 32
#define MULTI_WG_TMA_TILE_ELEMS 16
#define MULTI_WG_TMA_TILE_BYTES (MULTI_WG_TMA_TILE_ELEMS * 4)
#define MULTI_WG_TMA_MAX_WG 4
#define MULTI_WG_TMA_WG_SIZE 32
#define MULTI_WG_TMA_SUBBOX_ELEMS 64
#define MULTI_WG_TMA_SUBBOX_BYTES (MULTI_WG_TMA_SUBBOX_ELEMS * 4)
#define MULTI_WG_TENSOR_BLOCK_WORDS 64
#define MULTI_WG_TENSOR_BLOCK_BYTES (MULTI_WG_TENSOR_BLOCK_WORDS * 4)
#define MULTI_WG_TENSOR_PAGE_BLOCK_BYTES (3u * 4096u + MULTI_WG_TMA_TILE_BYTES)
#define MULTI_WG_TENSOR_SHARED_ELEMS 64

typedef struct {
  const char *name;
  unsigned rank;
  unsigned dataType;
  unsigned globalDim[5];
  unsigned globalStrides[5];
  unsigned boxDim[5];
  unsigned elementStrides[5];
  unsigned coord[5];
  size_t dst_offset_bytes;
  uint32_t pattern_base;
} multi_wg_tensor_case_t;

static uint32_t tma_g2s_word(unsigned wg, unsigned idx)
{
  return 0xA7000000u + wg * 0x1000u + idx;
}

static uint32_t tma_s2g_word(unsigned wg, unsigned idx)
{
  return 0xC2000000u + wg * 0x1000u + idx;
}

static void build_multi_wg_tma_desc(uint32_t *desc)
{
  memset(desc, 0, MULTI_WG_TMA_DESC_WORDS * sizeof(uint32_t));
  desc[0] = 0x56544d41u;
  desc[1] = (6u & 0xfu) | (2u << 4);
  desc[2] = 0;
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

static void build_multi_wg_tma_desc_case(uint32_t *desc,
                                         const multi_wg_tensor_case_t *c)
{
  memset(desc, 0, MULTI_WG_TMA_DESC_WORDS * sizeof(uint32_t));
  desc[0] = 0x56544d41u;
  desc[1] = ((c->dataType ? c->dataType : 6u) & 0xfu) |
            ((c->rank & 0xfu) << 4);
  desc[2] = 0;
  desc[3] = 128;
  for (unsigned i = 0; i < 5; i++) {
    desc[4 + i] = c->globalDim[i] ? c->globalDim[i] : 1;
    desc[9 + i] = i == 0 ? 4 : c->globalStrides[i - 1];
    desc[14 + i] = c->boxDim[i] ? c->boxDim[i] : 1;
    desc[19 + i] = c->elementStrides[i] ? c->elementStrides[i] : 1;
  }
}

static uint32_t multi_wg_tensor_pattern(const multi_wg_tensor_case_t *c,
                                        unsigned wg, unsigned idx)
{
  return c->pattern_base + wg * 0x1000u + idx;
}

static size_t multi_wg_tensor_dst_offset(const multi_wg_tensor_case_t *c,
                                         const unsigned coord[5])
{
  size_t off = (size_t)coord[0] * sizeof(uint32_t);
  for (unsigned d = 1; d < c->rank; d++) {
    off += (size_t)coord[d] * c->globalStrides[d - 1];
  }
  return off;
}

static void fill_multi_wg_tensor_expected(uint8_t *expected,
                                          size_t total_bytes,
                                          const multi_wg_tensor_case_t *cases,
                                          unsigned num_wg,
                                          size_t block_bytes)
{
  memset(expected, 0xcd, total_bytes);

  for (unsigned wg = 0; wg < num_wg; wg++) {
    const multi_wg_tensor_case_t *c = &cases[wg];
    unsigned outDim[5] = {1, 1, 1, 1, 1};
    size_t total = 1;
    for (unsigned d = 0; d < c->rank; d++) {
      outDim[d] = c->boxDim[d] ? c->boxDim[d] : 1;
      total *= outDim[d];
    }

    for (size_t lin = 0; lin < total; lin++) {
      unsigned idx[5] = {0, 0, 0, 0, 0};
      unsigned coord[5] = {0, 0, 0, 0, 0};
      size_t rem = lin;
      int oob = 0;

      for (unsigned d = 0; d < c->rank; d++) {
        idx[d] = (unsigned)(rem % outDim[d]);
        rem /= outDim[d];
      }
      for (unsigned d = 0; d < c->rank; d++) {
        unsigned stride = c->elementStrides[d] ? c->elementStrides[d] : 1;
        coord[d] = c->coord[d] + idx[d] * stride;
        if (coord[d] >= c->globalDim[d]) oob = 1;
      }
      if (oob) continue;

      size_t dst_off = multi_wg_tensor_dst_offset(c, coord);
      size_t shared_word = lin;
      size_t byte_off = (size_t)wg * block_bytes + c->dst_offset_bytes +
                        dst_off;
      uint32_t word = multi_wg_tensor_pattern(c, wg,
                                              (unsigned)shared_word);
      if (byte_off + sizeof(word) <= total_bytes) {
        memcpy(expected + byte_off, &word, sizeof(word));
      }
    }
  }
}

static int check_multi_wg_tensor_bytes(const char *label, const uint8_t *got,
                                       const uint8_t *expected,
                                       size_t total_bytes)
{
  for (size_t i = 0; i < total_bytes; i++) {
    if (got[i] != expected[i]) {
      fprintf(stderr,
              "FAIL %s byte %zu: got=%02x exp=%02x\n",
              label, i, got[i], expected[i]);
      return 1;
    }
  }
  printf("PASS %s bytes=%zu\n", label, total_bytes);
  return 0;
}

static void build_multi_wg_tma_inputs(uint32_t *desc, uint32_t *coords,
                                      unsigned num_wg)
{
  memset(coords, 0, num_wg * MULTI_WG_TMA_COORD_WORDS *
                      sizeof(uint32_t));
  for (unsigned wg = 0; wg < num_wg; wg++) {
    build_multi_wg_tma_desc(desc + wg * MULTI_WG_TMA_DESC_WORDS);
  }
}

static int
spike_backend(void)
{
  const char *backend = getenv("VENTUS_BACKEND");
  return backend && strcmp(backend, "spike") == 0;
}

static int
gvm_backend(void)
{
  const char *backend = getenv("VENTUS_BACKEND");
  return backend && (strcmp(backend, "gvm") == 0 ||
                     strcmp(backend, "gvm-nocache") == 0);
}

static int check_multi_wg_tma_words(const char *label, const uint32_t *got,
                                    unsigned num_wg,
                                    uint32_t (*expected)(unsigned, unsigned))
{
  for (unsigned wg = 0; wg < num_wg; wg++) {
    for (unsigned i = 0; i < MULTI_WG_TMA_TILE_ELEMS; i++) {
      unsigned idx = wg * MULTI_WG_TMA_TILE_ELEMS + i;
      uint32_t exp = expected(wg, i);
      if (got[idx] != exp) {
        fprintf(stderr,
                "FAIL %s WG%u [%u]: got=0x%08x exp=0x%08x\n",
                label, wg, i, got[idx], exp);
        return 1;
      }
    }
  }
  printf("PASS %s num_wg=%u elems_per_wg=%u\n", label,
         num_wg, MULTI_WG_TMA_TILE_ELEMS);
  return 0;
}

static int run_bulk_g2s_2wg(void)
{
  char *argv[] = {"multi_wg_dma_test", "16", "32", "2", NULL};
  return multi_wg_bulk_g2s_case_main(4, argv);
}

static int run_bulk_g2s_4wg(void)
{
  char *argv[] = {"multi_wg_dma_test", "16", "32", "4", NULL};
  return multi_wg_bulk_g2s_case_main(4, argv);
}

static int run_bulk_s2g_num_wg(unsigned num_wg)
{
  if (spike_backend()) {
    printf("SKIP bulk_s2g_%uwg: Spike multi-WG S2G/TMA path is not a stable oracle\n",
           num_wg);
    return SUITE_SKIP;
  }
  const int count = 16;
  const int wg_size = 32;
  const int total = count * (int)num_wg;
  cl_int err = CL_SUCCESS;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_mem dst_buf = NULL;
  int *dst = NULL;
  int exit_code = 1;

  dst = (int *)calloc((size_t)total, sizeof(int));
  if (!dst) {
    fprintf(stderr, "FAIL bulk_s2g_2wg host alloc\n");
    return 1;
  }

  err = ventus_get_default_device(&context, &device, &queue, NULL);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device(multi_wg_s2g)");
  err = ventus_build_program_from_source(context, device,
                                         "multi_wg_dma_s2g_test.cl", &program);
  CHECK_OPENCL_ERROR_IN("ventus_build_program_from_source(multi_wg_s2g)");
  kernel = clCreateKernel(program, "multi_wg_dma_s2g_copy", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(multi_wg_s2g)");
  dst_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
                           (size_t)total * sizeof(int), NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(multi_wg_s2g dst)");

  cl_int cl_count = count;
  err  = clSetKernelArg(kernel, 0, sizeof(dst_buf), &dst_buf);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_count), &cl_count);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(multi_wg_s2g)");

  size_t global = (size_t)wg_size * num_wg;
  size_t local = (size_t)wg_size;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                               0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel(multi_wg_s2g)");
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(multi_wg_s2g)");
  err = clEnqueueReadBuffer(queue, dst_buf, CL_TRUE, 0,
                            (size_t)total * sizeof(int), dst, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(multi_wg_s2g)");

  exit_code = 0;
  for (int wg = 0; wg < num_wg; wg++) {
    for (int i = 0; i < count; i++) {
      int idx = wg * count + i;
      int exp = (int)(0xB7000000u + (unsigned)wg * 0x1000u + (unsigned)i);
      if (dst[idx] != exp) {
        fprintf(stderr, "FAIL bulk_s2g_2wg WG%d [%d]: got=0x%08x exp=0x%08x\n",
                wg, i, dst[idx], exp);
        exit_code = 1;
        goto FINISH;
      }
    }
  }
  printf("PASS bulk_s2g_%uwg count=%d wg_size=%d num_wg=%u\n",
         num_wg, count, wg_size, num_wg);

FINISH:
  if (dst_buf) clReleaseMemObject(dst_buf);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
  free(dst);
  return exit_code;
}

static int run_bulk_s2g_2wg(void)
{
  return run_bulk_s2g_num_wg(2);
}

static int run_bulk_s2g_4wg(void)
{
  return run_bulk_s2g_num_wg(4);
}

static int run_bulk_s2g_2wg_group_wait(void)
{
  if (spike_backend()) {
    printf("SKIP bulk_s2g_2wg_group_wait: Spike multi-WG S2G/TMA path is not a stable oracle\n");
    return SUITE_SKIP;
  }
  const int count = 16;
  const unsigned num_wg = 2;
  const int words_per_wg = count * 2;
  const int total = words_per_wg * (int)num_wg;
  cl_int err = CL_SUCCESS;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_mem dst_buf = NULL;
  int *dst = NULL;
  int exit_code = 1;

  dst = (int *)calloc((size_t)total, sizeof(int));
  if (!dst) {
    fprintf(stderr, "FAIL bulk_s2g_2wg_group_wait host alloc\n");
    return 1;
  }
  for (int i = 0; i < total; i++) dst[i] = (int)0xcdcdcdcdu;

  err = ventus_get_default_device(&context, &device, &queue, NULL);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device(multi_wg_s2g_group)");
  err = ventus_build_program_from_source(context, device,
                                         "multi_wg_dma_s2g_test.cl", &program);
  CHECK_OPENCL_ERROR_IN("ventus_build_program_from_source(multi_wg_s2g_group)");
  kernel = clCreateKernel(program, "multi_wg_dma_s2g_group_wait", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(multi_wg_s2g_group)");
  dst_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           (size_t)total * sizeof(int), dst, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(multi_wg_s2g_group dst)");

  cl_int cl_count = count;
  err  = clSetKernelArg(kernel, 0, sizeof(dst_buf), &dst_buf);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_count), &cl_count);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(multi_wg_s2g_group)");

  size_t global = MULTI_WG_TMA_WG_SIZE * num_wg;
  size_t local = MULTI_WG_TMA_WG_SIZE;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                               0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel(multi_wg_s2g_group)");
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(multi_wg_s2g_group)");
  err = clEnqueueReadBuffer(queue, dst_buf, CL_TRUE, 0,
                            (size_t)total * sizeof(int), dst, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(multi_wg_s2g_group)");

  exit_code = 0;
  for (unsigned wg = 0; wg < num_wg; wg++) {
    for (int i = 0; i < count; i++) {
      int idx0 = (int)wg * words_per_wg + i;
      int idx1 = idx0 + count;
      int exp0 = (int)(0xB7100000u + wg * 0x1000u + (unsigned)i);
      int exp1 = (int)(0xB7200000u + wg * 0x1000u + (unsigned)i);
      if (dst[idx0] != exp0 || dst[idx1] != exp1) {
        fprintf(stderr,
                "FAIL bulk_s2g_2wg_group_wait WG%u [%d]: got=%08x,%08x exp=%08x,%08x\n",
                wg, i, dst[idx0], dst[idx1], exp0, exp1);
        exit_code = 1;
        goto FINISH;
      }
    }
  }
  printf("PASS bulk_s2g_2wg_group_wait count=%d num_wg=%u\n", count, num_wg);

FINISH:
  if (dst_buf) clReleaseMemObject(dst_buf);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
  free(dst);
  return exit_code;
}

static int run_multi_wg_cross_page(void)
{
  if (spike_backend()) {
    printf("SKIP multi_wg_cross_page: Spike multi-WG S2G/TMA path is not a stable oracle\n");
    return SUITE_SKIP;
  }
  const int count = 16;
  const unsigned num_wg = 2;
  const size_t block_bytes = 3u * 4096u + (size_t)count * sizeof(uint32_t);
  const size_t total_bytes = block_bytes * num_wg;
  cl_int err = CL_SUCCESS;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_mem dst_buf = NULL;
  uint8_t *dst = NULL;
  int exit_code = 1;

  dst = (uint8_t *)malloc(total_bytes);
  if (!dst) {
    fprintf(stderr, "FAIL multi_wg_cross_page host alloc\n");
    return 1;
  }
  memset(dst, 0xcd, total_bytes);

  err = ventus_get_default_device(&context, &device, &queue, NULL);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device(multi_wg_cross_page)");
  err = ventus_build_program_from_source(context, device,
                                         "multi_wg_dma_s2g_test.cl", &program);
  CHECK_OPENCL_ERROR_IN("ventus_build_program_from_source(multi_wg_cross_page)");
  kernel = clCreateKernel(program, "multi_wg_dma_s2g_cross_page", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(multi_wg_cross_page)");
  dst_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           total_bytes, dst, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(multi_wg_cross_page dst)");

  cl_int cl_count = count;
  err  = clSetKernelArg(kernel, 0, sizeof(dst_buf), &dst_buf);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_count), &cl_count);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(multi_wg_cross_page)");

  size_t global = MULTI_WG_TMA_WG_SIZE * num_wg;
  size_t local = MULTI_WG_TMA_WG_SIZE;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                               0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel(multi_wg_cross_page)");
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(multi_wg_cross_page)");
  err = clEnqueueReadBuffer(queue, dst_buf, CL_TRUE, 0, total_bytes, dst,
                            0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(multi_wg_cross_page)");

  exit_code = 0;
  for (unsigned wg = 0; wg < num_wg; wg++) {
    const size_t base = block_bytes * wg;
    const uint32_t seeds[3] = {0xB7300000u, 0xB7400000u, 0xB7500000u};
    const size_t offs[3] = {0u, 4096u, 8192u};
    for (unsigned page = 0; page < 3; page++) {
      for (int i = 0; i < count; i++) {
        uint32_t got;
        uint32_t exp = seeds[page] + wg * 0x1000u + (unsigned)i;
        memcpy(&got, dst + base + offs[page] + (size_t)i * sizeof(uint32_t),
               sizeof(got));
        if (got != exp) {
          fprintf(stderr,
                  "FAIL multi_wg_cross_page WG%u page%u [%d]: got=%08x exp=%08x\n",
                  wg, page, i, got, exp);
          exit_code = 1;
          goto FINISH;
        }
      }
    }
  }
  printf("PASS multi_wg_cross_page count=%d num_wg=%u bytes=%zu\n",
         count, num_wg, total_bytes);

FINISH:
  if (dst_buf) clReleaseMemObject(dst_buf);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
  free(dst);
  return exit_code;
}

static int run_bulk_s2g_4wg_group_page(void)
{
  if (spike_backend() || gvm_backend()) {
    printf("SKIP bulk_s2g_4wg_group_page: GVM/Spike multi-WG group/page stress is not a stable oracle\n");
    return SUITE_SKIP;
  }
  const int count = 16;
  const unsigned num_wg = 4;
  const size_t byte_count = (size_t)count * sizeof(uint32_t);
  const size_t block_bytes = 3u * 4096u + byte_count;
  const size_t total_bytes = block_bytes * num_wg;
  cl_int err = CL_SUCCESS;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_mem dst_buf = NULL;
  uint8_t *dst = NULL;
  int exit_code = 1;

  dst = (uint8_t *)malloc(total_bytes);
  if (!dst) {
    fprintf(stderr, "FAIL bulk_s2g_4wg_group_page host alloc\n");
    return 1;
  }
  memset(dst, 0xcd, total_bytes);

  err = ventus_get_default_device(&context, &device, &queue, NULL);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device(bulk_s2g_4wg_group_page)");
  err = ventus_build_program_from_source(context, device,
                                         "multi_wg_dma_s2g_test.cl", &program);
  CHECK_OPENCL_ERROR_IN("ventus_build_program_from_source(bulk_s2g_4wg_group_page)");
  kernel = clCreateKernel(program, "multi_wg_dma_s2g_4wg_group_page", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(bulk_s2g_4wg_group_page)");
  dst_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           total_bytes, dst, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(bulk_s2g_4wg_group_page dst)");

  cl_int cl_count = count;
  err  = clSetKernelArg(kernel, 0, sizeof(dst_buf), &dst_buf);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_count), &cl_count);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(bulk_s2g_4wg_group_page)");

  size_t global = MULTI_WG_TMA_WG_SIZE * num_wg;
  size_t local = MULTI_WG_TMA_WG_SIZE;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                               0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel(bulk_s2g_4wg_group_page)");
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(bulk_s2g_4wg_group_page)");
  err = clEnqueueReadBuffer(queue, dst_buf, CL_TRUE, 0, total_bytes, dst,
                            0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(bulk_s2g_4wg_group_page)");

  exit_code = 0;
  const uint32_t seeds[4] = {0xB7600000u, 0xB7700000u,
                             0xB7800000u, 0xB7900000u};
  const size_t offs[4] = {0u, 4096u, 8192u, 12288u};
  for (unsigned wg = 0; wg < num_wg; wg++) {
    const size_t base = block_bytes * wg;
    for (unsigned page = 0; page < 4; page++) {
      for (int i = 0; i < count; i++) {
        uint32_t got;
        uint32_t exp = seeds[page] + wg * 0x1000u + (unsigned)i;
        memcpy(&got, dst + base + offs[page] + (size_t)i * sizeof(uint32_t),
               sizeof(got));
        if (got != exp) {
          fprintf(stderr,
                  "FAIL bulk_s2g_4wg_group_page WG%u page%u [%d]: got=%08x exp=%08x\n",
                  wg, page, i, got, exp);
          exit_code = 1;
          goto FINISH;
        }
      }
      if (offs[page] + byte_count < block_bytes &&
          dst[base + offs[page] + byte_count] != 0xcd) {
        fprintf(stderr,
                "FAIL bulk_s2g_4wg_group_page WG%u page%u guard got=%02x\n",
                wg, page, dst[base + offs[page] + byte_count]);
        exit_code = 1;
        goto FINISH;
      }
    }
  }
  printf("PASS bulk_s2g_4wg_group_page count=%d num_wg=%u bytes=%zu\n",
         count, num_wg, total_bytes);

FINISH:
  if (dst_buf) clReleaseMemObject(dst_buf);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
  free(dst);
  return exit_code;
}

static int run_mixed_g2s_s2g_4wg(void)
{
  if (spike_backend() || gvm_backend()) {
    printf("SKIP mixed_g2s_s2g_4wg: GVM/Spike multi-WG mixed G2S/S2G stress is not a stable oracle\n");
    return SUITE_SKIP;
  }
  const int count = 16;
  const unsigned num_wg = 4;
  const int dst_words_per_wg = count * 2;
  cl_int err = CL_SUCCESS;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_mem src_buf = NULL;
  cl_mem dst_buf = NULL;
  uint32_t *src = NULL;
  uint32_t *dst = NULL;
  int exit_code = 1;

  src = (uint32_t *)malloc((size_t)num_wg * count * sizeof(uint32_t));
  dst = (uint32_t *)malloc((size_t)num_wg * dst_words_per_wg *
                           sizeof(uint32_t));
  if (!src || !dst) {
    fprintf(stderr, "FAIL mixed_g2s_s2g_4wg host alloc\n");
    goto FINISH;
  }
  for (unsigned wg = 0; wg < num_wg; wg++) {
    for (int i = 0; i < count; i++) {
      src[wg * count + i] = 0xA8000000u + wg * 0x1000u + (unsigned)i;
    }
  }
  for (unsigned i = 0; i < num_wg * (unsigned)dst_words_per_wg; i++) {
    dst[i] = 0xcdcdcdcdu;
  }

  err = ventus_get_default_device(&context, &device, &queue, NULL);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device(mixed_g2s_s2g_4wg)");
  err = ventus_build_program_from_source(context, device,
                                         "multi_wg_dma_s2g_test.cl", &program);
  CHECK_OPENCL_ERROR_IN("ventus_build_program_from_source(mixed_g2s_s2g_4wg)");
  kernel = clCreateKernel(program, "multi_wg_dma_mixed_g2s_s2g", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(mixed_g2s_s2g_4wg)");

  src_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           (size_t)num_wg * count * sizeof(uint32_t), src,
                           &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(mixed_g2s_s2g_4wg src)");
  dst_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           (size_t)num_wg * dst_words_per_wg *
                             sizeof(uint32_t), dst, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(mixed_g2s_s2g_4wg dst)");

  cl_int cl_count = count;
  err  = clSetKernelArg(kernel, 0, sizeof(src_buf), &src_buf);
  err |= clSetKernelArg(kernel, 1, sizeof(dst_buf), &dst_buf);
  err |= clSetKernelArg(kernel, 2, sizeof(cl_count), &cl_count);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(mixed_g2s_s2g_4wg)");

  size_t global = MULTI_WG_TMA_WG_SIZE * num_wg;
  size_t local = MULTI_WG_TMA_WG_SIZE;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                               0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel(mixed_g2s_s2g_4wg)");
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(mixed_g2s_s2g_4wg)");
  err = clEnqueueReadBuffer(queue, dst_buf, CL_TRUE, 0,
                            (size_t)num_wg * dst_words_per_wg *
                              sizeof(uint32_t), dst,
                            0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(mixed_g2s_s2g_4wg)");

  exit_code = 0;
  for (unsigned wg = 0; wg < num_wg; wg++) {
    for (int i = 0; i < count; i++) {
      unsigned base = wg * (unsigned)dst_words_per_wg;
      uint32_t exp_s2g = 0xB7A00000u + wg * 0x1000u + (unsigned)i;
      uint32_t exp_g2s = 0xA8000000u + wg * 0x1000u + (unsigned)i;
      if (dst[base + (unsigned)i] != exp_s2g ||
          dst[base + (unsigned)count + (unsigned)i] != exp_g2s) {
        fprintf(stderr,
                "FAIL mixed_g2s_s2g_4wg WG%u [%d]: got=%08x,%08x exp=%08x,%08x\n",
                wg, i, dst[base + (unsigned)i],
                dst[base + (unsigned)count + (unsigned)i],
                exp_s2g, exp_g2s);
        exit_code = 1;
        goto FINISH;
      }
    }
  }
  printf("PASS mixed_g2s_s2g_4wg count=%d num_wg=%u\n", count, num_wg);

FINISH:
  if (dst_buf) clReleaseMemObject(dst_buf);
  if (src_buf) clReleaseMemObject(src_buf);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
  free(dst);
  free(src);
  return exit_code;
}

static int run_tma_g2s_2wg(void)
{
  if (spike_backend()) {
    printf("SKIP tma_g2s_2wg: Spike multi-WG S2G/TMA path is not a stable oracle\n");
    return SUITE_SKIP;
  }
  cl_int err = CL_SUCCESS;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_mem desc_buf = NULL;
  cl_mem coords_buf = NULL;
  cl_mem src_buf = NULL;
  cl_mem dst_buf = NULL;
  int exit_code = 1;

  uint32_t desc[2 * MULTI_WG_TMA_DESC_WORDS];
  uint32_t coords[2 * MULTI_WG_TMA_COORD_WORDS];
  uint32_t src[2 * MULTI_WG_TMA_TILE_ELEMS];
  uint32_t dst[2 * MULTI_WG_TMA_TILE_ELEMS];

  build_multi_wg_tma_inputs(desc, coords, 2);
  for (unsigned wg = 0; wg < 2; wg++) {
    for (unsigned i = 0; i < MULTI_WG_TMA_TILE_ELEMS; i++) {
      src[wg * MULTI_WG_TMA_TILE_ELEMS + i] = tma_g2s_word(wg, i);
      dst[wg * MULTI_WG_TMA_TILE_ELEMS + i] = 0xcdcdcdcdu;
    }
  }

  err = ventus_get_default_device(&context, &device, &queue, NULL);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device(tma_g2s_2wg)");
  err = ventus_build_program_from_source(context, device,
                                         "multi_wg_tma_g2s_test.cl", &program);
  CHECK_OPENCL_ERROR_IN("ventus_build_program_from_source(tma_g2s_2wg)");
  kernel = clCreateKernel(program, "multi_wg_tma_g2s_copy", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(tma_g2s_2wg)");

  desc_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            sizeof(desc), desc, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(tma_g2s desc)");
  coords_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(coords), coords, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(tma_g2s coords)");
  src_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(src), src, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(tma_g2s src)");
  dst_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(dst), dst, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(tma_g2s dst)");

  err  = clSetKernelArg(kernel, 0, sizeof(desc_buf), &desc_buf);
  err |= clSetKernelArg(kernel, 1, sizeof(coords_buf), &coords_buf);
  err |= clSetKernelArg(kernel, 2, sizeof(src_buf), &src_buf);
  err |= clSetKernelArg(kernel, 3, sizeof(dst_buf), &dst_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(tma_g2s_2wg)");

  size_t global = MULTI_WG_TMA_WG_SIZE * 2;
  size_t local = MULTI_WG_TMA_WG_SIZE;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                               0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel(tma_g2s_2wg)");
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(tma_g2s_2wg)");
  err = clEnqueueReadBuffer(queue, dst_buf, CL_TRUE, 0, sizeof(dst), dst,
                            0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(tma_g2s_2wg)");

  exit_code = check_multi_wg_tma_words("tma_g2s_2wg", dst, 2, tma_g2s_word);

FINISH:
  if (dst_buf) clReleaseMemObject(dst_buf);
  if (src_buf) clReleaseMemObject(src_buf);
  if (coords_buf) clReleaseMemObject(coords_buf);
  if (desc_buf) clReleaseMemObject(desc_buf);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
  return exit_code;
}

static int run_tma_s2g_num_wg(unsigned num_wg)
{
  if (spike_backend()) {
    printf("SKIP tma_s2g_%uwg: Spike multi-WG S2G/TMA path is not a stable oracle\n",
           num_wg);
    return SUITE_SKIP;
  }
  cl_int err = CL_SUCCESS;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_mem desc_buf = NULL;
  cl_mem coords_buf = NULL;
  cl_mem dst_buf = NULL;
  int exit_code = 1;

  uint32_t desc[MULTI_WG_TMA_MAX_WG * MULTI_WG_TMA_DESC_WORDS];
  uint32_t coords[MULTI_WG_TMA_MAX_WG * MULTI_WG_TMA_COORD_WORDS];
  uint32_t dst[MULTI_WG_TMA_MAX_WG * MULTI_WG_TMA_TILE_ELEMS];

  if (num_wg > MULTI_WG_TMA_MAX_WG) {
    fprintf(stderr, "FAIL tma_s2g_%uwg exceeds max %u\n",
            num_wg, MULTI_WG_TMA_MAX_WG);
    return 1;
  }
  build_multi_wg_tma_inputs(desc, coords, num_wg);
  for (unsigned i = 0; i < num_wg * MULTI_WG_TMA_TILE_ELEMS; i++) {
    dst[i] = 0xcdcdcdcdu;
  }

  err = ventus_get_default_device(&context, &device, &queue, NULL);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device(tma_s2g_2wg)");
  err = ventus_build_program_from_source(context, device,
                                         "multi_wg_tma_s2g_test.cl", &program);
  CHECK_OPENCL_ERROR_IN("ventus_build_program_from_source(tma_s2g_2wg)");
  kernel = clCreateKernel(program, "multi_wg_tma_s2g_copy", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(tma_s2g_2wg)");

  desc_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            (size_t)num_wg * MULTI_WG_TMA_DESC_WORDS *
                              sizeof(uint32_t), desc, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(tma_s2g desc)");
  coords_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              (size_t)num_wg * MULTI_WG_TMA_COORD_WORDS *
                                sizeof(uint32_t), coords, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(tma_s2g coords)");
  dst_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           (size_t)num_wg * MULTI_WG_TMA_TILE_ELEMS *
                             sizeof(uint32_t), dst, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(tma_s2g dst)");

  err  = clSetKernelArg(kernel, 0, sizeof(desc_buf), &desc_buf);
  err |= clSetKernelArg(kernel, 1, sizeof(coords_buf), &coords_buf);
  err |= clSetKernelArg(kernel, 2, sizeof(dst_buf), &dst_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(tma_s2g_2wg)");

  size_t global = MULTI_WG_TMA_WG_SIZE * num_wg;
  size_t local = MULTI_WG_TMA_WG_SIZE;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                               0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel(tma_s2g_2wg)");
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(tma_s2g_2wg)");
  err = clEnqueueReadBuffer(queue, dst_buf, CL_TRUE, 0,
                            (size_t)num_wg * MULTI_WG_TMA_TILE_ELEMS *
                              sizeof(uint32_t), dst,
                            0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(tma_s2g_2wg)");

  exit_code = check_multi_wg_tma_words(num_wg == 2 ? "tma_s2g_2wg" :
                                       "tma_s2g_4wg",
                                       dst, num_wg, tma_s2g_word);

FINISH:
  if (dst_buf) clReleaseMemObject(dst_buf);
  if (coords_buf) clReleaseMemObject(coords_buf);
  if (desc_buf) clReleaseMemObject(desc_buf);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
  return exit_code;
}

static int run_tma_s2g_2wg(void)
{
  return run_tma_s2g_num_wg(2);
}

static int run_tma_s2g_4wg(void)
{
  return run_tma_s2g_num_wg(4);
}

static void build_multi_wg_tma_subbox_desc(uint32_t *desc)
{
  build_multi_wg_tma_desc(desc);
  desc[4] = 8;
  desc[5] = 8;
  desc[9] = 4;
  desc[10] = 32;
  desc[14] = 2;
  desc[15] = 2;
}

static int run_tma_s2g_2wg_subbox(void)
{
  if (spike_backend()) {
    printf("SKIP tma_s2g_2wg_subbox: Spike multi-WG S2G/TMA path is not a stable oracle\n");
    return SUITE_SKIP;
  }
  const unsigned num_wg = 2;
  const unsigned coords_xy[2][2] = {{1, 1}, {4, 3}};
  cl_int err = CL_SUCCESS;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_mem desc_guard_buf = NULL;
  cl_mem desc_buf = NULL;
  cl_mem coords_buf = NULL;
  cl_mem dst_buf = NULL;
  int exit_code = 1;

  uint32_t desc_guard[64];
  uint32_t desc[2 * MULTI_WG_TMA_DESC_WORDS];
  uint32_t coords[2 * MULTI_WG_TMA_COORD_WORDS];
  uint32_t dst[2 * MULTI_WG_TMA_SUBBOX_ELEMS];

  memset(desc_guard, 0, sizeof(desc_guard));
  memset(desc, 0, sizeof(desc));
  memset(coords, 0, sizeof(coords));
  for (unsigned wg = 0; wg < num_wg; wg++) {
    build_multi_wg_tma_subbox_desc(
        desc + wg * MULTI_WG_TMA_DESC_WORDS);
    coords[wg * MULTI_WG_TMA_COORD_WORDS + 0] = coords_xy[wg][0];
    coords[wg * MULTI_WG_TMA_COORD_WORDS + 1] = coords_xy[wg][1];
  }
  for (unsigned i = 0; i < num_wg * MULTI_WG_TMA_SUBBOX_ELEMS; i++) {
    dst[i] = 0xcdcdcdcdu;
  }

  err = ventus_get_default_device(&context, &device, &queue, NULL);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device(tma_s2g_subbox)");
  err = ventus_build_program_from_source(context, device,
                                         "multi_wg_tma_s2g_subbox_test.cl",
                                         &program);
  CHECK_OPENCL_ERROR_IN("ventus_build_program_from_source(tma_s2g_subbox)");
  kernel = clCreateKernel(program, "multi_wg_tma_s2g_subbox", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(tma_s2g_subbox)");

  /*
   * Keep this descriptor buffer on a fresh device page. The RTL descriptor
   * cache is not a coherent cache for host-side descriptor rewrites at the
   * same device address across back-to-back OpenCL buffers in one process.
   * CL_MEM_COPY_HOST_PTR forces Ventus to allocate/upload the guard buffer.
   */
  desc_guard_buf = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(desc_guard), desc_guard, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(tma_s2g_subbox desc guard)");
  desc_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            sizeof(desc), desc, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(tma_s2g_subbox desc)");
  coords_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(coords), coords, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(tma_s2g_subbox coords)");
  dst_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(dst), dst, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(tma_s2g_subbox dst)");

  err  = clSetKernelArg(kernel, 0, sizeof(desc_guard_buf), &desc_guard_buf);
  err |= clSetKernelArg(kernel, 1, sizeof(desc_buf), &desc_buf);
  err |= clSetKernelArg(kernel, 2, sizeof(coords_buf), &coords_buf);
  err |= clSetKernelArg(kernel, 3, sizeof(dst_buf), &dst_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(tma_s2g_subbox)");

  size_t global = MULTI_WG_TMA_WG_SIZE * num_wg;
  size_t local = MULTI_WG_TMA_WG_SIZE;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                               0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel(tma_s2g_subbox)");
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(tma_s2g_subbox)");
  err = clEnqueueReadBuffer(queue, dst_buf, CL_TRUE, 0, sizeof(dst), dst,
                            0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(tma_s2g_subbox)");

  exit_code = 0;
  for (unsigned wg = 0; wg < num_wg; wg++) {
    for (unsigned y = 0; y < 8; y++) {
      for (unsigned x = 0; x < 8; x++) {
        unsigned idx = wg * MULTI_WG_TMA_SUBBOX_ELEMS + y * 8 + x;
        uint32_t exp = 0xcdcdcdcdu;
        if (x >= coords_xy[wg][0] && x < coords_xy[wg][0] + 2 &&
            y >= coords_xy[wg][1] && y < coords_xy[wg][1] + 2) {
          unsigned lx = x - coords_xy[wg][0];
          unsigned ly = y - coords_xy[wg][1];
          exp = 0xC2100000u + wg * 0x1000u + ly * 2u + lx;
        }
        if (dst[idx] != exp) {
          fprintf(stderr,
                  "FAIL tma_s2g_2wg_subbox WG%u (%u,%u): got=%08x exp=%08x\n",
                  wg, x, y, dst[idx], exp);
          exit_code = 1;
          goto FINISH;
        }
      }
    }
  }
  printf("PASS tma_s2g_2wg_subbox num_wg=%u matrix=8x8 box=2x2\n", num_wg);

FINISH:
  if (dst_buf) clReleaseMemObject(dst_buf);
  if (coords_buf) clReleaseMemObject(coords_buf);
  if (desc_buf) clReleaseMemObject(desc_buf);
  if (desc_guard_buf) clReleaseMemObject(desc_guard_buf);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
  return exit_code;
}

static int run_tma_s2g_tensor_complex(const char *label,
                                      const char *source,
                                      const char *kernel_name,
                                      const multi_wg_tensor_case_t *cases,
                                      unsigned num_wg,
                                      size_t block_bytes,
                                      unsigned desc_word_offset,
                                      size_t guard_bytes,
                                      int skip_gvm)
{
  if (spike_backend() || (skip_gvm && gvm_backend())) {
    printf("SKIP %s: GVM/Spike multi-WG tensor stress is not a stable oracle\n",
           label);
    return SUITE_SKIP;
  }
  cl_int err = CL_SUCCESS;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_mem desc_guard_buf = NULL;
  cl_mem desc_buf = NULL;
  cl_mem coords_buf = NULL;
  cl_mem dst_buf = NULL;
  uint32_t *desc = NULL;
  uint32_t *coords = NULL;
  uint32_t *desc_guard = NULL;
  uint8_t *dst = NULL;
  uint8_t *expected = NULL;
  int exit_code = 1;
  const size_t total_bytes = block_bytes * num_wg;
  const size_t desc_words = desc_word_offset +
                            (size_t)num_wg * MULTI_WG_TMA_DESC_WORDS;
  if (guard_bytes < 64u * sizeof(uint32_t)) {
    guard_bytes = 64u * sizeof(uint32_t);
  }

  desc = (uint32_t *)calloc(desc_words, sizeof(uint32_t));
  coords = (uint32_t *)calloc((size_t)num_wg * MULTI_WG_TMA_COORD_WORDS,
                              sizeof(uint32_t));
  desc_guard = (uint32_t *)calloc(guard_bytes / sizeof(uint32_t),
                                  sizeof(uint32_t));
  dst = (uint8_t *)malloc(total_bytes);
  expected = (uint8_t *)malloc(total_bytes);
  if (!desc || !coords || !desc_guard || !dst || !expected) {
    fprintf(stderr, "FAIL %s host alloc\n", label);
    goto FINISH;
  }

  for (unsigned wg = 0; wg < num_wg; wg++) {
    build_multi_wg_tma_desc_case(
        desc + desc_word_offset + wg * MULTI_WG_TMA_DESC_WORDS,
        &cases[wg]);
    for (unsigned d = 0; d < 5; d++) {
      coords[wg * MULTI_WG_TMA_COORD_WORDS + d] = cases[wg].coord[d];
    }
  }
  memset(dst, 0xcd, total_bytes);
  fill_multi_wg_tensor_expected(expected, total_bytes, cases, num_wg,
                                block_bytes);

  err = ventus_get_default_device(&context, &device, &queue, NULL);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device(tma_s2g_tensor_complex)");
  err = ventus_build_program_from_source(context, device, source, &program);
  CHECK_OPENCL_ERROR_IN("ventus_build_program_from_source(tma_s2g_tensor_complex)");
  kernel = clCreateKernel(program, kernel_name, &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(tma_s2g_tensor_complex)");

  desc_guard_buf = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  guard_bytes, desc_guard, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(tma_s2g_complex desc guard)");
  desc_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            desc_words * sizeof(uint32_t), desc, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(tma_s2g_complex desc)");
  coords_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              (size_t)num_wg * MULTI_WG_TMA_COORD_WORDS *
                                sizeof(uint32_t), coords, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(tma_s2g_complex coords)");
  dst_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           total_bytes, dst, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(tma_s2g_complex dst)");

  err  = clSetKernelArg(kernel, 0, sizeof(desc_guard_buf), &desc_guard_buf);
  err |= clSetKernelArg(kernel, 1, sizeof(desc_buf), &desc_buf);
  err |= clSetKernelArg(kernel, 2, sizeof(coords_buf), &coords_buf);
  err |= clSetKernelArg(kernel, 3, sizeof(dst_buf), &dst_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(tma_s2g_tensor_complex)");

  size_t global = MULTI_WG_TMA_WG_SIZE * num_wg;
  size_t local = MULTI_WG_TMA_WG_SIZE;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                               0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel(tma_s2g_tensor_complex)");
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(tma_s2g_tensor_complex)");
  err = clEnqueueReadBuffer(queue, dst_buf, CL_TRUE, 0, total_bytes, dst,
                            0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(tma_s2g_tensor_complex)");

  exit_code = check_multi_wg_tensor_bytes(label, dst, expected, total_bytes);

FINISH:
  if (dst_buf) clReleaseMemObject(dst_buf);
  if (coords_buf) clReleaseMemObject(coords_buf);
  if (desc_buf) clReleaseMemObject(desc_buf);
  if (desc_guard_buf) clReleaseMemObject(desc_guard_buf);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
  free(expected);
  free(dst);
  free(desc_guard);
  free(coords);
  free(desc);
  return exit_code;
}

static int run_tma_s2g_2wg_stride_oob(void)
{
  const multi_wg_tensor_case_t cases[2] = {
    {
      .name = "wg0_stride_x_oob",
      .rank = 2,
      .globalDim = {8, 4, 1, 1, 1},
      .globalStrides = {32, 0, 0, 0, 0},
      .boxDim = {6, 3, 1, 1, 1},
      .elementStrides = {2, 1, 1, 1, 1},
      .coord = {1, 1, 0, 0, 0},
      .pattern_base = 0xC2200000u,
    },
    {
      .name = "wg1_stride_xy_oob",
      .rank = 2,
      .globalDim = {8, 4, 1, 1, 1},
      .globalStrides = {32, 0, 0, 0, 0},
      .boxDim = {5, 3, 1, 1, 1},
      .elementStrides = {2, 1, 1, 1, 1},
      .coord = {0, 2, 0, 0, 0},
      .pattern_base = 0xC2200000u,
    },
  };
  return run_tma_s2g_tensor_complex("tma_s2g_2wg_stride_oob",
                                    "multi_wg_tma_s2g_stride_oob_test.cl",
                                    "multi_wg_tma_s2g_stride_oob",
                                    cases, 2, MULTI_WG_TENSOR_BLOCK_BYTES,
                                    768u, 256u, 1);
}

static int run_tma_s2g_4wg_descriptor_mix(void)
{
  const multi_wg_tensor_case_t cases[4] = {
    {
      .name = "wg0_rank1",
      .rank = 1,
      .globalDim = {32, 1, 1, 1, 1},
      .boxDim = {16, 1, 1, 1, 1},
      .coord = {4, 0, 0, 0, 0},
      .pattern_base = 0xC2300000u,
    },
    {
      .name = "wg1_rank2_subbox",
      .rank = 2,
      .globalDim = {8, 8, 1, 1, 1},
      .globalStrides = {32, 0, 0, 0, 0},
      .boxDim = {4, 4, 1, 1, 1},
      .coord = {2, 3, 0, 0, 0},
      .pattern_base = 0xC2300000u,
    },
    {
      .name = "wg2_rank3_subbox",
      .rank = 3,
      .globalDim = {4, 4, 4, 1, 1},
      .globalStrides = {16, 64, 0, 0, 0},
      .boxDim = {2, 2, 2, 1, 1},
      .coord = {1, 1, 1, 0, 0},
      .pattern_base = 0xC2300000u,
    },
    {
      .name = "wg3_u32_stride",
      .rank = 2,
      .dataType = 2,
      .globalDim = {8, 8, 1, 1, 1},
      .globalStrides = {32, 0, 0, 0, 0},
      .boxDim = {4, 2, 1, 1, 1},
      .elementStrides = {2, 1, 1, 1, 1},
      .coord = {0, 5, 0, 0, 0},
      .pattern_base = 0xC2300000u,
    },
  };
  return run_tma_s2g_tensor_complex("tma_s2g_4wg_descriptor_mix",
                                    "multi_wg_tma_s2g_descriptor_mix_test.cl",
                                    "multi_wg_tma_s2g_descriptor_mix",
                                    cases, 4, MULTI_WG_TENSOR_BLOCK_BYTES,
                                    256u, 4096u + 256u, 1);
}

static int run_tma_s2g_4wg_group_page(void)
{
  const multi_wg_tensor_case_t cases[4] = {
    {
      .name = "wg0_page0",
      .rank = 1,
      .globalDim = {16, 1, 1, 1, 1},
      .boxDim = {16, 1, 1, 1, 1},
      .dst_offset_bytes = 0u,
      .pattern_base = 0xC2400000u,
    },
    {
      .name = "wg1_page1",
      .rank = 1,
      .globalDim = {16, 1, 1, 1, 1},
      .boxDim = {16, 1, 1, 1, 1},
      .dst_offset_bytes = 4096u,
      .pattern_base = 0xC2400000u,
    },
    {
      .name = "wg2_page2",
      .rank = 1,
      .globalDim = {16, 1, 1, 1, 1},
      .boxDim = {16, 1, 1, 1, 1},
      .dst_offset_bytes = 8192u,
      .pattern_base = 0xC2400000u,
    },
    {
      .name = "wg3_page3",
      .rank = 1,
      .globalDim = {16, 1, 1, 1, 1},
      .boxDim = {16, 1, 1, 1, 1},
      .dst_offset_bytes = 12288u,
      .pattern_base = 0xC2400000u,
    },
  };
  return run_tma_s2g_tensor_complex("tma_s2g_4wg_group_page",
                                    "multi_wg_tma_s2g_group_page_test.cl",
                                    "multi_wg_tma_s2g_group_page",
                                    cases, 4,
                                    MULTI_WG_TENSOR_PAGE_BLOCK_BYTES,
                                    512u, 2u * 4096u + 256u, 1);
}

static int matches_filter(const suite_t *suite, const char *filter)
{
  return strstr(suite->name, filter) || strstr(suite->tags, filter) ||
         strstr(suite->description, filter);
}

static int selected(const suite_t *suite, int full, int nfilters,
                    char **filters)
{
  if (nfilters == 0) return full || suite->default_run;
  for (int i = 0; i < nfilters; i++) {
    if (matches_filter(suite, filters[i])) return 1;
  }
  return 0;
}

int main(int argc, char **argv)
{
  suite_t suites[] = {
    {"bulk_g2s_2wg", "multi-wg|g2s|bulk|funct1|funct6|short",
     "two workgroups copy independent global segments through shared memory", 1,
     run_bulk_g2s_2wg},
    {"bulk_s2g_2wg", "multi-wg|s2g|bulk|funct3|funct6|short",
     "two workgroups write independent shared-memory segments to global", 1,
     run_bulk_s2g_2wg},
    {"bulk_s2g_4wg", "multi-wg|s2g|bulk|funct3|funct6|stress",
     "four workgroups write independent shared-memory segments to global", 1,
     run_bulk_s2g_4wg},
    {"bulk_s2g_2wg_group_wait",
     "multi-wg|s2g|bulk|funct3|group-wait|completion|stress",
     "two workgroups each commit two S2G groups and wait independently", 1,
     run_bulk_s2g_2wg_group_wait},
    {"multi_wg_cross_page",
     "multi-wg|s2g|bulk|funct3|page-boundary|tlb|stress",
     "two workgroups write independent sparse A/B/C pages", 1,
     run_multi_wg_cross_page},
    {"bulk_s2g_4wg_group_page",
     "multi-wg|s2g|bulk|funct3|4wg|group-wait|page-boundary|tlb|stress",
     "four workgroups commit independent S2G groups across four sparse pages",
     1, run_bulk_s2g_4wg_group_page},
    {"mixed_g2s_s2g_4wg",
     "multi-wg|mixed|g2s|s2g|bulk|funct1|funct3|4wg|completion|stress",
     "four workgroups keep G2S and S2G outstanding before independent checks",
     1, run_mixed_g2s_s2g_4wg},
    {"tma_g2s_2wg", "multi-wg|g2s|tensor|descriptor|funct2|short",
     "two workgroups copy independent descriptor tensors into shared memory", 1,
     run_tma_g2s_2wg},
    {"tma_s2g_2wg", "multi-wg|s2g|tensor|descriptor|funct4|short",
     "two workgroups write independent shared tensors through descriptors", 1,
     run_tma_s2g_2wg},
    {"tma_s2g_2wg_subbox",
     "multi-wg|s2g|tensor|descriptor|funct4|subbox|stress",
     "two workgroups write independent tensor subboxes through descriptors", 1,
     run_tma_s2g_2wg_subbox},
    {"tma_s2g_2wg_stride_oob",
     "multi-wg|s2g|tensor|descriptor|funct4|stride|oob|fallback|stress",
     "two workgroups write tensor stride/OOB fallback cases independently", 1,
     run_tma_s2g_2wg_stride_oob},
    {"tma_s2g_4wg", "multi-wg|s2g|tensor|descriptor|funct4|stress",
     "four workgroups write independent shared tensors through descriptors", 1,
     run_tma_s2g_4wg},
    {"tma_s2g_4wg_descriptor_mix",
     "multi-wg|s2g|tensor|descriptor|funct4|4wg|rank3|datatype|fallback|stress",
     "four workgroups use different descriptor shapes and coords", 1,
     run_tma_s2g_4wg_descriptor_mix},
    {"tma_s2g_4wg_group_page",
     "multi-wg|s2g|tensor|descriptor|funct4|4wg|group-wait|page-boundary|tlb|stress",
     "four workgroups write tensor groups to separate sparse pages", 1,
     run_tma_s2g_4wg_group_page},
    {"bulk_g2s_4wg", "multi-wg|g2s|bulk|funct1|funct6|short",
     "four workgroups version of the same independent-segment check", 1,
     run_bulk_g2s_4wg},
  };
  const size_t suite_count = sizeof(suites) / sizeof(suites[0]);
  int list = 0;
  int full = 0;
  char *filters[32];
  int nfilters = 0;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--list") == 0) {
      list = 1;
    } else if (strcmp(argv[i], "--full") == 0 ||
               strcmp(argv[i], "--mode=full") == 0) {
      full = 1;
    } else if (nfilters < (int)(sizeof(filters) / sizeof(filters[0]))) {
      filters[nfilters++] = argv[i];
    }
  }

  if (list) {
    printf("dma_tma_multi_wg_func_test suites:\n");
    for (size_t i = 0; i < suite_count; i++) {
      printf("  %-16s default=%d tags=%s\n      %s\n",
             suites[i].name, suites[i].default_run, suites[i].tags,
             suites[i].description);
    }
    return 0;
  }

  size_t pass = 0, fail = 0, skip = 0;
  for (size_t i = 0; i < suite_count; i++) {
    if (!selected(&suites[i], full, nfilters, filters)) {
      skip++;
      continue;
    }

    printf("\n[%zu/%zu] dma_tma_multi_wg_func_test.%s\n",
           i + 1, suite_count, suites[i].name);
    int rc = suites[i].run();
    if (rc == 0) {
      printf("[PASS] %s\n", suites[i].name);
      pass++;
    } else if (rc == SUITE_SKIP) {
      printf("[SKIP] %s\n", suites[i].name);
      skip++;
    } else {
      printf("[FAIL] %s rc=%d\n", suites[i].name, rc);
      fail++;
    }
  }

  printf("\n=== dma_tma_multi_wg_func_test summary ===\n");
  printf("  pass: %zu\n  fail: %zu\n  skip: %zu\n", pass, fail, skip);
  if (fail == 0 && pass > 0) {
    printf("OK\n");
    return 0;
  }
  printf("FAILED\n");
  return 1;
}
