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
#define MULTI_WG_TMA_NUM_WG 2
#define MULTI_WG_TMA_WG_SIZE 32

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

static void build_multi_wg_tma_inputs(uint32_t *desc, uint32_t *coords)
{
  memset(coords, 0, MULTI_WG_TMA_NUM_WG * MULTI_WG_TMA_COORD_WORDS *
                      sizeof(uint32_t));
  for (unsigned wg = 0; wg < MULTI_WG_TMA_NUM_WG; wg++) {
    build_multi_wg_tma_desc(desc + wg * MULTI_WG_TMA_DESC_WORDS);
  }
}

static int
spike_backend(void)
{
  const char *backend = getenv("VENTUS_BACKEND");
  return backend && strcmp(backend, "spike") == 0;
}

static int check_multi_wg_tma_words(const char *label, const uint32_t *got,
                                    uint32_t (*expected)(unsigned, unsigned))
{
  for (unsigned wg = 0; wg < MULTI_WG_TMA_NUM_WG; wg++) {
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
         MULTI_WG_TMA_NUM_WG, MULTI_WG_TMA_TILE_ELEMS);
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

static int run_bulk_s2g_2wg(void)
{
  if (spike_backend()) {
    printf("SKIP bulk_s2g_2wg: Spike multi-WG S2G/TMA path is not a stable oracle\n");
    return SUITE_SKIP;
  }
  const int count = 16;
  const int wg_size = 32;
  const int num_wg = 2;
  const int total = count * num_wg;
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
  printf("PASS bulk_s2g_2wg count=%d wg_size=%d num_wg=%d\n",
         count, wg_size, num_wg);

FINISH:
  if (dst_buf) clReleaseMemObject(dst_buf);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
  free(dst);
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

  uint32_t desc[MULTI_WG_TMA_NUM_WG * MULTI_WG_TMA_DESC_WORDS];
  uint32_t coords[MULTI_WG_TMA_NUM_WG * MULTI_WG_TMA_COORD_WORDS];
  uint32_t src[MULTI_WG_TMA_NUM_WG * MULTI_WG_TMA_TILE_ELEMS];
  uint32_t dst[MULTI_WG_TMA_NUM_WG * MULTI_WG_TMA_TILE_ELEMS];

  build_multi_wg_tma_inputs(desc, coords);
  for (unsigned wg = 0; wg < MULTI_WG_TMA_NUM_WG; wg++) {
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

  size_t global = MULTI_WG_TMA_WG_SIZE * MULTI_WG_TMA_NUM_WG;
  size_t local = MULTI_WG_TMA_WG_SIZE;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                               0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel(tma_g2s_2wg)");
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(tma_g2s_2wg)");
  err = clEnqueueReadBuffer(queue, dst_buf, CL_TRUE, 0, sizeof(dst), dst,
                            0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(tma_g2s_2wg)");

  exit_code = check_multi_wg_tma_words("tma_g2s_2wg", dst, tma_g2s_word);

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

static int run_tma_s2g_2wg(void)
{
  if (spike_backend()) {
    printf("SKIP tma_s2g_2wg: Spike multi-WG S2G/TMA path is not a stable oracle\n");
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

  uint32_t desc[MULTI_WG_TMA_NUM_WG * MULTI_WG_TMA_DESC_WORDS];
  uint32_t coords[MULTI_WG_TMA_NUM_WG * MULTI_WG_TMA_COORD_WORDS];
  uint32_t dst[MULTI_WG_TMA_NUM_WG * MULTI_WG_TMA_TILE_ELEMS];

  build_multi_wg_tma_inputs(desc, coords);
  for (unsigned i = 0; i < MULTI_WG_TMA_NUM_WG * MULTI_WG_TMA_TILE_ELEMS; i++) {
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
                            sizeof(desc), desc, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(tma_s2g desc)");
  coords_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(coords), coords, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(tma_s2g coords)");
  dst_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(dst), dst, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(tma_s2g dst)");

  err  = clSetKernelArg(kernel, 0, sizeof(desc_buf), &desc_buf);
  err |= clSetKernelArg(kernel, 1, sizeof(coords_buf), &coords_buf);
  err |= clSetKernelArg(kernel, 2, sizeof(dst_buf), &dst_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(tma_s2g_2wg)");

  size_t global = MULTI_WG_TMA_WG_SIZE * MULTI_WG_TMA_NUM_WG;
  size_t local = MULTI_WG_TMA_WG_SIZE;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                               0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel(tma_s2g_2wg)");
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(tma_s2g_2wg)");
  err = clEnqueueReadBuffer(queue, dst_buf, CL_TRUE, 0, sizeof(dst), dst,
                            0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(tma_s2g_2wg)");

  exit_code = check_multi_wg_tma_words("tma_s2g_2wg", dst, tma_s2g_word);

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
    {"tma_g2s_2wg", "multi-wg|g2s|tensor|descriptor|funct2|short",
     "two workgroups copy independent descriptor tensors into shared memory", 1,
     run_tma_g2s_2wg},
    {"tma_s2g_2wg", "multi-wg|s2g|tensor|descriptor|funct4|short",
     "two workgroups write independent shared tensors through descriptors", 1,
     run_tma_s2g_2wg},
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
