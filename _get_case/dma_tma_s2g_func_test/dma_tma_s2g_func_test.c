/*
 * Single-workgroup DMA/TMA S2G functional test.
 *
 * This project groups shared-to-global bulk DMA and tensor S2G coverage behind
 * one host entry point. Individual host case implementations live in named
 * case modules in this project, so the suite builds one executable directly.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dma_tma_s2g_cases.h"
#include "../common/ventus_opencl_test.h"

typedef int (*suite_fn_t)(void);

typedef struct {
  const char *name;
  const char *tags;
  const char *description;
  int default_run;
  suite_fn_t run;
} suite_t;

static int run_bulk_s2g(void)
{
  char *argv[] = {"shared_to_global_dma_test", NULL};
  return s2g_bulk_matrix_case_main(1, argv);
}

static int run_tensor_s2g(void)
{
  return s2g_tensor_matrix_case_main();
}

static uint8_t s2g_pattern_byte(unsigned idx)
{
  return (uint8_t)((idx * 7u + 0x23u) & 0xffu);
}

static int run_bulk_s2g_small_dual(void)
{
  const struct {
    const char *name;
    unsigned copy_a;
    unsigned copy_b;
    unsigned dst_offset;
  } cases[] = {
    {"dual_s2g_4B_8B", 4u, 8u, 0u},
    {"dual_s2g_16B_32B_offset", 16u, 32u, 16u},
  };

  cl_int err = CL_SUCCESS;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_mem dst_buf = NULL;
  int exit_code = 1;

  err = ventus_get_default_device(&context, &device, &queue, NULL);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device(s2g_small_dual)");
  err = ventus_build_program_from_source(context, device,
                                         "shared_to_global_dma_test.cl",
                                         &program);
  CHECK_OPENCL_ERROR_IN("ventus_build_program_from_source(s2g_small_dual)");
  kernel = clCreateKernel(program, "shared_to_global_dual_small_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(s2g_small_dual)");

  exit_code = 0;
  for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
    unsigned dst_bytes = cases[c].dst_offset + cases[c].copy_a +
                         cases[c].copy_b + 32u;
    uint8_t *init = (uint8_t *)malloc(dst_bytes);
    uint8_t *got = (uint8_t *)malloc(dst_bytes);
    if (!init || !got) {
      fprintf(stderr, "FAIL %s host alloc\n", cases[c].name);
      free(init);
      free(got);
      exit_code = 1;
      continue;
    }
    memset(init, 0xcd, dst_bytes);
    dst_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                             dst_bytes, init, &err);
    CHECK_OPENCL_ERROR_IN("clCreateBuffer(s2g_small_dual dst)");

    cl_uint copy_a = cases[c].copy_a;
    cl_uint copy_b = cases[c].copy_b;
    cl_uint dst_offset = cases[c].dst_offset;
    err  = clSetKernelArg(kernel, 0, sizeof(dst_buf), &dst_buf);
    err |= clSetKernelArg(kernel, 1, sizeof(copy_a), &copy_a);
    err |= clSetKernelArg(kernel, 2, sizeof(copy_b), &copy_b);
    err |= clSetKernelArg(kernel, 3, sizeof(dst_offset), &dst_offset);
    CHECK_OPENCL_ERROR_IN("clSetKernelArg(s2g_small_dual)");

    size_t global = 32, local = 32;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                                 0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel(s2g_small_dual)");
    err = clFinish(queue);
    CHECK_OPENCL_ERROR_IN("clFinish(s2g_small_dual)");
    err = clEnqueueReadBuffer(queue, dst_buf, CL_TRUE, 0, dst_bytes, got,
                              0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(s2g_small_dual)");

    int fail = 0;
    for (unsigned i = 0; i < dst_bytes; i++) {
      uint8_t exp = 0xcd;
      if (i >= cases[c].dst_offset &&
          i < cases[c].dst_offset + cases[c].copy_a) {
        exp = s2g_pattern_byte(i - cases[c].dst_offset);
      } else if (i >= cases[c].dst_offset + cases[c].copy_a &&
                 i < cases[c].dst_offset + cases[c].copy_a + cases[c].copy_b) {
        exp = s2g_pattern_byte(64u + i - cases[c].dst_offset - cases[c].copy_a);
      }
      if (got[i] != exp) {
        fprintf(stderr, "FAIL %s byte[%u]: got=%02x exp=%02x\n",
                cases[c].name, i, got[i], exp);
        fail = 1;
        break;
      }
    }
    if (fail) exit_code = 1;
    else printf("PASS %s bytes=%u\n", cases[c].name, dst_bytes);

    clReleaseMemObject(dst_buf);
    dst_buf = NULL;
    free(init);
    free(got);
  }

FINISH:
  if (dst_buf) clReleaseMemObject(dst_buf);
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
    {"bulk_s2g", "s2g|bulk|funct3|roundtrip|guard|short",
     "bulk shared-to-global directed matrix plus G2S/S2G roundtrip", 1,
     run_bulk_s2g},
    {"bulk_s2g_small_dual", "s2g|bulk|funct3|small|single-fence|short",
     "small 4/8/16/32B S2G copies with two requests drained by one fence", 1,
     run_bulk_s2g_small_dual},
    {"tma_s2g", "s2g|tma|tensor|descriptor|rank3|guard|short",
     "descriptor-form tensor shared-to-global origin/subbox/rank3 cases", 1,
     run_tensor_s2g},
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
    printf("dma_tma_s2g_func_test suites:\n");
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

    printf("\n[%zu/%zu] dma_tma_s2g_func_test.%s\n",
           i + 1, suite_count, suites[i].name);
    int rc = suites[i].run();
    if (rc == 0) {
      printf("[PASS] %s\n", suites[i].name);
      pass++;
    } else {
      printf("[FAIL] %s rc=%d\n", suites[i].name, rc);
      fail++;
    }
  }

  printf("\n=== dma_tma_s2g_func_test summary ===\n");
  printf("  pass: %zu\n  fail: %zu\n  skip: %zu\n", pass, fail, skip);
  if (fail == 0 && pass > 0) {
    printf("OK\n");
    return 0;
  }
  printf("FAILED\n");
  return 1;
}
