/*
 * Single-workgroup DMA/TMA G2S functional test.
 *
 * This is the suite entry for G2S functional coverage. Individual host case
 * implementations live in named case modules in this project, all device
 * kernels are built from dma_tma_g2s_func_test.cl, and the suite keeps running
 * after individual case failures so the final summary reports the full failing
 * surface.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dma_tma_g2s_cases.h"
#include "../common/ventus_opencl_test.h"

int tma_matrix_test_suite_main(int argc, char **argv);

typedef int (*suite_fn_t)(void);

typedef struct {
  const char *name;
  const char *tags;
  const char *description;
  int default_run;
  suite_fn_t run;
} suite_t;

static const char *g_program_path = "./dma_tma_g2s_func_test.out";

static int run_dma_basic(void)
{
  char *argv[] = {"dma_test", NULL};
  return g2s_dma_basic_case_main(1, argv);
}

#define G2S_EXTRA_SRC_BYTES 256u
#define G2S_EXTRA_GUARD_BYTES 32u

static void fill_g2s_bytes(uint8_t *dst, size_t n, unsigned seed)
{
  for (size_t i = 0; i < n; i++) dst[i] = (uint8_t)((i * 9u + seed) & 0xffu);
}

static int check_g2s_bytes(const char *label, const uint8_t *got,
                           const uint8_t *expected, size_t n)
{
  for (size_t i = 0; i < n; i++) {
    if (got[i] != expected[i]) {
      fprintf(stderr, "FAIL %s byte[%zu]: got=%02x exp=%02x\n",
              label, i, got[i], expected[i]);
      return 1;
    }
  }
  printf("PASS %s bytes=%zu\n", label, n);
  return 0;
}

static int run_copysize_matrix(void)
{
  const struct {
    const char *name;
    const char *kernel_name;
    unsigned bytes;
    unsigned src_offset;
  } cases[] = {
    {"copysize_4B", "dma_copysize_4b", 4u, 12u},
    {"copysize_8B", "dma_copysize_8b", 8u, 20u},
    {"copysize_16B", "dma_copysize_16b", 16u, 36u},
    {"copysize_32B", "dma_copysize_32b", 32u, 64u},
  };

  cl_int err = CL_SUCCESS;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_mem src_buf = NULL, dst_buf = NULL;
  uint8_t src[G2S_EXTRA_SRC_BYTES];
  uint8_t got[32];
  int exit_code = 1;

  fill_g2s_bytes(src, sizeof(src), 0x51u);
  err = ventus_get_default_device(&context, &device, &queue, NULL);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device(copysize_matrix)");
  err = ventus_build_program_from_source(context, device, "dma_tma_g2s_func_test.cl",
                                         &program);
  CHECK_OPENCL_ERROR_IN("ventus_build_program_from_source(copysize_matrix)");

  src_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(src), src, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(copysize src)");
  dst_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(got), NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(copysize dst)");

  exit_code = 0;
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
    const unsigned fill = 0xcdcdcdcdu;
    memset(got, 0xcd, sizeof(got));
    err = clEnqueueFillBuffer(queue, dst_buf, &fill, sizeof(fill),
                              0, sizeof(got), 0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueFillBuffer(copysize dst)");

    kernel = clCreateKernel(program, cases[i].kernel_name, &err);
    CHECK_OPENCL_ERROR_IN("clCreateKernel(copysize)");
    cl_uint src_offset = cases[i].src_offset;
    err  = clSetKernelArg(kernel, 0, sizeof(src_buf), &src_buf);
    err |= clSetKernelArg(kernel, 1, sizeof(dst_buf), &dst_buf);
    err |= clSetKernelArg(kernel, 2, sizeof(src_offset), &src_offset);
    CHECK_OPENCL_ERROR_IN("clSetKernelArg(copysize)");

    size_t global = 32, local = 32;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                                 0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel(copysize)");
    err = clFinish(queue);
    CHECK_OPENCL_ERROR_IN("clFinish(copysize)");
    err = clEnqueueReadBuffer(queue, dst_buf, CL_TRUE, 0, sizeof(got), got,
                              0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(copysize)");

    if (check_g2s_bytes(cases[i].name, got, src + cases[i].src_offset,
                        cases[i].bytes) != 0) {
      exit_code = 1;
    }
    for (unsigned j = cases[i].bytes; j < sizeof(got); j++) {
      if (got[j] != 0xcd) {
        fprintf(stderr, "FAIL %s guard byte[%u]: got=%02x exp=cd\n",
                cases[i].name, j, got[j]);
        exit_code = 1;
        break;
      }
    }
    clReleaseKernel(kernel);
    kernel = NULL;
  }

FINISH:
  if (kernel) clReleaseKernel(kernel);
  if (dst_buf) clReleaseMemObject(dst_buf);
  if (src_buf) clReleaseMemObject(src_buf);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
  return exit_code;
}

static int run_copysize(void)
{
  char *argv[] = {"copysize_test", NULL};
  int smoke_rc = g2s_copysize_smoke_case_main(1, argv);
  int matrix_rc = run_copysize_matrix();
  return smoke_rc || matrix_rc;
}

static int run_bulk_dual_g2s(void)
{
  cl_int err = CL_SUCCESS;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_mem src_a_buf = NULL, src_b_buf = NULL, dst_buf = NULL;
  uint8_t src_a[G2S_EXTRA_SRC_BYTES];
  uint8_t src_b[G2S_EXTRA_SRC_BYTES];
  uint8_t got[128];
  uint8_t expected[128];
  int exit_code = 1;

  const cl_uint copy_bytes = 64u;
  const cl_uint src_a_offset = 32u;
  const cl_uint src_b_offset = 120u;

  fill_g2s_bytes(src_a, sizeof(src_a), 0x23u);
  fill_g2s_bytes(src_b, sizeof(src_b), 0x95u);
  memcpy(expected, src_a + src_a_offset, copy_bytes);
  memcpy(expected + copy_bytes, src_b + src_b_offset, copy_bytes);
  memset(got, 0xcd, sizeof(got));

  err = ventus_get_default_device(&context, &device, &queue, NULL);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device(bulk_dual)");
  err = ventus_build_program_from_source(context, device,
                                         "dma_tma_g2s_func_test.cl", &program);
  CHECK_OPENCL_ERROR_IN("ventus_build_program_from_source(bulk_dual)");
  kernel = clCreateKernel(program, "bulk_dual_g2s_single_fence_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(bulk_dual)");

  src_a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             sizeof(src_a), src_a, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(src_a bulk_dual)");
  src_b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             sizeof(src_b), src_b, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(src_b bulk_dual)");
  dst_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(got), got, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(dst bulk_dual)");

  err  = clSetKernelArg(kernel, 0, sizeof(src_a_buf), &src_a_buf);
  err |= clSetKernelArg(kernel, 1, sizeof(src_b_buf), &src_b_buf);
  err |= clSetKernelArg(kernel, 2, sizeof(dst_buf), &dst_buf);
  err |= clSetKernelArg(kernel, 3, sizeof(copy_bytes), &copy_bytes);
  err |= clSetKernelArg(kernel, 4, sizeof(src_a_offset), &src_a_offset);
  err |= clSetKernelArg(kernel, 5, sizeof(src_b_offset), &src_b_offset);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(bulk_dual)");

  size_t global = 32, local = 32;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                               0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel(bulk_dual)");
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(bulk_dual)");
  err = clEnqueueReadBuffer(queue, dst_buf, CL_TRUE, 0, sizeof(got), got,
                            0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(bulk_dual)");

  exit_code = check_g2s_bytes("bulk_dual_g2s_single_fence", got, expected,
                              sizeof(expected));

FINISH:
  if (dst_buf) clReleaseMemObject(dst_buf);
  if (src_b_buf) clReleaseMemObject(src_b_buf);
  if (src_a_buf) clReleaseMemObject(src_a_buf);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
  return exit_code;
}

static int run_bulk_matrix(void)
{
  char *argv[] = {"bulk_dma_matrix_test", NULL};
  return g2s_bulk_matrix_case_main(1, argv);
}

static int run_tensor_smoke(void)
{
  char *argv[] = {"tensor_dma_test", NULL};
  return g2s_tensor_smoke_case_main(1, argv);
}

static int run_tma_descriptor(void)
{
  return g2s_tma_descriptor_case_main();
}

static int run_tma_matrix_functional(void)
{
  char *argv[] = {(char *)g_program_path, "--functional-suite", NULL};
  return tma_matrix_test_suite_main(2, argv);
}

static int run_mixed_async_fence(void)
{
  return g2s_mixed_async_fence_case_main();
}

static int run_routing_conflict(void)
{
  char *argv[] = {"dma_shared_routing_conflict_test", NULL};
  return g2s_routing_conflict_case_main(1, argv);
}

static int run_multi_warp_fence(void)
{
  char *argv[] = {"multi_warp_dma_fence_test", NULL};
  return g2s_multi_warp_fence_case_main(1, argv);
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
  if (argc > 0 && argv[0]) g_program_path = argv[0];

  if (argc == 3 && strcmp(argv[1], "--tma-matrix-case") == 0) {
    char *matrix_argv[] = {"tma_matrix_test", argv[2], NULL};
    return tma_matrix_test_suite_main(2, matrix_argv);
  }

  suite_t suites[] = {
    {"bulk_basic", "g2s|bulk|funct1|funct6|short",
     "CP_ASYNC_BULK G2S smoke", 1, run_dma_basic},
    {"bulk_copysize", "g2s|bulk|funct0|funct6|short",
     "CP_ASYNC_COPYSIZE funct0 smoke", 1, run_copysize},
    {"bulk_matrix", "g2s|bulk|cacheline|offset|short",
     "bulk G2S cacheline and shared-offset directed matrix", 1, run_bulk_matrix},
    {"bulk_dual_g2s", "g2s|bulk|dual|single-fence|short",
     "two CP_ASYNC_BULK G2S requests drained by one fence", 1,
     run_bulk_dual_g2s},
    {"tma_smoke", "g2s|tma|tensor|descriptor|short",
     "descriptor-form TMA G2S 4x4 smoke", 1, run_tensor_smoke},
    {"tma_descriptor", "g2s|tma|descriptor|prefetch|single-fence|short",
     "descriptor prefetch and dual tensor single-fence coverage", 1,
     run_tma_descriptor},
    {"tma_matrix_functional",
     "g2s|tma|tensor|matrix|datatype|subbox|stride|swizzle|oob",
     "curated TMA matrix functional coverage", 1,
     run_tma_matrix_functional},
    {"mixed_async_fence",
     "g2s|s2g|bulk|tma|tensor|prefetch|bidirectional|fence|stress",
     "single-workgroup mixed bulk/TMA async fence coverage", 1,
     run_mixed_async_fence},
    {"routing_conflict",
     "g2s|bulk|routing|shared|bank-conflict|stress",
     "DMA shared response routing under bank-conflict pressure", 1,
     run_routing_conflict},
    {"multi_warp_fence",
     "g2s|bulk|multi-warp|fence|stress",
     "same-workgroup multi-warp DMA bursts and fence drain", 1,
     run_multi_warp_fence},
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
    printf("dma_tma_g2s_func_test suites:\n");
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

    printf("\n[%zu/%zu] dma_tma_g2s_func_test.%s\n",
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

  printf("\n=== dma_tma_g2s_func_test summary ===\n");
  printf("  pass: %zu\n  fail: %zu\n  skip: %zu\n", pass, fail, skip);
  if (fail == 0 && pass > 0) {
    printf("OK\n");
    return 0;
  }
  printf("FAILED\n");
  return 1;
}
