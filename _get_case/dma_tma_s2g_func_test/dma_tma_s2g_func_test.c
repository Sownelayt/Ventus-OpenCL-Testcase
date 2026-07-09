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

static int run_mixed_s2g_routing(void)
{
  char *argv[] = {"s2g_mixed_routing", NULL};
  return s2g_mixed_routing_case_main(1, argv);
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

static int check_stress_bytes(const char *name, const uint8_t *got,
                              size_t bytes, int mode)
{
  for (size_t i = 0; i < bytes; i++) {
    uint8_t exp = 0xcd;
    if (mode == 0) {
      exp = s2g_pattern_byte((unsigned)i);
    } else if (mode == 1) {
      exp = s2g_pattern_byte((unsigned)i);
    } else if (mode == 5) {
      if (i < 128u) {
        exp = s2g_pattern_byte(384u + (unsigned)i);
      } else if (i >= 4096u && i < 4224u) {
        exp = s2g_pattern_byte(128u + (unsigned)i - 4096u);
      } else if (i >= 8192u && i < 8320u) {
        exp = s2g_pattern_byte(256u + (unsigned)i - 8192u);
      }
    }
    if (got[i] != exp) {
      fprintf(stderr, "FAIL %s byte[%zu]: got=%02x exp=%02x\n",
              name, i, got[i], exp);
      return 1;
    }
  }
  return 0;
}

static int run_stress_kernel(cl_context context, cl_command_queue queue,
                             cl_kernel kernel, const char *name,
                             size_t dst_bytes, size_t local_size, int mode)
{
  cl_int err = CL_SUCCESS;
  cl_mem dst_buf = NULL;
  uint8_t *init = (uint8_t *)malloc(dst_bytes);
  uint8_t *got = (uint8_t *)malloc(dst_bytes);
  int rc = 1;

  if (!init || !got) {
    fprintf(stderr, "FAIL %s host alloc\n", name);
    goto FINISH;
  }
  memset(init, 0xcd, dst_bytes);
  dst_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           dst_bytes, init, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(s2g_stress dst)");
  err = clSetKernelArg(kernel, 0, sizeof(dst_buf), &dst_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(s2g_stress)");

  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &local_size,
                               &local_size, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel(s2g_stress)");
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(s2g_stress)");
  err = clEnqueueReadBuffer(queue, dst_buf, CL_TRUE, 0, dst_bytes, got,
                            0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(s2g_stress)");

  rc = check_stress_bytes(name, got, dst_bytes, mode);
  if (rc == 0) printf("PASS %s bytes=%zu\n", name, dst_bytes);

FINISH:
  if (dst_buf) clReleaseMemObject(dst_buf);
  free(init);
  free(got);
  return rc;
}

static int stress_case_selected(const char *filter, const char *name)
{
  return !filter || !filter[0] || strstr(name, filter);
}

static void run_selected_stress_kernel(cl_context context,
                                       cl_command_queue queue,
                                       cl_kernel kernel, const char *name,
                                       size_t dst_bytes, size_t local_size,
                                       int mode, const char *filter,
                                       size_t *selected_count, int *exit_code)
{
  if (!stress_case_selected(filter, name)) return;
  (*selected_count)++;
  if (run_stress_kernel(context, queue, kernel, name, dst_bytes, local_size,
                        mode) != 0) {
    *exit_code = 1;
  }
}

static int run_bulk_s2g_stress(void)
{
  cl_int err = CL_SUCCESS;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel four_issue_kernel = NULL;
  cl_kernel six_issue_kernel = NULL;
  cl_kernel eight_issue_kernel = NULL;
  cl_kernel empty_commit_kernel = NULL;
  cl_kernel wait_oldest_kernel = NULL;
  cl_kernel wait_group_kernel = NULL;
  cl_kernel group_wrap_kernel = NULL;
  cl_kernel group_multi_issue_wrap_kernel = NULL;
  cl_kernel cross_page_tlb_kernel = NULL;
  cl_kernel multi_warp_kernel = NULL;
  const char *case_filter = getenv("S2G_STRESS_CASE_FILTER");
  size_t selected_count = 0;
  int exit_code = 1;

  err = ventus_get_default_device(&context, &device, &queue, NULL);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device(s2g_stress)");
  err = ventus_build_program_from_source(context, device,
                                         "shared_to_global_dma_test.cl",
                                         &program);
  CHECK_OPENCL_ERROR_IN("ventus_build_program_from_source(s2g_stress)");

  four_issue_kernel = clCreateKernel(program,
      "shared_to_global_four_issue_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(s2g_four_issue)");
  six_issue_kernel = clCreateKernel(program,
      "shared_to_global_six_issue_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(s2g_six_issue)");
  eight_issue_kernel = clCreateKernel(program,
      "shared_to_global_eight_issue_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(s2g_eight_issue)");
  empty_commit_kernel = clCreateKernel(program,
      "shared_to_global_empty_commit_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(s2g_empty_commit)");
  wait_oldest_kernel = clCreateKernel(program,
      "shared_to_global_wait_oldest_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(s2g_wait_oldest)");
  wait_group_kernel = clCreateKernel(program,
      "shared_to_global_wait_group_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(s2g_wait_group)");
  group_wrap_kernel = clCreateKernel(program,
      "shared_to_global_group_wrap_wait0123_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(s2g_group_wrap)");
  group_multi_issue_wrap_kernel = clCreateKernel(program,
      "shared_to_global_group_multi_issue_wrap_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(s2g_group_multi_issue_wrap)");
  cross_page_tlb_kernel = clCreateKernel(program,
      "shared_to_global_cross_page_tlb_abc_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(s2g_cross_page_tlb)");
  multi_warp_kernel = clCreateKernel(program,
      "shared_to_global_multi_warp_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(s2g_multi_warp)");

  exit_code = 0;
  run_selected_stress_kernel(context, queue, four_issue_kernel,
      "four_issue_one_fence", 512u, 32u, 0, case_filter,
      &selected_count, &exit_code);
  run_selected_stress_kernel(context, queue, six_issue_kernel,
      "line_entry_pressure_6_lines", 768u, 32u, 1, case_filter,
      &selected_count, &exit_code);
  run_selected_stress_kernel(context, queue, eight_issue_kernel,
      "line_entry_pressure_8_lines", 1024u, 32u, 1, case_filter,
      &selected_count, &exit_code);
  run_selected_stress_kernel(context, queue, empty_commit_kernel,
      "group_empty_commit", 128u, 32u, 4, case_filter,
      &selected_count, &exit_code);
  run_selected_stress_kernel(context, queue, wait_oldest_kernel,
      "wait_oldest_one_reuse", 384u, 32u, 1, case_filter,
      &selected_count, &exit_code);
  run_selected_stress_kernel(context, queue, wait_group_kernel,
      "wait_group_reuse", 512u, 32u, 1, case_filter,
      &selected_count, &exit_code);
  run_selected_stress_kernel(context, queue, group_wrap_kernel,
      "group_wrap_wait0123", 896u, 32u, 1, case_filter,
      &selected_count, &exit_code);
  run_selected_stress_kernel(context, queue, group_multi_issue_wrap_kernel,
      "group_multi_issue_wrap", 896u, 32u, 1, case_filter,
      &selected_count, &exit_code);
  run_selected_stress_kernel(context, queue, cross_page_tlb_kernel,
      "cross_page_tlb_ABC", 8320u, 32u, 5, case_filter,
      &selected_count, &exit_code);
  run_selected_stress_kernel(context, queue, multi_warp_kernel,
      "multi_warp_s2g_fence", 512u, 128u, 1, case_filter,
      &selected_count, &exit_code);

  if (selected_count == 0) {
    fprintf(stderr, "FAIL no bulk S2G stress case matched filter '%s'\n",
            case_filter);
    exit_code = 1;
  }

FINISH:
  if (multi_warp_kernel) clReleaseKernel(multi_warp_kernel);
  if (cross_page_tlb_kernel) clReleaseKernel(cross_page_tlb_kernel);
  if (group_multi_issue_wrap_kernel) clReleaseKernel(group_multi_issue_wrap_kernel);
  if (group_wrap_kernel) clReleaseKernel(group_wrap_kernel);
  if (wait_group_kernel) clReleaseKernel(wait_group_kernel);
  if (wait_oldest_kernel) clReleaseKernel(wait_oldest_kernel);
  if (empty_commit_kernel) clReleaseKernel(empty_commit_kernel);
  if (eight_issue_kernel) clReleaseKernel(eight_issue_kernel);
  if (six_issue_kernel) clReleaseKernel(six_issue_kernel);
  if (four_issue_kernel) clReleaseKernel(four_issue_kernel);
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
    {"bulk_s2g_stress",
     "s2g|bulk|funct3|multi-issue|multi-warp|fence|wait-oldest|wait-group|stress",
     "line pressure, wait-oldest, wait-group, cross-page, and multi-warp fence",
     1, run_bulk_s2g_stress},
    {"tma_s2g", "s2g|tma|tensor|descriptor|rank3|guard|short",
     "descriptor-form tensor shared-to-global origin/subbox/rank3 cases", 1,
     run_tensor_s2g},
    {"mixed_s2g_routing",
     "s2g|mixed|routing|completion|g2s-wait|s2g-wait|bulk|tensor|descriptor|fence|stress",
     "mixed bulk/tensor S2G routing plus independent G2S/S2G completion domains",
     1, run_mixed_s2g_routing},
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
