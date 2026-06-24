/*
 * Temporary marker overhead test.
 *
 * Background:
 *   Quantify how much cycle overhead comes from the marker instrumentation used
 *   by dma_tma_serial_pipeline_perf_test.
 *
 * Flow:
 *   The parent launches one child per marker variant. Each child builds the same
 *   kernel source with a different MARK_MODE, runs one manual load/compute/store
 *   tile, and writes full GVM stdout into log/. The parent parses GVM PMU active
 *   cycles from the child log and prints a compact table.
 *
 * Usage:
 *   ./temp_marker_overhead_test.out
 *   ./temp_marker_overhead_test.out single <rows> <cols>
 *   ./temp_marker_overhead_test.out child plain|csr_only|barrier_only|csr_barrier <rows> <cols>
 *
 * Maintenance:
 *   This is a temporary diagnostic case. Keep generated logs and runtime build
 *   products inside this project's log/ directory.
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "../common/ventus_opencl_test.h"

#define WG_SIZE 32u
#define DEFAULT_TILE_ROWS 16u
#define DEFAULT_TILE_COLS 16u
#define MAX_TILE_ROWS 64u
#define MAX_TILE_COLS 64u
#define LOG_DIR "log"
#define DESC_WORDS 32u

typedef enum {
  VARIANT_PLAIN = 0,
  VARIANT_CSR_ONLY,
  VARIANT_BARRIER_ONLY,
  VARIANT_CSR_BARRIER,
  VARIANT_COUNT
} Variant;

typedef struct {
  uint32_t rows;
  uint32_t cols;
} TestCase;

typedef struct {
  Variant variant;
  TestCase c;
  uint32_t elements;
  uint64_t ns;
  uint64_t cycles;
  int cycle_valid;
  int passed;
  int status;
  char log_path[256];
  char run_dir[256];
} RunResult;

static const char *variant_name(Variant v)
{
  switch (v) {
  case VARIANT_PLAIN: return "plain";
  case VARIANT_CSR_ONLY: return "csr_only";
  case VARIANT_BARRIER_ONLY: return "barrier_only";
  case VARIANT_CSR_BARRIER: return "csr_barrier";
  default: return "unknown";
  }
}

static int parse_variant(const char *text, Variant *out)
{
  for (int i = 0; i < (int)VARIANT_COUNT; i++) {
    if (strcmp(text, variant_name((Variant)i)) == 0) {
      *out = (Variant)i;
      return 0;
    }
  }
  fprintf(stderr, "unknown variant '%s'\n", text);
  return 1;
}

static int parse_u32(const char *text, uint32_t min_value, uint32_t max_value,
                     const char *name, uint32_t *out)
{
  char *end = NULL;
  long parsed = strtol(text, &end, 0);
  if (!text[0] || *end || parsed < (long)min_value ||
      parsed > (long)max_value) {
    fprintf(stderr, "%s must be in [%u, %u], got '%s'\n",
            name, min_value, max_value, text);
    return 1;
  }
  *out = (uint32_t)parsed;
  return 0;
}

static uint32_t case_elements(TestCase c)
{
  return c.rows * c.cols;
}

static uint32_t desc_control(unsigned data_type, unsigned rank)
{
  return (data_type & 0xfu) | ((rank & 0xfu) << 4);
}

static void build_desc(uint32_t *desc, TestCase c)
{
  memset(desc, 0, DESC_WORDS * sizeof(uint32_t));
  desc[0] = 0x56544d41u;
  desc[1] = desc_control(6, 2);
  desc[2] = 0;
  desc[3] = 128;
  desc[4] = c.cols;
  desc[5] = c.rows;
  desc[6] = desc[7] = desc[8] = 1;
  desc[9] = sizeof(float);
  desc[10] = c.cols * sizeof(float);
  desc[11] = desc[12] = desc[13] = 0;
  desc[14] = c.cols;
  desc[15] = c.rows;
  desc[16] = desc[17] = desc[18] = 1;
  for (uint32_t i = 0; i < 5; i++) desc[19 + i] = 1;
}

static float abs_f32(float x)
{
  return x < 0.0f ? -x : x;
}

static float compute_ref_value(float x)
{
  return x * 1.0009765625f + 0.000244140625f;
}

static void fill_input(float *input, uint32_t elements)
{
  for (uint32_t i = 0; i < elements; i++) {
    int v = (int)((i * 7u + 11u) % 31u) - 15;
    input[i] = (float)v * 0.03125f;
  }
}

static int check_result(const float *got, const float *input, uint32_t elements)
{
  uint32_t bad = 0;
  float max_abs = 0.0f;
  for (uint32_t i = 0; i < elements; i++) {
    float exp = compute_ref_value(input[i]);
    float diff = abs_f32(got[i] - exp);
    if (diff > max_abs) max_abs = diff;
    if (diff > 1.0e-4f) {
      if (bad < 8) {
        fprintf(stderr, "FAIL at %u: got=%+.8f exp=%+.8f diff=%g\n",
                i, got[i], exp, diff);
      }
      bad++;
    }
  }
  if (bad) {
    fprintf(stderr, "FAIL bad=%u/%u max_abs=%g\n", bad, elements, max_abs);
    return 1;
  }
  printf("PASS elements=%u max_abs=%g\n", elements, max_abs);
  return 0;
}

static int event_duration_ns(cl_event event, uint64_t *duration_ns)
{
  cl_ulong start = 0, end = 0;
  cl_int err;
  err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                sizeof(start), &start, NULL);
  if (err != CL_SUCCESS) return 1;
  err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                sizeof(end), &end, NULL);
  if (err != CL_SUCCESS || end < start) return 1;
  *duration_ns = (uint64_t)(end - start);
  return 0;
}

static cl_int build_program_with_options(cl_context context,
                                         cl_device_id device,
                                         TestCase c,
                                         Variant variant,
                                         cl_program *program)
{
  const char *source_path = getenv("VENTUS_MARKER_OVERHEAD_SOURCE");
  if (!source_path || !source_path[0]) source_path = "temp_marker_overhead_test.cl";
  size_t source_size = 0;
  char *source = ventus_read_text_file(source_path, &source_size);
  if (!source) return CL_INVALID_PROGRAM;

  cl_int err;
  const char *sources[] = {source};
  const size_t sizes[] = {source_size};
  cl_program prog = clCreateProgramWithSource(context, 1, sources, sizes, &err);
  free(source);
  if (err != CL_SUCCESS) return err;

  char options[160];
  snprintf(options, sizeof(options),
           "-DTILE_ROWS=%uu -DTILE_COLS=%uu -DMARK_MODE=%d",
           c.rows, c.cols, (int)variant);
  err = clBuildProgram(prog, 1, &device, options, NULL, NULL);
  if (err != CL_SUCCESS) {
    ventus_print_build_log(prog, device);
    clReleaseProgram(prog);
    return err;
  }
  *program = prog;
  return CL_SUCCESS;
}

static int run_child(Variant variant, TestCase c)
{
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_mem input_buf = NULL;
  cl_mem output_buf = NULL;
  cl_event event = NULL;
  float *input = NULL, *got = NULL;
  uint32_t elements = case_elements(c);
  size_t bytes = (size_t)elements * sizeof(float);
  uint64_t ns = 0;
  cl_int err = CL_SUCCESS;
  int rc = 1;

  input = (float *)malloc(bytes);
  got = (float *)malloc(bytes);
  if (!input || !got) goto FINISH;
  fill_input(input, elements);
  memset(got, 0, bytes);

  err = ventus_get_default_device(&context, &device, &queue, NULL);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device");
  clReleaseCommandQueue(queue);
  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
  CHECK_OPENCL_ERROR_IN("clCreateCommandQueue(profiled)");
  err = build_program_with_options(context, device, c, variant, &program);
  CHECK_OPENCL_ERROR_IN("build_program_with_options");
  kernel = clCreateKernel(program, "marker_overhead_manual_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel");

  input_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             bytes, input, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(input)");
  output_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              bytes, got, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(output)");
  err  = clSetKernelArg(kernel, 0, sizeof(input_buf), &input_buf);
  err |= clSetKernelArg(kernel, 1, sizeof(output_buf), &output_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg");

  size_t global = WG_SIZE;
  size_t local = WG_SIZE;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                               0, NULL, &event);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel");
  err = clWaitForEvents(1, &event);
  CHECK_OPENCL_ERROR_IN("clWaitForEvents");
  if (event_duration_ns(event, &ns) != 0) goto FINISH;
  err = clEnqueueReadBuffer(queue, output_buf, CL_TRUE, 0, bytes, got,
                            0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer");
  if (check_result(got, input, elements) != 0) goto FINISH;

  printf("PATH_RESULT variant=%s rows=%u cols=%u elements=%u ns=%" PRIu64 "\n",
         variant_name(variant), c.rows, c.cols, elements, ns);
  printf("OK child variant=%s rows=%u cols=%u\n",
         variant_name(variant), c.rows, c.cols);
  rc = 0;

FINISH:
  if (event) clReleaseEvent(event);
  if (output_buf) clReleaseMemObject(output_buf);
  if (input_buf) clReleaseMemObject(input_buf);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
  free(got);
  free(input);
  return rc;
}


static int run_child_tma(Variant variant, TestCase c)
{
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel setup_kernel = NULL;
  cl_kernel kernel = NULL;
  cl_mem desc_buf = NULL;
  cl_mem input_buf = NULL;
  cl_mem output_buf = NULL;
  cl_event event = NULL;
  float *input = NULL, *got = NULL;
  uint32_t desc[DESC_WORDS];
  uint32_t elements = case_elements(c);
  size_t bytes = (size_t)elements * sizeof(float);
  uint64_t ns = 0;
  cl_int err = CL_SUCCESS;
  int rc = 1;

  input = (float *)malloc(bytes);
  got = (float *)malloc(bytes);
  if (!input || !got) goto FINISH;
  fill_input(input, elements);
  memset(got, 0, bytes);
  build_desc(desc, c);

  err = ventus_get_default_device(&context, &device, &queue, NULL);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device");
  clReleaseCommandQueue(queue);
  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
  CHECK_OPENCL_ERROR_IN("clCreateCommandQueue(profiled)");
  err = build_program_with_options(context, device, c, variant, &program);
  CHECK_OPENCL_ERROR_IN("build_program_with_options");
  setup_kernel = clCreateKernel(program, "setup_desc_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(setup)");
  kernel = clCreateKernel(program, "marker_overhead_tma_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(tma)");

  desc_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            sizeof(desc), desc, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(desc)");
  input_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             bytes, input, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(input)");
  output_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              bytes, got, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(output)");

  size_t setup_global = 1;
  size_t setup_local = 1;
  err  = clSetKernelArg(setup_kernel, 0, sizeof(desc_buf), &desc_buf);
  err |= clSetKernelArg(setup_kernel, 1, sizeof(input_buf), &input_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(setup)");
  err = clEnqueueNDRangeKernel(queue, setup_kernel, 1, NULL,
                               &setup_global, &setup_local, 0, NULL, &event);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel(setup)");
  err = clWaitForEvents(1, &event);
  CHECK_OPENCL_ERROR_IN("clWaitForEvents(setup)");
  clReleaseEvent(event);
  event = NULL;
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(setup)");

  err  = clSetKernelArg(kernel, 0, sizeof(desc_buf), &desc_buf);
  err |= clSetKernelArg(kernel, 1, sizeof(input_buf), &input_buf);
  err |= clSetKernelArg(kernel, 2, sizeof(output_buf), &output_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(tma)");
  size_t global = WG_SIZE;
  size_t local = WG_SIZE;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                               0, NULL, &event);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel(tma)");
  err = clWaitForEvents(1, &event);
  CHECK_OPENCL_ERROR_IN("clWaitForEvents(tma)");
  if (event_duration_ns(event, &ns) != 0) goto FINISH;
  err = clEnqueueReadBuffer(queue, output_buf, CL_TRUE, 0, bytes, got,
                            0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer");
  if (check_result(got, input, elements) != 0) goto FINISH;

  printf("PATH_RESULT variant=%s path=tma rows=%u cols=%u elements=%u ns=%" PRIu64 "\n",
         variant_name(variant), c.rows, c.cols, elements, ns);
  printf("OK child variant=%s path=tma rows=%u cols=%u\n",
         variant_name(variant), c.rows, c.cols);
  rc = 0;

FINISH:
  if (event) clReleaseEvent(event);
  if (output_buf) clReleaseMemObject(output_buf);
  if (input_buf) clReleaseMemObject(input_buf);
  if (desc_buf) clReleaseMemObject(desc_buf);
  if (kernel) clReleaseKernel(kernel);
  if (setup_kernel) clReleaseKernel(setup_kernel);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
  free(got);
  free(input);
  return rc;
}

static int ensure_dir(const char *path)
{
  struct stat st;
  if (mkdir(path, 0775) == 0) return 0;
  if (errno == EEXIST && stat(path, &st) == 0 && S_ISDIR(st.st_mode)) return 0;
  return 1;
}

static void make_timestamp(char *stamp, size_t stamp_size)
{
  time_t now = time(NULL);
  struct tm tm_now;
  if (localtime_r(&now, &tm_now) &&
      strftime(stamp, stamp_size, "%Y%m%d_%H%M%S", &tm_now) > 0) return;
  snprintf(stamp, stamp_size, "%ld", (long)now);
}

static int make_absolute_path(char *out, size_t out_size, const char *cwd,
                              const char *path)
{
  if (!path || !path[0]) return 1;
  if (path[0] == '/') snprintf(out, out_size, "%s", path);
  else snprintf(out, out_size, "%s/%s", cwd, path);
  return 0;
}

static int parse_cycles_after_colon(const char *line, uint64_t *cycles)
{
  const char *colon = strchr(line, ':');
  char *end = NULL;
  unsigned long long parsed = 0;
  if (!colon) return 1;
  parsed = strtoull(colon + 1, &end, 10);
  if (end == colon + 1) return 1;
  *cycles = (uint64_t)parsed;
  return 0;
}

static int parse_path_result_line(const char *line, RunResult *result)
{
  char variant_text[32];
  unsigned rows = 0, cols = 0, elements = 0;
  unsigned long long ns = 0;
  Variant variant;
  int fields = sscanf(line,
                      "PATH_RESULT variant=%31s rows=%u cols=%u elements=%u ns=%llu",
                      variant_text, &rows, &cols, &elements, &ns);
  if (fields != 5) return 1;
  if (parse_variant(variant_text, &variant) != 0) return 1;
  memset(result, 0, sizeof(*result));
  result->variant = variant;
  result->c.rows = rows;
  result->c.cols = cols;
  result->elements = elements;
  result->ns = (uint64_t)ns;
  result->passed = 1;
  return 0;
}

static int parse_child_log(RunResult *result)
{
  FILE *f = fopen(result->log_path, "r");
  char line[1024];
  RunResult parsed;
  uint64_t cycles = 0;
  memset(&parsed, 0, sizeof(parsed));
  if (!f) return 1;
  while (fgets(line, sizeof(line), f)) {
    if (strstr(line, "PATH_RESULT variant=")) (void)parse_path_result_line(line, &parsed);
    if (strstr(line, "[PROGRAM          0] [INST+CYCLE] active cycles")) {
      (void)parse_cycles_after_colon(line, &cycles);
    } else if (strstr(line, "[TESTCASE TOTAL] [INST+CYCLE] active cycles")) {
      if (!cycles) (void)parse_cycles_after_colon(line, &cycles);
    }
  }
  fclose(f);
  if (!parsed.passed) return 1;
  parsed.cycles = cycles;
  parsed.cycle_valid = cycles != 0;
  snprintf(parsed.log_path, sizeof(parsed.log_path), "%s", result->log_path);
  snprintf(parsed.run_dir, sizeof(parsed.run_dir), "%s", result->run_dir);
  *result = parsed;
  return 0;
}

static int launch_and_wait(const char *prog, Variant variant, TestCase c,
                           RunResult *result)
{
  char stamp[32], cwd[256], exe_abs[512], source_abs[512];
  char rows_arg[16], cols_arg[16];
  pid_t pid;
  int status = 0;
  FILE *header = NULL;

  memset(result, 0, sizeof(*result));
  result->variant = variant;
  result->c = c;
  result->elements = case_elements(c);
  make_timestamp(stamp, sizeof(stamp));
  snprintf(result->log_path, sizeof(result->log_path),
           LOG_DIR "/marker_overhead_%s_%s_r%u_c%u.log",
           stamp, variant_name(variant), c.rows, c.cols);
  snprintf(result->run_dir, sizeof(result->run_dir),
           LOG_DIR "/marker_overhead_run_%s_%s_r%u_c%u",
           stamp, variant_name(variant), c.rows, c.cols);
  if (!getcwd(cwd, sizeof(cwd))) return 1;
  if (make_absolute_path(exe_abs, sizeof(exe_abs), cwd, prog) != 0) return 1;
  snprintf(source_abs, sizeof(source_abs), "%s/temp_marker_overhead_test.cl", cwd);
  if (ensure_dir(LOG_DIR) != 0 || mkdir(result->run_dir, 0775) != 0) return 1;
  header = fopen(result->log_path, "w");
  if (!header) return 1;
  fprintf(header, "# command: %s child %s %u %u\n",
          prog, variant_name(variant), c.rows, c.cols);
  fclose(header);
  snprintf(rows_arg, sizeof(rows_arg), "%u", c.rows);
  snprintf(cols_arg, sizeof(cols_arg), "%u", c.cols);

  pid = fork();
  if (pid < 0) return 1;
  if (pid == 0) {
    FILE *out = fopen(result->log_path, "a");
    setenv("VENTUS_MARKER_OVERHEAD_SOURCE", source_abs, 1);
    if (out) {
      dup2(fileno(out), STDOUT_FILENO);
      dup2(fileno(out), STDERR_FILENO);
    }
    if (chdir(result->run_dir) != 0) _exit(127);
    execlp(exe_abs, exe_abs, "child", variant_name(variant),
           rows_arg, cols_arg, (char *)NULL);
    _exit(127);
  }
  if (waitpid(pid, &status, 0) < 0) return 1;
  result->status = status;
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) return 1;
  return parse_child_log(result);
}

static void print_result_header(void)
{
  printf("| variant | rows | cols | cycles | delta_vs_plain | ns | log | run_dir |\n");
  printf("|---|---:|---:|---:|---:|---:|---|---|\n");
}

static void print_result_row(const RunResult *r, uint64_t plain_cycles)
{
  int64_t delta = 0;
  if (r->cycle_valid && plain_cycles) delta = (int64_t)r->cycles - (int64_t)plain_cycles;
  printf("| `%s` | %u | %u | ", variant_name(r->variant), r->c.rows, r->c.cols);
  if (r->cycle_valid) printf("%" PRIu64, r->cycles);
  else printf("NA");
  printf(" | ");
  if (r->cycle_valid && plain_cycles) printf("%+" PRId64, delta);
  else printf("NA");
  printf(" | %" PRIu64 " | `%s` | `%s` |\n", r->ns, r->log_path, r->run_dir);
}

static int run_parent(const char *prog, TestCase c)
{
  RunResult results[VARIANT_COUNT];
  uint64_t plain_cycles = 0;
  int failed = 0;
  for (int i = 0; i < (int)VARIANT_COUNT; i++) {
    if (launch_and_wait(prog, (Variant)i, c, &results[i]) != 0) {
      fprintf(stderr, "failed variant=%s log=%s\n",
              variant_name((Variant)i), results[i].log_path);
      failed = 1;
    }
    if (i == (int)VARIANT_PLAIN && results[i].cycle_valid) {
      plain_cycles = results[i].cycles;
    }
  }
  print_result_header();
  for (int i = 0; i < (int)VARIANT_COUNT; i++) print_result_row(&results[i], plain_cycles);
  return failed ? 1 : 0;
}

static void usage(const char *prog)
{
  fprintf(stderr,
          "Usage:\n"
          "  %s\n"
          "  %s single <rows> <cols>\n"
          "  %s child plain|csr_only|barrier_only|csr_barrier <rows> <cols>\n"
          "  %s child_tma plain|csr_only|barrier_only|csr_barrier <rows> <cols>\n",
          prog, prog, prog, prog);
}

int main(int argc, char **argv)
{
  TestCase c = {DEFAULT_TILE_ROWS, DEFAULT_TILE_COLS};
  if (argc == 1) return run_parent(argv[0], c);
  if (strcmp(argv[1], "single") == 0 && argc == 4) {
    if (parse_u32(argv[2], 1, MAX_TILE_ROWS, "rows", &c.rows) ||
        parse_u32(argv[3], 1, MAX_TILE_COLS, "cols", &c.cols)) return 1;
    return run_parent(argv[0], c);
  }
  if (strcmp(argv[1], "child") == 0 && argc == 5) {
    Variant variant;
    if (parse_variant(argv[2], &variant) ||
        parse_u32(argv[3], 1, MAX_TILE_ROWS, "rows", &c.rows) ||
        parse_u32(argv[4], 1, MAX_TILE_COLS, "cols", &c.cols)) return 1;
    return run_child(variant, c);
  }
  if (strcmp(argv[1], "child_tma") == 0 && argc == 5) {
    Variant variant;
    if (parse_variant(argv[2], &variant) ||
        parse_u32(argv[3], 1, MAX_TILE_ROWS, "rows", &c.rows) ||
        parse_u32(argv[4], 1, MAX_TILE_COLS, "cols", &c.cols)) return 1;
    return run_child_tma(variant, c);
  }
  usage(argv[0]);
  return 1;
}
