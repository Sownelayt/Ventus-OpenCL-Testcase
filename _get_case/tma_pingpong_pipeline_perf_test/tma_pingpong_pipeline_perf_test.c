/*
 * TMA ping-pong pipeline performance test.
 *
 * Background:
 *   Measure a ping-pong double-buffer tile path that includes input movement, shared-memory
 *   compute, and output movement. The manual path performs
 *   global -> register -> shared, compute, and shared -> register -> global.
 *   The TMA path uses descriptor CP_ASYNC_TENSOR G2S for the input side,
 *   then uses the same ordinary shared/register/global store as the manual path.
 *
 * Implementation:
 *   The test fixes shared-memory buffers at two and scans the number of tiles
 *   processed by one work-group. Compute intensity is fixed to one FP32 FMA-like
 *   update per element. Parent modes spawn isolated child runs so each child has
 *   its own POCL/GVM temporary files and log. Existing GVM PMU summaries are
 *   parsed from those logs when available.
 *
 * Usage:
 *   ./tma_pingpong_pipeline_perf_test.out
 *   ./tma_pingpong_pipeline_perf_test.out sweep
 *   ./tma_pingpong_pipeline_perf_test.out single <buffers> <stages>
 *   ./tma_pingpong_pipeline_perf_test.out single <rows> <cols> <buffers> <stages>
 *   ./tma_pingpong_pipeline_perf_test.out child manual|tma <rows> <cols> <buffers> <stages>
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <signal.h>
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

#define DESC_WORDS 32u
#define COORD_WORDS 32u
#define DEFAULT_TILE_ROWS 16u
#define DEFAULT_TILE_COLS 16u
#define MAX_TILE_ROWS 64u
#define MAX_TILE_COLS 64u
#define MAX_BUFFERS 8u
#define MAX_STAGES 16u
#define WG_SIZE 32u
#define DEFAULT_BUFFERS 2u
#define DEFAULT_STAGES 4u
#define LOG_DIR "log"

typedef enum {
  PATH_MANUAL = 0,
  PATH_TMA = 1,
  PATH_COUNT = 2
} PathKind;

typedef struct {
  uint32_t rows;
  uint32_t cols;
} TileSize;

typedef struct {
  uint32_t rows;
  uint32_t cols;
  uint32_t buffers;
  uint32_t stages;
} PipelineCase;

typedef struct {
  PathKind kind;
  PipelineCase c;
  uint32_t elements;
  uint64_t ns;
  uint64_t cycles;
  int cycle_valid;
  double elements_per_ns;
  int seen;
} PathResult;

typedef struct {
  PipelineCase c;
  PathResult manual;
  PathResult tma;
} CaseResult;

typedef struct {
  PathKind kind;
  PipelineCase c;
  pid_t pid;
  int active;
  char log_path[256];
  char run_dir[256];
} ChildRun;

static const TileSize sweep_tiles[] = {
  {16u, 16u},
  {32u, 16u},
  {32u, 32u},
  {64u, 32u},
  {64u, 64u},
};
static const uint32_t sweep_buffers[] = {2u};
static const uint32_t sweep_stages[] = {1u, 2u, 4u, 8u, 16u};
static const char *sweep_tile_text = "16x16, 32x16, 32x32, 64x32, 64x64";

static const char *path_name(PathKind kind)
{
  return kind == PATH_TMA ? "tma_pingpong_g2s_manual_store" : "manual";
}

static const char *path_arg(PathKind kind)
{
  return kind == PATH_TMA ? "tma" : "manual";
}

static int parse_path_kind(const char *text, PathKind *kind)
{
  if (strcmp(text, "manual") == 0) {
    *kind = PATH_MANUAL;
    return 0;
  }
  if (strcmp(text, "tma") == 0 || strcmp(text, "tma_pingpong") == 0 ||
      strcmp(text, "tma_g2s") == 0 ||
      strcmp(text, "tma_pingpong_g2s_manual_store") == 0) {
    *kind = PATH_TMA;
    return 0;
  }
  fprintf(stderr, "unknown path '%s', expected manual or tma\n", text);
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

static int is_supported_value(uint32_t value, const uint32_t *values,
                              size_t count);
static int is_supported_tile(uint32_t rows, uint32_t cols);

static int validate_case(PipelineCase c)
{
  if (!is_supported_tile(c.rows, c.cols) ||
      !is_supported_value(c.buffers, sweep_buffers,
                          sizeof(sweep_buffers) / sizeof(sweep_buffers[0])) ||
      !is_supported_value(c.stages, sweep_stages,
                          sizeof(sweep_stages) / sizeof(sweep_stages[0]))) {
    fprintf(stderr,
            "invalid case: rows=%u cols=%u buffers=%u stages=%u; supported tiles are %s, supported buffers are 2 and supported stages are 1,2,4,8,16\n",
            c.rows, c.cols, c.buffers, c.stages, sweep_tile_text);
    return 1;
  }
  return 0;
}

static uint32_t desc_control(unsigned data_type, unsigned rank)
{
  return (data_type & 0xfu) | ((rank & 0xfu) << 4);
}

static uint32_t tile_words(PipelineCase c)
{
  return c.rows * c.cols;
}

static uint32_t case_elements(PipelineCase c)
{
  return tile_words(c) * c.stages;
}

static int is_supported_value(uint32_t value, const uint32_t *values,
                              size_t count)
{
  for (size_t i = 0; i < count; i++) {
    if (value == values[i]) return 1;
  }
  return 0;
}

static int is_supported_tile(uint32_t rows, uint32_t cols)
{
  for (size_t i = 0; i < sizeof(sweep_tiles) / sizeof(sweep_tiles[0]); i++) {
    if (rows == sweep_tiles[i].rows && cols == sweep_tiles[i].cols) return 1;
  }
  return 0;
}

static const char *manual_kernel_name(PipelineCase c)
{
  if (c.buffers != 2u) return NULL;
  switch (c.stages) {
  case 1: return "manual_pingpong_b2_s1_kernel";
  case 2: return "manual_pingpong_b2_s2_kernel";
  case 4: return "manual_pingpong_b2_s4_kernel";
  case 8: return "manual_pingpong_b2_s8_kernel";
  case 16: return "manual_pingpong_b2_s16_kernel";
  default: return NULL;
  }
}

static const char *tma_kernel_name(PipelineCase c)
{
  if (c.buffers != 2u) return NULL;
  switch (c.stages) {
  case 1: return "tma_pingpong_b2_s1_kernel";
  case 2: return "tma_pingpong_b2_s2_kernel";
  case 4: return "tma_pingpong_b2_s4_kernel";
  case 8: return "tma_pingpong_b2_s8_kernel";
  case 16: return "tma_pingpong_b2_s16_kernel";
  default: return NULL;
  }
}



static void build_desc(uint32_t *desc, PipelineCase c)
{
  memset(desc, 0, DESC_WORDS * sizeof(uint32_t));
  desc[0] = 0x56544d41u;
  desc[1] = desc_control(6, 2);  /* FP32, rank=2 */
  desc[2] = 0;                   /* setup kernel patches runtime base */
  desc[3] = 128;
  desc[4] = c.cols;
  desc[5] = c.rows * c.stages;
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

static void fill_input(float *input, uint32_t elements)
{
  for (uint32_t i = 0; i < elements; i++) {
    int v = (int)((i * 7u + 11u) % 31u) - 15;
    input[i] = (float)v * 0.03125f;
  }
}

static float compute_ref_value(float x)
{
  return x * 1.0009765625f + 0.000244140625f;
}

static void cpu_ref(const float *input, float *output, PipelineCase c)
{
  uint32_t elements = case_elements(c);
  for (uint32_t i = 0; i < elements; i++) {
    output[i] = compute_ref_value(input[i]);
  }
}

static int check_result(const char *label, const float *got, const float *ref,
                        uint32_t elements)
{
  float max_abs = 0.0f;
  uint32_t bad = 0;
  for (uint32_t i = 0; i < elements; i++) {
    float diff = abs_f32(got[i] - ref[i]);
    if (diff > max_abs) max_abs = diff;
    if (diff > 1.0e-4f) {
      if (bad < 8) {
        fprintf(stderr, "FAIL %s at %u: got=%+.8f exp=%+.8f diff=%g\n",
                label, i, got[i], ref[i], diff);
      }
      bad++;
    }
  }
  if (bad) {
    fprintf(stderr, "FAIL %s bad=%u/%u max_abs=%g\n",
            label, bad, elements, max_abs);
    return 1;
  }
  printf("PASS %s elements=%u max_abs=%g\n", label, elements, max_abs);
  return 0;
}

static int event_duration_ns(cl_event event, uint64_t *duration_ns)
{
  cl_ulong start = 0;
  cl_ulong end = 0;
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

static int run_setup_kernel(cl_command_queue queue, cl_kernel kernel,
                            cl_mem g2s_desc, cl_mem input_buf)
{
  cl_int err;
  size_t global = 1;
  size_t local = 1;
  cl_event event = NULL;
  err  = clSetKernelArg(kernel, 0, sizeof(g2s_desc), &g2s_desc);
  err |= clSetKernelArg(kernel, 1, sizeof(input_buf), &input_buf);
  if (err != CL_SUCCESS) return 1;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                               0, NULL, &event);
  if (err != CL_SUCCESS) return 1;
  err = clWaitForEvents(1, &event);
  clReleaseEvent(event);
  return err != CL_SUCCESS;
}

static int run_measured_kernel(cl_command_queue queue, PathKind kind,
                               cl_kernel manual_kernel, cl_kernel tma_kernel,
                               cl_mem g2s_desc,
                               cl_mem input_buf, cl_mem output_buf,
                               PipelineCase c, uint64_t *ns)
{
  cl_int err;
  cl_kernel kernel = kind == PATH_TMA ? tma_kernel : manual_kernel;
  size_t global = WG_SIZE;
  size_t local = WG_SIZE;
  cl_event event = NULL;
  if (kind == PATH_TMA) {
    err  = clSetKernelArg(kernel, 0, sizeof(g2s_desc), &g2s_desc);
    err |= clSetKernelArg(kernel, 1, sizeof(input_buf), &input_buf);
    err |= clSetKernelArg(kernel, 2, sizeof(output_buf), &output_buf);
  } else {
    err  = clSetKernelArg(kernel, 0, sizeof(input_buf), &input_buf);
    err |= clSetKernelArg(kernel, 1, sizeof(output_buf), &output_buf);
    err |= clSetKernelArg(kernel, 2, sizeof(c.buffers), &c.buffers);
    err |= clSetKernelArg(kernel, 3, sizeof(c.stages), &c.stages);
  }
  if (err != CL_SUCCESS) return 1;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                               0, NULL, &event);
  if (err != CL_SUCCESS) return 1;
  err = clWaitForEvents(1, &event);
  if (err != CL_SUCCESS) {
    clReleaseEvent(event);
    return 1;
  }
  if (event_duration_ns(event, ns) != 0) {
    clReleaseEvent(event);
    return 1;
  }
  clReleaseEvent(event);
  return 0;
}

static cl_int build_program_with_tile_options(cl_context context,
                                              cl_device_id device,
                                              const char *source_path,
                                              PipelineCase c,
                                              cl_program *program)
{
  size_t source_size = 0;
  char *source = ventus_read_text_file(source_path, &source_size);
  if (!source) return CL_INVALID_PROGRAM;

  cl_int err;
  const char *sources[] = {source};
  const size_t sizes[] = {source_size};
  cl_program prog = clCreateProgramWithSource(context, 1, sources, sizes, &err);
  free(source);
  if (err != CL_SUCCESS) return err;

  char options[128];
  snprintf(options, sizeof(options), "-DTILE_ROWS=%uu -DTILE_COLS=%uu",
           c.rows, c.cols);
  err = clBuildProgram(prog, 1, &device, options, NULL, NULL);
  if (err != CL_SUCCESS) {
    ventus_print_build_log(prog, device);
    clReleaseProgram(prog);
    return err;
  }

  *program = prog;
  return CL_SUCCESS;
}

static int run_child(PathKind kind, PipelineCase c, PathResult *result)
{
  cl_int err = CL_SUCCESS;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel setup_kernel = NULL;
  cl_kernel manual_kernel = NULL;
  cl_kernel tma_kernel = NULL;
  cl_mem g2s_desc_buf = NULL;
  cl_mem input_buf = NULL;
  cl_mem output_buf = NULL;
  float *input = NULL, *ref = NULL, *got = NULL;
  uint32_t g2s_desc[DESC_WORDS];
  uint32_t elements = case_elements(c);
  uint64_t ns = 0;
  int exit_code = 1;
  size_t bytes = (size_t)elements * sizeof(float);
  const char *source_path = getenv("VENTUS_TMA_PINGPONG_SOURCE");
  if (!source_path || !source_path[0]) {
    source_path = "tma_pingpong_pipeline_perf_test.cl";
  }

  printf("CASE path=%s rows=%u cols=%u buffers=%u stages=%u elements=%u compute_iters=1 wg_size=%u single_wg=1\n",
         path_name(kind), c.rows, c.cols, c.buffers, c.stages,
         elements, WG_SIZE);

  input = (float *)malloc(bytes);
  ref = (float *)malloc(bytes);
  got = (float *)malloc(bytes);
  if (!input || !ref || !got) goto FINISH;
  fill_input(input, elements);
  cpu_ref(input, ref, c);
  memset(got, 0, bytes);
  build_desc(g2s_desc, c);

  err = ventus_get_default_device(&context, &device, &queue, NULL);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device");
  clReleaseCommandQueue(queue);
  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE,
                               &err);
  CHECK_OPENCL_ERROR_IN("clCreateCommandQueue(profiled)");
  err = build_program_with_tile_options(context, device, source_path, c, &program);
  CHECK_OPENCL_ERROR_IN("build_program_with_tile_options");
  setup_kernel = clCreateKernel(program, "setup_desc_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(setup_desc)");
  if (kind == PATH_TMA) {
    const char *kernel_name = tma_kernel_name(c);
    if (!kernel_name) goto FINISH;
    tma_kernel = clCreateKernel(program, kernel_name, &err);
    CHECK_OPENCL_ERROR_IN("clCreateKernel(tma)");
  } else {
    const char *kernel_name = manual_kernel_name(c);
    if (!kernel_name) goto FINISH;
    manual_kernel = clCreateKernel(program, kernel_name, &err);
    CHECK_OPENCL_ERROR_IN("clCreateKernel(manual)");
  }

  g2s_desc_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                sizeof(g2s_desc), g2s_desc, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(g2s_desc)");
  input_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             bytes, input, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(input)");
  output_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              bytes, got, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(output)");

  if (run_setup_kernel(queue, setup_kernel, g2s_desc_buf,
                       input_buf) != 0) goto FINISH;
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(setup)");
  if (run_measured_kernel(queue, kind, manual_kernel, tma_kernel,
                          g2s_desc_buf,
                          input_buf, output_buf, c, &ns) != 0) goto FINISH;
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(measured)");
  err = clEnqueueReadBuffer(queue, output_buf, CL_TRUE, 0, bytes,
                            got, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(output)");
  if (check_result(path_name(kind), got, ref, elements) != 0) goto FINISH;

  result->kind = kind;
  result->c = c;
  result->elements = elements;
  result->ns = ns;
  result->cycles = 0;
  result->cycle_valid = 0;
  result->elements_per_ns = ns ? (double)elements / (double)ns : 0.0;
  result->seen = 1;
  printf("PATH_RESULT path=%s rows=%u cols=%u buffers=%u stages=%u elements=%u compute_iters=1 ns=%" PRIu64
         " elements_per_ns=%.6e\n",
         path_name(kind), c.rows, c.cols, c.buffers, c.stages,
         elements, ns, result->elements_per_ns);
  printf("OK child path=%s buffers=%u stages=%u\n",
         path_name(kind), c.buffers, c.stages);
  exit_code = 0;

FINISH:
  if (output_buf) clReleaseMemObject(output_buf);
  if (input_buf) clReleaseMemObject(input_buf);
  if (g2s_desc_buf) clReleaseMemObject(g2s_desc_buf);
  if (tma_kernel) clReleaseKernel(tma_kernel);
  if (manual_kernel) clReleaseKernel(manual_kernel);
  if (setup_kernel) clReleaseKernel(setup_kernel);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
  free(got);
  free(ref);
  free(input);
  return exit_code;
}

static void make_timestamp(char *stamp, size_t stamp_size)
{
  time_t now = time(NULL);
  struct tm tm_now;
  if (localtime_r(&now, &tm_now) &&
      strftime(stamp, stamp_size, "%Y%m%d_%H%M%S", &tm_now) > 0) return;
  snprintf(stamp, stamp_size, "%ld", (long)now);
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

static int parse_path_result_line(const char *line, PathResult *result)
{
  char path[32];
  unsigned rows = 0, cols = 0, buffers = 0, stages = 0, elements = 0;
  unsigned compute_iters = 0;
  unsigned long long ns = 0;
  double elements_per_ns = 0.0;
  PathKind kind;
  int fields = sscanf(line,
                      "PATH_RESULT path=%31s rows=%u cols=%u buffers=%u stages=%u elements=%u compute_iters=%u ns=%llu elements_per_ns=%lf",
                      path, &rows, &cols, &buffers, &stages, &elements,
                      &compute_iters, &ns, &elements_per_ns);
  if (fields != 9) return 1;
  if (compute_iters != 1 || rows == 0 || cols == 0) return 1;
  if (elements != rows * cols * stages) return 1;
  if (parse_path_kind(path, &kind) != 0) return 1;
  result->kind = kind;
  result->c.rows = rows;
  result->c.cols = cols;
  result->c.buffers = buffers;
  result->c.stages = stages;
  result->elements = elements;
  result->ns = (uint64_t)ns;
  result->cycles = 0;
  result->cycle_valid = 0;
  result->elements_per_ns = elements_per_ns;
  result->seen = 1;
  return 0;
}

static int make_absolute_path(char *out, size_t out_size, const char *cwd,
                              const char *path)
{
  if (!path || !path[0]) return 1;
  if (path[0] == '/') snprintf(out, out_size, "%s", path);
  else snprintf(out, out_size, "%s/%s", cwd, path);
  return 0;
}

static int ensure_dir(const char *path)
{
  struct stat st;
  if (mkdir(path, 0775) == 0) return 0;
  if (errno == EEXIST && stat(path, &st) == 0 && S_ISDIR(st.st_mode)) {
    return 0;
  }
  return 1;
}

static void make_child_paths(ChildRun *child)
{
  char stamp[32];
  make_timestamp(stamp, sizeof(stamp));
  snprintf(child->log_path, sizeof(child->log_path),
           LOG_DIR "/tma_pingpong_pipeline_perf_child_%s_%ld_%s_r%u_c%u_b%u_s%u.log",
           stamp, (long)getpid(), path_name(child->kind),
           child->c.rows, child->c.cols, child->c.buffers, child->c.stages);
  snprintf(child->run_dir, sizeof(child->run_dir),
           LOG_DIR "/tma_pingpong_pipeline_perf_run_%s_%ld_%s_r%u_c%u_b%u_s%u",
           stamp, (long)getpid(), path_name(child->kind),
           child->c.rows, child->c.cols, child->c.buffers, child->c.stages);
}

static int launch_child(const char *prog, PathKind kind, PipelineCase c,
                        ChildRun *child)
{
  FILE *header = NULL;
  pid_t pid;
  char cwd[256];
  char exe_abs[512];
  char source_abs[512];
  char rows_arg[16];
  char cols_arg[16];
  char buffers_arg[16];
  char stages_arg[16];

  child->kind = kind;
  child->c = c;
  make_child_paths(child);
  if (!getcwd(cwd, sizeof(cwd))) return 1;
  if (make_absolute_path(exe_abs, sizeof(exe_abs), cwd, prog) != 0) return 1;
  snprintf(source_abs, sizeof(source_abs),
           "%s/tma_pingpong_pipeline_perf_test.cl", cwd);
  if (ensure_dir(LOG_DIR) != 0) return 1;
  if (mkdir(child->run_dir, 0775) != 0) return 1;
  header = fopen(child->log_path, "w");
  if (!header) return 1;
  fprintf(header, "# command: %s child %s %u %u %u %u\n",
          prog, path_arg(kind), c.rows, c.cols, c.buffers, c.stages);
  fclose(header);
  snprintf(rows_arg, sizeof(rows_arg), "%u", c.rows);
  snprintf(cols_arg, sizeof(cols_arg), "%u", c.cols);
  snprintf(buffers_arg, sizeof(buffers_arg), "%u", c.buffers);
  snprintf(stages_arg, sizeof(stages_arg), "%u", c.stages);

  pid = fork();
  if (pid < 0) return 1;
  if (pid == 0) {
    FILE *out = fopen(child->log_path, "a");
    setenv("VENTUS_TMA_PINGPONG_SOURCE", source_abs, 1);
    if (out) {
      dup2(fileno(out), STDOUT_FILENO);
      dup2(fileno(out), STDERR_FILENO);
    }
    if (chdir(child->run_dir) != 0) _exit(127);
    execlp(exe_abs, exe_abs, "child", path_arg(kind),
           rows_arg, cols_arg, buffers_arg, stages_arg, (char *)NULL);
    _exit(127);
  }
  child->pid = pid;
  child->active = 1;
  printf("PINGPONG_CHILD_START path=%s pid=%ld rows=%u cols=%u buffers=%u stages=%u log=%s run_dir=%s\n",
         path_name(kind), (long)pid, c.rows, c.cols, c.buffers, c.stages,
         child->log_path, child->run_dir);
  fflush(stdout);
  return 0;
}

static int parse_child_log(const ChildRun *child, PathResult *result)
{
  FILE *f = fopen(child->log_path, "r");
  char line[1024];
  uint64_t program1_cycles = 0;
  uint64_t total_cycles = 0;
  PathResult parsed;
  memset(&parsed, 0, sizeof(parsed));
  if (!f) return 1;
  while (fgets(line, sizeof(line), f)) {
    if (strstr(line, "PATH_RESULT path=")) {
      (void)parse_path_result_line(line, &parsed);
    }
    if (strstr(line, "[PROGRAM          1] [INST+CYCLE] active cycles")) {
      (void)parse_cycles_after_colon(line, &program1_cycles);
    } else if (strstr(line, "[TESTCASE TOTAL] [INST+CYCLE] active cycles")) {
      (void)parse_cycles_after_colon(line, &total_cycles);
    }
  }
  fclose(f);
  if (!parsed.seen) return 1;
  *result = parsed;
  if (program1_cycles) {
    result->cycles = program1_cycles;
    result->cycle_valid = 1;
  } else if (total_cycles) {
    result->cycles = total_cycles;
    result->cycle_valid = 1;
  }
  return 0;
}

static int wait_child_and_parse(const ChildRun *child, PathResult *result)
{
  int status = 0;
  if (waitpid(child->pid, &status, 0) < 0) return 1;
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    fprintf(stderr, "child failed path=%s rows=%u cols=%u buffers=%u stages=%u log=%s status=%d\n",
            path_name(child->kind), child->c.rows, child->c.cols,
            child->c.buffers, child->c.stages, child->log_path, status);
    return 1;
  }
  if (parse_child_log(child, result) != 0) {
    fprintf(stderr, "failed to parse child log %s\n", child->log_path);
    return 1;
  }
  printf("PINGPONG_CHILD_DONE path=%s rows=%u cols=%u buffers=%u stages=%u ns=%" PRIu64,
         path_name(result->kind), result->c.rows, result->c.cols,
         result->c.buffers, result->c.stages, result->ns);
  if (result->cycle_valid) printf(" cycles=%" PRIu64, result->cycles);
  printf("\n");
  return 0;
}

static void stop_children(ChildRun *children)
{
  for (uint32_t i = 0; i < PATH_COUNT; i++) {
    if (children[i].active && children[i].pid > 0) kill(children[i].pid, SIGTERM);
  }
  for (uint32_t i = 0; i < PATH_COUNT; i++) {
    if (children[i].active && children[i].pid > 0) {
      waitpid(children[i].pid, NULL, 0);
      children[i].active = 0;
    }
  }
}

static int run_pair(const char *prog, PipelineCase c, CaseResult *out)
{
  ChildRun children[PATH_COUNT];
  uint32_t completed = 0;
  memset(children, 0, sizeof(children));
  memset(out, 0, sizeof(*out));
  out->c = c;

  if (launch_child(prog, PATH_MANUAL, c, &children[PATH_MANUAL]) != 0 ||
      launch_child(prog, PATH_TMA, c, &children[PATH_TMA]) != 0) {
    stop_children(children);
    return 1;
  }

  while (completed < PATH_COUNT) {
    int status = 0;
    pid_t done = waitpid(-1, &status, 0);
    if (done < 0) {
      stop_children(children);
      return 1;
    }

    uint32_t slot = 0;
    while (slot < PATH_COUNT &&
           (!children[slot].active || children[slot].pid != done)) {
      slot++;
    }
    if (slot == PATH_COUNT) continue;

    children[slot].active = 0;
    completed++;
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      fprintf(stderr, "child failed path=%s rows=%u cols=%u buffers=%u stages=%u log=%s status=%d\n",
              path_name(children[slot].kind), children[slot].c.rows,
              children[slot].c.cols, children[slot].c.buffers,
              children[slot].c.stages, children[slot].log_path, status);
      stop_children(children);
      return 1;
    }

    PathResult *target = children[slot].kind == PATH_TMA ? &out->tma : &out->manual;
    if (parse_child_log(&children[slot], target) != 0) {
      fprintf(stderr, "failed to parse child log %s\n", children[slot].log_path);
      stop_children(children);
      return 1;
    }

    printf("PINGPONG_CHILD_DONE path=%s rows=%u cols=%u buffers=%u stages=%u ns=%" PRIu64,
           path_name(target->kind), target->c.rows, target->c.cols,
           target->c.buffers, target->c.stages, target->ns);
    if (target->cycle_valid) printf(" cycles=%" PRIu64, target->cycles);
    printf("\n");
  }

  return 0;
}

static double speedup_u64(uint64_t base, uint64_t opt)
{
  if (!base || !opt) return 0.0;
  return (double)base / (double)opt;
}

static double improvement_percent(double speedup)
{
  if (speedup <= 0.0) return 0.0;
  return (1.0 - 1.0 / speedup) * 100.0;
}

static void print_result_row(FILE *f, const CaseResult *r)
{
  double cycle_speed = 0.0;
  double cycle_improve = 0.0;
  double ns_speed = speedup_u64(r->manual.ns, r->tma.ns);
  double ns_improve = improvement_percent(ns_speed);
  if (r->manual.cycle_valid && r->tma.cycle_valid) {
    cycle_speed = speedup_u64(r->manual.cycles, r->tma.cycles);
    cycle_improve = improvement_percent(cycle_speed);
    fprintf(f, "| %u | %u | %u | %u | %u | %" PRIu64 " | %" PRIu64 " | %.4f | %.2f | %" PRIu64 " | %" PRIu64 " | %.4f | %.2f |\n",
            r->c.rows, r->c.cols, r->c.buffers, r->c.stages,
            r->manual.elements, r->manual.cycles, r->tma.cycles,
            cycle_speed, cycle_improve, r->manual.ns, r->tma.ns,
            ns_speed, ns_improve);
  } else {
    fprintf(f, "| %u | %u | %u | %u | %u | NA | NA | NA | NA | %" PRIu64 " | %" PRIu64 " | %.4f | %.2f |\n",
            r->c.rows, r->c.cols, r->c.buffers, r->c.stages,
            r->manual.elements, r->manual.ns, r->tma.ns,
            ns_speed, ns_improve);
  }
}

static int write_report(const CaseResult *results, size_t count)
{
  char stamp[32];
  char path[128];
  FILE *f = NULL;
  const CaseResult *best = NULL;
  double best_speed = 0.0;
  make_timestamp(stamp, sizeof(stamp));
  if (ensure_dir(LOG_DIR) != 0) return 1;
  snprintf(path, sizeof(path),
           LOG_DIR "/tma_pingpong_pipeline_perf_report_%s.md", stamp);
  f = fopen(path, "w");
  if (!f) return 1;
  fprintf(f, "# TMA Ping-Pong Pipeline Performance Report\n\n");
  fprintf(f, "- tile_sweep: %s FP32 tiles, one work-group\n", sweep_tile_text);
  fprintf(f, "- compute_iters: 1 fixed\n");
  fprintf(f, "- buffer_sweep: 2 active shared-memory tile buffers\n");
  fprintf(f, "- stage_sweep: 1, 2, 4, 8, 16 tiles per work-group\n");
  fprintf(f, "- manual_path: global/L2 -> register -> shared -> compute -> register -> global\n");
  fprintf(f, "- tma_path: descriptor TMA G2S -> shared compute -> ordinary shared/register/global store\n");
  fprintf(f, "- primary_metric: GVM PMU active cycles when available; host ns is auxiliary\n\n");
  fprintf(f, "PINGPONG_SWEEP_TABLE_BEGIN\n");
  fprintf(f, "| rows | cols | buffers | stages | elements | manual_cycles | tma_cycles | cycle_speedup | cycle_improvement_percent | manual_ns | tma_ns | ns_speedup | ns_improvement_percent |\n");
  fprintf(f, "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
  for (size_t i = 0; i < count; i++) {
    double speed = 0.0;
    print_result_row(f, &results[i]);
    if (results[i].manual.cycle_valid && results[i].tma.cycle_valid) {
      speed = speedup_u64(results[i].manual.cycles, results[i].tma.cycles);
    } else {
      speed = speedup_u64(results[i].manual.ns, results[i].tma.ns);
    }
    if (!best || speed > best_speed) {
      best = &results[i];
      best_speed = speed;
    }
  }
  fprintf(f, "PINGPONG_SWEEP_TABLE_END\n\n");
  if (best) {
    fprintf(f, "BEST_PINGPONG rows=%u cols=%u buffers=%u stages=%u speedup=%.4fx improvement_percent=%.2f\n",
            best->c.rows, best->c.cols, best->c.buffers, best->c.stages,
            best_speed, improvement_percent(best_speed));
  }
  fclose(f);
  printf("REPORT %s\n", path);
  return 0;
}

static int run_single_or_sweep(const char *prog, int sweep,
                               PipelineCase single_case)
{
  CaseResult results[sizeof(sweep_tiles) / sizeof(sweep_tiles[0]) *
                     sizeof(sweep_buffers) / sizeof(sweep_buffers[0]) *
                     sizeof(sweep_stages) / sizeof(sweep_stages[0])];
  size_t count = 0;
  if (sweep) {
    for (size_t ti = 0; ti < sizeof(sweep_tiles) / sizeof(sweep_tiles[0]); ti++) {
      for (size_t bi = 0; bi < sizeof(sweep_buffers) / sizeof(sweep_buffers[0]); bi++) {
        for (size_t si = 0; si < sizeof(sweep_stages) / sizeof(sweep_stages[0]); si++) {
          PipelineCase c = {sweep_tiles[ti].rows, sweep_tiles[ti].cols,
                            sweep_buffers[bi], sweep_stages[si]};
          if (run_pair(prog, c, &results[count]) != 0) return 1;
          print_result_row(stdout, &results[count]);
          count++;
        }
      }
    }
  } else {
    if (run_pair(prog, single_case, &results[count]) != 0) return 1;
    print_result_row(stdout, &results[count]);
    count++;
  }
  return write_report(results, count);
}

static void usage(const char *prog)
{
  fprintf(stderr,
          "Usage:\n"
          "  %s\n"
          "  %s sweep\n"
          "  %s single <buffers> <stages>                 # default 16x16\n"
          "  %s single <rows> <cols> <buffers> <stages>\n"
          "  %s child manual|tma <rows> <cols> <buffers> <stages>\n",
          prog, prog, prog, prog, prog);
}

int main(int argc, char **argv)
{
  PipelineCase c = {DEFAULT_TILE_ROWS, DEFAULT_TILE_COLS,
                    DEFAULT_BUFFERS, DEFAULT_STAGES};
  if (argc == 1) {
    return run_single_or_sweep(argv[0], 1, c);
  }
  if (strcmp(argv[1], "sweep") == 0 && argc == 2) {
    return run_single_or_sweep(argv[0], 1, c);
  }
  if (strcmp(argv[1], "single") == 0 && (argc == 4 || argc == 6)) {
    int arg = 2;
    if (argc == 6) {
      if (parse_u32(argv[arg++], 1, MAX_TILE_ROWS, "rows", &c.rows) ||
          parse_u32(argv[arg++], 1, MAX_TILE_COLS, "cols", &c.cols)) {
        return 1;
      }
    }
    if (parse_u32(argv[arg++], 1, MAX_BUFFERS, "buffers", &c.buffers) ||
        parse_u32(argv[arg++], 1, MAX_STAGES, "stages", &c.stages) ||
        validate_case(c)) {
      return 1;
    }
    return run_single_or_sweep(argv[0], 0, c);
  }
  if (strcmp(argv[1], "child") == 0 && (argc == 5 || argc == 7)) {
    PathKind kind;
    PathResult result;
    int arg = 3;
    memset(&result, 0, sizeof(result));
    if (parse_path_kind(argv[2], &kind)) return 1;
    if (argc == 7) {
      if (parse_u32(argv[arg++], 1, MAX_TILE_ROWS, "rows", &c.rows) ||
          parse_u32(argv[arg++], 1, MAX_TILE_COLS, "cols", &c.cols)) {
        return 1;
      }
    }
    if (parse_u32(argv[arg++], 1, MAX_BUFFERS, "buffers", &c.buffers) ||
        parse_u32(argv[arg++], 1, MAX_STAGES, "stages", &c.stages) ||
        validate_case(c)) {
      return 1;
    }
    return run_child(kind, c, &result);
  }
  usage(argv[0]);
  return 1;
}
