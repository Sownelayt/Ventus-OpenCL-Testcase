/*
 * TMA serial movement performance test.
 *
 * Background:
 *   This is the non-overlap counterpart of tma_pingpong_pipeline_perf_test.
 *   Each measured child run executes one tile in one work-group and reports
 *   movement, ordinary shared-memory compute, and ordinary global writeback.
 *
 * Implementation:
 *   The sweep uses the ping-pong test's tile sizes: 16x16, 32x16, 32x32,
 *   64x32, and 64x64 FP32 tiles. The manual path directly builds and runs
 *   tma_pingpong_pipeline_perf_test's manual_pingpong_b2_s1_kernel so its
 *   whole-kernel PMU cycles are the same baseline, not a copy with a different
 *   program layout. The TMA path is intentionally different:
 *   it is a blocking serial path that issues the current tile, waits, computes,
 *   then writes back. The measured kernels contain no marker instructions;
 *   per-behavior cycles are recovered out-of-band from GVM retire PC timestamps
 *   and the generated object disassembly.
 *
 * Usage:
 *   ./dma_tma_serial_pipeline_perf_test.out
 *   ./dma_tma_serial_pipeline_perf_test.out sweep
 *   ./dma_tma_serial_pipeline_perf_test.out single <rows> <cols>
 *   ./dma_tma_serial_pipeline_perf_test.out child manual|tma <rows> <cols>
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

#define DESC_WORDS 32u
#define DEFAULT_TILE_ROWS 16u
#define DEFAULT_TILE_COLS 16u
#define MAX_TILE_ROWS 64u
#define MAX_TILE_COLS 64u
#define WG_SIZE 32u
#define MANUAL_COMPAT_BUFFERS 2u
#define MANUAL_COMPAT_STAGES 1u
#define LOG_DIR "log"
/* GVM retire-log @time uses 5 time units per active cycle. */
#define GVM_CYCLE_TIME_UNITS 5ull

typedef enum {
  PATH_MANUAL = 0,
  PATH_TMA,
  PATH_COUNT
} PathKind;

typedef enum {
  CYCLE_SOURCE_NONE = 0,
  CYCLE_SOURCE_GVM_PC_TRACE
} CycleSource;

typedef struct {
  uint32_t rows;
  uint32_t cols;
} SerialCase;

typedef struct {
  PathKind kind;
  SerialCase c;
  uint32_t elements;
  uint64_t ns;
  uint64_t kernel_cycles;
  uint64_t movement_cycles;
  uint64_t compute_cycles;
  uint64_t writeback_cycles;
  uint64_t behavior_total_cycles;
  int kernel_cycle_valid;
  CycleSource behavior_cycle_source;
  int seen;
  int passed;
  int exit_status;
  char log_path[256];
} PathResult;

typedef struct {
  SerialCase c;
  PathResult paths[PATH_COUNT];
} CaseResult;

typedef struct {
  PathKind kind;
  SerialCase c;
  pid_t pid;
  int active;
  char log_path[256];
  char run_dir[256];
} ChildRun;

typedef struct {
  uint64_t movement_start_pc;
  uint64_t compute_start_pc;
  uint64_t writeback_start_pc;
  uint64_t behavior_end_pc;
  int valid;
} PcBoundaries;

typedef struct {
  uint64_t movement_start_time;
  uint64_t compute_start_time;
  uint64_t writeback_start_time;
  uint64_t behavior_end_time;
  int movement_seen;
  int compute_seen;
  int writeback_seen;
  int end_seen;
} PcTimes;

static const SerialCase sweep_tiles[] = {
  {16u, 16u},
  {32u, 16u},
  {32u, 32u},
  {64u, 32u},
  {64u, 64u},
};
static const char *sweep_tile_text = "16x16, 32x16, 32x32, 64x32, 64x64";

static const char *path_arg(PathKind kind)
{
  switch (kind) {
  case PATH_MANUAL: return "manual";
  case PATH_TMA: return "tma";
  default: return "unknown";
  }
}

static const char *path_desc(PathKind kind)
{
  switch (kind) {
  case PATH_MANUAL: return "ordinary global load";
  case PATH_TMA: return "descriptor CP_ASYNC_TENSOR G2S";
  default: return "unknown";
  }
}

static const char *kernel_name(PathKind kind)
{
  switch (kind) {
  case PATH_MANUAL: return "manual_pingpong_b2_s1_kernel";
  case PATH_TMA: return "tma_serial_pipeline_kernel";
  default: return "unknown";
  }
}

static const char *default_source_path(PathKind kind)
{
  switch (kind) {
  case PATH_MANUAL: return "../tma_pingpong_pipeline_perf_test/tma_pingpong_pipeline_perf_test.cl";
  case PATH_TMA: return "dma_tma_serial_pipeline_perf_test.cl";
  default: return "dma_tma_serial_pipeline_perf_test.cl";
  }
}

static const char *cycle_source_text(CycleSource source)
{
  switch (source) {
  case CYCLE_SOURCE_GVM_PC_TRACE: return "gvm_pc_trace";
  default: return "unavailable";
  }
}

static int parse_path_kind(const char *text, PathKind *kind)
{
  if (strcmp(text, "manual") == 0) {
    *kind = PATH_MANUAL;
    return 0;
  }
  if (strcmp(text, "tma") == 0 || strcmp(text, "tma_g2s") == 0) {
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

static int is_supported_tile(uint32_t rows, uint32_t cols)
{
  for (size_t i = 0; i < sizeof(sweep_tiles) / sizeof(sweep_tiles[0]); i++) {
    if (rows == sweep_tiles[i].rows && cols == sweep_tiles[i].cols) return 1;
  }
  return 0;
}

static int validate_case(SerialCase c)
{
  if (!is_supported_tile(c.rows, c.cols)) {
    fprintf(stderr,
            "invalid case: rows=%u cols=%u; supported tiles are %s\n",
            c.rows, c.cols, sweep_tile_text);
    return 1;
  }
  return 0;
}

static uint32_t desc_control(unsigned data_type, unsigned rank)
{
  return (data_type & 0xfu) | ((rank & 0xfu) << 4);
}

static uint32_t case_elements(SerialCase c)
{
  return c.rows * c.cols;
}

static void build_desc(uint32_t *desc, SerialCase c)
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

static void cpu_ref(const float *input, float *output, uint32_t elements)
{
  for (uint32_t i = 0; i < elements; i++) output[i] = compute_ref_value(input[i]);
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
    fprintf(stderr, "FAIL %s bad=%u/%u max_abs=%g\n", label, bad, elements, max_abs);
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

static cl_int build_program_with_options(cl_context context,
                                         cl_device_id device,
                                         const char *source_path,
                                         SerialCase c,
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

static int run_setup_kernel(cl_command_queue queue, cl_kernel kernel,
                            cl_mem desc_buf, cl_mem input_buf)
{
  cl_int err;
  size_t global = 1;
  size_t local = 1;
  cl_event event = NULL;
  err  = clSetKernelArg(kernel, 0, sizeof(desc_buf), &desc_buf);
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
                               cl_kernel kernel, cl_mem desc_buf,
                               cl_mem input_buf, cl_mem output_buf,
                               uint64_t *ns)
{
  cl_int err;
  size_t global = WG_SIZE;
  size_t local = WG_SIZE;
  cl_event event = NULL;
  if (kind == PATH_TMA) {
    err  = clSetKernelArg(kernel, 0, sizeof(desc_buf), &desc_buf);
    err |= clSetKernelArg(kernel, 1, sizeof(input_buf), &input_buf);
    err |= clSetKernelArg(kernel, 2, sizeof(output_buf), &output_buf);
  } else {
    uint32_t buffers = MANUAL_COMPAT_BUFFERS;
    uint32_t stages = MANUAL_COMPAT_STAGES;
    err  = clSetKernelArg(kernel, 0, sizeof(input_buf), &input_buf);
    err |= clSetKernelArg(kernel, 1, sizeof(output_buf), &output_buf);
    err |= clSetKernelArg(kernel, 2, sizeof(buffers), &buffers);
    err |= clSetKernelArg(kernel, 3, sizeof(stages), &stages);
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

static int run_child(PathKind kind, SerialCase c, PathResult *result)
{
  cl_int err = CL_SUCCESS;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel setup_kernel = NULL;
  cl_kernel measured_kernel = NULL;
  cl_mem desc_buf = NULL;
  cl_mem input_buf = NULL;
  cl_mem output_buf = NULL;
  float *input = NULL, *ref = NULL, *got = NULL;
  uint32_t desc[DESC_WORDS];
  uint32_t elements = case_elements(c);
  uint64_t ns = 0;
  int exit_code = 1;
  size_t bytes = (size_t)elements * sizeof(float);
  const char *source_path = getenv("VENTUS_SERIAL_PIPELINE_SOURCE");
  if (!source_path || !source_path[0]) source_path = default_source_path(kind);

  memset(result, 0, sizeof(*result));
  result->kind = kind;
  result->c = c;
  result->elements = elements;
  printf("CASE path=%s rows=%u cols=%u elements=%u compute_iters=1 wg_size=%u single_wg=1\n",
         path_arg(kind), c.rows, c.cols, elements, WG_SIZE);

  input = (float *)malloc(bytes);
  ref = (float *)malloc(bytes);
  got = (float *)malloc(bytes);
  if (!input || !ref || !got) goto FINISH;
  fill_input(input, elements);
  cpu_ref(input, ref, elements);
  memset(got, 0, bytes);
  build_desc(desc, c);

  err = ventus_get_default_device(&context, &device, &queue, NULL);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device");
  clReleaseCommandQueue(queue);
  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
  CHECK_OPENCL_ERROR_IN("clCreateCommandQueue(profiled)");
  err = build_program_with_options(context, device, source_path, c, &program);
  CHECK_OPENCL_ERROR_IN("build_program_with_options");
  measured_kernel = clCreateKernel(program, kernel_name(kind), &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(measured)");
  setup_kernel = clCreateKernel(program, "setup_desc_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(setup_desc)");

  desc_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            sizeof(desc), desc, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(desc)");
  input_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             bytes, input, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(input)");
  output_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              bytes, got, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(output)");

  if (run_setup_kernel(queue, setup_kernel, desc_buf, input_buf) != 0) goto FINISH;
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(setup)");
  if (run_measured_kernel(queue, kind, measured_kernel, desc_buf, input_buf,
                          output_buf, &ns) != 0) goto FINISH;
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(measured)");
  err = clEnqueueReadBuffer(queue, output_buf, CL_TRUE, 0, bytes, got,
                            0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(output)");
  if (check_result(path_arg(kind), got, ref, elements) != 0) goto FINISH;

  result->ns = ns;
  result->behavior_cycle_source = CYCLE_SOURCE_NONE;
  result->seen = 1;
  result->passed = 1;
  printf("PATH_RESULT path=%s rows=%u cols=%u elements=%u ns=%" PRIu64 "\n",
         path_arg(kind), c.rows, c.cols, elements, ns);
  printf("OK child path=%s rows=%u cols=%u\n", path_arg(kind), c.rows, c.cols);
  exit_code = 0;

FINISH:
  if (output_buf) clReleaseMemObject(output_buf);
  if (input_buf) clReleaseMemObject(input_buf);
  if (desc_buf) clReleaseMemObject(desc_buf);
  if (setup_kernel) clReleaseKernel(setup_kernel);
  if (measured_kernel) clReleaseKernel(measured_kernel);
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

static int ensure_dir(const char *path)
{
  struct stat st;
  if (mkdir(path, 0775) == 0) return 0;
  if (errno == EEXIST && stat(path, &st) == 0 && S_ISDIR(st.st_mode)) return 0;
  return 1;
}

static int make_absolute_path(char *out, size_t out_size, const char *cwd,
                              const char *path)
{
  if (!path || !path[0]) return 1;
  if (path[0] == '/') snprintf(out, out_size, "%s", path);
  else snprintf(out, out_size, "%s/%s", cwd, path);
  return 0;
}

static void copy_text(char *dst, size_t dst_size, const char *src)
{
  if (!dst_size) return;
  snprintf(dst, dst_size, "%s", src ? src : "");
}

static void make_child_paths(ChildRun *child)
{
  char stamp[32];
  make_timestamp(stamp, sizeof(stamp));
  snprintf(child->log_path, sizeof(child->log_path),
           LOG_DIR "/dma_tma_serial_pipeline_perf_child_%s_%ld_%s_r%u_c%u.log",
           stamp, (long)getpid(), path_arg(child->kind), child->c.rows, child->c.cols);
  snprintf(child->run_dir, sizeof(child->run_dir),
           LOG_DIR "/dma_tma_serial_pipeline_perf_run_%s_%ld_%s_r%u_c%u",
           stamp, (long)getpid(), path_arg(child->kind), child->c.rows, child->c.cols);
}

static void init_failed_result(const ChildRun *child, int status, PathResult *result)
{
  memset(result, 0, sizeof(*result));
  result->kind = child->kind;
  result->c = child->c;
  result->elements = case_elements(child->c);
  result->seen = 1;
  result->passed = 0;
  result->exit_status = status;
  copy_text(result->log_path, sizeof(result->log_path), child->log_path);
}

static int launch_child(const char *prog, PathKind kind, SerialCase c,
                        ChildRun *child)
{
  FILE *header = NULL;
  pid_t pid;
  char cwd[256];
  char exe_abs[512];
  char source_abs[512];
  char rows_arg[16];
  char cols_arg[16];

  memset(child, 0, sizeof(*child));
  child->kind = kind;
  child->c = c;
  make_child_paths(child);
  if (!getcwd(cwd, sizeof(cwd))) return 1;
  if (make_absolute_path(exe_abs, sizeof(exe_abs), cwd, prog) != 0) return 1;
  if (kind == PATH_MANUAL) {
    snprintf(source_abs, sizeof(source_abs),
             "%s/../tma_pingpong_pipeline_perf_test/tma_pingpong_pipeline_perf_test.cl", cwd);
  } else {
    snprintf(source_abs, sizeof(source_abs), "%s/dma_tma_serial_pipeline_perf_test.cl", cwd);
  }
  if (ensure_dir(LOG_DIR) != 0) return 1;
  if (mkdir(child->run_dir, 0775) != 0) return 1;
  header = fopen(child->log_path, "w");
  if (!header) return 1;
  fprintf(header, "# command: %s child %s %u %u\n", prog, path_arg(kind), c.rows, c.cols);
  fclose(header);
  snprintf(rows_arg, sizeof(rows_arg), "%u", c.rows);
  snprintf(cols_arg, sizeof(cols_arg), "%u", c.cols);

  pid = fork();
  if (pid < 0) return 1;
  if (pid == 0) {
    FILE *out = fopen(child->log_path, "a");
    setenv("VENTUS_SERIAL_PIPELINE_SOURCE", source_abs, 1);
    if (out) {
      dup2(fileno(out), STDOUT_FILENO);
      dup2(fileno(out), STDERR_FILENO);
    }
    if (chdir(child->run_dir) != 0) _exit(127);
    execlp(exe_abs, exe_abs, "child", path_arg(kind), rows_arg, cols_arg, (char *)NULL);
    _exit(127);
  }
  child->pid = pid;
  child->active = 1;
  printf("SERIAL_CHILD_START path=%s pid=%ld rows=%u cols=%u log=%s run_dir=%s\n",
         path_arg(kind), (long)pid, c.rows, c.cols, child->log_path, child->run_dir);
  fflush(stdout);
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

static int parse_path_result_line(const char *line, PathResult *result)
{
  char path[32];
  unsigned rows = 0, cols = 0, elements = 0;
  unsigned long long ns = 0;
  PathKind kind;
  int fields = sscanf(line,
                      "PATH_RESULT path=%31s rows=%u cols=%u elements=%u ns=%llu",
                      path, &rows, &cols, &elements, &ns);
  if (fields != 5) return 1;
  if (rows == 0 || cols == 0 || elements != rows * cols) return 1;
  if (parse_path_kind(path, &kind) != 0) return 1;
  memset(result, 0, sizeof(*result));
  result->kind = kind;
  result->c.rows = rows;
  result->c.cols = cols;
  result->elements = elements;
  result->ns = (uint64_t)ns;
  result->behavior_cycle_source = CYCLE_SOURCE_NONE;
  result->seen = 1;
  result->passed = 1;
  return 0;
}

static int parse_retire_pc_line(const char *line, uint64_t *time_out, uint64_t *pc_out)
{
  const char *at = NULL;
  const char *pc_text = NULL;
  const char *hex = NULL;
  char *end = NULL;
  unsigned long long t = 0;
  unsigned long long pc = 0;
  if (!strstr(line, "GVM retire:")) return 1;
  at = strchr(line, '@');
  if (!at) return 1;
  t = strtoull(at + 1, &end, 10);
  if (end == at + 1) return 1;
  pc_text = strstr(line, "pc:");
  if (!pc_text) return 1;
  hex = strstr(pc_text, "0x");
  if (!hex) return 1;
  pc = strtoull(hex, &end, 16);
  if (end == hex) return 1;
  *time_out = (uint64_t)t;
  *pc_out = (uint64_t)pc;
  return 0;
}

static uint64_t pc_delta_cycles(uint64_t begin, uint64_t end)
{
  if (end <= begin) return 0;
  return (end - begin) / GVM_CYCLE_TIME_UNITS;
}

static int parse_insn_pc(const char *line, uint64_t *pc_out)
{
  char *end = NULL;
  unsigned long long pc = strtoull(line, &end, 16);
  if (end == line || *end != ':') return 1;
  *pc_out = (uint64_t)pc;
  return 0;
}

static int parse_symbol_line(const char *line, char *sym, size_t sym_size)
{
  unsigned long long pc = 0;
  char tmp[128];
  if (sscanf(line, " %llx <%127[^>]>", &pc, tmp) != 2) return 1;
  snprintf(sym, sym_size, "%s", tmp);
  return 0;
}

static int line_is_barrier(const char *line)
{
  return strstr(line, "\tbarrier") || strstr(line, " barrier");
}

static int program_index_for_path(PathKind kind)
{
  (void)kind;
  return 1;
}

static void barrier_indices_for_path(PathKind kind, int *first, int *second, int *third)
{
  if (kind == PATH_TMA) {
    *first = 3;
    *second = 4;
    *third = 5;
  } else {
    *first = 1;
    *second = 2;
    *third = 3;
  }
}

static void make_objdump_command(const ChildRun *child, char *cmd, size_t cmd_size)
{
  const char *objdump = getenv("VENTUS_LLVM_OBJDUMP");
  const char *env_path = getenv("VENTUS_ENV_PATH");
  char objdump_path[512];
  if (objdump && objdump[0]) {
    snprintf(objdump_path, sizeof(objdump_path), "%s", objdump);
  } else if (env_path && env_path[0]) {
    snprintf(objdump_path, sizeof(objdump_path), "%s/install/bin/llvm-objdump", env_path);
  } else {
    snprintf(objdump_path, sizeof(objdump_path), "../../../install/bin/llvm-objdump");
  }
  snprintf(cmd, cmd_size, "%s -d --mattr=+v,+zfinx %s/object0.riscv",
           objdump_path, child->run_dir);
}

static int find_pc_boundaries(const ChildRun *child, PcBoundaries *bounds)
{
  FILE *pipe = NULL;
  char cmd[1024];
  char line[1024];
  char sym[128];
  int in_kernel = 0;
  int need_next_after_get_local_id = 0;
  int pending_boundary = 0;
  int barrier_count = 0;
  int first_barrier = 0, second_barrier = 0, third_barrier = 0;
  memset(bounds, 0, sizeof(*bounds));
  barrier_indices_for_path(child->kind, &first_barrier, &second_barrier, &third_barrier);
  make_objdump_command(child, cmd, sizeof(cmd));
  pipe = popen(cmd, "r");
  if (!pipe) return 1;
  while (fgets(line, sizeof(line), pipe)) {
    uint64_t pc = 0;
    if (parse_symbol_line(line, sym, sizeof(sym)) == 0) {
      if (strcmp(sym, kernel_name(child->kind)) == 0) {
        in_kernel = 1;
        continue;
      }
      if (in_kernel && sym[0] != '.') break;
    }
    if (!in_kernel) continue;
    if (parse_insn_pc(line, &pc) != 0) continue;
    if (need_next_after_get_local_id && bounds->movement_start_pc == 0) {
      bounds->movement_start_pc = pc;
      need_next_after_get_local_id = 0;
    }
    if (pending_boundary == 1 && bounds->compute_start_pc == 0) {
      bounds->compute_start_pc = pc;
      pending_boundary = 0;
    } else if (pending_boundary == 2 && bounds->writeback_start_pc == 0) {
      bounds->writeback_start_pc = pc;
      pending_boundary = 0;
    } else if (pending_boundary == 3 && bounds->behavior_end_pc == 0) {
      bounds->behavior_end_pc = pc;
      pending_boundary = 0;
    }
    if (strstr(line, "_Z12get_local_idj")) need_next_after_get_local_id = 1;
    if (line_is_barrier(line)) {
      barrier_count++;
      if (barrier_count == first_barrier) pending_boundary = 1;
      else if (barrier_count == second_barrier) pending_boundary = 2;
      else if (barrier_count == third_barrier) pending_boundary = 3;
    }
  }
  (void)pclose(pipe);
  bounds->valid = bounds->movement_start_pc && bounds->compute_start_pc &&
                  bounds->writeback_start_pc && bounds->behavior_end_pc;
  return bounds->valid ? 0 : 1;
}

static int parse_program_cycles_line(const char *line, int program_index, uint64_t *cycles)
{
  const char *program = strstr(line, "[PROGRAM");
  char *end = NULL;
  long parsed_index = 0;
  if (!program || !strstr(line, "[INST+CYCLE] active cycles")) return 1;
  parsed_index = strtol(program + 8, &end, 10);
  if (end == program + 8 || parsed_index != program_index) return 1;
  return parse_cycles_after_colon(line, cycles);
}

static void update_pc_times(const PcBoundaries *bounds, PcTimes *times,
                            uint64_t pc, uint64_t time)
{
  if (pc == bounds->movement_start_pc && !times->movement_seen) {
    times->movement_start_time = time;
    times->movement_seen = 1;
  } else if (pc == bounds->compute_start_pc && !times->compute_seen) {
    times->compute_start_time = time;
    times->compute_seen = 1;
  } else if (pc == bounds->writeback_start_pc && !times->writeback_seen) {
    times->writeback_start_time = time;
    times->writeback_seen = 1;
  } else if (pc == bounds->behavior_end_pc && !times->end_seen) {
    times->behavior_end_time = time;
    times->end_seen = 1;
  }
}

static int pc_times_complete(const PcTimes *times)
{
  return times->movement_seen && times->compute_seen &&
         times->writeback_seen && times->end_seen;
}

static int parse_child_log(const ChildRun *child, PathResult *result)
{
  FILE *f = fopen(child->log_path, "r");
  char line[1024];
  uint64_t program_cycles = 0;
  uint64_t total_cycles = 0;
  int program_index = program_index_for_path(child->kind);
  PcBoundaries bounds;
  PcTimes times;
  PathResult parsed;
  memset(&parsed, 0, sizeof(parsed));
  memset(&bounds, 0, sizeof(bounds));
  memset(&times, 0, sizeof(times));
  if (!f) return 1;
  (void)find_pc_boundaries(child, &bounds);
  while (fgets(line, sizeof(line), f)) {
    uint64_t retire_time = 0;
    uint64_t retire_pc = 0;
    if (strstr(line, "PATH_RESULT path=")) (void)parse_path_result_line(line, &parsed);
    if (parse_program_cycles_line(line, program_index, &program_cycles) != 0 &&
        strstr(line, "[TESTCASE TOTAL] [INST+CYCLE] active cycles")) {
      (void)parse_cycles_after_colon(line, &total_cycles);
    }
    if (bounds.valid && parse_retire_pc_line(line, &retire_time, &retire_pc) == 0) {
      update_pc_times(&bounds, &times, retire_pc, retire_time);
    }
  }
  fclose(f);
  if (!parsed.seen) return 1;
  *result = parsed;
  copy_text(result->log_path, sizeof(result->log_path), child->log_path);
  if (program_cycles) {
    result->kernel_cycles = program_cycles;
    result->kernel_cycle_valid = 1;
  } else if (total_cycles) {
    result->kernel_cycles = total_cycles;
    result->kernel_cycle_valid = 1;
  }
  if (bounds.valid && pc_times_complete(&times)) {
    result->movement_cycles = pc_delta_cycles(times.movement_start_time, times.compute_start_time);
    result->compute_cycles = pc_delta_cycles(times.compute_start_time, times.writeback_start_time);
    result->writeback_cycles = pc_delta_cycles(times.writeback_start_time, times.behavior_end_time);
    result->behavior_total_cycles = pc_delta_cycles(times.movement_start_time, times.behavior_end_time);
    result->behavior_cycle_source = CYCLE_SOURCE_GVM_PC_TRACE;
  }
  return 0;
}

static int wait_one_child(ChildRun *child, PathResult *result)
{
  int status = 0;
  if (waitpid(child->pid, &status, 0) < 0) {
    init_failed_result(child, -1, result);
    return 1;
  }
  child->active = 0;
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    fprintf(stderr, "child failed path=%s rows=%u cols=%u log=%s status=%d\n",
            path_arg(child->kind), child->c.rows, child->c.cols,
            child->log_path, status);
    init_failed_result(child, status, result);
    return 1;
  }
  if (parse_child_log(child, result) != 0) {
    fprintf(stderr, "failed to parse child log %s\n", child->log_path);
    init_failed_result(child, status, result);
    return 1;
  }
  result->exit_status = status;
  printf("SERIAL_CHILD_DONE status=PASS path=%s rows=%u cols=%u cycle_source=%s movement_cycles=%" PRIu64 " compute_cycles=%" PRIu64 " writeback_cycles=%" PRIu64 " behavior_total_cycles=%" PRIu64 " ns=%" PRIu64,
         path_arg(result->kind), result->c.rows, result->c.cols,
         cycle_source_text(result->behavior_cycle_source),
         result->movement_cycles, result->compute_cycles,
         result->writeback_cycles, result->behavior_total_cycles, result->ns);
  if (result->kernel_cycle_valid) printf(" kernel_cycles=%" PRIu64, result->kernel_cycles);
  printf("\n");
  return 0;
}

static int run_case(const char *prog, SerialCase c, CaseResult *out)
{
  int failed = 0;
  memset(out, 0, sizeof(*out));
  out->c = c;
  for (int i = 0; i < (int)PATH_COUNT; i++) {
    ChildRun child;
    PathResult result;
    memset(&child, 0, sizeof(child));
    memset(&result, 0, sizeof(result));
    if (launch_child(prog, (PathKind)i, c, &child) != 0) {
      child.kind = (PathKind)i;
      child.c = c;
      make_child_paths(&child);
      init_failed_result(&child, -1, &result);
      fprintf(stderr, "failed to launch child path=%s rows=%u cols=%u\n",
              path_arg((PathKind)i), c.rows, c.cols);
      failed = 1;
      out->paths[i] = result;
      continue;
    }
    if (wait_one_child(&child, &result) != 0) failed = 1;
    out->paths[i] = result;
  }
  return failed;
}

static double per_cycle(uint32_t elements, uint64_t cycles)
{
  return cycles ? (double)elements / (double)cycles : 0.0;
}

static const char *kernel_cycle_source(const PathResult *r)
{
  return r->kernel_cycle_valid ? "gvm_pmu" : "unavailable";
}

static void print_cycle_or_na(FILE *f, int valid, uint64_t value)
{
  if (valid) fprintf(f, "%" PRIu64, value);
  else fprintf(f, "NA");
}

static void print_rate_or_na(FILE *f, uint32_t elements, int valid, uint64_t cycles)
{
  if (valid && cycles) fprintf(f, "%.6f", per_cycle(elements, cycles));
  else fprintf(f, "NA");
}

static void print_result_row(FILE *f, const PathResult *r)
{
  int behavior_valid = r->passed && r->behavior_cycle_source != CYCLE_SOURCE_NONE;
  fprintf(f, "| %s | %u | %u | %u | `%s` | %s | `%s` | ",
          r->passed ? "PASS" : "FAIL", r->c.rows, r->c.cols, r->elements,
          path_arg(r->kind), path_desc(r->kind), cycle_source_text(r->behavior_cycle_source));
  print_cycle_or_na(f, behavior_valid, r->movement_cycles);
  fprintf(f, " | ");
  print_cycle_or_na(f, behavior_valid, r->compute_cycles);
  fprintf(f, " | ");
  print_cycle_or_na(f, behavior_valid, r->writeback_cycles);
  fprintf(f, " | ");
  print_cycle_or_na(f, behavior_valid, r->behavior_total_cycles);
  fprintf(f, " | `%s` | ", kernel_cycle_source(r));
  print_cycle_or_na(f, r->passed && r->kernel_cycle_valid, r->kernel_cycles);
  fprintf(f, " | ");
  print_cycle_or_na(f, r->passed, r->ns);
  fprintf(f, " | ");
  print_rate_or_na(f, r->elements, behavior_valid, r->behavior_total_cycles);
  fprintf(f, " | ");
  print_rate_or_na(f, r->elements, r->passed && r->kernel_cycle_valid, r->kernel_cycles);
  fprintf(f, " | `%s` |\n", r->log_path[0] ? r->log_path : "NA");
}

static int write_report(const CaseResult *results, size_t count)
{
  char stamp[32];
  char path[160];
  FILE *f = NULL;
  make_timestamp(stamp, sizeof(stamp));
  if (ensure_dir(LOG_DIR) != 0) return 1;
  snprintf(path, sizeof(path), LOG_DIR "/dma_tma_serial_pipeline_perf_report_%s.md", stamp);
  f = fopen(path, "w");
  if (!f) return 1;
  fprintf(f, "# TMA Serial Movement Performance Report\n\n");
  fprintf(f, "- tile_sweep: %s FP32 tiles, one work-group; tile sizes match tma_pingpong_pipeline_perf_test.\n", sweep_tile_text);
  fprintf(f, "- serial_logical_stages: 1 tile per measured kernel.\n");
  fprintf(f, "- manual_baseline: builds tma_pingpong_pipeline_perf_test.cl and runs manual_pingpong_b2_s1_kernel with buffers=2, stages=1.\n");
  fprintf(f, "- measured_order: movement -> compute -> writeback; no marker instructions are inserted into measured kernels.\n");
  fprintf(f, "- movement_paths: ordinary global load, blocking descriptor TMA G2S. TMA descriptor base setup is a separate setup kernel and is excluded from measured-kernel cycles.\n");
  fprintf(f, "- behavior_cycles: parsed out-of-band from GVM retire PC timestamps and object0.riscv disassembly.\n");
  fprintf(f, "- behavior_total_cycles: behavior_end_pc - movement_start_pc. It should equal movement + compute + writeback when GVM retire times are aligned to the cycle unit.\n");
  fprintf(f, "- kernel_cycles: whole measured-kernel active cycles parsed from GVM PMU when available.\n");
  fprintf(f, "- logs: child simulator logs and per-child build products are under this test's log/ directory.\n\n");
  fprintf(f, "SERIAL_PIPELINE_SWEEP_TABLE_BEGIN\n");
  fprintf(f, "| status | rows | cols | elements | path | movement | behavior_cycle_source | movement_cycles | compute_cycles | writeback_cycles | behavior_total_cycles | kernel_cycle_source | kernel_cycles | host_ns | elements_per_behavior_cycle | elements_per_kernel_cycle | child_log |\n");
  fprintf(f, "|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|\n");
  for (size_t i = 0; i < count; i++) {
    for (int p = 0; p < (int)PATH_COUNT; p++) print_result_row(f, &results[i].paths[p]);
  }
  fprintf(f, "SERIAL_PIPELINE_SWEEP_TABLE_END\n");
  fclose(f);
  printf("REPORT %s\n", path);
  return 0;
}

static int run_single_or_sweep(const char *prog, int sweep, SerialCase single_case)
{
  CaseResult results[sizeof(sweep_tiles) / sizeof(sweep_tiles[0])];
  size_t count = 0;
  int failed = 0;
  if (sweep) {
    for (size_t ti = 0; ti < sizeof(sweep_tiles) / sizeof(sweep_tiles[0]); ti++) {
      SerialCase c = sweep_tiles[ti];
      if (run_case(prog, c, &results[count]) != 0) failed = 1;
      for (int p = 0; p < (int)PATH_COUNT; p++) print_result_row(stdout, &results[count].paths[p]);
      count++;
    }
  } else {
    if (run_case(prog, single_case, &results[count]) != 0) failed = 1;
    for (int p = 0; p < (int)PATH_COUNT; p++) print_result_row(stdout, &results[count].paths[p]);
    count++;
  }
  if (write_report(results, count) != 0) return 1;
  return failed ? 1 : 0;
}

static void usage(const char *prog)
{
  fprintf(stderr,
          "Usage:\n"
          "  %s\n"
          "  %s sweep\n"
          "  %s single <rows> <cols>\n"
          "  %s child manual|tma <rows> <cols>\n",
          prog, prog, prog, prog);
}

int main(int argc, char **argv)
{
  SerialCase c = {DEFAULT_TILE_ROWS, DEFAULT_TILE_COLS};
  if (argc == 1) return run_single_or_sweep(argv[0], 1, c);
  if (strcmp(argv[1], "sweep") == 0 && argc == 2) {
    return run_single_or_sweep(argv[0], 1, c);
  }
  if (strcmp(argv[1], "single") == 0 && argc == 4) {
    if (parse_u32(argv[2], 1, MAX_TILE_ROWS, "rows", &c.rows) ||
        parse_u32(argv[3], 1, MAX_TILE_COLS, "cols", &c.cols) ||
        validate_case(c)) {
      return 1;
    }
    return run_single_or_sweep(argv[0], 0, c);
  }
  if (strcmp(argv[1], "child") == 0 && argc == 5) {
    PathKind kind;
    PathResult result;
    memset(&result, 0, sizeof(result));
    if (parse_path_kind(argv[2], &kind) ||
        parse_u32(argv[3], 1, MAX_TILE_ROWS, "rows", &c.rows) ||
        parse_u32(argv[4], 1, MAX_TILE_COLS, "cols", &c.cols) ||
        validate_case(c)) {
      return 1;
    }
    return run_child(kind, c, &result);
  }
  usage(argv[0]);
  return 1;
}
