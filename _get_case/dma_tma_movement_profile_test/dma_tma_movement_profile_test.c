/*
 * DMA/TMA movement profile test.
 *
 * Background:
 *   This replaces the older G2S serial movement/compute/writeback profile with
 *   a pure movement microbench.  The measured kernels no longer expose
 *   buffers/stages parameters and no longer reuse the ping-pong performance
 *   kernels.
 *
 * Implementation:
 *   The host runs paired baseline/DMA children for bulk G2S, tensor G2S, bulk
 *   S2G, tensor S2G, and tensor G2S+S2G roundtrip.  Every child executes the
 *   same measured kernel skeleton with runtime pair/use_dma selectors, so setup,
 *   timer barriers, tensor coordinate load policy, and post-copy policy stay
 *   aligned within a pair.  The kernel returns an in-kernel mcycle delta for the
 *   movement window.  S2G-only cases fill shared memory before that window
 *   because local memory cannot persist across OpenCL kernels.
 *
 * Usage:
 *   ./dma_tma_movement_profile_test.out
 *   ./dma_tma_movement_profile_test.out sweep
 *   ./dma_tma_movement_profile_test.out sweep bulk_g2s|tensor_g2s|bulk_s2g|tensor_s2g|tensor_g2s_s2g
 *   ./dma_tma_movement_profile_test.out single <mode> <rows> <cols>
 *   ./dma_tma_movement_profile_test.out child <path> <rows> <cols>
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
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
#define DESC_SETS 2u
#define COORD_WORDS 32u
#define MAX_TILE_ROWS 64u
#define MAX_TILE_COLS 64u
#define WG_SIZE 32u
#define LOG_DIR "log"
#define TEMP_DIR "/home/liyb/ventus-env/temp"

typedef enum {
  PAIR_BULK_G2S = 0,
  PAIR_TENSOR_G2S,
  PAIR_BULK_S2G,
  PAIR_TENSOR_S2G,
  PAIR_TENSOR_G2S_S2G,
  PAIR_COUNT
} PairKind;

typedef enum {
  PATH_MANUAL_G2S = 0,
  PATH_BULK_G2S,
  PATH_TENSOR_G2S,
  PATH_MANUAL_S2G,
  PATH_BULK_S2G,
  PATH_TENSOR_S2G,
  PATH_MANUAL_G2S_S2G,
  PATH_TENSOR_G2S_S2G,
  PATH_COUNT
} PathKind;

typedef struct {
  uint32_t rows;
  uint32_t cols;
} MoveCase;

typedef struct {
  PathKind path;
  PairKind pair;
  MoveCase c;
  uint32_t bytes;
  uint64_t ns;
  uint64_t cycles;
  uint64_t g2s_cycles;
  uint64_t s2g_cycles;
  uint64_t tail_cycles;
  uint32_t tma_status;
  uint64_t active_cycles;
  uint64_t total_issued;
  uint64_t data_dep_stall;
  uint64_t barrier_stall;
  uint64_t frontend_stall;
  uint64_t lsu_backpressure;
  uint64_t ibuffer_full;
  uint64_t dma_wait_stall;
  int cycle_valid;
  int segmented_valid;
  int active_valid;
  int seen;
  int passed;
  int exit_status;
  char log_path[256];
} MoveResult;

typedef struct {
  PairKind pair;
  MoveCase c;
  MoveResult baseline;
  MoveResult dma;
} PairResult;

typedef struct {
  PairKind pair;
  PathKind path;
  MoveCase c;
  pid_t pid;
  int active;
  char log_path[256];
  char log_abs[512];
  char run_dir[256];
  char run_dir_abs[512];
  char source_abs[512];
} ChildRun;

static const MoveCase sweep_tiles[] = {
  {16u, 16u},
  {32u, 16u},
  {32u, 32u},
  {64u, 32u},
  {64u, 64u},
};
static const char *sweep_tile_text = "16x16, 32x16, 32x32, 64x32, 64x64";

static const char *pair_arg(PairKind pair)
{
  switch (pair) {
  case PAIR_BULK_G2S: return "bulk_g2s";
  case PAIR_TENSOR_G2S: return "tensor_g2s";
  case PAIR_BULK_S2G: return "bulk_s2g";
  case PAIR_TENSOR_S2G: return "tensor_s2g";
  case PAIR_TENSOR_G2S_S2G: return "tensor_g2s_s2g";
  default: return "unknown";
  }
}

static const char *path_arg(PathKind path)
{
  switch (path) {
  case PATH_MANUAL_G2S: return "manual_g2s";
  case PATH_BULK_G2S: return "bulk_g2s";
  case PATH_TENSOR_G2S: return "tensor_g2s";
  case PATH_MANUAL_S2G: return "manual_s2g";
  case PATH_BULK_S2G: return "bulk_s2g";
  case PATH_TENSOR_S2G: return "tensor_s2g";
  case PATH_MANUAL_G2S_S2G: return "manual_g2s_s2g";
  case PATH_TENSOR_G2S_S2G: return "tensor_g2s_s2g";
  default: return "unknown";
  }
}

static const char *path_desc(PathKind path)
{
  switch (path) {
  case PATH_MANUAL_G2S: return "warp global load to shared";
  case PATH_BULK_G2S: return "CP_ASYNC_BULK_G2S";
  case PATH_TENSOR_G2S: return "CP_ASYNC_TENSOR_G2S";
  case PATH_MANUAL_S2G: return "warp shared store to global";
  case PATH_BULK_S2G: return "CP_ASYNC_BULK_S2G";
  case PATH_TENSOR_S2G: return "CP_ASYNC_TENSOR_S2G";
  case PATH_MANUAL_G2S_S2G: return "warp global load to shared, then shared store to global";
  case PATH_TENSOR_G2S_S2G: return "CP_ASYNC_TENSOR_G2S, then CP_ASYNC_TENSOR_S2G";
  default: return "unknown";
  }
}

static const char *kernel_name(PathKind path)
{
  (void)path;
  return "movement_profile_kernel";
}

static PairKind default_pair_for_path(PathKind path)
{
  switch (path) {
  case PATH_TENSOR_G2S: return PAIR_TENSOR_G2S;
  case PATH_TENSOR_S2G: return PAIR_TENSOR_S2G;
  case PATH_MANUAL_G2S_S2G:
  case PATH_TENSOR_G2S_S2G:
    return PAIR_TENSOR_G2S_S2G;
  case PATH_MANUAL_S2G:
  case PATH_BULK_S2G:
    return PAIR_BULK_S2G;
  case PATH_MANUAL_G2S:
  case PATH_BULK_G2S:
  default:
    return PAIR_BULK_G2S;
  }
}

static int path_is_s2g(PathKind path)
{
  return path == PATH_MANUAL_S2G ||
         path == PATH_BULK_S2G ||
         path == PATH_TENSOR_S2G;
}

static PathKind baseline_for_pair(PairKind pair)
{
  switch (pair) {
  case PAIR_BULK_G2S:
  case PAIR_TENSOR_G2S:
    return PATH_MANUAL_G2S;
  case PAIR_BULK_S2G:
  case PAIR_TENSOR_S2G:
    return PATH_MANUAL_S2G;
  case PAIR_TENSOR_G2S_S2G:
    return PATH_MANUAL_G2S_S2G;
  default:
    return PATH_MANUAL_G2S;
  }
}

static PathKind dma_for_pair(PairKind pair)
{
  switch (pair) {
  case PAIR_BULK_G2S: return PATH_BULK_G2S;
  case PAIR_TENSOR_G2S: return PATH_TENSOR_G2S;
  case PAIR_BULK_S2G: return PATH_BULK_S2G;
  case PAIR_TENSOR_S2G: return PATH_TENSOR_S2G;
  case PAIR_TENSOR_G2S_S2G: return PATH_TENSOR_G2S_S2G;
  default: return PATH_BULK_G2S;
  }
}

static int parse_pair_kind(const char *text, PairKind *pair)
{
  if (strcmp(text, "bulk_g2s") == 0) {
    *pair = PAIR_BULK_G2S;
    return 0;
  }
  if (strcmp(text, "tensor_g2s") == 0 || strcmp(text, "tma_g2s") == 0) {
    *pair = PAIR_TENSOR_G2S;
    return 0;
  }
  if (strcmp(text, "bulk_s2g") == 0) {
    *pair = PAIR_BULK_S2G;
    return 0;
  }
  if (strcmp(text, "tensor_s2g") == 0 || strcmp(text, "tma_s2g") == 0) {
    *pair = PAIR_TENSOR_S2G;
    return 0;
  }
  if (strcmp(text, "tensor_g2s_s2g") == 0 ||
      strcmp(text, "tensor_roundtrip") == 0 ||
      strcmp(text, "tma_g2s_s2g") == 0) {
    *pair = PAIR_TENSOR_G2S_S2G;
    return 0;
  }
  fprintf(stderr, "unknown mode '%s'\n", text);
  return 1;
}

static int parse_path_kind(const char *text, PathKind *path)
{
  for (int i = 0; i < (int)PATH_COUNT; i++) {
    if (strcmp(text, path_arg((PathKind)i)) == 0) {
      *path = (PathKind)i;
      return 0;
    }
  }
  fprintf(stderr, "unknown path '%s'\n", text);
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

static int validate_case(MoveCase c)
{
  if (!is_supported_tile(c.rows, c.cols)) {
    fprintf(stderr,
            "invalid case: rows=%u cols=%u; supported tiles are %s\n",
            c.rows, c.cols, sweep_tile_text);
    return 1;
  }
  return 0;
}

static uint32_t case_words(MoveCase c)
{
  return c.rows * c.cols;
}

static uint32_t case_bytes(MoveCase c)
{
  return case_words(c) * (uint32_t)sizeof(uint32_t);
}

static uint32_t desc_control(unsigned data_type, unsigned rank)
{
  return (data_type & 0x1fu) | ((rank & 0x7u) << 5);
}

static void build_desc(uint32_t *desc, MoveCase c)
{
  memset(desc, 0, DESC_WORDS * sizeof(uint32_t));
  desc[0] = 0x544d4103u;
  desc[1] = desc_control(7u, 2u);
  desc[4] = c.cols;
  desc[5] = c.rows;
  desc[9] = c.cols * sizeof(uint32_t);
  desc[17] = c.cols;
  desc[18] = c.rows;
  desc[22] = 1u;
  desc[23] = 1u;
}

static uint32_t pattern_word(uint32_t idx)
{
  uint32_t base = idx << 2;
  uint32_t b0 = (base * 7u + 0x23u) & 0xffu;
  uint32_t b1 = ((base + 1u) * 7u + 0x23u) & 0xffu;
  uint32_t b2 = ((base + 2u) * 7u + 0x23u) & 0xffu;
  uint32_t b3 = ((base + 3u) * 7u + 0x23u) & 0xffu;
  return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

static void fill_input(uint32_t *input, uint32_t words)
{
  for (uint32_t i = 0; i < words; i++) input[i] = pattern_word(i);
}

static int check_linear_copy(const char *label, const uint32_t *got, uint32_t words)
{
  uint32_t bad = 0;
  for (uint32_t i = 0; i < words; i++) {
    uint32_t exp = pattern_word(i);
    if (got[i] != exp) {
      if (bad < 8u) {
        fprintf(stderr, "FAIL %s word%u got=0x%08x exp=0x%08x\n",
                label, i, got[i], exp);
      }
      bad++;
    }
  }
  if (bad) {
    fprintf(stderr, "FAIL %s bad=%u/%u\n", label, bad, words);
    return 1;
  }
  printf("PASS %s words=%u\n", label, words);
  return 0;
}

static int check_result(PathKind path, const uint32_t *got, uint32_t words)
{
  return check_linear_copy(path_arg(path), got, words);
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
                                         MoveCase c,
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

  char source_dir[PATH_MAX];
  snprintf(source_dir, sizeof(source_dir), "%s", source_path);
  char *slash = strrchr(source_dir, '/');
  if (slash) *slash = '\0';
  else snprintf(source_dir, sizeof(source_dir), ".");
  char options[PATH_MAX + 160];
  snprintf(options, sizeof(options),
           "-I%s/../common -DTILE_ROWS=%u -DTILE_COLS=%u",
           source_dir, c.rows, c.cols);
  err = clBuildProgram(prog, 1, &device, options, NULL, NULL);
  if (err != CL_SUCCESS) {
    ventus_print_build_log(prog, device);
    clReleaseProgram(prog);
    return err;
  }
  *program = prog;
  return CL_SUCCESS;
}

static int run_setup_kernel(cl_command_queue queue,
                            cl_kernel setup_kernel,
                            cl_mem desc_buf,
                            cl_mem input_buf,
                            cl_mem output_buf,
                            uint32_t desc_base_select)
{
  cl_int err = CL_SUCCESS;
  size_t global = 1;
  err = clSetKernelArg(setup_kernel, 0, sizeof(desc_buf), &desc_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(setup.desc)");
  err = clSetKernelArg(setup_kernel, 1, sizeof(input_buf), &input_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(setup.input)");
  err = clSetKernelArg(setup_kernel, 2, sizeof(output_buf), &output_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(setup.output)");
  err = clSetKernelArg(setup_kernel, 3, sizeof(desc_base_select), &desc_base_select);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(setup.select)");
  err = clEnqueueNDRangeKernel(queue, setup_kernel, 1, NULL, &global, NULL,
                               0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel(setup)");
  return 0;
FINISH:
  return 1;
}

static int run_measured_kernel(cl_command_queue queue,
                               cl_kernel kernel,
                               cl_mem desc_buf,
                               cl_mem coords_buf,
                               cl_mem input_buf,
                               cl_mem output_buf,
                               cl_mem cycles_buf,
                               PathKind path,
                               PairKind pair,
                               uint64_t *ns,
                               uint32_t cycles[5])
{
  cl_int err = CL_SUCCESS;
  cl_event event = NULL;
  uint32_t zero_cycles[5] = {0, 0, 0, 0, 0};
  size_t global = WG_SIZE;
  size_t local = WG_SIZE;
  err = clEnqueueWriteBuffer(queue, cycles_buf, CL_TRUE, 0,
                             sizeof(zero_cycles), zero_cycles, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueWriteBuffer(cycles.zero)");
  err = clSetKernelArg(kernel, 0, sizeof(desc_buf), &desc_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(measured.desc)");
  err = clSetKernelArg(kernel, 1, sizeof(coords_buf), &coords_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(measured.coords)");
  err = clSetKernelArg(kernel, 2, sizeof(input_buf), &input_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(measured.input)");
  err = clSetKernelArg(kernel, 3, sizeof(output_buf), &output_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(measured.output)");
  err = clSetKernelArg(kernel, 4, sizeof(cycles_buf), &cycles_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(measured.cycles)");
  {
    cl_uint path_arg_value = (cl_uint)path;
    cl_uint pair_arg_value = (cl_uint)pair;
    cl_uint use_dma_arg_value = (path == dma_for_pair(pair)) ? 1u : 0u;
    err = clSetKernelArg(kernel, 5, sizeof(path_arg_value), &path_arg_value);
    CHECK_OPENCL_ERROR_IN("clSetKernelArg(measured.path)");
    err = clSetKernelArg(kernel, 6, sizeof(pair_arg_value), &pair_arg_value);
    CHECK_OPENCL_ERROR_IN("clSetKernelArg(measured.pair)");
    err = clSetKernelArg(kernel, 7, sizeof(use_dma_arg_value), &use_dma_arg_value);
    CHECK_OPENCL_ERROR_IN("clSetKernelArg(measured.use_dma)");
  }
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                               0, NULL, &event);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel(measured)");
  err = clWaitForEvents(1, &event);
  CHECK_OPENCL_ERROR_IN("clWaitForEvents(measured)");
  if (event_duration_ns(event, ns) != 0) {
    fprintf(stderr, "failed to read measured event duration\n");
    goto FINISH;
  }
  err = clEnqueueReadBuffer(queue, cycles_buf, CL_TRUE, 0,
                            sizeof(zero_cycles), cycles, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(cycles)");
  clReleaseEvent(event);
  return 0;
FINISH:
  if (event) clReleaseEvent(event);
  return 1;
}

static int run_child(PairKind pair, PathKind path, MoveCase c, const char *source_path)
{
  cl_int err = CL_SUCCESS;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel setup_kernel = NULL;
  cl_kernel measured_kernel = NULL;
  cl_mem desc_buf = NULL;
  cl_mem coords_buf = NULL;
  cl_mem input_buf = NULL;
  cl_mem output_buf = NULL;
  cl_mem cycles_buf = NULL;
  uint32_t *input = NULL;
  uint32_t *got = NULL;
  uint32_t desc[DESC_WORDS * DESC_SETS];
  uint32_t coords[COORD_WORDS];
  uint32_t cycles[5] = {0, 0, 0, 0, 0};
  uint32_t cycle_init[5] = {0, 0, 0, 0, 0};
  uint32_t words = case_words(c);
  uint32_t bytes = case_bytes(c);
  uint64_t ns = 0;
  int exit_code = 1;
  uint32_t desc_base_select = path_is_s2g(path) ? 1u : 0u;

  printf("CASE path=%s rows=%u cols=%u words=%u bytes=%u wg_size=%u\n",
         path_arg(path), c.rows, c.cols, words, bytes, WG_SIZE);
  printf("CASE_PAIR mode=%s path=%s same_skeleton=1\n",
         pair_arg(pair), path_arg(path));

  input = (uint32_t *)malloc(bytes);
  got = (uint32_t *)malloc(bytes);
  if (!input || !got) goto FINISH;
  fill_input(input, words);
  memset(got, 0xcd, bytes);
  build_desc(desc, c);
  build_desc(desc + DESC_WORDS, c);
  memset(coords, 0, sizeof(coords));

  err = ventus_get_default_device(&context, &device, &queue, NULL);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device");
  clReleaseCommandQueue(queue);
  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
  CHECK_OPENCL_ERROR_IN("clCreateCommandQueue(profiled)");

  err = build_program_with_options(context, device, source_path, c, &program);
  CHECK_OPENCL_ERROR_IN("build_program_with_options");
  measured_kernel = clCreateKernel(program, kernel_name(path), &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(measured)");
  setup_kernel = clCreateKernel(program, "setup_desc_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(setup_desc)");

  desc_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            sizeof(desc), desc, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(desc)");
  coords_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(coords), coords, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(coords)");
  input_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             bytes, input, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(input)");
  output_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              bytes, got, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(output)");
  cycles_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              sizeof(cycle_init), cycle_init, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(cycles)");

  if (run_setup_kernel(queue, setup_kernel, desc_buf, input_buf, output_buf,
                       desc_base_select) != 0) goto FINISH;
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(setup)");
  if (run_measured_kernel(queue, measured_kernel, desc_buf, coords_buf,
                          input_buf, output_buf, cycles_buf, path, pair,
                          &ns, cycles) != 0) goto FINISH;
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(measured)");
  err = clEnqueueReadBuffer(queue, output_buf, CL_TRUE, 0, bytes, got,
                            0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(output)");
  if (check_result(path, got, words) != 0) goto FINISH;

  if (cycles[0] == 0u) {
    fprintf(stderr, "FAIL %s movement cycle counter returned zero\n", path_arg(path));
    goto FINISH;
  }
  if (cycles[4] != 0u) {
    fprintf(stderr, "FAIL %s TMA status=0x%x\n", path_arg(path), cycles[4]);
    goto FINISH;
  }

  printf("MOVE_RESULT path=%s rows=%u cols=%u bytes=%u cycles=%u ns=%" PRIu64
         " status=0x%x",
         path_arg(path), c.rows, c.cols, bytes, cycles[0], ns, cycles[4]);
  if (cycles[1] || cycles[2] || cycles[3]) {
    printf(" g2s_cycles=%u s2g_cycles=%u tail_cycles=%u",
           cycles[1], cycles[2], cycles[3]);
  }
  printf("\n");
  exit_code = 0;

FINISH:
  if (cycles_buf) clReleaseMemObject(cycles_buf);
  if (output_buf) clReleaseMemObject(output_buf);
  if (input_buf) clReleaseMemObject(input_buf);
  if (coords_buf) clReleaseMemObject(coords_buf);
  if (desc_buf) clReleaseMemObject(desc_buf);
  if (setup_kernel) clReleaseKernel(setup_kernel);
  if (measured_kernel) clReleaseKernel(measured_kernel);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
  free(got);
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

static int make_abs_path(char *out, size_t out_size, const char *cwd,
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

static void make_child_paths(ChildRun *child, const char *cwd)
{
  char stamp[32];
  struct timespec ts;
  long nsec = 0;
  make_timestamp(stamp, sizeof(stamp));
  if (clock_gettime(CLOCK_REALTIME, &ts) == 0) nsec = ts.tv_nsec;
  snprintf(child->log_path, sizeof(child->log_path),
           LOG_DIR "/dma_tma_movement_profile_sameskeleton_child_%s_%09ld_%s_%s_r%u_c%u.log",
           stamp, nsec, pair_arg(child->pair), path_arg(child->path),
           child->c.rows, child->c.cols);
  snprintf(child->run_dir, sizeof(child->run_dir),
           LOG_DIR "/dma_tma_movement_profile_sameskeleton_run_%s_%09ld_%s_%s_r%u_c%u",
           stamp, nsec, pair_arg(child->pair), path_arg(child->path),
           child->c.rows, child->c.cols);
  make_abs_path(child->log_abs, sizeof(child->log_abs), cwd, child->log_path);
  make_abs_path(child->run_dir_abs, sizeof(child->run_dir_abs), cwd, child->run_dir);
  make_abs_path(child->source_abs, sizeof(child->source_abs), cwd,
                "dma_tma_movement_profile_test.cl");
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

static int parse_program_cycles_line(const char *line, uint64_t *cycles)
{
  if (!strstr(line, "[PROGRAM") ||
      !strstr(line, "[INST+CYCLE] active cycles")) return 1;
  return parse_cycles_after_colon(line, cycles);
}

static int parse_move_result_line(const char *line, MoveResult *result)
{
  char path_text[64];
  unsigned rows = 0;
  unsigned cols = 0;
  unsigned bytes = 0;
  unsigned cycles = 0;
  unsigned g2s_cycles = 0;
  unsigned s2g_cycles = 0;
  unsigned tail_cycles = 0;
  unsigned tma_status = 0;
  unsigned long long ns = 0;
  PathKind path;
  if (sscanf(line,
             "MOVE_RESULT path=%63s rows=%u cols=%u bytes=%u cycles=%u ns=%llu",
             path_text, &rows, &cols, &bytes, &cycles, &ns) != 6) {
    return 1;
  }
  if (parse_path_kind(path_text, &path) != 0) return 1;
  result->path = path;
  result->c.rows = rows;
  result->c.cols = cols;
  result->bytes = bytes;
  result->cycles = cycles;
  result->cycle_valid = cycles != 0u;
  result->ns = (uint64_t)ns;
  const char *status_text = strstr(line, "status=");
  if (status_text) sscanf(status_text, "status=%x", &tma_status);
  result->tma_status = tma_status;
  const char *seg = strstr(line, "g2s_cycles=");
  if (seg && sscanf(seg, "g2s_cycles=%u s2g_cycles=%u tail_cycles=%u",
                    &g2s_cycles, &s2g_cycles, &tail_cycles) == 3) {
    result->g2s_cycles = g2s_cycles;
    result->s2g_cycles = s2g_cycles;
    result->tail_cycles = tail_cycles;
    result->segmented_valid = 1;
  }
  result->seen = 1;
  result->passed = 1;
  return 0;
}

static void parse_pmu_line(const char *line, MoveResult *result)
{
  uint64_t value = 0;
  if (!strstr(line, "[PROGRAM")) return;
  if (strstr(line, "[INST+CYCLE] active cycles")) {
    if (parse_cycles_after_colon(line, &value) == 0) {
      result->active_cycles = value;
      result->active_valid = 1;
    }
  } else if (strstr(line, "[INST+CYCLE] total issued")) {
    if (parse_cycles_after_colon(line, &value) == 0) result->total_issued = value;
  } else if (strstr(line, "[STALL] data dependency")) {
    if (parse_cycles_after_colon(line, &value) == 0) result->data_dep_stall = value;
  } else if (strstr(line, "[STALL] barrier stall cycles")) {
    if (parse_cycles_after_colon(line, &value) == 0) result->barrier_stall = value;
  } else if (strstr(line, "[STALL] frontend stall cycles")) {
    if (parse_cycles_after_colon(line, &value) == 0) result->frontend_stall = value;
  } else if (strstr(line, "[STALL] lsu backpressure cyc")) {
    if (parse_cycles_after_colon(line, &value) == 0) result->lsu_backpressure = value;
  } else if (strstr(line, "[STALL] ibuffer full cycles")) {
    if (parse_cycles_after_colon(line, &value) == 0) result->ibuffer_full = value;
  } else if (strstr(line, "[STALL] dma fence/group wait")) {
    if (parse_cycles_after_colon(line, &value) == 0) result->dma_wait_stall = value;
  }
}

static uint64_t other_cycles(const MoveResult *r)
{
  if (!r->active_valid || !r->cycle_valid) return 0;
  if (r->active_cycles < r->cycles) return 0;
  return r->active_cycles - r->cycles;
}

static double pct_u64(uint64_t part, uint64_t total)
{
  if (!total) return 0.0;
  return 100.0 * (double)part / (double)total;
}

static uint64_t pmu_unclassified_cycles(const MoveResult *r)
{
  uint64_t known = r->data_dep_stall + r->barrier_stall + r->frontend_stall +
                   r->lsu_backpressure + r->ibuffer_full + r->dma_wait_stall;
  if (!r->active_valid || known >= r->active_cycles) return 0;
  return r->active_cycles - known;
}

static void init_failed_result(const ChildRun *child, int status, MoveResult *result)
{
  memset(result, 0, sizeof(*result));
  result->pair = child->pair;
  result->path = child->path;
  result->c = child->c;
  result->bytes = case_bytes(child->c);
  result->exit_status = status;
  result->passed = 0;
  copy_text(result->log_path, sizeof(result->log_path), child->log_path);
}

static int parse_child_log(const ChildRun *child, MoveResult *result)
{
  FILE *f = fopen(child->log_abs, "r");
  char line[1024];
  MoveResult parsed;
  memset(&parsed, 0, sizeof(parsed));
  parsed.pair = child->pair;
  parsed.path = child->path;
  parsed.c = child->c;
  parsed.bytes = case_bytes(child->c);
  if (!f) return 1;
  while (fgets(line, sizeof(line), f)) {
    if (strstr(line, "MOVE_RESULT path=")) {
      (void)parse_move_result_line(line, &parsed);
    } else {
      parse_pmu_line(line, &parsed);
    }
  }
  fclose(f);
  if (!parsed.seen) return 1;
  *result = parsed;
  result->pair = child->pair;
  result->path = child->path;
  result->c = child->c;
  result->bytes = case_bytes(child->c);
  copy_text(result->log_path, sizeof(result->log_path), child->log_path);
  return 0;
}

static int wait_child(ChildRun *child, MoveResult *result)
{
  int status = 0;
  if (waitpid(child->pid, &status, 0) < 0) {
    init_failed_result(child, -1, result);
    return 1;
  }
  child->active = 0;
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    fprintf(stderr, "child failed path=%s rows=%u cols=%u log=%s status=%d\n",
            path_arg(child->path), child->c.rows, child->c.cols,
            child->log_path, status);
    init_failed_result(child, status, result);
    return 1;
  }
  if (parse_child_log(child, result) != 0) {
    fprintf(stderr, "failed to parse child log path=%s log=%s\n",
            path_arg(child->path), child->log_path);
    init_failed_result(child, status, result);
    return 1;
  }
  return 0;
}

static int spawn_child(PairKind pair, PathKind path, MoveCase c,
                       const char *cwd, ChildRun *child)
{
  memset(child, 0, sizeof(*child));
  child->pair = pair;
  child->path = path;
  child->c = c;
  make_child_paths(child, cwd);
  if (ensure_dir(LOG_DIR) != 0 || ensure_dir(child->run_dir) != 0) {
    perror("mkdir");
    return 1;
  }

  printf("MOVE_CHILD_START mode=%s path=%s rows=%u cols=%u log=%s run_dir=%s\n",
         pair_arg(pair), path_arg(path), c.rows, c.cols,
         child->log_path, child->run_dir);
  fflush(stdout);

  pid_t pid = fork();
  if (pid < 0) {
    perror("fork");
    return 1;
  }
  if (pid == 0) {
    char cache_dir[640];
    char tmp_dir[640];
    int fd = open(child->log_abs, O_CREAT | O_TRUNC | O_WRONLY, 0664);
    if (fd < 0) _exit(127);
    (void)dup2(fd, STDOUT_FILENO);
    (void)dup2(fd, STDERR_FILENO);
    close(fd);
    snprintf(cache_dir, sizeof(cache_dir), "%s/pocl-cache", child->run_dir_abs);
    snprintf(tmp_dir, sizeof(tmp_dir), "%s/tmp", child->run_dir_abs);
    (void)mkdir(cache_dir, 0775);
    (void)mkdir(tmp_dir, 0775);
    setenv("POCL_CACHE_DIR", cache_dir, 1);
    setenv("TMPDIR", tmp_dir, 1);
    setenv("TMP", tmp_dir, 1);
    setenv("TEMP", tmp_dir, 1);
    if (chdir(child->run_dir_abs) != 0) _exit(126);
    int code = run_child(pair, path, c, child->source_abs);
    fflush(stdout);
    fflush(stderr);
    _exit(code);
  }
  child->pid = pid;
  child->active = 1;
  return 0;
}

static void print_result_line(FILE *f, const MoveResult *r,
                              const MoveResult *baseline)
{
  double bytes_per_cycle = 0.0;
  double speedup = 0.0;
  uint64_t other = other_cycles(r);
  if (r->cycle_valid && r->cycles) {
    bytes_per_cycle = (double)r->bytes / (double)r->cycles;
  }
  if (baseline && baseline->cycle_valid && r->cycle_valid && r->cycles) {
    speedup = (double)baseline->cycles / (double)r->cycles;
  }

  fprintf(f, "| %s | `%s` | %u | %u | %u | `%s` | %s | ",
          r->passed ? "PASS" : "FAIL", pair_arg(r->pair),
          r->c.rows, r->c.cols, r->bytes, path_arg(r->path), path_desc(r->path));
  if (r->cycle_valid) fprintf(f, "%" PRIu64, r->cycles);
  else fprintf(f, "NA");
  fprintf(f, " | ");
  if (r->active_valid) fprintf(f, "%" PRIu64, r->active_cycles);
  else fprintf(f, "NA");
  fprintf(f, " | ");
  if (r->active_valid && r->cycle_valid) fprintf(f, "%" PRIu64, other);
  else fprintf(f, "NA");
  fprintf(f, " | ");
  if (r->active_valid && r->cycle_valid) fprintf(f, "%.2f%%", pct_u64(r->cycles, r->active_cycles));
  else fprintf(f, "NA");
  fprintf(f, " | ");
  if (r->active_valid && r->cycle_valid) fprintf(f, "%.2f%%", pct_u64(other, r->active_cycles));
  else fprintf(f, "NA");
  fprintf(f, " | %" PRIu64 " | ", r->ns);
  if (r->cycle_valid) fprintf(f, "%.6f", bytes_per_cycle);
  else fprintf(f, "NA");
  fprintf(f, " | ");
  if (speedup > 0.0) fprintf(f, "%.4fx", speedup);
  else fprintf(f, "baseline");
  fprintf(f, " | `%s` |\n", r->log_path[0] ? r->log_path : "NA");
}

static int run_pair_case(PairKind pair, MoveCase c, PairResult *result)
{
  char cwd[512];
  ChildRun child;
  int failed = 0;
  if (!getcwd(cwd, sizeof(cwd))) return 1;
  memset(result, 0, sizeof(*result));
  result->pair = pair;
  result->c = c;

  if (spawn_child(pair, baseline_for_pair(pair), c, cwd, &child) != 0) return 1;
  if (wait_child(&child, &result->baseline) != 0) failed = 1;
  printf("MOVE_CHILD_DONE status=%s mode=%s path=%s rows=%u cols=%u",
         result->baseline.passed ? "PASS" : "FAIL", pair_arg(pair),
         path_arg(result->baseline.path), c.rows, c.cols);
  if (result->baseline.cycle_valid) {
    printf(" cycles=%" PRIu64, result->baseline.cycles);
  }
  if (result->baseline.segmented_valid) {
    printf(" g2s=%" PRIu64 " s2g=%" PRIu64 " tail=%" PRIu64,
           result->baseline.g2s_cycles, result->baseline.s2g_cycles,
           result->baseline.tail_cycles);
  }
  if (result->baseline.active_valid) {
    printf(" active=%" PRIu64 " other=%" PRIu64,
           result->baseline.active_cycles, other_cycles(&result->baseline));
  }
  printf(" ns=%" PRIu64 "\n", result->baseline.ns);

  if (spawn_child(pair, dma_for_pair(pair), c, cwd, &child) != 0) return 1;
  if (wait_child(&child, &result->dma) != 0) failed = 1;
  printf("MOVE_CHILD_DONE status=%s mode=%s path=%s rows=%u cols=%u",
         result->dma.passed ? "PASS" : "FAIL", pair_arg(pair),
         path_arg(result->dma.path), c.rows, c.cols);
  if (result->dma.cycle_valid) {
    printf(" cycles=%" PRIu64, result->dma.cycles);
  }
  if (result->dma.segmented_valid) {
    printf(" g2s=%" PRIu64 " s2g=%" PRIu64 " tail=%" PRIu64,
           result->dma.g2s_cycles, result->dma.s2g_cycles,
           result->dma.tail_cycles);
  }
  if (result->dma.active_valid) {
    printf(" active=%" PRIu64 " other=%" PRIu64,
           result->dma.active_cycles, other_cycles(&result->dma));
  }
  printf(" ns=%" PRIu64, result->dma.ns);
  if (result->baseline.cycle_valid && result->dma.cycle_valid &&
      result->dma.cycles) {
    printf(" speedup=%.4fx", (double)result->baseline.cycles /
                             (double)result->dma.cycles);
  }
  printf("\n");
  print_result_line(stdout, &result->baseline, NULL);
  print_result_line(stdout, &result->dma, &result->baseline);
  return failed;
}

static void write_report_body(FILE *f, const PairResult *results, size_t count)
{
  fprintf(f, "# DMA/TMA Same-Skeleton Movement Profile Report\n\n");
  fprintf(f, "- tile_sweep: %s FP32/uint32 tiles, one work-group.\n", sweep_tile_text);
  fprintf(f, "- measured_scope: one shared measured kernel skeleton; runtime pair/use_dma selectors choose the movement branch.\n");
  fprintf(f, "- compared_pairs: bulk_g2s, tensor_g2s, bulk_s2g, tensor_s2g, tensor_g2s_s2g.\n");
  fprintf(f, "- baseline_policy: G2S compares against warp global-load-to-shared; S2G compares against warp shared-store-to-global with the same tile size.\n");
  fprintf(f, "- roundtrip_policy: tensor_g2s_s2g compares manual global->shared->global against tensor G2S followed by tensor S2G in the same timed window.\n");
  fprintf(f, "- tensor_setup_policy: tensor pair baselines execute the same coordinate-load setup as tensor DMA paths before timing.\n");
  fprintf(f, "- s2g_setup_note: S2G paths initialize their shared source before the timed movement window.\n");
  fprintf(f, "- cycle_source: window_cycles is in-kernel mcycle delta; active_cycles is RTL/GVM PMU PROGRAM1 active cycles parsed from child logs; host_ns is diagnostic only.\n");
  fprintf(f, "- other_cycles: active_cycles - window_cycles. It includes common setup, timer barriers, branch/control overhead, G2S validation copy, S2G source fill, and kernel prologue/epilogue.\n\n");

  fprintf(f, "MOVEMENT_PROFILE_SUMMARY_BEGIN\n");
  fprintf(f, "| mode | rows | cols | bytes | baseline path | baseline window | baseline active | baseline other | dma path | dma window | dma active | dma other | window speedup | active speedup |\n");
  fprintf(f, "|---|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|\n");
  for (size_t i = 0; i < count; i++) {
    const MoveResult *b = &results[i].baseline;
    const MoveResult *d = &results[i].dma;
    fprintf(f, "| `%s` | %u | %u | %u | `%s` | ",
            pair_arg(results[i].pair), results[i].c.rows,
            results[i].c.cols, case_bytes(results[i].c), path_arg(b->path));
    if (b->cycle_valid) fprintf(f, "%" PRIu64, b->cycles);
    else fprintf(f, "NA");
    fprintf(f, " | ");
    if (b->active_valid) fprintf(f, "%" PRIu64, b->active_cycles);
    else fprintf(f, "NA");
    fprintf(f, " | ");
    if (b->active_valid && b->cycle_valid) fprintf(f, "%" PRIu64, other_cycles(b));
    else fprintf(f, "NA");
    fprintf(f, " | `%s` | ", path_arg(d->path));
    if (d->cycle_valid) fprintf(f, "%" PRIu64, d->cycles);
    else fprintf(f, "NA");
    fprintf(f, " | ");
    if (d->active_valid) fprintf(f, "%" PRIu64, d->active_cycles);
    else fprintf(f, "NA");
    fprintf(f, " | ");
    if (d->active_valid && d->cycle_valid) fprintf(f, "%" PRIu64, other_cycles(d));
    else fprintf(f, "NA");
    fprintf(f, " | ");
    if (b->cycle_valid && d->cycle_valid && d->cycles) {
      fprintf(f, "%.4fx", (double)b->cycles / (double)d->cycles);
    } else {
      fprintf(f, "NA");
    }
    fprintf(f, " | ");
    if (b->active_valid && d->active_valid && d->active_cycles) {
      fprintf(f, "%.4fx", (double)b->active_cycles / (double)d->active_cycles);
    } else {
      fprintf(f, "NA");
    }
    fprintf(f, " |\n");
  }
  fprintf(f, "MOVEMENT_PROFILE_SUMMARY_END\n\n");

  fprintf(f, "MOVEMENT_PROFILE_DETAIL_BEGIN\n");
  fprintf(f, "| status | mode | rows | cols | bytes | path | movement | window_cycles | active_cycles | other_cycles | window_pct | other_pct | host_ns | bytes_per_cycle | speedup_vs_baseline | child_log |\n");
  fprintf(f, "|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n");
  for (size_t i = 0; i < count; i++) {
    print_result_line(f, &results[i].baseline, NULL);
    print_result_line(f, &results[i].dma, &results[i].baseline);
  }
  fprintf(f, "MOVEMENT_PROFILE_DETAIL_END\n\n");

  fprintf(f, "MOVEMENT_PROFILE_SEGMENTS_BEGIN\n");
  fprintf(f, "Segment cycles are only emitted by paired G2S+S2G roundtrip modes. total_window is the same movement window reported in the detail table.\n\n");
  fprintf(f, "| mode | rows | cols | path | total_window | g2s_segment | s2g_segment | tail_segment | g2s_pct | s2g_pct | tail_pct | active | active_minus_total |\n");
  fprintf(f, "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
  for (size_t i = 0; i < count; i++) {
    const MoveResult *rows[2] = {&results[i].baseline, &results[i].dma};
    for (int ri = 0; ri < 2; ri++) {
      const MoveResult *r = rows[ri];
      if (!r->segmented_valid) continue;
      fprintf(f, "| `%s` | %u | %u | `%s` | %" PRIu64 " | %" PRIu64 " | %" PRIu64 " | %" PRIu64 " | %.2f%% | %.2f%% | %.2f%% | ",
              pair_arg(r->pair), r->c.rows, r->c.cols, path_arg(r->path),
              r->cycles, r->g2s_cycles, r->s2g_cycles, r->tail_cycles,
              pct_u64(r->g2s_cycles, r->cycles),
              pct_u64(r->s2g_cycles, r->cycles),
              pct_u64(r->tail_cycles, r->cycles));
      if (r->active_valid) {
        fprintf(f, "%" PRIu64 " | %" PRIu64 " |\n",
                r->active_cycles, other_cycles(r));
      } else {
        fprintf(f, "NA | NA |\n");
      }
    }
  }
  fprintf(f, "MOVEMENT_PROFILE_SEGMENTS_END\n\n");

  fprintf(f, "MOVEMENT_PROFILE_ACTIVE_BREAKDOWN_BEGIN\n");
  fprintf(f, "Approximate PMU proportions are whole measured-kernel counters, not window-scoped counters. They explain where active_cycles broadly went after enforcing the common skeleton.\n\n");
  fprintf(f, "| mode | rows | cols | path | active | window_pct | active_minus_window_pct | frontend_pct | data_dep_pct | dma_wait_pct | ibuffer_pct | barrier_pct | lsu_pct | pmu_other_pct | issued |\n");
  fprintf(f, "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
  for (size_t i = 0; i < count; i++) {
    const MoveResult *rows[2] = {&results[i].baseline, &results[i].dma};
    for (int ri = 0; ri < 2; ri++) {
      const MoveResult *r = rows[ri];
      uint64_t active = r->active_cycles;
      fprintf(f, "| `%s` | %u | %u | `%s` | ",
              pair_arg(r->pair), r->c.rows, r->c.cols, path_arg(r->path));
      if (r->active_valid) fprintf(f, "%" PRIu64, active);
      else fprintf(f, "NA");
      if (r->active_valid && r->cycle_valid) {
        fprintf(f, " | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %" PRIu64 " |\n",
                pct_u64(r->cycles, active),
                pct_u64(other_cycles(r), active),
                pct_u64(r->frontend_stall, active),
                pct_u64(r->data_dep_stall, active),
                pct_u64(r->dma_wait_stall, active),
                pct_u64(r->ibuffer_full, active),
                pct_u64(r->barrier_stall, active),
                pct_u64(r->lsu_backpressure, active),
                pct_u64(pmu_unclassified_cycles(r), active),
                r->total_issued);
      } else {
        fprintf(f, " | NA | NA | NA | NA | NA | NA | NA | NA | NA | %" PRIu64 " |\n",
                r->total_issued);
      }
    }
  }
  fprintf(f, "MOVEMENT_PROFILE_ACTIVE_BREAKDOWN_END\n");
}

static int write_report(const PairResult *results, size_t count)
{
  char stamp[32];
  char path[192];
  char temp_path[256];
  const char *backend = getenv("VENTUS_BACKEND");
  FILE *f = NULL;
  make_timestamp(stamp, sizeof(stamp));
  if (ensure_dir(LOG_DIR) != 0) return 1;
  if (ensure_dir(TEMP_DIR) != 0) return 1;

  snprintf(path, sizeof(path), LOG_DIR "/dma_tma_movement_profile_sameskeleton_report_%s.md", stamp);
  snprintf(temp_path, sizeof(temp_path),
           TEMP_DIR "/dma_tma_movement_profile_sameskeleton_%s_%s.md",
           backend && backend[0] ? backend : "default", stamp);

  f = fopen(path, "w");
  if (!f) return 1;
  write_report_body(f, results, count);
  fclose(f);

  f = fopen(temp_path, "w");
  if (!f) return 1;
  write_report_body(f, results, count);
  fclose(f);

  printf("REPORT %s\n", path);
  printf("TEMP_REPORT %s\n", temp_path);
  return 0;
}

static int run_sweep_or_single(int sweep, int filter_valid, PairKind filter_pair,
                               MoveCase single_case)
{
  PairResult results[PAIR_COUNT * (sizeof(sweep_tiles) / sizeof(sweep_tiles[0]))];
  size_t count = 0;
  int failed = 0;
  if (sweep) {
    for (int p = 0; p < (int)PAIR_COUNT; p++) {
      PairKind pair = (PairKind)p;
      if (filter_valid && pair != filter_pair) continue;
      for (size_t ti = 0; ti < sizeof(sweep_tiles) / sizeof(sweep_tiles[0]); ti++) {
        if (run_pair_case(pair, sweep_tiles[ti], &results[count]) != 0) failed = 1;
        count++;
      }
    }
  } else {
    if (run_pair_case(filter_pair, single_case, &results[count]) != 0) failed = 1;
    count++;
  }
  if (write_report(results, count) != 0) failed = 1;
  return failed ? 1 : 0;
}

static void usage(const char *prog)
{
  fprintf(stderr,
          "Usage:\n"
          "  %s\n"
          "  %s sweep [bulk_g2s|tensor_g2s|bulk_s2g|tensor_s2g|tensor_g2s_s2g]\n"
          "  %s single <mode> <rows> <cols>\n"
          "  %s child <path> <rows> <cols>\n"
          "  %s child <mode> <path> <rows> <cols>\n",
          prog, prog, prog, prog, prog);
}

int main(int argc, char **argv)
{
  if (ensure_dir(LOG_DIR) != 0) {
    perror("mkdir log");
    return 1;
  }

  if (argc == 1 || (argc >= 2 && strcmp(argv[1], "sweep") == 0)) {
    PairKind pair = PAIR_BULK_G2S;
    int filter_valid = 0;
    if (argc == 3) {
      if (parse_pair_kind(argv[2], &pair) != 0) return 1;
      filter_valid = 1;
    } else if (argc > 3) {
      usage(argv[0]);
      return 1;
    }
    return run_sweep_or_single(1, filter_valid, pair, (MoveCase){0u, 0u});
  }

  if (argc == 5 && strcmp(argv[1], "single") == 0) {
    PairKind pair;
    MoveCase c;
    if (parse_pair_kind(argv[2], &pair) ||
        parse_u32(argv[3], 1u, MAX_TILE_ROWS, "rows", &c.rows) ||
        parse_u32(argv[4], 1u, MAX_TILE_COLS, "cols", &c.cols) ||
        validate_case(c)) {
      return 1;
    }
    return run_sweep_or_single(0, 1, pair, c);
  }

  if ((argc == 5 || argc == 6) && strcmp(argv[1], "child") == 0) {
    PairKind pair;
    PathKind path;
    MoveCase c;
    char cwd[512];
    char source_abs[512];
    const char *source_env = getenv("VENTUS_DMA_TMA_MOVEMENT_SOURCE");
    if (argc == 6) {
      if (parse_pair_kind(argv[2], &pair) ||
          parse_path_kind(argv[3], &path) ||
          parse_u32(argv[4], 1u, MAX_TILE_ROWS, "rows", &c.rows) ||
          parse_u32(argv[5], 1u, MAX_TILE_COLS, "cols", &c.cols) ||
          validate_case(c)) {
        return 1;
      }
    } else {
      if (parse_path_kind(argv[2], &path) ||
          parse_u32(argv[3], 1u, MAX_TILE_ROWS, "rows", &c.rows) ||
          parse_u32(argv[4], 1u, MAX_TILE_COLS, "cols", &c.cols) ||
          validate_case(c)) {
        return 1;
      }
      pair = default_pair_for_path(path);
    }
    if (source_env && source_env[0]) {
      snprintf(source_abs, sizeof(source_abs), "%s", source_env);
    } else {
      if (!getcwd(cwd, sizeof(cwd))) return 1;
      make_abs_path(source_abs, sizeof(source_abs), cwd,
                    "dma_tma_movement_profile_test.cl");
    }
    return run_child(pair, path, c, source_abs);
  }

  usage(argv[0]);
  return 1;
}
