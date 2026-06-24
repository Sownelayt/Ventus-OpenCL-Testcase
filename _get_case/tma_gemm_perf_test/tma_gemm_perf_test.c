/*
 * TMA GEMM performance sweep host program.
 *
 * Measures two paths that both compute:
 *   C[(m_tiles*16)x(n_tiles*16)] =
 *     A[(m_tiles*16)x(k_tiles*16)] * B[(k_tiles*16)x(n_tiles*16)]
 *
 * manual_gemm_kernel computes the full K dimension in one device launch.
 * tma_gemm_kernel processes one K tile per launch with descriptor-form
 * CP_ASYNC_TENSOR G2S and writes each partial sum tile into a separate global
 * partial buffer. A plain device reduce kernel then sums partials into C. Both
 * paths are checked against the same CPU reference.
 *
 * The default sweep launches one work-group per dispatch. For matrices with
 * multiple output tiles or K tiles, the host walks those launches serially and
 * accumulates event time, so every reported row stays on the stable 1WG RTL/GVM path. Sweep
 * mode runs each row as a child single-case process so GVM/RTL starts from
 * clean device state for every measurement row.
 *
 * Known RTL/GVM limitation: parallel TMA with two or more work-groups is
 * currently unstable/flaky. Set VENTUS_TMA_GEMM_PARALLEL_WG=1 and
 * VENTUS_TMA_GEMM_ALLOW_UNSTABLE_MULTI_WG=1 only when intentionally
 * reproducing that bug.
 *
 * Usage:
 *   ./tma_gemm_perf_test.out
 *   ./tma_gemm_perf_test.out sweep [iterations]
 *   ./tma_gemm_perf_test.out single <m_tiles> <n_tiles> <k_tiles> [iterations]
 *   ./tma_gemm_perf_test.out <tiles> [iterations]   # legacy square K=16
 *
 * Run from this directory after source env.sh. Top-level sweep/single runs wrap
 * each case in a child process so PMU cycles can be parsed. Sweep launches
 * up to VENTUS_TMA_GEMM_JOBS child processes in parallel, default 1. Set
 * VENTUS_TMA_GEMM_IN_CHILD=1 only for raw GVM debug output. Top-level runs
 * also write tma_gemm_perf_report_<timestamp>.md under log/.
 */

#define _POSIX_C_SOURCE 200809L

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "../common/ventus_opencl_test.h"

#define DESC_WORDS 32u
#define DESC_PAIR_WORDS (DESC_WORDS * 2u)
#define COORD_WORDS 32u
#define TILE_M 16u
#define TILE_N 16u
#define TILE_K 16u
#define C_TILE_WORDS (TILE_M * TILE_N)
#define WG_SIZE 32u
#define DEFAULT_ITERATIONS 1u
#define MAX_TILES 16u
#define MAX_K_TILES 8u
#define MAX_CASES 32u
#define MAX_STABLE_PARALLEL_WGS 1u
#define DEFAULT_SWEEP_JOBS 1u
#define MAX_SWEEP_JOBS 16u
#define LOG_DIR "log"

typedef struct {
  uint32_t m_tiles;
  uint32_t n_tiles;
  uint32_t k_tiles;
  uint32_t iterations;
} GemmCase;

typedef struct {
  pid_t pid;
  uint32_t case_index;
  int active;
  char log_path[192];
} ChildRun;

typedef struct {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t output_tiles;
  uint32_t launch_workgroups;
  uint32_t iterations;
  uint64_t manual_ns;
  uint64_t tma_ns;
  uint64_t manual_cycles;
  uint64_t tma_cycles;
  int cycle_valid;
  double speedup;
  double improvement;
  double manual_gflops;
  double tma_gflops;
} GemmResult;

static const GemmCase default_cases[] = {
  {1, 1, 1, DEFAULT_ITERATIONS},
  {1, 1, 2, DEFAULT_ITERATIONS},
  {1, 2, 1, DEFAULT_ITERATIONS},
  {1, 2, 2, DEFAULT_ITERATIONS},
  {2, 1, 1, DEFAULT_ITERATIONS},
  {2, 1, 2, DEFAULT_ITERATIONS},
  {2, 2, 1, DEFAULT_ITERATIONS},
  {2, 2, 2, DEFAULT_ITERATIONS},
};

static uint32_t
desc_control(unsigned data_type, unsigned rank)
{
  return (data_type & 0xfu) | ((rank & 0xfu) << 4);
}

static float
abs_f32(float x)
{
  return x < 0.0f ? -x : x;
}

static int
parse_u32(const char *text, uint32_t min_value, uint32_t max_value,
          const char *name, uint32_t *out)
{
  char *end = NULL;
  long parsed = strtol(text, &end, 0);
  if (!text[0] || *end || parsed < (long)min_value || parsed > (long)max_value) {
    fprintf(stderr, "%s must be in [%u, %u], got '%s'\n",
            name, min_value, max_value, text);
    return 1;
  }
  *out = (uint32_t)parsed;
  return 0;
}

static void
print_usage(const char *prog)
{
  fprintf(stderr,
          "Usage:\n"
          "  %s\n"
          "  %s sweep [iterations]\n"
          "  %s single <m_tiles> <n_tiles> <k_tiles> [iterations]\n"
          "  %s <tiles> [iterations]\n",
          prog, prog, prog, prog);
}

static int env_enabled(const char *name);
static int use_parallel_workgroups(void);
static int allow_unstable_multi_wg(void);

static int
validate_case(GemmCase c)
{
  uint32_t output_tiles = c.m_tiles * c.n_tiles;

  if (c.m_tiles == 0 || c.m_tiles > MAX_TILES ||
      c.n_tiles == 0 || c.n_tiles > MAX_TILES ||
      c.k_tiles == 0 || c.k_tiles > MAX_K_TILES ||
      c.iterations == 0) {
    fprintf(stderr, "invalid case: m_tiles=%u n_tiles=%u k_tiles=%u iterations=%u\n",
            c.m_tiles, c.n_tiles, c.k_tiles, c.iterations);
    return 1;
  }
  if (use_parallel_workgroups() &&
      output_tiles > MAX_STABLE_PARALLEL_WGS &&
      !allow_unstable_multi_wg()) {
    fprintf(stderr,
            "known RTL/GVM issue: parallel TMA with %u work-groups is unstable; "
            "use <=%u work-groups for performance numbers, set "
            "VENTUS_TMA_GEMM_PARALLEL_WG=0 for serial tile launches, or set "
            "VENTUS_TMA_GEMM_ALLOW_UNSTABLE_MULTI_WG=1 only to reproduce the bug\n",
            output_tiles, MAX_STABLE_PARALLEL_WGS);
    return 1;
  }
  return 0;
}

static int
env_enabled(const char *name)
{
  const char *value = getenv(name);
  return value && value[0] && value[0] != '0';
}

static int
use_parallel_workgroups(void)
{
  return env_enabled("VENTUS_TMA_GEMM_PARALLEL_WG");
}

static int
allow_unstable_multi_wg(void)
{
  return env_enabled("VENTUS_TMA_GEMM_ALLOW_UNSTABLE_MULTI_WG");
}

static int
skip_tma_path(void)
{
  return env_enabled("VENTUS_TMA_GEMM_SKIP_TMA");
}

static int
skip_measure_tma_path(void)
{
  return env_enabled("VENTUS_TMA_GEMM_SKIP_MEASURE_TMA");
}

static int
skip_manual_check_path(void)
{
  return env_enabled("VENTUS_TMA_GEMM_SKIP_MANUAL_CHECK");
}

static int
build_case_list(int argc, char **argv, GemmCase *cases, uint32_t *case_count)
{
  *case_count = 0;

  if (argc == 1 || (argc >= 2 && strcmp(argv[1], "sweep") == 0)) {
    uint32_t iterations = DEFAULT_ITERATIONS;
    if (argc > 3) {
      print_usage(argv[0]);
      return 1;
    }
    if (argc == 3 && parse_u32(argv[2], 1, 1000000u, "iterations", &iterations)) {
      return 1;
    }

    uint32_t n = (uint32_t)(sizeof(default_cases) / sizeof(default_cases[0]));
    for (uint32_t i = 0; i < n; i++) {
      cases[i] = default_cases[i];
      cases[i].iterations = iterations;
    }
    *case_count = n;
    return 0;
  }

  if (strcmp(argv[1], "single") == 0) {
    GemmCase c = {0, 0, 0, DEFAULT_ITERATIONS};
    if (argc != 5 && argc != 6) {
      print_usage(argv[0]);
      return 1;
    }
    if (parse_u32(argv[2], 1, MAX_TILES, "m_tiles", &c.m_tiles) ||
        parse_u32(argv[3], 1, MAX_TILES, "n_tiles", &c.n_tiles) ||
        parse_u32(argv[4], 1, MAX_K_TILES, "k_tiles", &c.k_tiles)) {
      return 1;
    }
    if (argc == 6 && parse_u32(argv[5], 1, 1000000u, "iterations", &c.iterations)) {
      return 1;
    }
    if (validate_case(c)) return 1;
    cases[0] = c;
    *case_count = 1;
    return 0;
  }

  if (argc == 2 || argc == 3) {
    GemmCase c = {0, 0, 1, DEFAULT_ITERATIONS};
    if (parse_u32(argv[1], 1, MAX_TILES, "tiles", &c.m_tiles)) return 1;
    c.n_tiles = c.m_tiles;
    if (argc == 3 && parse_u32(argv[2], 1, 1000000u, "iterations", &c.iterations)) {
      return 1;
    }
    cases[0] = c;
    *case_count = 1;
    return 0;
  }

  print_usage(argv[0]);
  return 1;
}

static void
build_desc(uint32_t *desc, uint32_t dim0, uint32_t dim1,
           uint32_t stride1, uint32_t box0, uint32_t box1)
{
  memset(desc, 0, DESC_WORDS * sizeof(uint32_t));
  desc[0] = 0x56544d41u;          /* "VTMA" */
  desc[1] = desc_control(6, 2);   /* FP32-sized element, rank=2 */
  desc[2] = 0;                    /* kernel patches runtime base pointer */
  desc[3] = 128;
  desc[4] = dim0;                 /* globalDim[0] */
  desc[5] = dim1;                 /* globalDim[1] */
  desc[6] = desc[7] = desc[8] = 1;
  desc[9] = 4;                    /* byteStride[0] */
  desc[10] = stride1;             /* byteStride[1] */
  desc[14] = box0;                /* boxDim[0] */
  desc[15] = box1;                /* boxDim[1] */
  desc[16] = desc[17] = desc[18] = 1;
  for (uint32_t i = 0; i < 5; i++) desc[19 + i] = 1;
}

static void
build_coords_pair(uint32_t *coords_pair, uint32_t m_tiles,
                  uint32_t n_tiles, uint32_t k_tiles)
{
  uint32_t output_tiles = m_tiles * n_tiles;
  size_t words = (size_t)output_tiles * k_tiles * 2u * COORD_WORDS;
  memset(coords_pair, 0, words * sizeof(uint32_t));

  for (uint32_t tile_id = 0; tile_id < output_tiles; tile_id++) {
    uint32_t tile_m = tile_id / n_tiles;
    uint32_t tile_n = tile_id - tile_m * n_tiles;
    uint32_t m_base = tile_m * TILE_M;
    uint32_t n_base = tile_n * TILE_N;

    for (uint32_t kt = 0; kt < k_tiles; kt++) {
      uint32_t *coords_a =
        coords_pair + ((tile_id * k_tiles + kt) * 2u * COORD_WORDS);
      uint32_t *coords_b = coords_a + COORD_WORDS;

      coords_a[0] = kt * TILE_K;
      coords_a[1] = m_base;
      coords_b[0] = n_base;
      coords_b[1] = kt * TILE_K;
    }
  }
}

static void
fill_inputs(float *A, float *B, uint32_t M, uint32_t N, uint32_t K)
{
  for (uint32_t i = 0; i < M * K; i++) {
    int v = (int)(i % 17u) - 8;
    A[i] = (float)v * 0.03125f;
  }
  for (uint32_t i = 0; i < K * N; i++) {
    int v = (int)((i * 3u + 5u) % 19u) - 9;
    B[i] = (float)v * 0.0234375f;
  }
}

static void
cpu_gemm_ref(const float *A, const float *B, float *C,
             uint32_t M, uint32_t N, uint32_t K)
{
  for (uint32_t row = 0; row < M; row++) {
    for (uint32_t col = 0; col < N; col++) {
      float acc = 0.0f;
      for (uint32_t k = 0; k < K; k++) {
        acc += A[row * K + k] * B[k * N + col];
      }
      C[row * N + col] = acc;
    }
  }
}

static int
check_result(const char *label, const float *got, const float *ref,
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

static int
event_duration_ns(cl_event event, uint64_t *duration_ns)
{
  cl_ulong start = 0;
  cl_ulong end = 0;
  cl_int err;

  err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                sizeof(start), &start, NULL);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "clGetEventProfilingInfo(START) failed: %d\n", err);
    return 1;
  }
  err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                sizeof(end), &end, NULL);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "clGetEventProfilingInfo(END) failed: %d\n", err);
    return 1;
  }
  if (end < start) {
    fprintf(stderr, "invalid profiling timestamps: start=%" PRIu64
                    " end=%" PRIu64 "\n",
            (uint64_t)start, (uint64_t)end);
    return 1;
  }

  *duration_ns = (uint64_t)(end - start);
  return 0;
}

static int
run_setup_kernel(cl_command_queue queue, cl_kernel kernel,
                 cl_mem desc_pair_buf, cl_mem a_buf, cl_mem b_buf)
{
  cl_int err;
  size_t global = 1;
  size_t local = 1;
  cl_event event = NULL;

  err  = clSetKernelArg(kernel, 0, sizeof(desc_pair_buf), &desc_pair_buf);
  err |= clSetKernelArg(kernel, 1, sizeof(a_buf), &a_buf);
  err |= clSetKernelArg(kernel, 2, sizeof(b_buf), &b_buf);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "clSetKernelArg(setup_desc) failed: %d\n", err);
    return 1;
  }

  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                               0, NULL, &event);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "clEnqueueNDRangeKernel(setup_desc) failed: %d\n", err);
    return 1;
  }
  err = clWaitForEvents(1, &event);
  clReleaseEvent(event);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "clWaitForEvents(setup_desc) failed: %d\n", err);
    return 1;
  }
  return 0;
}

static int
run_manual_kernel(cl_command_queue queue, cl_kernel kernel,
                  cl_mem a_buf, cl_mem b_buf, cl_mem c_buf,
                  GemmCase c, uint64_t *duration_ns)
{
  cl_int err;
  size_t local = WG_SIZE;
  uint32_t output_tiles = c.m_tiles * c.n_tiles;
  int parallel_wg = use_parallel_workgroups() && output_tiles > 1;
  size_t global = parallel_wg ? (size_t)WG_SIZE * output_tiles : WG_SIZE;

  err  = clSetKernelArg(kernel, 0, sizeof(a_buf), &a_buf);
  err |= clSetKernelArg(kernel, 1, sizeof(b_buf), &b_buf);
  err |= clSetKernelArg(kernel, 2, sizeof(c_buf), &c_buf);
  err |= clSetKernelArg(kernel, 3, sizeof(c.m_tiles), &c.m_tiles);
  err |= clSetKernelArg(kernel, 4, sizeof(c.n_tiles), &c.n_tiles);
  err |= clSetKernelArg(kernel, 5, sizeof(c.k_tiles), &c.k_tiles);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "clSetKernelArg(manual) failed: %d\n", err);
    return 1;
  }

  if (duration_ns) *duration_ns = 0;
  for (uint32_t rep = 0; rep < c.iterations; rep++) {
    uint32_t launches = parallel_wg ? 1u : output_tiles;
    for (uint32_t launch = 0; launch < launches; launch++) {
      cl_event event = NULL;
      uint64_t this_ns = 0;
      uint32_t tile_offset = parallel_wg ? 0u : launch;

      err = clSetKernelArg(kernel, 6, sizeof(tile_offset), &tile_offset);
      if (err != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg(manual tile_offset) failed: %d\n", err);
        return 1;
      }
      err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                                   0, NULL, &event);
      if (err != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueNDRangeKernel(manual) failed: %d\n", err);
        return 1;
      }
      err = clWaitForEvents(1, &event);
      if (err != CL_SUCCESS) {
        fprintf(stderr, "clWaitForEvents(manual) failed: %d\n", err);
        clReleaseEvent(event);
        return 1;
      }

      if (duration_ns && event_duration_ns(event, &this_ns) != 0) {
        clReleaseEvent(event);
        return 1;
      }
      if (duration_ns) *duration_ns += this_ns;
      clReleaseEvent(event);
    }
  }
  return 0;
}

static int
run_tma_kernel(cl_command_queue queue, cl_kernel tma_kernel,
               cl_kernel reduce_kernel, cl_mem desc_pair_buf,
               cl_mem coords_pair_buf, cl_mem a_buf, cl_mem b_buf,
               cl_mem partial_buf, cl_mem c_buf, GemmCase c,
               uint64_t *duration_ns)
{
  cl_int err;
  size_t local = WG_SIZE;
  uint32_t output_tiles = c.m_tiles * c.n_tiles;
  int parallel_wg = use_parallel_workgroups() && output_tiles > 1;
  size_t global = parallel_wg ? (size_t)WG_SIZE * output_tiles : WG_SIZE;

  err  = clSetKernelArg(tma_kernel, 0, sizeof(desc_pair_buf), &desc_pair_buf);
  err |= clSetKernelArg(tma_kernel, 1, sizeof(coords_pair_buf), &coords_pair_buf);
  err |= clSetKernelArg(tma_kernel, 2, sizeof(a_buf), &a_buf);
  err |= clSetKernelArg(tma_kernel, 3, sizeof(b_buf), &b_buf);
  err |= clSetKernelArg(tma_kernel, 4, sizeof(partial_buf), &partial_buf);
  err |= clSetKernelArg(tma_kernel, 5, sizeof(c.m_tiles), &c.m_tiles);
  err |= clSetKernelArg(tma_kernel, 6, sizeof(c.n_tiles), &c.n_tiles);
  err |= clSetKernelArg(tma_kernel, 7, sizeof(c.k_tiles), &c.k_tiles);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "clSetKernelArg(tma) failed: %d\n", err);
    return 1;
  }

  err  = clSetKernelArg(reduce_kernel, 0, sizeof(partial_buf), &partial_buf);
  err |= clSetKernelArg(reduce_kernel, 1, sizeof(c_buf), &c_buf);
  err |= clSetKernelArg(reduce_kernel, 2, sizeof(c.m_tiles), &c.m_tiles);
  err |= clSetKernelArg(reduce_kernel, 3, sizeof(c.n_tiles), &c.n_tiles);
  err |= clSetKernelArg(reduce_kernel, 4, sizeof(c.k_tiles), &c.k_tiles);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "clSetKernelArg(reduce) failed: %d\n", err);
    return 1;
  }

  if (duration_ns) *duration_ns = 0;
  for (uint32_t rep = 0; rep < c.iterations; rep++) {
    uint32_t launches = parallel_wg ? 1u : output_tiles;
    for (uint32_t kt_offset = 0; kt_offset < c.k_tiles; kt_offset++) {
      for (uint32_t launch = 0; launch < launches; launch++) {
        cl_event event = NULL;
        uint64_t this_ns = 0;
        uint32_t tile_offset = parallel_wg ? 0u : launch;

        err  = clSetKernelArg(tma_kernel, 8, sizeof(kt_offset), &kt_offset);
        err |= clSetKernelArg(tma_kernel, 9, sizeof(tile_offset), &tile_offset);
        if (err != CL_SUCCESS) {
          fprintf(stderr, "clSetKernelArg(tma dynamic args) failed: %d\n", err);
          return 1;
        }
        err = clEnqueueNDRangeKernel(queue, tma_kernel, 1, NULL, &global, &local,
                                     0, NULL, &event);
        if (err != CL_SUCCESS) {
          fprintf(stderr, "clEnqueueNDRangeKernel(tma) failed: %d\n", err);
          return 1;
        }
        err = clWaitForEvents(1, &event);
        if (err != CL_SUCCESS) {
          fprintf(stderr, "clWaitForEvents(tma) failed: %d\n", err);
          clReleaseEvent(event);
          return 1;
        }

        if (duration_ns && event_duration_ns(event, &this_ns) != 0) {
          clReleaseEvent(event);
          return 1;
        }
        if (duration_ns) *duration_ns += this_ns;
        clReleaseEvent(event);
      }
    }

    for (uint32_t launch = 0; launch < launches; launch++) {
      cl_event event = NULL;
      uint64_t this_ns = 0;
      uint32_t tile_offset = parallel_wg ? 0u : launch;

      err = clSetKernelArg(reduce_kernel, 5, sizeof(tile_offset), &tile_offset);
      if (err != CL_SUCCESS) {
        fprintf(stderr, "clSetKernelArg(reduce tile_offset) failed: %d\n", err);
        return 1;
      }
      err = clEnqueueNDRangeKernel(queue, reduce_kernel, 1, NULL, &global, &local,
                                   0, NULL, &event);
      if (err != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueNDRangeKernel(reduce) failed: %d\n", err);
        return 1;
      }
      err = clWaitForEvents(1, &event);
      if (err != CL_SUCCESS) {
        fprintf(stderr, "clWaitForEvents(reduce) failed: %d\n", err);
        clReleaseEvent(event);
        return 1;
      }

      if (duration_ns && event_duration_ns(event, &this_ns) != 0) {
        clReleaseEvent(event);
        return 1;
      }
      if (duration_ns) *duration_ns += this_ns;
      clReleaseEvent(event);
    }
  }
  return 0;
}

static int
fill_buffer_f32(cl_command_queue queue, cl_mem buf, size_t bytes, float value)
{
  cl_int err = clEnqueueFillBuffer(queue, buf, &value, sizeof(value),
                                   0, bytes, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "clEnqueueFillBuffer failed: %d\n", err);
    return 1;
  }
  err = clFinish(queue);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "clFinish(fill) failed: %d\n", err);
    return 1;
  }
  return 0;
}

static int
run_case(cl_context context, cl_command_queue queue,
         cl_kernel setup_kernel, cl_kernel manual_kernel, cl_kernel tma_kernel,
         cl_kernel reduce_kernel, GemmCase c, GemmResult *result)
{
  cl_int err = CL_SUCCESS;
  cl_mem desc_pair_buf = NULL;
  cl_mem coords_pair_buf = NULL;
  cl_mem a_buf = NULL, b_buf = NULL;
  cl_mem c_manual_buf = NULL, c_tma_buf = NULL, partial_buf = NULL;
  float *A = NULL, *B = NULL, *C_ref = NULL, *C_manual = NULL, *C_tma = NULL;
  uint32_t *coords_pair = NULL;
  uint32_t desc_pair[DESC_PAIR_WORDS];
  uint32_t M = c.m_tiles * TILE_M;
  uint32_t N = c.n_tiles * TILE_N;
  uint32_t K = c.k_tiles * TILE_K;
  uint32_t output_tiles = c.m_tiles * c.n_tiles;
  uint32_t launch_workgroups = use_parallel_workgroups() ? output_tiles : 1;
  uint32_t c_elements = M * N;
  size_t a_bytes = (size_t)M * K * sizeof(float);
  size_t b_bytes = (size_t)K * N * sizeof(float);
  size_t c_bytes = (size_t)c_elements * sizeof(float);
  size_t coords_pair_words =
    (size_t)output_tiles * c.k_tiles * 2u * COORD_WORDS;
  size_t coords_pair_bytes = coords_pair_words * sizeof(uint32_t);
  size_t partial_words = (size_t)output_tiles * c.k_tiles * C_TILE_WORDS;
  size_t partial_bytes = partial_words * sizeof(uint32_t);
  uint64_t manual_ns = 0;
  uint64_t tma_ns = 0;
  int exit_code = 1;
  int skip_tma = skip_tma_path();
  int skip_measure_tma = skip_measure_tma_path();
  int skip_manual_check = skip_manual_check_path();
  char manual_label[96];
  char tma_label[96];

  printf("CASE M=%u N=%u K=%u m_tiles=%u n_tiles=%u k_tiles=%u output_tiles=%u launch_workgroups=%u iterations=%u\n",
         M, N, K, c.m_tiles, c.n_tiles, c.k_tiles, output_tiles,
         launch_workgroups, c.iterations);

  A = (float *)malloc(a_bytes);
  B = (float *)malloc(b_bytes);
  C_ref = (float *)malloc(c_bytes);
  C_manual = (float *)malloc(c_bytes);
  C_tma = (float *)malloc(c_bytes);
  coords_pair = (uint32_t *)malloc(coords_pair_bytes);
  if (!A || !B || !C_ref || !C_manual || !C_tma || !coords_pair) {
    fprintf(stderr, "host allocation failed\n");
    goto FINISH;
  }

  fill_inputs(A, B, M, N, K);
  cpu_gemm_ref(A, B, C_ref, M, N, K);
  build_desc(desc_pair, K, M, K * sizeof(float), TILE_K, TILE_M);
  build_desc(desc_pair + DESC_WORDS, N, K, N * sizeof(float), TILE_N, TILE_K);
  build_coords_pair(coords_pair, c.m_tiles, c.n_tiles, c.k_tiles);
  memset(C_manual, 0, c_bytes);
  memset(C_tma, 0, c_bytes);

  desc_pair_buf = clCreateBuffer(context,
                                 CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                 sizeof(desc_pair), desc_pair, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(desc_pair)");
  a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         a_bytes, A, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(A)");
  b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         b_bytes, B, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(B)");
  coords_pair_buf = clCreateBuffer(context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   coords_pair_bytes, coords_pair, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(coords_pair)");
  c_manual_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                c_bytes, NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(C_manual)");
  c_tma_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
                             c_bytes, NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(C_tma)");
  partial_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
                               partial_bytes, NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(partials)");

  if (run_setup_kernel(queue, setup_kernel, desc_pair_buf,
                       a_buf, b_buf) != 0) {
    goto FINISH;
  }
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(setup_desc)");

  if (fill_buffer_f32(queue, c_manual_buf, c_bytes, 0.0f) != 0) goto FINISH;
  if (fill_buffer_f32(queue, c_tma_buf, c_bytes, 0.0f) != 0) goto FINISH;
  if (fill_buffer_f32(queue, partial_buf, partial_bytes, 0.0f) != 0) goto FINISH;

  GemmCase warmup_case = c;
  warmup_case.iterations = 1;
  if (run_manual_kernel(queue, manual_kernel, a_buf, b_buf, c_manual_buf,
                        warmup_case, NULL) != 0) {
    goto FINISH;
  }
  if (!skip_tma && run_tma_kernel(queue, tma_kernel, reduce_kernel,
                                  desc_pair_buf, coords_pair_buf, a_buf,
                                  b_buf, partial_buf, c_tma_buf,
                                  warmup_case, NULL) != 0) {
    goto FINISH;
  }
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(warmup)");

  if (fill_buffer_f32(queue, c_manual_buf, c_bytes, 0.0f) != 0) goto FINISH;
  if (fill_buffer_f32(queue, c_tma_buf, c_bytes, 0.0f) != 0) goto FINISH;
  if (fill_buffer_f32(queue, partial_buf, partial_bytes, 0.0f) != 0) goto FINISH;

  if (run_manual_kernel(queue, manual_kernel, a_buf, b_buf, c_manual_buf,
                        c, &manual_ns) != 0) {
    goto FINISH;
  }
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(manual)");

  if (!skip_tma && !skip_measure_tma) {
    if (run_tma_kernel(queue, tma_kernel, reduce_kernel, desc_pair_buf,
                       coords_pair_buf, a_buf, b_buf, partial_buf,
                       c_tma_buf, c, &tma_ns) != 0) {
      goto FINISH;
    }
    err = clFinish(queue);
    CHECK_OPENCL_ERROR_IN("clFinish(tma)");
  }

  err = clEnqueueReadBuffer(queue, c_manual_buf, CL_TRUE, 0, c_bytes,
                            C_manual, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(C_manual)");
  if (!skip_tma && !skip_measure_tma) {
    err = clEnqueueReadBuffer(queue, c_tma_buf, CL_TRUE, 0, c_bytes,
                              C_tma, 0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(C_tma)");
  }

  snprintf(manual_label, sizeof(manual_label), "manual_gemm_%ux%ux%u", M, N, K);
  snprintf(tma_label, sizeof(tma_label), "tma_gemm_%ux%ux%u", M, N, K);
  if (!skip_manual_check && check_result(manual_label, C_manual, C_ref, c_elements) != 0) goto FINISH;
  if (!skip_tma && !skip_measure_tma && check_result(tma_label, C_tma, C_ref, c_elements) != 0) goto FINISH;

  double speedup = tma_ns ? (double)manual_ns / (double)tma_ns : 0.0;
  double improvement = manual_ns
                         ? ((double)manual_ns - (double)tma_ns) *
                             100.0 / (double)manual_ns
                         : 0.0;
  double ops = 2.0 * (double)M * (double)N * (double)K *
               (double)c.iterations;
  double manual_gflops = manual_ns ? ops / (double)manual_ns : 0.0;
  double tma_gflops = tma_ns ? ops / (double)tma_ns : 0.0;

  result->M = M;
  result->N = N;
  result->K = K;
  result->output_tiles = output_tiles;
  result->launch_workgroups = launch_workgroups;
  result->iterations = c.iterations;
  result->manual_ns = manual_ns;
  result->tma_ns = tma_ns;
  result->manual_cycles = 0;
  result->tma_cycles = 0;
  result->cycle_valid = 0;
  result->speedup = speedup;
  result->improvement = improvement;
  result->manual_gflops = manual_gflops;
  result->tma_gflops = tma_gflops;

  printf("RESULT M=%u N=%u K=%u output_tiles=%u launch_workgroups=%u iterations=%u"
         " manual_ns=%" PRIu64 " tma_ns=%" PRIu64
         " speedup=%.4fx improvement_percent=%.2f"
         " manual_gflops=%.6e tma_gflops=%.6e\n",
         M, N, K, output_tiles, launch_workgroups, c.iterations,
         manual_ns, tma_ns, speedup, improvement,
         manual_gflops, tma_gflops);
  exit_code = 0;

FINISH:
  if (partial_buf) clReleaseMemObject(partial_buf);
  if (c_tma_buf) clReleaseMemObject(c_tma_buf);
  if (c_manual_buf) clReleaseMemObject(c_manual_buf);
  if (b_buf) clReleaseMemObject(b_buf);
  if (a_buf) clReleaseMemObject(a_buf);
  if (coords_pair_buf) clReleaseMemObject(coords_pair_buf);
  if (desc_pair_buf) clReleaseMemObject(desc_pair_buf);
  free(coords_pair);
  free(C_tma);
  free(C_manual);
  free(C_ref);
  free(B);
  free(A);
  return exit_code;
}

static int
parse_result_line(const char *line, GemmResult *result)
{
  unsigned M = 0, N = 0, K = 0;
  unsigned output_tiles = 0, launch_workgroups = 0, iterations = 0;
  unsigned long long manual_ns = 0, tma_ns = 0;
  double speedup = 0.0, improvement = 0.0;
  double manual_gflops = 0.0, tma_gflops = 0.0;

  int fields = sscanf(line,
                      "RESULT M=%u N=%u K=%u output_tiles=%u launch_workgroups=%u iterations=%u manual_ns=%llu tma_ns=%llu speedup=%lfx improvement_percent=%lf manual_gflops=%lf tma_gflops=%lf",
                      &M, &N, &K, &output_tiles, &launch_workgroups,
                      &iterations, &manual_ns, &tma_ns, &speedup,
                      &improvement, &manual_gflops, &tma_gflops);
  if (fields != 12) return 1;

  result->M = M;
  result->N = N;
  result->K = K;
  result->output_tiles = output_tiles;
  result->launch_workgroups = launch_workgroups;
  result->iterations = iterations;
  result->manual_ns = (uint64_t)manual_ns;
  result->tma_ns = (uint64_t)tma_ns;
  result->manual_cycles = 0;
  result->tma_cycles = 0;
  result->cycle_valid = 0;
  result->speedup = speedup;
  result->improvement = improvement;
  result->manual_gflops = manual_gflops;
  result->tma_gflops = tma_gflops;
  return 0;
}

static void
make_timestamp(char *stamp, size_t stamp_size)
{
  time_t now = time(NULL);
  struct tm tm_now;

  if (localtime_r(&now, &tm_now) &&
      strftime(stamp, stamp_size, "%Y%m%d_%H%M%S", &tm_now) > 0) {
    return;
  }
  snprintf(stamp, stamp_size, "%ld", (long)now);
}

static int
parse_active_cycles_line(const char *line, uint64_t *cycles)
{
  const char *key = "[TESTCASE TOTAL] [INST+CYCLE] active cycles";
  const char *hit = strstr(line, key);
  const char *colon = NULL;
  char *end = NULL;
  unsigned long long parsed = 0;

  if (!hit) return 1;
  colon = strchr(hit, ':');
  if (!colon) return 1;
  parsed = strtoull(colon + 1, &end, 10);
  if (end == colon + 1) return 1;
  *cycles = (uint64_t)parsed;
  return 0;
}

static void
attach_measured_cycles(GemmResult *result, const uint64_t *pmu_active_cycles,
                       uint32_t pmu_count)
{
  uint32_t launches_per_iteration =
    result->launch_workgroups > 1 ? 1u : result->output_tiles;
  uint32_t manual_launches = launches_per_iteration * result->iterations;
  uint32_t k_tiles = result->K / TILE_K;
  uint32_t tma_launches = launches_per_iteration * (k_tiles + 1u) * result->iterations;
  uint32_t final_idx = 0;
  uint32_t after_manual_idx = 0;
  uint32_t before_manual_idx = 0;
  uint64_t before_manual = 0;
  uint64_t after_manual = 0;
  uint64_t after_tma = 0;

  if (manual_launches == 0 || tma_launches == 0) return;
  if (pmu_count < manual_launches + tma_launches + 1u) return;

  final_idx = pmu_count - 1u;
  after_manual_idx = final_idx - tma_launches;
  before_manual_idx = after_manual_idx - manual_launches;

  before_manual = pmu_active_cycles[before_manual_idx];
  after_manual = pmu_active_cycles[after_manual_idx];
  after_tma = pmu_active_cycles[final_idx];
  if (after_manual < before_manual || after_tma < after_manual) return;

  result->manual_cycles = after_manual - before_manual;
  result->tma_cycles = after_tma - after_manual;
  result->cycle_valid = result->manual_cycles != 0 && result->tma_cycles != 0;
}

static uint32_t
sweep_jobs(void)
{
  const char *value = getenv("VENTUS_TMA_GEMM_JOBS");
  char *end = NULL;
  long parsed = 0;

  if (!value || !value[0]) return DEFAULT_SWEEP_JOBS;
  parsed = strtol(value, &end, 0);
  if (*end || parsed < 1) return DEFAULT_SWEEP_JOBS;
  if (parsed > (long)MAX_SWEEP_JOBS) return MAX_SWEEP_JOBS;
  return (uint32_t)parsed;
}

static void
make_child_log_path(char *path, size_t path_size, uint32_t case_index,
                    GemmCase c)
{
  char stamp[32];
  make_timestamp(stamp, sizeof(stamp));
  snprintf(path, path_size,
           LOG_DIR "/tma_gemm_perf_child_%s_%ld_case%02u_m%u_n%u_k%u.log",
           stamp, (long)getpid(), case_index, c.m_tiles, c.n_tiles,
           c.k_tiles);
}

static int
launch_child_case(const char *prog, GemmCase c, uint32_t case_index,
                  ChildRun *child)
{
  char m_arg[16];
  char n_arg[16];
  char k_arg[16];
  char iter_arg[16];
  FILE *header = NULL;
  pid_t pid;

  make_child_log_path(child->log_path, sizeof(child->log_path), case_index, c);
  mkdir(LOG_DIR, 0775);
  header = fopen(child->log_path, "w");
  if (!header) {
    fprintf(stderr, "failed to create child log for case %u: %s\n",
            case_index, child->log_path);
    return 1;
  }
  fprintf(header, "# command: %s single %u %u %u %u\n",
          prog, c.m_tiles, c.n_tiles, c.k_tiles, c.iterations);
  fclose(header);

  snprintf(m_arg, sizeof(m_arg), "%u", c.m_tiles);
  snprintf(n_arg, sizeof(n_arg), "%u", c.n_tiles);
  snprintf(k_arg, sizeof(k_arg), "%u", c.k_tiles);
  snprintf(iter_arg, sizeof(iter_arg), "%u", c.iterations);

  pid = fork();
  if (pid < 0) {
    fprintf(stderr, "fork failed for case %u\n", case_index);
    return 1;
  }

  if (pid == 0) {
    FILE *out = NULL;
    setenv("VENTUS_TMA_GEMM_IN_CHILD", "1", 1);
    out = fopen(child->log_path, "a");
    if (out) {
      dup2(fileno(out), STDOUT_FILENO);
      dup2(fileno(out), STDERR_FILENO);
    }
    execlp(prog, prog, "single", m_arg, n_arg, k_arg, iter_arg, (char *)NULL);
    perror("execlp");
    _exit(127);
  }

  child->pid = pid;
  child->case_index = case_index;
  child->active = 1;
  printf("CHILD_START case=%u pid=%ld path=%s\n",
         case_index, (long)pid, child->log_path);
  fflush(stdout);
  return 0;
}

static int
parse_child_log(uint32_t case_index, const char *log_path, GemmResult *result)
{
  FILE *log = fopen(log_path, "r");
  char line[512];
  uint64_t pmu_active_cycles[64];
  uint32_t pmu_count = 0;
  int seen_result = 0;

  if (!log) {
    fprintf(stderr, "failed to open child log for case %u: %s\n",
            case_index, log_path);
    return 1;
  }

  while (fgets(line, sizeof(line), log)) {
    uint64_t active_cycles = 0;
    if (parse_active_cycles_line(line, &active_cycles) == 0 &&
        pmu_count < (uint32_t)(sizeof(pmu_active_cycles) / sizeof(pmu_active_cycles[0]))) {
      pmu_active_cycles[pmu_count++] = active_cycles;
      printf("PMU_TOTAL_ACTIVE case=%u snapshot=%u cycles=%" PRIu64 "\n",
             case_index, pmu_count, active_cycles);
    }

    if (strncmp(line, "CASE ", 5) == 0 ||
        strncmp(line, "PASS ", 5) == 0 ||
        strncmp(line, "FAIL ", 5) == 0 ||
        strncmp(line, "RESULT ", 7) == 0) {
      fputs(line, strncmp(line, "FAIL ", 5) == 0 ? stderr : stdout);
    }

    if (strncmp(line, "RESULT ", 7) == 0 &&
        parse_result_line(line, result) == 0) {
      seen_result = 1;
    }
  }
  fclose(log);

  if (!seen_result) {
    fprintf(stderr, "case %u did not produce RESULT; child_log=%s\n",
            case_index, log_path);
    return 1;
  }

  attach_measured_cycles(result, pmu_active_cycles, pmu_count);
  if (result->cycle_valid) {
    printf("RESULT_CYCLES M=%u N=%u K=%u manual_cycles=%" PRIu64
           " tma_cycles=%" PRIu64 " speedup=%.4fx improvement_percent=%.2f\n",
           result->M, result->N, result->K, result->manual_cycles,
           result->tma_cycles,
           (double)result->manual_cycles / (double)result->tma_cycles,
           ((double)result->manual_cycles - (double)result->tma_cycles) *
             100.0 / (double)result->manual_cycles);
  } else {
    printf("RESULT_CYCLES M=%u N=%u K=%u manual_cycles=NA tma_cycles=NA\n",
           result->M, result->N, result->K);
  }
  return 0;
}


static void
stop_active_children(ChildRun *children, uint32_t jobs)
{
  for (uint32_t i = 0; i < jobs; i++) {
    if (children[i].active && children[i].pid > 0) {
      kill(children[i].pid, SIGTERM);
    }
  }
  for (uint32_t i = 0; i < jobs; i++) {
    if (children[i].active && children[i].pid > 0) {
      waitpid(children[i].pid, NULL, 0);
      children[i].active = 0;
    }
  }
}


static int
run_sweep_as_child_processes(const char *prog, const GemmCase *cases,
                             uint32_t count, GemmResult *results,
                             uint32_t *result_count)
{
  ChildRun children[MAX_SWEEP_JOBS];
  uint32_t jobs = sweep_jobs();
  uint32_t next_case = 0;
  uint32_t active = 0;
  uint32_t completed = 0;

  if (jobs > count) jobs = count ? count : 1u;
  for (uint32_t i = 0; i < MAX_SWEEP_JOBS; i++) {
    children[i].active = 0;
  }
  *result_count = 0;
  printf("SWEEP_JOBS %u\n", jobs);

  while (completed < count) {
    while (next_case < count && active < jobs) {
      uint32_t slot = 0;
      while (slot < jobs && children[slot].active) slot++;
      if (slot == jobs) break;
      if (launch_child_case(prog, cases[next_case], next_case,
                            &children[slot]) != 0) {
        stop_active_children(children, jobs);
        return 1;
      }
      next_case++;
      active++;
    }

    int status = 0;
    pid_t done = waitpid(-1, &status, 0);
    if (done < 0) {
      fprintf(stderr, "waitpid failed while %u children were active\n", active);
      stop_active_children(children, jobs);
      return 1;
    }

    uint32_t slot = 0;
    while (slot < jobs && (!children[slot].active || children[slot].pid != done)) {
      slot++;
    }
    if (slot == jobs) continue;

    ChildRun child = children[slot];
    children[slot].active = 0;
    active--;
    completed++;

    printf("CHILD_DONE case=%u pid=%ld status=%d path=%s\n",
           child.case_index, (long)done, status, child.log_path);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      fprintf(stderr, "sweep child case %u failed status=%d child_log=%s\n",
              child.case_index, status, child.log_path);
      stop_active_children(children, jobs);
      return 1;
    }
    if (parse_child_log(child.case_index, child.log_path,
                        &results[child.case_index]) != 0) {
      stop_active_children(children, jobs);
      return 1;
    }
  }

  *result_count = count;
  return 0;
}


static void
print_result_table_legacy(const GemmResult *results, uint32_t count)
{
  printf("\nGEMM_SWEEP_TABLE_BEGIN\n");
  printf("| M | N | K | output_tiles | launch_wgs | iters | manual_ns | tma_ns | speedup | improvement_percent | manual_gflops | tma_gflops |\n");
  printf("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
  for (uint32_t i = 0; i < count; i++) {
    printf("| %u | %u | %u | %u | %u | %u | %" PRIu64 " | %" PRIu64
           " | %.4f | %.2f | %.6e | %.6e |\n",
           results[i].M, results[i].N, results[i].K,
           results[i].output_tiles, results[i].launch_workgroups,
           results[i].iterations, results[i].manual_ns, results[i].tma_ns,
           results[i].speedup, results[i].improvement,
           results[i].manual_gflops, results[i].tma_gflops);
  }
  printf("GEMM_SWEEP_TABLE_END\n");
}


static void
write_result_table(FILE *out, const GemmResult *results, uint32_t count)
{
  fprintf(out, "\nGEMM_SWEEP_TABLE_BEGIN\n");
  fprintf(out, "| M | N | K | output_tiles | launch_wgs | iters | manual_cycles | tma_cycles | cycle_speedup | cycle_improvement_percent | manual_ns | tma_ns | ns_speedup | ns_improvement_percent | manual_gflops | tma_gflops |\n");
  fprintf(out, "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
  for (uint32_t i = 0; i < count; i++) {
    if (results[i].cycle_valid) {
      double cycle_speedup = (double)results[i].manual_cycles /
                             (double)results[i].tma_cycles;
      double cycle_improvement =
        ((double)results[i].manual_cycles - (double)results[i].tma_cycles) *
        100.0 / (double)results[i].manual_cycles;
      fprintf(out, "| %u | %u | %u | %u | %u | %u | %" PRIu64 " | %" PRIu64
              " | %.4f | %.2f | %" PRIu64 " | %" PRIu64
              " | %.4f | %.2f | %.6e | %.6e |\n",
              results[i].M, results[i].N, results[i].K,
              results[i].output_tiles, results[i].launch_workgroups,
              results[i].iterations, results[i].manual_cycles,
              results[i].tma_cycles, cycle_speedup, cycle_improvement,
              results[i].manual_ns, results[i].tma_ns,
              results[i].speedup, results[i].improvement,
              results[i].manual_gflops, results[i].tma_gflops);
    } else {
      fprintf(out, "| %u | %u | %u | %u | %u | %u | NA | NA | NA | NA | %" PRIu64
              " | %" PRIu64 " | %.4f | %.2f | %.6e | %.6e |\n",
              results[i].M, results[i].N, results[i].K,
              results[i].output_tiles, results[i].launch_workgroups,
              results[i].iterations, results[i].manual_ns, results[i].tma_ns,
              results[i].speedup, results[i].improvement,
              results[i].manual_gflops, results[i].tma_gflops);
    }
  }
  fprintf(out, "GEMM_SWEEP_TABLE_END\n");
}

static void
print_result_table(const GemmResult *results, uint32_t count)
{
  write_result_table(stdout, results, count);
}

static FILE *
open_timestamped_report(char *path, size_t path_size)
{
  time_t now = time(NULL);
  struct tm tm_now;
  char stamp[32];

  if (localtime_r(&now, &tm_now) &&
      strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", &tm_now) > 0) {
    /* stamp filled */
  } else {
    snprintf(stamp, sizeof(stamp), "%ld", (long)now);
  }

  mkdir(LOG_DIR, 0775);
  for (uint32_t suffix = 0; suffix < 1000u; suffix++) {
    FILE *existing = NULL;
    if (suffix == 0) {
      snprintf(path, path_size, LOG_DIR "/tma_gemm_perf_report_%s.md", stamp);
    } else {
      snprintf(path, path_size, LOG_DIR "/tma_gemm_perf_report_%s_%03u.md", stamp, suffix);
    }
    existing = fopen(path, "r");
    if (existing) {
      fclose(existing);
      continue;
    }
    return fopen(path, "w");
  }

  return NULL;
}

static int
write_results_report(const GemmResult *results, uint32_t count)
{
  char path[160];
  FILE *out = open_timestamped_report(path, sizeof(path));
  if (!out) {
    fprintf(stderr, "failed to create timestamped report file\n");
    return 1;
  }

  fprintf(out, "# TMA GEMM Performance Report\n\n");
  fprintf(out, "- primary_metric: GVM PMU active cycles\n");
  fprintf(out, "- cycle_extraction: child cumulative PMU active cycles; measured manual/TMA are detected from the final measured kernel-window group, summed across serial tile launches and iterations\n");
  fprintf(out, "- host_ns_columns: OpenCL event profiling, kept only as auxiliary timing\n");
  fprintf(out, "- stable_path: default serial 1WG launches; parallel TMA with >=2 work-groups is marked as known RTL/GVM unstable\n");
  fprintf(out, "- child_logs: full per-case stdout/stderr saved as log/tma_gemm_perf_child_<timestamp>_<pid>_case*.log\n");
  write_result_table(out, results, count);
  fclose(out);

  printf("REPORT_FILE %s\n", path);
  return 0;
}

int
main(int argc, char **argv)
{
  GemmCase cases[MAX_CASES];
  GemmResult results[MAX_CASES];
  uint32_t case_count = 0;
  uint32_t result_count = 0;
  cl_int err = CL_SUCCESS;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel setup_kernel = NULL;
  cl_kernel manual_kernel = NULL;
  cl_kernel tma_kernel = NULL;
  cl_kernel reduce_kernel = NULL;
  int exit_code = 1;

  if (build_case_list(argc, argv, cases, &case_count) != 0) {
    return 1;
  }
  for (uint32_t i = 0; i < case_count; i++) {
    if (validate_case(cases[i])) return 1;
  }

  printf("TMA GEMM perf sweep: tile=%ux%ux%u wg_size=%u cases=%u parallel_wg=%s stable_parallel_wgs<=%u\n",
         TILE_M, TILE_N, TILE_K, WG_SIZE, case_count,
         use_parallel_workgroups() ? "on" : "off",
         MAX_STABLE_PARALLEL_WGS);

  if (!env_enabled("VENTUS_TMA_GEMM_IN_CHILD")) {
    if (run_sweep_as_child_processes(argv[0], cases, case_count, results,
                                     &result_count) != 0) {
      return 1;
    }
    print_result_table(results, result_count);
    if (write_results_report(results, result_count) != 0) {
      return 1;
    }
    printf("OK tma_gemm_perf_test cases=%u\n", result_count);
    return 0;
  }

  err = ventus_get_default_device(&context, &device, &queue, NULL);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device");
  clReleaseCommandQueue(queue);
  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE,
                               &err);
  CHECK_OPENCL_ERROR_IN("clCreateCommandQueue(profiled)");

  err = ventus_build_program_from_source(context, device,
                                         "tma_gemm_perf_test.cl", &program);
  CHECK_OPENCL_ERROR_IN("ventus_build_program_from_source");
  setup_kernel = clCreateKernel(program, "setup_desc_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(setup_desc)");
  manual_kernel = clCreateKernel(program, "manual_gemm_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(manual)");
  tma_kernel = clCreateKernel(program, "tma_gemm_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(tma)");
  reduce_kernel = clCreateKernel(program, "sum_partials_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(sum_partials)");

  for (uint32_t i = 0; i < case_count; i++) {
    if (run_case(context, queue, setup_kernel, manual_kernel, tma_kernel,
                 reduce_kernel, cases[i], &results[result_count]) != 0) {
      fprintf(stderr, "case %u failed\n", i);
      goto FINISH;
    }
    result_count++;
  }

  print_result_table(results, result_count);
  if (!env_enabled("VENTUS_TMA_GEMM_IN_CHILD") &&
      write_results_report(results, result_count) != 0) {
    goto FINISH;
  }
  printf("OK tma_gemm_perf_test cases=%u\n", result_count);
  exit_code = 0;

FINISH:
  if (reduce_kernel) clReleaseKernel(reduce_kernel);
  if (tma_kernel) clReleaseKernel(tma_kernel);
  if (manual_kernel) clReleaseKernel(manual_kernel);
  if (setup_kernel) clReleaseKernel(setup_kernel);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
  return exit_code;
}
