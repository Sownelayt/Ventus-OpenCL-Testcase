/*
 * V1 bidirectional contention performance test.
 *
 * Usage:
 *   ./dma_tma_bidirectional_contention_perf_test.out
 *   ./dma_tma_bidirectional_contention_perf_test.out sweep
 *   ./dma_tma_bidirectional_contention_perf_test.out single \
 *       <bulk|tensor> <g2s_only|s2g_only|g2s_s2g|s2g_g2s|dualwarp|burst>
 *       <bytes> <wg-or-depth> [same_set|spread]
 */

#include <CL/cl.h>
#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits.h>
#include <sys/wait.h>
#include <unistd.h>
#include <string>
#include <vector>

#include "../common/ventus_opencl_test.h"

namespace {

constexpr uint32_t kDescWords = 32;
constexpr uint32_t kDescSets = 2;
constexpr uint32_t kResultWords = 16;
constexpr uint32_t kSmallRegionBytes = 4096;
constexpr uint32_t kLargeRegionBytes = 16384;

enum Path : uint32_t { kBulk = 0, kTensor = 1 };
enum Scenario : uint32_t {
  kG2sS2g = 0,
  kS2gG2s = 1,
  kDualWarp = 2,
  kBurst = 3,
  kG2sOnly = 4,
  kS2gOnly = 5,
};
enum AddressMode : uint32_t { kSameSet = 0, kSpread = 1 };

struct Variant {
  uint32_t max_region_bytes = 0;
  cl_program program = nullptr;
  cl_kernel setup = nullptr;
  cl_kernel samewarp[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
  cl_kernel dualwarp[2] = {nullptr, nullptr};
  cl_kernel burst[2] = {nullptr, nullptr};
  cl_kernel single_direction[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
};

struct Runtime {
  cl_context context = nullptr;
  cl_device_id device = nullptr;
  cl_command_queue queue = nullptr;
  Variant small;
  Variant large;
  std::vector<cl_mem> retained_desc_buffers;
};

struct Metrics {
  uint64_t total_sum = 0;
  uint32_t total_max = 0;
  uint32_t total_min = UINT32_MAX;
  uint64_t g2s_issue_sum = 0;
  uint64_t s2g_issue_sum = 0;
  uint32_t g2s_issue_max = 0;
  uint32_t s2g_issue_max = 0;
  uint64_t completion_tail_sum = 0;
  uint64_t g2s_tail_sum = 0;
  uint64_t s2g_tail_sum = 0;
  uint64_t commit_sum = 0;
  uint64_t arrive_sum = 0;
};

const char *path_name(Path path)
{
  return path == kBulk ? "bulk" : "tensor";
}

const char *scenario_name(Scenario scenario)
{
  switch (scenario) {
    case kG2sS2g: return "g2s_s2g";
    case kS2gG2s: return "s2g_g2s";
    case kDualWarp: return "dualwarp";
    case kBurst: return "burst";
    case kG2sOnly: return "g2s_only";
    case kS2gOnly: return "s2g_only";
  }
  return "unknown";
}

bool parse_path(const char *text, Path *path)
{
  if (std::strcmp(text, "bulk") == 0) {
    *path = kBulk;
    return true;
  }
  if (std::strcmp(text, "tensor") == 0) {
    *path = kTensor;
    return true;
  }
  return false;
}

bool parse_scenario(const char *text, Scenario *scenario)
{
  if (std::strcmp(text, "g2s_s2g") == 0) *scenario = kG2sS2g;
  else if (std::strcmp(text, "s2g_g2s") == 0) *scenario = kS2gG2s;
  else if (std::strcmp(text, "dualwarp") == 0) *scenario = kDualWarp;
  else if (std::strcmp(text, "burst") == 0) *scenario = kBurst;
  else if (std::strcmp(text, "g2s_only") == 0) *scenario = kG2sOnly;
  else if (std::strcmp(text, "s2g_only") == 0) *scenario = kS2gOnly;
  else return false;
  return true;
}

const char *address_mode_name(AddressMode mode)
{
  return mode == kSameSet ? "same_set" : "spread";
}

bool parse_address_mode(const char *text, AddressMode *mode)
{
  if (std::strcmp(text, "same_set") == 0) {
    *mode = kSameSet;
    return true;
  }
  if (std::strcmp(text, "spread") == 0) {
    *mode = kSpread;
    return true;
  }
  return false;
}

uint32_t region_stride(uint32_t bytes, AddressMode mode)
{
  uint32_t same_set_stride = (bytes + 4095u) & ~4095u;
  return mode == kSameSet ? same_set_stride : same_set_stride + 128u;
}

bool parse_u32(const char *text, uint32_t min_value, uint32_t max_value,
               uint32_t *value)
{
  char *end = nullptr;
  unsigned long parsed = std::strtoul(text, &end, 0);
  if (!text[0] || !end || *end || parsed < min_value || parsed > max_value) {
    return false;
  }
  *value = static_cast<uint32_t>(parsed);
  return true;
}

uint32_t pattern_word(uint32_t idx)
{
  uint32_t base = idx << 2;
  uint32_t b0 = (base * 13u + 0x31u) & 0xffu;
  uint32_t b1 = ((base + 1u) * 13u + 0x31u) & 0xffu;
  uint32_t b2 = ((base + 2u) * 13u + 0x31u) & 0xffu;
  uint32_t b3 = ((base + 3u) * 13u + 0x31u) & 0xffu;
  return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

uint32_t desc_control(uint32_t data_type, uint32_t rank)
{
  return (data_type & 0x1fu) | ((rank & 0x7u) << 5);
}

void build_desc(uint32_t *desc, uint32_t columns, uint32_t global_rows,
                uint32_t box_rows)
{
  std::memset(desc, 0, kDescWords * sizeof(uint32_t));
  desc[0] = 0x544d4103u;
  desc[1] = desc_control(7u, 2u);
  desc[4] = columns;
  desc[5] = global_rows;
  desc[9] = columns * sizeof(uint32_t);
  desc[17] = columns;
  desc[18] = box_rows;
  desc[22] = 1u;
  desc[23] = 1u;
}

void release_variant(Variant *variant)
{
  for (auto &path_kernels : variant->single_direction) {
    for (cl_kernel kernel : path_kernels) if (kernel) clReleaseKernel(kernel);
  }
  for (cl_kernel kernel : variant->burst) if (kernel) clReleaseKernel(kernel);
  for (cl_kernel kernel : variant->dualwarp) if (kernel) clReleaseKernel(kernel);
  for (auto &path_kernels : variant->samewarp) {
    for (cl_kernel kernel : path_kernels) if (kernel) clReleaseKernel(kernel);
  }
  if (variant->setup) clReleaseKernel(variant->setup);
  if (variant->program) clReleaseProgram(variant->program);
  *variant = Variant{};
}

void release_runtime(Runtime *runtime)
{
  for (cl_mem buffer : runtime->retained_desc_buffers) {
    if (buffer) clReleaseMemObject(buffer);
  }
  runtime->retained_desc_buffers.clear();
  release_variant(&runtime->large);
  release_variant(&runtime->small);
  if (runtime->queue) clReleaseCommandQueue(runtime->queue);
  if (runtime->context) clReleaseContext(runtime->context);
  runtime->queue = nullptr;
  runtime->context = nullptr;
}

cl_int build_variant(Runtime *runtime, const char *source_path,
                     uint32_t max_region_bytes, Variant *variant)
{
  size_t source_size = 0;
  char *source = ventus_read_text_file(source_path, &source_size);
  if (!source) return CL_INVALID_PROGRAM;
  cl_int err = CL_SUCCESS;
  const char *sources[] = {source};
  const size_t sizes[] = {source_size};
  variant->program = clCreateProgramWithSource(runtime->context, 1, sources,
                                               sizes, &err);
  std::free(source);
  if (err != CL_SUCCESS) return err;

  char source_dir[PATH_MAX];
  std::snprintf(source_dir, sizeof(source_dir), "%s", source_path);
  char *slash = std::strrchr(source_dir, '/');
  if (slash) *slash = '\0';
  char options[PATH_MAX + 128];
  std::snprintf(options, sizeof(options), "-I%s/../common -DMAX_REGION_BYTES=%u",
                source_dir, max_region_bytes);
  err = clBuildProgram(variant->program, 1, &runtime->device, options,
                       nullptr, nullptr);
  if (err != CL_SUCCESS) {
    ventus_print_build_log(variant->program, runtime->device);
    return err;
  }
  variant->setup = clCreateKernel(variant->program, "setup_desc_kernel", &err);
  if (err != CL_SUCCESS) return err;
  const char *samewarp_names[2][2] = {
      {"contention_g2s_s2g_bulk_kernel", "contention_s2g_g2s_bulk_kernel"},
      {"contention_g2s_s2g_tensor_kernel", "contention_s2g_g2s_tensor_kernel"}};
  const char *dualwarp_names[2] = {
      "contention_dualwarp_bulk_kernel", "contention_dualwarp_tensor_kernel"};
  const char *burst_names[2] = {
      "contention_burst_bulk_kernel", "contention_burst_tensor_kernel"};
  const char *single_direction_names[2][2] = {
      {"contention_g2s_only_bulk_kernel", "contention_s2g_only_bulk_kernel"},
      {"contention_g2s_only_tensor_kernel", "contention_s2g_only_tensor_kernel"}};
  for (uint32_t path = 0; path < 2u; ++path) {
    for (uint32_t order = 0; order < 2u; ++order) {
      variant->samewarp[path][order] = clCreateKernel(
          variant->program, samewarp_names[path][order], &err);
      if (err != CL_SUCCESS) return err;
    }
    variant->dualwarp[path] = clCreateKernel(variant->program,
                                             dualwarp_names[path], &err);
    if (err != CL_SUCCESS) return err;
    variant->burst[path] = clCreateKernel(variant->program,
                                          burst_names[path], &err);
    if (err != CL_SUCCESS) return err;
    for (uint32_t direction = 0; direction < 2u; ++direction) {
      variant->single_direction[path][direction] = clCreateKernel(
          variant->program, single_direction_names[path][direction], &err);
      if (err != CL_SUCCESS) return err;
    }
  }
  variant->max_region_bytes = max_region_bytes;
  return CL_SUCCESS;
}

int init_runtime(Runtime *runtime, const char *source_path)
{
  cl_int err = ventus_get_default_device(&runtime->context, &runtime->device,
                                         &runtime->queue, nullptr);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "ventus_get_default_device failed: %d\n", err);
    return 1;
  }
  clReleaseCommandQueue(runtime->queue);
  runtime->queue = clCreateCommandQueue(runtime->context, runtime->device,
                                        CL_QUEUE_PROFILING_ENABLE, &err);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "clCreateCommandQueue failed: %d\n", err);
    return 1;
  }
  err = build_variant(runtime, source_path, kSmallRegionBytes, &runtime->small);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "small kernel build failed: %d\n", err);
    return 1;
  }
  err = build_variant(runtime, source_path, kLargeRegionBytes, &runtime->large);
  if (err != CL_SUCCESS) {
    std::fprintf(stderr, "large kernel build failed: %d\n", err);
    return 1;
  }
  return 0;
}

uint64_t event_ns(cl_event event)
{
  cl_ulong start = 0;
  cl_ulong end = 0;
  if (clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                              sizeof(start), &start, nullptr) != CL_SUCCESS) return 0;
  if (clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                              sizeof(end), &end, nullptr) != CL_SUCCESS) return 0;
  return end >= start ? static_cast<uint64_t>(end - start) : 0;
}

bool validate_copy(const char *name, const std::vector<uint32_t> &got,
                   uint32_t words, uint32_t blocks, uint32_t stride_words)
{
  uint32_t failures = 0;
  for (uint32_t block = 0; block < blocks; ++block) {
    for (uint32_t i = 0; i < words; ++i) {
      uint32_t physical = block * stride_words + i;
      uint32_t logical = block * words + i;
      uint32_t expected = pattern_word(logical);
      if (got[physical] == expected) continue;
      if (failures < 4u) {
        std::fprintf(stderr,
            "FAIL %s block=%u word=%u physical=%u got=0x%08x expected=0x%08x\n",
            name, block, i, physical, got[physical], expected);
      }
      ++failures;
    }
  }
  if (failures) {
    std::fprintf(stderr, "FAIL %s mismatches=%u/%u\n", name, failures,
                 blocks * words);
    return false;
  }
  return true;
}

int run_case(Runtime *runtime, Path path, Scenario scenario, uint32_t bytes,
             uint32_t parallel, AddressMode address_mode = kSpread)
{
  cl_int err = CL_SUCCESS;
  cl_mem desc_buf = nullptr;
  cl_mem input_buf = nullptr;
  cl_mem output_buf = nullptr;
  cl_mem result_buf = nullptr;
  cl_event event = nullptr;
  int rc = 1;
  uint32_t depth = scenario == kBurst ? parallel : 1u;
  uint32_t workgroups = scenario == kBurst ? 1u : parallel;
  uint32_t local_size = scenario == kDualWarp ? 64u : 32u;
  uint32_t regions_per_wg = scenario == kBurst ? depth : 1u;
  uint32_t words = bytes / sizeof(uint32_t);
  uint32_t columns = 32u;
  uint32_t box_rows = words / columns;
  uint32_t blocks = workgroups * regions_per_wg;
  uint32_t stride_bytes = scenario == kBurst ? bytes
      : region_stride(bytes, address_mode);
  uint32_t stride_words = stride_bytes / sizeof(uint32_t);
  uint32_t total_words = (blocks - 1u) * stride_words + words;
  uint32_t region_footprint = bytes * regions_per_wg;
  Variant *variant = region_footprint <= kSmallRegionBytes
                         ? &runtime->small : &runtime->large;
  cl_kernel kernel = scenario == kBurst ? variant->burst[path]
      : (scenario == kDualWarp ? variant->dualwarp[path]
         : (scenario == kG2sOnly || scenario == kS2gOnly
              ? variant->single_direction[path][scenario == kS2gOnly ? 1 : 0]
              : variant->samewarp[path][scenario]));

  if ((bytes & 127u) || bytes == 0u || bytes > kLargeRegionBytes ||
      region_footprint > variant->max_region_bytes ||
      (scenario == kBurst && depth != 2u && depth != 4u) ||
      (scenario != kBurst && (workgroups < 1u || workgroups > 4u))) {
    std::fprintf(stderr, "invalid case bytes=%u scenario=%s parallel=%u\n",
                 bytes, scenario_name(scenario), parallel);
    return 1;
  }

  std::vector<uint32_t> input(total_words);
  std::vector<uint32_t> output(total_words, 0u);
  std::vector<uint32_t> results(workgroups * kResultWords, 0u);
  uint32_t coord_blocks = scenario == kBurst ? depth : workgroups;
  std::vector<uint32_t> coords(coord_blocks * 32u, 0u);
  std::vector<uint32_t> desc(kDescWords * kDescSets + coords.size(), 0u);
  for (uint32_t block = 0; block < blocks; ++block) {
    for (uint32_t i = 0; i < words; ++i) {
      input[block * stride_words + i] = pattern_word(block * words + i);
    }
  }
  for (uint32_t block = 0; block < coord_blocks; ++block) {
    coords[block * 32u] = 0u;
    coords[block * 32u + 1u] = block * (stride_bytes / 128u);
  }
  uint32_t global_rows = (coord_blocks - 1u) * (stride_bytes / 128u) + box_rows;
  build_desc(desc.data(), columns, global_rows, box_rows);
  build_desc(desc.data() + kDescWords, columns,
             global_rows, box_rows);
  std::copy(coords.begin(), coords.end(),
            desc.begin() + kDescWords * kDescSets);

  desc_buf = clCreateBuffer(runtime->context,
      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      desc.size() * sizeof(uint32_t), desc.data(), &err);
  if (err != CL_SUCCESS) goto FINISH;
  input_buf = clCreateBuffer(runtime->context,
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input.size() * sizeof(uint32_t),
      input.data(), &err);
  if (err != CL_SUCCESS) goto FINISH;
  output_buf = clCreateBuffer(runtime->context,
      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      output.size() * sizeof(uint32_t), output.data(), &err);
  if (err != CL_SUCCESS) goto FINISH;
  result_buf = clCreateBuffer(runtime->context,
      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      results.size() * sizeof(uint32_t), results.data(), &err);
  if (err != CL_SUCCESS) goto FINISH;

  err = clSetKernelArg(variant->setup, 0, sizeof(desc_buf), &desc_buf);
  if (err != CL_SUCCESS) goto FINISH;
  err = clSetKernelArg(variant->setup, 1, sizeof(input_buf), &input_buf);
  if (err != CL_SUCCESS) goto FINISH;
  err = clSetKernelArg(variant->setup, 2, sizeof(output_buf), &output_buf);
  if (err != CL_SUCCESS) goto FINISH;
  {
    size_t one = 1;
    err = clEnqueueNDRangeKernel(runtime->queue, variant->setup, 1, nullptr,
                                 &one, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) goto FINISH;
  }
  err = clFinish(runtime->queue);
  if (err != CL_SUCCESS) goto FINISH;

  if (path == kBulk) {
    if (scenario == kS2gOnly) {
      err  = clSetKernelArg(kernel, 0, sizeof(output_buf), &output_buf);
      err |= clSetKernelArg(kernel, 1, sizeof(result_buf), &result_buf);
      err |= clSetKernelArg(kernel, 2, sizeof(bytes), &bytes);
      err |= clSetKernelArg(kernel, 3, sizeof(stride_bytes), &stride_bytes);
    } else {
      err  = clSetKernelArg(kernel, 0, sizeof(input_buf), &input_buf);
      err |= clSetKernelArg(kernel, 1, sizeof(output_buf), &output_buf);
      err |= clSetKernelArg(kernel, 2, sizeof(result_buf), &result_buf);
      err |= clSetKernelArg(kernel, 3, sizeof(bytes), &bytes);
      if (scenario == kBurst) {
        err |= clSetKernelArg(kernel, 4, sizeof(depth), &depth);
      } else {
        err |= clSetKernelArg(kernel, 4, sizeof(stride_bytes), &stride_bytes);
      }
    }
    if (err != CL_SUCCESS) goto FINISH;
  } else {
    if (scenario == kG2sOnly) {
      err  = clSetKernelArg(kernel, 0, sizeof(desc_buf), &desc_buf);
      err |= clSetKernelArg(kernel, 1, sizeof(output_buf), &output_buf);
      err |= clSetKernelArg(kernel, 2, sizeof(result_buf), &result_buf);
      err |= clSetKernelArg(kernel, 3, sizeof(bytes), &bytes);
      err |= clSetKernelArg(kernel, 4, sizeof(stride_bytes), &stride_bytes);
    } else {
      err  = clSetKernelArg(kernel, 0, sizeof(desc_buf), &desc_buf);
      err |= clSetKernelArg(kernel, 1, sizeof(result_buf), &result_buf);
      err |= clSetKernelArg(kernel, 2, sizeof(bytes), &bytes);
      if (scenario == kBurst) {
        err |= clSetKernelArg(kernel, 3, sizeof(depth), &depth);
      }
    }
    if (err != CL_SUCCESS) goto FINISH;
  }
  {
    size_t global = static_cast<size_t>(workgroups) * local_size;
    size_t local = local_size;
    err = clEnqueueNDRangeKernel(runtime->queue, kernel, 1, nullptr,
                                 &global, &local, 0, nullptr, &event);
    if (err != CL_SUCCESS) goto FINISH;
  }
  err = clWaitForEvents(1, &event);
  if (err != CL_SUCCESS) goto FINISH;
  err = clEnqueueReadBuffer(runtime->queue, output_buf, CL_TRUE, 0,
                            output.size() * sizeof(uint32_t), output.data(),
                            0, nullptr, nullptr);
  if (err != CL_SUCCESS) goto FINISH;
  err = clEnqueueReadBuffer(runtime->queue, result_buf, CL_TRUE, 0,
                            results.size() * sizeof(uint32_t), results.data(),
                            0, nullptr, nullptr);
  if (err != CL_SUCCESS) goto FINISH;

  if (!validate_copy(scenario == kG2sOnly ? "g2s-readback" : "s2g-output",
                     output, words, blocks, stride_words)) goto FINISH;

  {
    Metrics metrics;
    uint32_t samples = workgroups;
    bool status_ok = true;
    for (uint32_t wg = 0; wg < workgroups; ++wg) {
      const uint32_t *r = &results[wg * kResultWords];
      uint32_t total = 0;
      uint32_t g2s_issue = 0;
      uint32_t s2g_issue = 0;
      uint32_t completion_tail = 0;
      uint32_t g2s_tail = 0;
      uint32_t s2g_tail = 0;
      if (scenario == kG2sS2g || scenario == kS2gG2s) {
        total = r[0];
        g2s_issue = scenario == kG2sS2g ? r[1] : r[2];
        s2g_issue = scenario == kG2sS2g ? r[2] : r[1];
        completion_tail = r[3];
        status_ok &= r[4] == 0u;
      } else if (scenario == kDualWarp) {
        total = std::max(r[0], r[4]);
        g2s_issue = r[1];
        s2g_issue = r[5];
        g2s_tail = r[2];
        s2g_tail = r[6];
        status_ok &= r[3] == 0u && r[7] == 0u;
      } else if (scenario == kG2sOnly) {
        total = r[0];
        g2s_issue = r[1];
        completion_tail = r[2];
        g2s_tail = r[2];
        status_ok &= r[3] == 0u;
      } else if (scenario == kS2gOnly) {
        total = r[4];
        s2g_issue = r[5];
        completion_tail = r[6];
        s2g_tail = r[6];
        status_ok &= r[7] == 0u;
      } else {
        total = r[0];
        g2s_issue = r[1];
        s2g_issue = r[2];
        completion_tail = r[5] + r[6];
        metrics.g2s_issue_max = std::max(metrics.g2s_issue_max, r[3]);
        metrics.s2g_issue_max = std::max(metrics.s2g_issue_max, r[4]);
        metrics.commit_sum += r[8];
        metrics.arrive_sum += r[9];
        status_ok &= r[7] == 0u;
      }
      metrics.total_sum += total;
      metrics.total_max = std::max(metrics.total_max, total);
      metrics.total_min = std::min(metrics.total_min, total);
      metrics.g2s_issue_sum += g2s_issue;
      metrics.s2g_issue_sum += s2g_issue;
      if (scenario != kBurst) {
        metrics.g2s_issue_max = std::max(metrics.g2s_issue_max, g2s_issue);
        metrics.s2g_issue_max = std::max(metrics.s2g_issue_max, s2g_issue);
      }
      metrics.completion_tail_sum += completion_tail;
      metrics.g2s_tail_sum += g2s_tail;
      metrics.s2g_tail_sum += s2g_tail;
    }
    if (!status_ok || metrics.total_max == 0u) {
      std::fprintf(stderr, "FAIL status/cycle path=%s scenario=%s\n",
                   path_name(path), scenario_name(scenario));
      for (uint32_t wg = 0; wg < workgroups; ++wg) {
        const uint32_t *r = &results[wg * kResultWords];
        std::fprintf(stderr,
            "  wg=%u r0..r9=%u,%u,%u,%u,%u,%u,%u,%u,%u,%u\n",
            wg, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9]);
      }
      goto FINISH;
    }
    std::printf(
        "CONTENTION_RESULT status=PASS path=%s scenario=%s bytes=%u "
        "address_mode=%s region_stride=%u workgroups=%u warps_per_wg=%u "
        "depth=%u total_avg=%.2f total_min=%u total_max=%u fairness=%.4f "
        "g2s_issue_avg=%.2f s2g_issue_avg=%.2f "
        "g2s_issue_max=%u s2g_issue_max=%u completion_tail_avg=%.2f "
        "g2s_tail_avg=%.2f s2g_tail_avg=%.2f commit_avg=%.2f "
        "arrive_expect_avg=%.2f "
        "host_ns=%" PRIu64 "\n",
        path_name(path), scenario_name(scenario), bytes,
        scenario == kBurst ? "compact" : address_mode_name(address_mode),
        stride_bytes, workgroups, local_size / 32u, depth,
        static_cast<double>(metrics.total_sum) / samples, metrics.total_min,
        metrics.total_max,
        static_cast<double>(metrics.total_max) * samples / metrics.total_sum,
        static_cast<double>(metrics.g2s_issue_sum) / samples,
        static_cast<double>(metrics.s2g_issue_sum) / samples,
        metrics.g2s_issue_max, metrics.s2g_issue_max,
        static_cast<double>(metrics.completion_tail_sum) / samples,
        static_cast<double>(metrics.g2s_tail_sum) / samples,
        static_cast<double>(metrics.s2g_tail_sum) / samples,
        static_cast<double>(metrics.commit_sum) / samples,
        static_cast<double>(metrics.arrive_sum) / samples, event_ns(event));
  }
  rc = 0;

FINISH:
  if (rc != 0 && err != CL_SUCCESS) {
    std::fprintf(stderr, "OpenCL failure path=%s scenario=%s error=%d\n",
                 path_name(path), scenario_name(scenario), err);
  }
  if (event) clReleaseEvent(event);
  if (result_buf) clReleaseMemObject(result_buf);
  if (output_buf) clReleaseMemObject(output_buf);
  if (input_buf) clReleaseMemObject(input_buf);
  if (desc_buf) runtime->retained_desc_buffers.push_back(desc_buf);
  return rc;
}

int run_child_case(const char *executable, Path path, Scenario scenario,
                   uint32_t bytes, uint32_t parallel,
                   AddressMode address_mode = kSpread)
{
  char bytes_text[16];
  char parallel_text[16];
  std::snprintf(bytes_text, sizeof(bytes_text), "%u", bytes);
  std::snprintf(parallel_text, sizeof(parallel_text), "%u", parallel);
  std::printf(
      "CONTENTION_CHILD_START path=%s scenario=%s bytes=%u parallel=%u address_mode=%s\n",
      path_name(path), scenario_name(scenario), bytes, parallel,
      address_mode_name(address_mode));
  std::fflush(nullptr);
  pid_t pid = fork();
  if (pid < 0) {
    std::perror("fork contention child");
    return 1;
  }
  if (pid == 0) {
    execl(executable, executable, "single", path_name(path),
          scenario_name(scenario), bytes_text, parallel_text,
          address_mode_name(address_mode), static_cast<char *>(nullptr));
    std::perror("exec contention child");
    _exit(127);
  }

  int status = 0;
  while (waitpid(pid, &status, 0) < 0) {
    if (errno == EINTR) continue;
    std::perror("waitpid contention child");
    return 1;
  }
  if (WIFEXITED(status)) {
    const int rc = WEXITSTATUS(status);
    std::printf(
        "CONTENTION_CHILD_DONE status=%s path=%s scenario=%s bytes=%u parallel=%u address_mode=%s rc=%d\n",
        rc == 0 ? "PASS" : "FAIL", path_name(path), scenario_name(scenario),
        bytes, parallel, address_mode_name(address_mode), rc);
    return rc == 0 ? 0 : 1;
  }
  if (WIFSIGNALED(status)) {
    std::fprintf(stderr,
        "CONTENTION_CHILD_SIGNAL path=%s scenario=%s bytes=%u parallel=%u signal=%d\n",
        path_name(path), scenario_name(scenario), bytes, parallel,
        WTERMSIG(status));
  }
  return 1;
}

int run_sweep(const char *executable, int path_filter = -1)
{
  const uint32_t sizes[] = {128u, 1024u, 4096u, 16384u};
  const char *backend = std::getenv("VENTUS_BACKEND");
  const bool gvm_backend = backend && std::strncmp(backend, "gvm", 3) == 0;
  int failures = 0;
  int cases = 0;
  for (Path path : {kBulk, kTensor}) {
    if (path_filter >= 0 && static_cast<int>(path) != path_filter) continue;

    // Direction-isolated pressure matrix.  Work-group 1 has no set-phase
    // distinction; work-groups 2..4 run both aligned-set and rotated-set bases.
    for (uint32_t bytes : sizes) {
      for (uint32_t workgroups : {1u, 2u, 3u, 4u}) {
        for (AddressMode mode : {kSpread, kSameSet}) {
          if (workgroups == 1u && mode == kSameSet) continue;
          failures += run_child_case(executable, path, kG2sOnly, bytes,
                                     workgroups, mode);
          failures += run_child_case(executable, path, kS2gOnly, bytes,
                                     workgroups, mode);
          cases += 2;
        }
      }
    }

    // Bidirectional overlap is intentionally smaller than the isolated
    // matrix; it targets the known large-command and 4WG pressure points.
    for (uint32_t bytes : {4096u, 16384u}) {
      for (uint32_t workgroups : {1u, 2u, 4u}) {
        failures += run_child_case(executable, path, kG2sS2g, bytes,
                                   workgroups, kSpread);
        failures += run_child_case(executable, path, kS2gG2s, bytes,
                                   workgroups, kSpread);
        cases += 2;
      }
      failures += run_child_case(executable, path, kG2sS2g, bytes, 4u,
                                 kSameSet);
      failures += run_child_case(executable, path, kS2gG2s, bytes, 4u,
                                 kSameSet);
      cases += 2;
    }

    for (uint32_t bytes : {4096u, 16384u}) {
      for (uint32_t workgroups : {1u, 2u, 4u}) {
        // RTL completes these points, but GVM lockstep checking loses PC/data
        // agreement under 8 concurrent warps. Keep them available through
        // `single`; 4WG and dual-warp pressure remain covered independently.
        if (gvm_backend && workgroups == 4u) {
          std::printf(
              "CONTENTION_SKIP path=%s scenario=dualwarp bytes=%u "
              "workgroups=4 reason=gvm_lockstep_limit\n",
              path_name(path), bytes);
          continue;
        }
        failures += run_child_case(executable, path, kDualWarp, bytes,
                                   workgroups, kSpread);
        ++cases;
      }
    }
    for (uint32_t bytes : {128u, 1024u, 4096u}) {
      for (uint32_t depth : {2u, 4u}) {
        if (gvm_backend && path == kTensor && bytes >= 4096u) {
          std::printf(
              "CONTENTION_SKIP path=tensor scenario=burst bytes=%u depth=%u "
              "reason=gvm_lockstep_limit\n",
              bytes, depth);
          continue;
        }
        // Bulk depth=4 intentionally fills the command/group window and can
        // stop GVM lockstep checking at the backpressure point. Keep it
        // available through `single`, but isolate it from the main sweep.
        if (path == kBulk && (depth == 4u || bytes >= 4096u)) continue;
        if (bytes * depth <= kLargeRegionBytes) {
          failures += run_child_case(executable, path, kBurst, bytes, depth);
          ++cases;
        }
      }
    }
  }
  std::printf("CONTENTION_SUMMARY cases=%d failures=%d\n", cases, failures);
  return failures ? 1 : 0;
}

void usage(const char *argv0)
{
  std::fprintf(stderr,
      "usage:\n"
      "  %s [sweep]\n"
      "  %s sweep <bulk|tensor>\n"
      "  %s single <bulk|tensor> "
      "<g2s_only|s2g_only|g2s_s2g|s2g_g2s|dualwarp|burst> "
      "<bytes> <wg-or-depth> [same_set|spread]\n",
      argv0, argv0, argv0);
}

}  // namespace

int main(int argc, char **argv)
{
  char executable[PATH_MAX];
  if (!realpath(argv[0], executable)) {
    std::perror("realpath(contention executable)");
    return 1;
  }

  if (argc == 1 || (argc == 2 && std::strcmp(argv[1], "sweep") == 0)) {
    return run_sweep(executable);
  }
  if (argc == 3 && std::strcmp(argv[1], "sweep") == 0) {
    Path path;
    if (!parse_path(argv[2], &path)) {
      usage(argv[0]);
      return 2;
    }
    return run_sweep(executable, static_cast<int>(path));
  }

  char source_path[PATH_MAX];
  if (!realpath("dma_tma_bidirectional_contention_perf_test.cl", source_path)) {
    std::perror("realpath(kernel source)");
    return 1;
  }
  Runtime runtime;
  if (init_runtime(&runtime, source_path) != 0) {
    release_runtime(&runtime);
    return 1;
  }

  int rc = 0;
  if ((argc == 6 || argc == 7) &&
             std::strcmp(argv[1], "single") == 0) {
    Path path;
    Scenario scenario;
    AddressMode address_mode = kSpread;
    uint32_t bytes = 0;
    uint32_t parallel = 0;
    if (!parse_path(argv[2], &path) || !parse_scenario(argv[3], &scenario) ||
        !parse_u32(argv[4], 128u, kLargeRegionBytes, &bytes) ||
        !parse_u32(argv[5], 1u, 4u, &parallel) ||
        (argc == 7 && !parse_address_mode(argv[6], &address_mode))) {
      usage(argv[0]);
      rc = 1;
    } else {
      rc = run_case(&runtime, path, scenario, bytes, parallel, address_mode);
    }
  } else {
    usage(argv[0]);
    rc = 1;
  }

  release_runtime(&runtime);
  return rc;
}
