#include <CL/cl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../common/ventus_opencl_test.h"
#include "../common/ventus_tma_v2_spec.h"

namespace {

constexpr size_t kWgSize = 32;
constexpr uint32_t kMaxWords = 128;
constexpr uint32_t kMaxSplits = 8;
constexpr uint32_t kInitialValue = 100;
constexpr int kMeasuredRuns = 3;

struct Env {
  cl_context context = nullptr;
  cl_device_id device = nullptr;
  cl_command_queue queue = nullptr;
  cl_program program = nullptr;
  cl_kernel patch = nullptr;
  cl_kernel tma = nullptr;
  cl_kernel partial = nullptr;
  cl_kernel finalize = nullptr;

  bool init() {
    cl_int error =
        ventus_get_default_device(&context, &device, &queue, nullptr);
    if (error != CL_SUCCESS) return false;
    error = ventus_build_program_from_source(
        context, device, "dma_tma_v2_reduce_perf_test.cl", &program);
    if (error != CL_SUCCESS) return false;
    patch = clCreateKernel(program, "patch_descriptor_base", &error);
    if (error != CL_SUCCESS) return false;
    tma = clCreateKernel(program, "tma_reduce_add_perf", &error);
    if (error != CL_SUCCESS) return false;
    partial = clCreateKernel(program, "write_partial_perf", &error);
    if (error != CL_SUCCESS) return false;
    finalize = clCreateKernel(program, "finalize_partial_perf", &error);
    return error == CL_SUCCESS;
  }

  ~Env() {
    if (finalize) clReleaseKernel(finalize);
    if (partial) clReleaseKernel(partial);
    if (tma) clReleaseKernel(tma);
    if (patch) clReleaseKernel(patch);
    if (program) clReleaseProgram(program);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
  }
};

struct Buffers {
  std::vector<cl_mem> items;
  ~Buffers() {
    for (cl_mem item : items) clReleaseMemObject(item);
  }
};

cl_mem make_buffer(Env &env, Buffers &buffers, cl_mem_flags flags,
                   size_t bytes, void *data = nullptr) {
  cl_int error = CL_SUCCESS;
  cl_mem buffer = clCreateBuffer(env.context, flags, bytes, data, &error);
  if (error != CL_SUCCESS) return nullptr;
  buffers.items.push_back(buffer);
  return buffer;
}

bool launch(Env &env, cl_kernel kernel, size_t global, size_t local) {
  cl_int error = clEnqueueNDRangeKernel(
      env.queue, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);
  return error == CL_SUCCESS && clFinish(env.queue) == CL_SUCCESS;
}

template <typename T>
bool write_buffer(Env &env, cl_mem buffer, const std::vector<T> &data) {
  return clEnqueueWriteBuffer(env.queue, buffer, CL_TRUE, 0,
                              data.size() * sizeof(T), data.data(),
                              0, nullptr, nullptr) == CL_SUCCESS;
}

template <typename T>
bool read_buffer(Env &env, cl_mem buffer, std::vector<T> *data) {
  return clEnqueueReadBuffer(env.queue, buffer, CL_TRUE, 0,
                             data->size() * sizeof(T), data->data(),
                             0, nullptr, nullptr) == CL_SUCCESS;
}

uint32_t elapsed(uint32_t begin, uint32_t end) {
  return end - begin;
}

uint32_t workgroup_span(const std::vector<uint32_t> &times,
                        uint32_t workgroups) {
  uint32_t earliest = times[0];
  uint32_t latest = times[1];
  for (uint32_t group = 1; group < workgroups; ++group) {
    earliest = std::min(earliest, times[group * 2]);
    latest = std::max(latest, times[group * 2 + 1]);
  }
  return elapsed(earliest, latest);
}

uint32_t median(std::vector<uint32_t> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

bool set_arg(cl_kernel kernel, cl_uint index, size_t bytes,
             const void *value) {
  return clSetKernelArg(kernel, index, bytes, value) == CL_SUCCESS;
}

bool reset_output(Env &env, cl_mem output) {
  std::vector<uint32_t> initial(kMaxWords, kInitialValue);
  return write_buffer(env, output, initial);
}

bool validate_output(Env &env, cl_mem output, uint32_t words,
                     uint32_t splits, const char *path) {
  std::vector<uint32_t> values(kMaxWords);
  if (!read_buffer(env, output, &values)) return false;
  const uint32_t expected =
      kInitialValue + splits * (splits + 1u) / 2u;
  for (uint32_t index = 0; index < words; ++index) {
    if (values[index] != expected) {
      std::cerr << "FAIL reduce_perf path=" << path
                << " words=" << words << " splits=" << splits
                << " index=" << index << " got=" << values[index]
                << " expected=" << expected << '\n';
      return false;
    }
  }
  return true;
}

bool patch_descriptor(Env &env, cl_mem descriptor, cl_mem output) {
  return set_arg(env.patch, 0, sizeof(descriptor), &descriptor) &&
         set_arg(env.patch, 1, sizeof(output), &output) &&
         launch(env, env.patch, 1, 1);
}

bool configure_descriptor(Env &env, cl_mem descriptor, cl_mem output,
                          uint32_t words) {
  std::array<uint32_t, 32> data{};
  data[0] = VENTUS_TMA_V2_MAGIC;
  data[1] = VENTUS_TMA_DTYPE_U32 | (1u << 5);
  data[4] = words;
  data[17] = words;
  data[22] = 1u;
  if (clEnqueueWriteBuffer(env.queue, descriptor, CL_TRUE, 0, sizeof(data),
                           data.data(), 0, nullptr, nullptr) != CL_SUCCESS) {
    return false;
  }
  return patch_descriptor(env, descriptor, output);
}

bool run_tma(Env &env, cl_mem descriptor, cl_mem output, cl_mem status,
             cl_mem times_buffer, uint32_t words, uint32_t splits,
             uint32_t *cycles) {
  std::vector<uint32_t> status_data(kMaxSplits, 0xffffffffu);
  std::vector<uint32_t> times(kMaxSplits * 2u, 0u);
  if (!reset_output(env, output) ||
      !write_buffer(env, status, status_data) ||
      !write_buffer(env, times_buffer, times) ||
      !set_arg(env.tma, 0, sizeof(descriptor), &descriptor) ||
      !set_arg(env.tma, 1, sizeof(status), &status) ||
      !set_arg(env.tma, 2, sizeof(times_buffer), &times_buffer) ||
      !set_arg(env.tma, 3, sizeof(words), &words) ||
      !launch(env, env.tma, splits * kWgSize, kWgSize) ||
      !read_buffer(env, status, &status_data) ||
      !read_buffer(env, times_buffer, &times)) {
    return false;
  }
  for (uint32_t group = 0; group < splits; ++group) {
    if (status_data[group] != VENTUS_TMA_STATUS_OK) {
      std::cerr << "FAIL reduce_perf TMA status group=" << group
                << " value=0x" << std::hex << status_data[group]
                << std::dec << '\n';
      return false;
    }
  }
  *cycles = workgroup_span(times, splits);
  return validate_output(env, output, words, splits, "tma");
}

bool run_baseline(Env &env, cl_mem output, cl_mem partials,
                  cl_mem partial_times_buffer, cl_mem final_times_buffer,
                  uint32_t words, uint32_t splits, uint32_t *cycles) {
  std::vector<uint32_t> partial_times(kMaxSplits * 2u, 0u);
  std::vector<uint32_t> final_times(2u, 0u);
  if (!reset_output(env, output) ||
      !write_buffer(env, partial_times_buffer, partial_times) ||
      !write_buffer(env, final_times_buffer, final_times) ||
      !set_arg(env.partial, 0, sizeof(partials), &partials) ||
      !set_arg(env.partial, 1, sizeof(partial_times_buffer),
               &partial_times_buffer) ||
      !set_arg(env.partial, 2, sizeof(words), &words) ||
      !launch(env, env.partial, splits * kWgSize, kWgSize) ||
      !read_buffer(env, partial_times_buffer, &partial_times) ||
      !set_arg(env.finalize, 0, sizeof(partials), &partials) ||
      !set_arg(env.finalize, 1, sizeof(output), &output) ||
      !set_arg(env.finalize, 2, sizeof(final_times_buffer),
               &final_times_buffer) ||
      !set_arg(env.finalize, 3, sizeof(words), &words) ||
      !set_arg(env.finalize, 4, sizeof(splits), &splits) ||
      !launch(env, env.finalize, kWgSize, kWgSize) ||
      !read_buffer(env, final_times_buffer, &final_times)) {
    return false;
  }
  *cycles = workgroup_span(partial_times, splits) +
            elapsed(final_times[0], final_times[1]);
  return validate_output(env, output, words, splits, "temporary");
}

}  // namespace

int main(int argc, char **argv) {
  static_assert(VENTUS_TMA_V2_REDUCE_ATOMIC == 1,
                "benchmark requires the shared atomic endpoint");
  if (argc != 1 && argc != 3) {
    std::cerr << "usage: " << argv[0] << " [words splits]\n";
    return 2;
  }
  const uint32_t requested_words =
      argc == 3 ? static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 0)) : 0;
  const uint32_t requested_splits =
      argc == 3 ? static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 0)) : 0;
  if (argc == 3 &&
      ((requested_words != 8 && requested_words != 32 &&
        requested_words != 128) ||
       (requested_splits != 1 && requested_splits != 2 &&
        requested_splits != 4 && requested_splits != 8))) {
    std::cerr << "FAIL reduce_perf unsupported case words=" << requested_words
              << " splits=" << requested_splits << '\n';
    return 2;
  }
  Env env;
  Buffers buffers;
  if (!env.init()) {
    std::cerr << "FAIL reduce_perf OpenCL initialization\n";
    return 1;
  }

  std::array<cl_mem, 3> descriptors{};
  for (cl_mem &descriptor : descriptors) {
    descriptor = make_buffer(
        env, buffers, CL_MEM_READ_WRITE,
        VENTUS_TMA_V2_DESCRIPTOR_STORAGE_BYTES);
  }
  cl_mem output = make_buffer(
      env, buffers, CL_MEM_READ_WRITE, kMaxWords * sizeof(uint32_t));
  cl_mem partials = make_buffer(
      env, buffers, CL_MEM_READ_WRITE,
      kMaxWords * kMaxSplits * sizeof(uint32_t));
  cl_mem status = make_buffer(
      env, buffers, CL_MEM_READ_WRITE, kMaxSplits * sizeof(uint32_t));
  cl_mem tma_times = make_buffer(
      env, buffers, CL_MEM_READ_WRITE,
      kMaxSplits * 2u * sizeof(uint32_t));
  cl_mem partial_times = make_buffer(
      env, buffers, CL_MEM_READ_WRITE,
      kMaxSplits * 2u * sizeof(uint32_t));
  cl_mem final_times = make_buffer(
      env, buffers, CL_MEM_READ_WRITE, 2u * sizeof(uint32_t));
  if (std::any_of(descriptors.begin(), descriptors.end(),
                  [](cl_mem descriptor) { return descriptor == nullptr; }) ||
      !output || !partials || !status || !tma_times ||
      !partial_times || !final_times) {
    std::cerr << "FAIL reduce_perf buffer allocation\n";
    return 1;
  }

  const std::array<uint32_t, 3> word_cases{{8, 32, 128}};
  const std::array<uint32_t, 4> split_cases{{1, 2, 4, 8}};
  for (size_t index = 0; index < word_cases.size(); ++index) {
    if (!configure_descriptor(
          env, descriptors[index], output, word_cases[index])) {
      std::cerr << "FAIL reduce_perf descriptor initialization\n";
      return 1;
    }
  }
  std::array<uint32_t, split_cases.size()> first_crossover{};
  double min_speedup = 1.0e30;
  double max_speedup = 0.0;
  int failures = 0;

  for (size_t split_index = 0; split_index < split_cases.size();
       ++split_index) {
    const uint32_t splits = split_cases[split_index];
    if (requested_splits != 0 && splits != requested_splits) continue;
    for (size_t word_index = 0; word_index < word_cases.size(); ++word_index) {
      const uint32_t words = word_cases[word_index];
      const cl_mem descriptor = descriptors[word_index];
      if (requested_words != 0 && words != requested_words) continue;
      uint32_t warm_cycles = 0;
      bool ok = run_tma(env, descriptor, output, status, tma_times,
                        words, splits, &warm_cycles) &&
                run_baseline(env, output, partials, partial_times, final_times,
                             words, splits, &warm_cycles);
      std::vector<uint32_t> tma_samples;
      std::vector<uint32_t> baseline_samples;
      for (int run = 0; ok && run < kMeasuredRuns; ++run) {
        uint32_t tma_cycles = 0;
        uint32_t baseline_cycles = 0;
        ok = run_tma(env, descriptor, output, status, tma_times,
                     words, splits, &tma_cycles) &&
             run_baseline(env, output, partials, partial_times, final_times,
                          words, splits, &baseline_cycles);
        if (ok) {
          tma_samples.push_back(tma_cycles);
          baseline_samples.push_back(baseline_cycles);
        }
      }
      if (!ok) {
        ++failures;
        continue;
      }
      const uint32_t tma_cycles = median(tma_samples);
      const uint32_t baseline_cycles = median(baseline_samples);
      const uint32_t atomics = words * splits;
      const double throughput =
          static_cast<double>(atomics) / static_cast<double>(tma_cycles);
      const double speedup =
          static_cast<double>(baseline_cycles) / static_cast<double>(tma_cycles);
      min_speedup = std::min(min_speedup, speedup);
      max_speedup = std::max(max_speedup, speedup);
      if (speedup >= 1.0 && first_crossover[split_index] == 0)
        first_crossover[split_index] = words;
      std::cout << std::dec
                << "REDUCE_PERF status=PASS backend="
                << (std::getenv("VENTUS_BACKEND")
                        ? std::getenv("VENTUS_BACKEND") : "default")
                << " words=" << words
                << " splits=" << splits
                << " atomics=" << atomics
                << " tma_cycles=" << tma_cycles
                << " temp_two_kernel_cycles=" << baseline_cycles
                << " atomics_per_cycle=" << std::fixed
                << std::setprecision(4) << throughput
                << " speedup=" << speedup << "x\n";
    }
  }

  for (size_t index = 0; index < split_cases.size(); ++index) {
    if (requested_splits != 0 && split_cases[index] != requested_splits)
      continue;
    std::cout << std::dec
              << "REDUCE_CROSSOVER splits=" << split_cases[index]
              << " first_words=";
    if (first_crossover[index] == 0)
      std::cout << "none";
    else
      std::cout << first_crossover[index];
    std::cout << '\n';
  }
  const int expected_cases = requested_words == 0 ? 12 : 1;
  std::cout << std::dec
            << "REDUCE_PERF_SUMMARY cases=" << expected_cases
            << " failures=" << failures
            << " min_speedup=" << std::fixed << std::setprecision(4)
            << (failures == expected_cases ? 0.0 : min_speedup)
            << " max_speedup=" << max_speedup << '\n';
  return failures == 0 ? 0 : 1;
}
