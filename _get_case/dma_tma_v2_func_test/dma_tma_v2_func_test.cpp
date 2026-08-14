#include <CL/cl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tma_model.h"
#include "ventus_opencl_test.h"

static_assert(VENTUS_TMA_V2_REDUCE_ATOMIC == 1,
              "functional model expects atomic element reduction");
static_assert(VENTUS_TMA_V2_REDUCE_REQUIRES_EXCLUSIVE_DESTINATION == 0,
              "atomic reduce destinations may overlap");

using ventus::tma::DecodeAndPlan;
using ventus::tma::Descriptor;
using ventus::tma::Request;
using ventus::tma::Status;

namespace {

constexpr size_t kWgSize = 32;
constexpr size_t kTensorSharedCapacity = 8192;
constexpr size_t kGlobalCapacity = 65536;
constexpr uint8_t kSentinel = 0xcd;

struct TensorCase {
  std::string name;
  uint8_t dtype = 0;
  uint8_t rank = 1;
  uint8_t interleave = 0;
  uint8_t swizzle = 0;
  uint8_t oob = 0;
  std::array<uint32_t, 5> dims{};
  std::array<uint32_t, 4> strides{};
  std::array<uint32_t, 5> box{};
  std::array<int32_t, 5> coords{};
  bool run_g2s = true;
  bool run_s2g = true;
  bool prefetch = false;
};

struct InvalidCase {
  TensorCase tensor;
  uint8_t direction;
};

struct ReduceCase {
  std::string name;
  TensorCase tensor;
  uint32_t mode;
  std::vector<uint32_t> initial;
  std::vector<uint32_t> operand;
  std::vector<uint32_t> expected;
};

struct BulkReduceCase {
  std::string name;
  uint32_t mode;
  uint32_t type;
  std::vector<uint32_t> initial;
  std::vector<uint32_t> operand;
  std::vector<uint32_t> expected;
};

struct OpenClEnv {
  cl_context context = nullptr;
  cl_device_id device = nullptr;
  cl_command_queue queue = nullptr;
  cl_program program = nullptr;

  bool Init() {
    cl_int err = ventus_get_default_device(&context, &device, &queue, nullptr);
    if (err != CL_SUCCESS) {
      std::cerr << "FAIL OpenCL initialization error=" << err << '\n';
      return false;
    }
    err = ventus_build_program_from_source(context, device,
                                           "dma_tma_v2_func_test.cl", &program);
    if (err != CL_SUCCESS) {
      std::cerr << "FAIL OpenCL kernel build error=" << err << '\n';
      return false;
    }
    return true;
  }

  ~OpenClEnv() {
    if (program) clReleaseProgram(program);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
  }
};

struct BufferPool {
  std::vector<cl_mem> buffers;
  ~BufferPool() {
    for (cl_mem buffer : buffers) clReleaseMemObject(buffer);
  }
};

Descriptor Encode(const TensorCase &c) {
  Descriptor descriptor{};
  descriptor.words[VENTUS_TMA_V2_WORD_MAGIC] = VENTUS_TMA_V2_MAGIC;
  descriptor.words[VENTUS_TMA_V2_WORD_CONTROL] =
      c.dtype | (c.rank << 5) | (c.interleave << 8) |
      (c.swizzle << 10) | (c.oob << 18);
  descriptor.words[VENTUS_TMA_V2_WORD_GLOBAL_BASE] = 0;
  for (unsigned dim = 0; dim < c.rank; ++dim) {
    descriptor.words[VENTUS_TMA_V2_WORD_GLOBAL_DIMS + dim] = c.dims[dim];
    descriptor.words[VENTUS_TMA_V2_WORD_BOX_DIMS + dim] = c.box[dim];
    descriptor.words[VENTUS_TMA_V2_WORD_ELEMENT_STRIDES + dim] = 1;
    if (dim + 1 < c.rank)
      descriptor.words[VENTUS_TMA_V2_WORD_GLOBAL_STRIDES + dim * 2] =
          c.strides[dim];
  }
  return descriptor;
}

std::vector<TensorCase> LegalCases() {
  std::vector<TensorCase> cases;
  const std::array<uint8_t, 14> dtypes = {
      VENTUS_TMA_DTYPE_U8, VENTUS_TMA_DTYPE_U16, VENTUS_TMA_DTYPE_U32,
      VENTUS_TMA_DTYPE_S32, VENTUS_TMA_DTYPE_U64, VENTUS_TMA_DTYPE_S64,
      VENTUS_TMA_DTYPE_FP16, VENTUS_TMA_DTYPE_FP32,
      VENTUS_TMA_DTYPE_FP32_FTZ, VENTUS_TMA_DTYPE_FP64,
      VENTUS_TMA_DTYPE_BF16, VENTUS_TMA_DTYPE_TF32,
      VENTUS_TMA_DTYPE_TF32_FTZ, VENTUS_TMA_DTYPE_B4X16};
  for (uint8_t dtype : dtypes) {
    const uint32_t bits = dtype == VENTUS_TMA_DTYPE_B4X16
                              ? 4
                              : (dtype == VENTUS_TMA_DTYPE_U8
                                     ? 8
                                     : ((dtype == VENTUS_TMA_DTYPE_U32 ||
                                         dtype == VENTUS_TMA_DTYPE_S32 ||
                                         dtype == VENTUS_TMA_DTYPE_FP32 ||
                                         dtype == VENTUS_TMA_DTYPE_FP32_FTZ ||
                                         dtype == VENTUS_TMA_DTYPE_TF32 ||
                                         dtype == VENTUS_TMA_DTYPE_TF32_FTZ)
                                            ? 32
                                            : ((dtype == VENTUS_TMA_DTYPE_U64 ||
                                                dtype == VENTUS_TMA_DTYPE_S64 ||
                                                dtype == VENTUS_TMA_DTYPE_FP64)
                                                   ? 64
                                                   : 16)));
    const uint32_t elements = 256 / bits;
    TensorCase c{};
    c.name = "dtype_" + std::to_string(dtype);
    c.dtype = dtype;
    c.rank = 1;
    c.dims = {elements * 2, 0, 0, 0, 0};
    c.box = {elements, 0, 0, 0, 0};
    cases.push_back(c);
  }

  cases.push_back({"rank2_fp32_pitch", VENTUS_TMA_DTYPE_FP32, 2, 0, 0, 0,
                   {32, 8, 0, 0, 0}, {160, 0, 0, 0},
                   {16, 4, 0, 0, 0},
                   {4, 1, 0, 0, 0}, true, true, true});
  cases.push_back({"rank3_u8_plain", VENTUS_TMA_DTYPE_U8, 3, 0, 0, 0,
                   {64, 8, 3, 0, 0}, {80, 768, 0, 0},
                   {32, 4, 2, 0, 0},
                   {0, 2, 1, 0, 0}});
  cases.push_back({"rank4_u16", VENTUS_TMA_DTYPE_U16, 4, 0, 0, 0,
                   {16, 4, 3, 2, 0}, {32, 128, 384, 0},
                   {8, 3, 2, 2, 0},
                   {0, 1, 0, 0, 0}});
  cases.push_back({"rank5_bf16", VENTUS_TMA_DTYPE_BF16, 5, 0, 0, 0,
                   {16, 2, 2, 2, 2}, {32, 64, 128, 256},
                   {16, 2, 2, 2, 2},
                   {0, 0, 0, 0, 0}, true, true, true});

  cases.push_back({"swizzle32_rank3", VENTUS_TMA_DTYPE_FP32, 3, 0, 1, 0,
                   {16, 8, 2, 0, 0}, {64, 512, 0, 0},
                   {8, 4, 2, 0, 0},
                   {0, 1, 0, 0, 0}});
  cases.push_back({"swizzle64_rank3", VENTUS_TMA_DTYPE_FP32, 3, 0, 2, 0,
                   {32, 8, 2, 0, 0}, {128, 1024, 0, 0},
                   {16, 4, 2, 0, 0},
                   {4, 1, 0, 0, 0}});
  cases.push_back({"swizzle128_rank3", VENTUS_TMA_DTYPE_FP32, 3, 0, 3, 0,
                   {64, 8, 2, 0, 0}, {256, 2048, 0, 0},
                   {32, 4, 2, 0, 0},
                   {8, 1, 0, 0, 0}});
  cases.push_back({"interleave16_rank3", VENTUS_TMA_DTYPE_FP32, 3, 1, 0, 0,
                   {4, 8, 2, 0, 0}, {16, 128, 0, 0},
                   {4, 8, 2, 0, 0},
                   {0, 0, 0, 0, 0}});
  cases.push_back({"interleave32_swizzle32_rank3", VENTUS_TMA_DTYPE_FP32,
                   3, 2, 1, 0, {8, 4, 2, 0, 0}, {32, 128, 0, 0},
                   {8, 4, 2, 0, 0},
                   {0, 0, 0, 0, 0}, true, true, true});

  cases.push_back({"interleave16_u8_whole_atom", VENTUS_TMA_DTYPE_U8,
                   3, 1, 0, 0, {16, 8, 1, 0, 0}, {16, 128, 0, 0},
                   {16, 8, 1, 0, 0},
                   {0, 0, 0, 0, 0}});
  cases.push_back({"interleave16_u16_whole_atom", VENTUS_TMA_DTYPE_U16,
                   3, 1, 0, 0, {8, 8, 1, 0, 0}, {16, 128, 0, 0},
                   {8, 8, 1, 0, 0},
                   {0, 0, 0, 0, 0}});
  cases.push_back({"interleave16_fp32_C4_W64_whole_atom", VENTUS_TMA_DTYPE_FP32,
                   3, 1, 0, 0, {4, 64, 1, 0, 0}, {16, 1024, 0, 0},
                   {4, 64, 1, 0, 0},
                   {0, 0, 0, 0, 0}});
  cases.push_back({"interleave32_fp32_C8_W32_whole_atom", VENTUS_TMA_DTYPE_FP32,
                   3, 2, 1, 0, {8, 32, 1, 0, 0}, {32, 1024, 0, 0},
                   {8, 32, 1, 0, 0},
                   {0, 0, 0, 0, 0}});

  cases.push_back({"g2s_negative_oob_zero", VENTUS_TMA_DTYPE_U8,
                   2, 0, 0, VENTUS_TMA_OOB_ZERO,
                   {32, 4, 0, 0, 0}, {32, 0, 0, 0},
                   {32, 3, 0, 0, 0},
                   {-16, 0, 0, 0, 0}, true, false});
  cases.push_back({"g2s_positive_oob_zero", VENTUS_TMA_DTYPE_U16,
                   2, 0, 0, VENTUS_TMA_OOB_ZERO,
                   {16, 4, 0, 0, 0}, {32, 0, 0, 0},
                   {16, 3, 0, 0, 0},
                   {8, 2, 0, 0, 0}, true, false});
  cases.push_back({"g2s_fp16_oob_nan", VENTUS_TMA_DTYPE_FP16,
                   2, 0, 0, VENTUS_TMA_OOB_NAN,
                   {16, 4, 0, 0, 0}, {32, 0, 0, 0},
                   {16, 2, 0, 0, 0},
                   {-8, 3, 0, 0, 0}, true, false, true});
  cases.push_back({"g2s_bf16_oob_nan", VENTUS_TMA_DTYPE_BF16,
                   2, 0, 0, VENTUS_TMA_OOB_NAN,
                   {16, 4, 0, 0, 0}, {32, 0, 0, 0},
                   {16, 2, 0, 0, 0},
                   {-8, 3, 0, 0, 0}, true, false});
  cases.push_back({"g2s_fp32_oob_nan", VENTUS_TMA_DTYPE_FP32,
                   2, 0, 0, VENTUS_TMA_OOB_NAN,
                   {16, 4, 0, 0, 0}, {64, 0, 0, 0},
                   {8, 3, 0, 0, 0},
                   {12, 3, 0, 0, 0}, true, false});
  cases.push_back({"s2g_partial_oob_suppress", VENTUS_TMA_DTYPE_FP32,
                   2, 0, 0, VENTUS_TMA_OOB_ZERO,
                   {16, 4, 0, 0, 0}, {64, 0, 0, 0},
                   {8, 3, 0, 0, 0},
                   {12, 3, 0, 0, 0}, false, true});
  cases.push_back({"b4x16_p64_swizzle128", VENTUS_TMA_DTYPE_B4X16_P64,
                   1, 0, VENTUS_TMA_SWIZZLE_128B, VENTUS_TMA_OOB_ZERO,
                   {128, 0, 0, 0, 0}, {}, {128, 0, 0, 0, 0},
                   {0, 0, 0, 0, 0}, true, false});
  cases.push_back({"fp6_plain", VENTUS_TMA_DTYPE_B6,
                   1, 0, VENTUS_TMA_SWIZZLE_NONE, VENTUS_TMA_OOB_ZERO,
                   {128, 0, 0, 0, 0}, {}, {128, 0, 0, 0, 0},
                   {0, 0, 0, 0, 0}, true, true});
  return cases;
}

std::vector<InvalidCase> InvalidCases() {
  std::vector<InvalidCase> cases;
  TensorCase padded_store{"unsupported_b4x16_p64_s2g",
                          VENTUS_TMA_DTYPE_B4X16_P64,
                          1, 0, 0, 0, {128, 0, 0, 0, 0}, {},
                          {128, 0, 0, 0, 0}};
  cases.push_back({padded_store, VENTUS_TMA_S2G});
  TensorCase interleave_rank2{"unsupported_interleave_rank2",
                              VENTUS_TMA_DTYPE_FP32, 2, 1, 0, 0,
                              {16, 4, 0, 0, 0}, {64, 0, 0, 0},
                              {4, 4, 0, 0, 0}};
  cases.push_back({interleave_rank2, VENTUS_TMA_S2G});
  TensorCase s2g_negative{"unsupported_s2g_negative", VENTUS_TMA_DTYPE_U8,
                          1, 0, 0, 0, {64, 0, 0, 0, 0}, {},
                          {16, 0, 0, 0, 0},
                          {-1, 0, 0, 0, 0}};
  cases.push_back({s2g_negative, VENTUS_TMA_S2G});
  TensorCase dynamic_alignment{"invalid_dynamic_bounding_box_alignment",
                               VENTUS_TMA_DTYPE_U8,
                               1, 0, 0, 0, {64, 0, 0, 0, 0}, {},
                               {16, 0, 0, 0, 0}, {1, 0, 0, 0, 0}};
  cases.push_back({dynamic_alignment, VENTUS_TMA_G2S});
  TensorCase packed_b4_phase{"invalid_b4_coordinate_phase",
                             VENTUS_TMA_DTYPE_B4X16,
                             1, 0, 0, 0, {64, 0, 0, 0, 0}, {},
                             {32, 0, 0, 0, 0}, {16, 0, 0, 0, 0}};
  cases.push_back({packed_b4_phase, VENTUS_TMA_G2S});
  TensorCase packed_b4_odd{"invalid_b4_odd_dim0",
                           VENTUS_TMA_DTYPE_B4X16,
                           1, 0, 0, 0, {65, 0, 0, 0, 0}, {},
                           {32, 0, 0, 0, 0}};
  cases.push_back({packed_b4_odd, VENTUS_TMA_G2S});
  TensorCase l2{"unsupported_l2_promotion", VENTUS_TMA_DTYPE_U8,
                1, 0, 0, 0, {64, 0, 0, 0, 0}, {},
                {16, 0, 0, 0, 0}};
  cases.push_back({l2, VENTUS_TMA_G2S});
  TensorCase invalid_magic = l2;
  invalid_magic.name = "invalid_magic";
  cases.push_back({invalid_magic, VENTUS_TMA_G2S});
  TensorCase reserved = l2;
  reserved.name = "invalid_reserved_word27";
  cases.push_back({reserved, VENTUS_TMA_G2S});
  TensorCase fp6_nan{"unsupported_fp6_oob_nan", VENTUS_TMA_DTYPE_B6,
                     1, 0, 0, VENTUS_TMA_OOB_NAN,
                     {128, 0, 0, 0, 0}, {}, {128, 0, 0, 0, 0}};
  cases.push_back({fp6_nan, VENTUS_TMA_G2S});
  TensorCase fp6_interleave32{"unsupported_fp6_interleave32",
                              VENTUS_TMA_DTYPE_B6, 3,
                              VENTUS_TMA_INTERLEAVE_32B,
                              VENTUS_TMA_SWIZZLE_32B, VENTUS_TMA_OOB_ZERO,
                              {128, 1, 1, 0, 0}, {96, 96, 0, 0},
                              {128, 1, 1, 0, 0}};
  cases.push_back({fp6_interleave32, VENTUS_TMA_G2S});
  TensorCase fp6_interleave16{"unsupported_fp6_interleave16",
                              VENTUS_TMA_DTYPE_B6, 3,
                              VENTUS_TMA_INTERLEAVE_16B,
                              VENTUS_TMA_SWIZZLE_NONE, VENTUS_TMA_OOB_ZERO,
                              {128, 1, 1, 0, 0}, {96, 96, 0, 0},
                              {128, 1, 1, 0, 0}};
  cases.push_back({fp6_interleave16, VENTUS_TMA_G2S});
  return cases;
}

cl_mem MakeBuffer(OpenClEnv &env, BufferPool &pool, cl_mem_flags flags,
                  size_t bytes, void *host = nullptr) {
  cl_int err = CL_SUCCESS;
  cl_mem buffer = clCreateBuffer(env.context, flags, bytes, host, &err);
  if (err != CL_SUCCESS) return nullptr;
  pool.buffers.push_back(buffer);
  return buffer;
}

bool Write(OpenClEnv &env, cl_mem buffer, const void *data, size_t bytes) {
  return clEnqueueWriteBuffer(env.queue, buffer, CL_TRUE, 0, bytes, data,
                              0, nullptr, nullptr) == CL_SUCCESS;
}

bool Read(OpenClEnv &env, cl_mem buffer, void *data, size_t bytes) {
  return clEnqueueReadBuffer(env.queue, buffer, CL_TRUE, 0, bytes, data,
                             0, nullptr, nullptr) == CL_SUCCESS;
}

bool Launch(OpenClEnv &env, const char *name, const std::vector<cl_mem> &buffers,
            const std::vector<uint32_t> &scalars, size_t global = kWgSize,
            size_t local = kWgSize) {
  cl_int err = CL_SUCCESS;
  cl_kernel kernel = clCreateKernel(env.program, name, &err);
  if (err != CL_SUCCESS) return false;
  cl_uint arg = 0;
  for (cl_mem buffer : buffers) {
    err = clSetKernelArg(kernel, arg++, sizeof(buffer), &buffer);
    if (err != CL_SUCCESS) break;
  }
  for (uint32_t scalar : scalars) {
    err = clSetKernelArg(kernel, arg++, sizeof(scalar), &scalar);
    if (err != CL_SUCCESS) break;
  }
  if (err == CL_SUCCESS) {
    err = clEnqueueNDRangeKernel(env.queue, kernel, 1, nullptr, &global,
                                 &local, 0, nullptr, nullptr);
  }
  if (err == CL_SUCCESS) err = clFinish(env.queue);
  clReleaseKernel(kernel);
  if (err != CL_SUCCESS) {
    std::cerr << "FAIL launch " << name << " error=" << err << '\n';
    return false;
  }
  return true;
}

bool PatchDescriptor(OpenClEnv &env, cl_mem descriptor, cl_mem base) {
  return Launch(env, "patch_descriptor_base", {descriptor, base}, {}, 1, 1);
}

uint32_t PopCount(uint16_t value) {
  return static_cast<uint32_t>(__builtin_popcount(value));
}

bool Compare(const std::string &name, const std::vector<uint8_t> &got,
             const std::vector<uint8_t> &expected, size_t bytes) {
  size_t mismatches = 0;
  for (size_t i = 0; i < bytes; ++i) {
    if (got[i] != expected[i]) {
      if (mismatches < 16) {
        std::cerr << "FAIL " << name << " byte=" << i
                  << " got=0x" << std::hex << unsigned(got[i])
                  << " expected=0x" << unsigned(expected[i]) << std::dec
                  << '\n';
      }
      ++mismatches;
    }
  }
  if (mismatches != 0) {
    std::cerr << "FAIL " << name << " mismatches=" << mismatches
              << "/" << bytes << '\n';
    return false;
  }
  std::cout << "PASS " << name << " bytes=" << bytes << '\n';
  return true;
}

struct Planned {
  ventus::tma::Result result;
  size_t shared_bytes = 0;
  uint32_t transaction_bytes = 0;
};

Planned Plan(const TensorCase &c, uint8_t direction) {
  // The runtime kernel patches the TensorMap to a real device buffer before
  // issue.  Give the host model the same nonzero-address property so a legal
  // aligned negative G2S origin does not underflow the 32-bit address space.
  // Normalize planned non-fill addresses back to buffer-relative offsets for
  // the byte-for-byte golden comparison below.
  constexpr uint32_t kModelGlobalBase = 0x10000;
  Descriptor descriptor = Encode(c);
  descriptor.words[VENTUS_TMA_V2_WORD_GLOBAL_BASE] = kModelGlobalBase;
  descriptor.words[VENTUS_TMA_V2_WORD_GLOBAL_BASE + 1] = 0;
  Request request;
  request.direction = direction;
  request.shared_base = 0;
  request.coordinates = c.coords;
  Planned planned;
  planned.result = DecodeAndPlan(descriptor, request);
  for (auto &atom : planned.result.atoms) {
    if (!atom.fill) atom.global_atom -= kModelGlobalBase;
    planned.shared_bytes = std::max(planned.shared_bytes,
                                    size_t(atom.shared_atom) + 16);
    planned.transaction_bytes += PopCount(atom.shared_mask);
  }
  planned.shared_bytes = std::max<size_t>(planned.shared_bytes, 16);
  return planned;
}

int RunTensorDirection(OpenClEnv &env, BufferPool &pool,
                       const TensorCase &c, uint8_t direction,
                       const std::vector<uint8_t> &input,
                       const std::vector<uint8_t> &seed) {
  const std::string suffix = direction == VENTUS_TMA_G2S ? "_g2s" : "_s2g";
  Planned planned = Plan(c, direction);
  if (planned.result.status != Status::kOk ||
      planned.shared_bytes > kTensorSharedCapacity) {
    std::cerr << "FAIL " << c.name << suffix << " cmodel_status="
              << ventus::tma::StatusName(planned.result.status)
              << " detail=" << planned.result.detail
              << " shared_bytes=" << planned.shared_bytes << '\n';
    return 1;
  }

  Descriptor descriptor = Encode(c);
  std::array<int32_t, 32> coordinates{};
  std::copy(c.coords.begin(), c.coords.end(), coordinates.begin());
  uint32_t status = 0xffffffffu;
  std::vector<uint8_t> output(kGlobalCapacity, kSentinel);
  std::vector<uint8_t> readback(planned.shared_bytes, 0xee);

  cl_mem descriptor_buffer = MakeBuffer(
      env, pool, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(descriptor.words), descriptor.words.data());
  cl_mem coordinates_buffer = MakeBuffer(
      env, pool, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      sizeof(coordinates), coordinates.data());
  cl_mem status_buffer = MakeBuffer(
      env, pool, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(status), &status);
  if (!descriptor_buffer || !coordinates_buffer || !status_buffer) return 1;

  bool ok = false;
  if (direction == VENTUS_TMA_G2S) {
    cl_mem input_buffer = MakeBuffer(
        env, pool, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        input.size(), const_cast<uint8_t *>(input.data()));
    cl_mem readback_buffer = MakeBuffer(env, pool, CL_MEM_READ_WRITE,
                                        readback.size());
    if (!input_buffer || !readback_buffer ||
        !PatchDescriptor(env, descriptor_buffer, input_buffer)) return 1;
    ok = Launch(env, "tensor_g2s",
                {descriptor_buffer, coordinates_buffer, readback_buffer,
                 status_buffer},
                {static_cast<uint32_t>(planned.shared_bytes),
                 planned.transaction_bytes, c.prefetch ? 1u : 0u});
    ok &= Read(env, readback_buffer, readback.data(), readback.size());

    std::vector<uint8_t> expected(planned.shared_bytes, 0);
    for (const auto &atom : planned.result.atoms) {
      for (unsigned lane = 0; lane < 16; ++lane) {
        if (((atom.shared_mask >> lane) & 1u) == 0) continue;
        if (atom.fill) {
          expected[atom.shared_atom + lane] = atom.fill_bytes[lane];
        } else {
          for (unsigned global_lane = 0; global_lane < 16; ++global_lane) {
            if (((atom.global_mask >> global_lane) & 1u) != 0 &&
                atom.global_to_shared[global_lane] == lane) {
              expected[atom.shared_atom + lane] =
                  input[atom.global_atom + global_lane];
            }
          }
        }
      }
    }
    ok &= Compare(c.name + suffix, readback, expected, expected.size());
  } else {
    cl_mem output_buffer = MakeBuffer(
        env, pool, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        output.size(), output.data());
    cl_mem seed_buffer = MakeBuffer(
        env, pool, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        planned.shared_bytes, const_cast<uint8_t *>(seed.data()));
    if (!output_buffer || !seed_buffer ||
        !PatchDescriptor(env, descriptor_buffer, output_buffer)) return 1;
    ok = Launch(env, "tensor_s2g",
                {descriptor_buffer, coordinates_buffer, seed_buffer,
                 status_buffer},
                {static_cast<uint32_t>(planned.shared_bytes)});
    ok &= Read(env, output_buffer, output.data(), output.size());

    std::vector<uint8_t> expected(kGlobalCapacity, kSentinel);
    if (c.dtype == VENTUS_TMA_DTYPE_B6) {
      for (unsigned lane = 0; lane < 8; ++lane) {
        uint32_t accumulator = 0;
        unsigned accumulator_bits = 0;
        unsigned output_byte = 0;
        for (unsigned element = 0; element < 16; ++element) {
          accumulator |=
              (static_cast<uint32_t>(seed[lane * 16 + element]) & 0x3f)
              << accumulator_bits;
          accumulator_bits += 6;
          while (accumulator_bits >= 8) {
            expected[lane * 12 + output_byte] = accumulator & 0xff;
            accumulator >>= 8;
            accumulator_bits -= 8;
            ++output_byte;
          }
        }
      }
    } else {
      for (const auto &atom : planned.result.atoms) {
        for (unsigned global_lane = 0; global_lane < 16; ++global_lane) {
          if (((atom.global_mask >> global_lane) & 1u) == 0) continue;
          const unsigned shared_lane = atom.global_to_shared[global_lane];
          expected[atom.global_atom + global_lane] =
              seed[atom.shared_atom + shared_lane];
        }
      }
    }
    ok &= Compare(c.name + suffix, output, expected, output.size());
  }
  ok &= Read(env, status_buffer, &status, sizeof(status));
  if (status != VENTUS_TMA_STATUS_OK) {
    std::cerr << "FAIL " << c.name << suffix << " status=" << status << '\n';
    ok = false;
  }
  return ok ? 0 : 1;
}

uint32_t ApplyReduce(uint32_t mode, uint8_t dtype, uint32_t old_value,
                     uint32_t operand) {
  switch (mode) {
    case VENTUS_TMA_V2_REDUCE_ADD:
      return old_value + operand;
    case VENTUS_TMA_V2_REDUCE_MIN:
      if (dtype == VENTUS_TMA_DTYPE_S32) {
        return static_cast<int32_t>(old_value) < static_cast<int32_t>(operand)
                   ? old_value : operand;
      }
      return std::min(old_value, operand);
    case VENTUS_TMA_V2_REDUCE_MAX:
      if (dtype == VENTUS_TMA_DTYPE_S32) {
        return static_cast<int32_t>(old_value) > static_cast<int32_t>(operand)
                   ? old_value : operand;
      }
      return std::max(old_value, operand);
    case VENTUS_TMA_V2_REDUCE_AND:
      return old_value & operand;
    case VENTUS_TMA_V2_REDUCE_OR:
      return old_value | operand;
    case VENTUS_TMA_V2_REDUCE_XOR:
      return old_value ^ operand;
    default:
      return old_value;
  }
}

int RunBulkReduceCase(OpenClEnv &env, BufferPool &pool,
                      const BulkReduceCase &test) {
  uint32_t status = 0xffffffffu;
  std::vector<uint32_t> output = test.initial;
  cl_mem operand_buffer = MakeBuffer(
      env, pool, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      test.operand.size() * sizeof(uint32_t),
      const_cast<uint32_t *>(test.operand.data()));
  cl_mem output_buffer = MakeBuffer(
      env, pool, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      output.size() * sizeof(uint32_t), output.data());
  cl_mem status_buffer = MakeBuffer(
      env, pool, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(status), &status);
  bool ok = operand_buffer && output_buffer && status_buffer &&
            Launch(env, "bulk_s2g_reduce",
                   {output_buffer, operand_buffer, status_buffer},
                   {static_cast<uint32_t>(test.operand.size()),
                    test.mode, test.type});
  ok &= Read(env, output_buffer, output.data(),
             output.size() * sizeof(uint32_t));
  ok &= Read(env, status_buffer, &status, sizeof(status));
  size_t mismatches = 0;
  for (size_t i = 0; i < output.size(); ++i) {
    if (output[i] != test.expected[i]) {
      if (mismatches < 8) {
        std::cerr << "FAIL " << test.name << " word=" << i
                  << " got=0x" << std::hex << output[i]
                  << " expected=0x" << test.expected[i] << std::dec << '\n';
      }
      ++mismatches;
    }
  }
  ok &= mismatches == 0 && status == VENTUS_TMA_STATUS_OK;
  if (ok) {
    std::cout << "PASS " << test.name << " words=" << output.size() << '\n';
  } else if (status != VENTUS_TMA_STATUS_OK) {
    std::cerr << "FAIL " << test.name << " status=" << status << '\n';
  }
  return ok ? 0 : 1;
}

int RunBulkReduceCases(OpenClEnv &env, BufferPool &pool) {
  std::vector<BulkReduceCase> cases;
  auto add = [&](const std::string &name, uint32_t mode, uint32_t type,
                 uint32_t old_value, uint32_t operand) {
    const uint8_t dtype =
        type == VENTUS_TMA_V2_BULK_REDUCE_TYPE_S32
            ? VENTUS_TMA_DTYPE_S32
            : VENTUS_TMA_DTYPE_U32;
    BulkReduceCase test{name, mode, type,
                        std::vector<uint32_t>(8, old_value),
                        std::vector<uint32_t>(8, operand), {}};
    test.expected.resize(8);
    for (size_t i = 0; i < test.expected.size(); ++i)
      test.expected[i] = ApplyReduce(mode, dtype, old_value, operand);
    cases.push_back(std::move(test));
  };

  add("bulk_reduce_add_u32", VENTUS_TMA_V2_REDUCE_ADD,
      VENTUS_TMA_V2_BULK_REDUCE_TYPE_U32, 0xffffffffu, 1u);
  add("bulk_reduce_add_s32", VENTUS_TMA_V2_REDUCE_ADD,
      VENTUS_TMA_V2_BULK_REDUCE_TYPE_S32, 0x7fffffffu, 1u);
  add("bulk_reduce_min_u32", VENTUS_TMA_V2_REDUCE_MIN,
      VENTUS_TMA_V2_BULK_REDUCE_TYPE_U32, 0x80000000u, 0x7fffffffu);
  add("bulk_reduce_min_s32", VENTUS_TMA_V2_REDUCE_MIN,
      VENTUS_TMA_V2_BULK_REDUCE_TYPE_S32, 0x80000000u, 0x7fffffffu);
  add("bulk_reduce_max_u32", VENTUS_TMA_V2_REDUCE_MAX,
      VENTUS_TMA_V2_BULK_REDUCE_TYPE_U32, 0x80000000u, 0x7fffffffu);
  add("bulk_reduce_max_s32", VENTUS_TMA_V2_REDUCE_MAX,
      VENTUS_TMA_V2_BULK_REDUCE_TYPE_S32, 0x80000000u, 0x7fffffffu);
  add("bulk_reduce_and_b32", VENTUS_TMA_V2_REDUCE_AND,
      VENTUS_TMA_V2_BULK_REDUCE_TYPE_B32, 0x33333333u, 0x0ff00ff0u);
  add("bulk_reduce_or_b32", VENTUS_TMA_V2_REDUCE_OR,
      VENTUS_TMA_V2_BULK_REDUCE_TYPE_B32, 0x30303030u, 0x0ff00ff0u);
  add("bulk_reduce_xor_b32", VENTUS_TMA_V2_REDUCE_XOR,
      VENTUS_TMA_V2_BULK_REDUCE_TYPE_B32, 0x33333333u, 0x0ff00ff0u);

  int failures = 0;
  for (const BulkReduceCase &test : cases)
    failures += RunBulkReduceCase(env, pool, test);

  const std::vector<uint32_t> initial(8, 0x12345678u);
  const std::vector<uint32_t> operand(8, 0xffffffffu);
  const uint32_t expected_status =
      (uint32_t(VENTUS_TMA_V2_FUNCT_BULK_S2G) << 8) |
      VENTUS_TMA_STATUS_UNSUPPORTED_FEATURE;
  for (uint32_t variant = 0; variant < 2; ++variant) {
    std::vector<uint32_t> output = initial;
    uint32_t status = 0xffffffffu;
    cl_mem output_buffer = MakeBuffer(
        env, pool, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        output.size() * sizeof(uint32_t), output.data());
    cl_mem operand_buffer = MakeBuffer(
        env, pool, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        operand.size() * sizeof(uint32_t),
        const_cast<uint32_t *>(operand.data()));
    cl_mem status_buffer = MakeBuffer(
        env, pool, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(status), &status);
    bool ok = output_buffer && operand_buffer && status_buffer &&
        Launch(env, "invalid_bulk_reduce_encoding",
               {output_buffer, operand_buffer, status_buffer}, {variant});
    ok &= Read(env, output_buffer, output.data(),
               output.size() * sizeof(uint32_t));
    ok &= Read(env, status_buffer, &status, sizeof(status));
    ok &= output == initial && status == expected_status;
    if (ok) {
      std::cout << "PASS invalid_bulk_reduce_encoding_" << variant
                << " status=" << status << '\n';
    } else {
      std::cerr << "FAIL invalid_bulk_reduce_encoding_" << variant
                << " status=" << status << '\n';
    }
    failures += !ok;
  }
  return failures;
}

int RunReduceCase(OpenClEnv &env, BufferPool &pool, const ReduceCase &test) {
  Descriptor descriptor = Encode(test.tensor);
  std::array<int32_t, 32> coordinates{};
  std::copy(test.tensor.coords.begin(), test.tensor.coords.end(),
            coordinates.begin());
  uint32_t status = 0xffffffffu;
  std::vector<uint32_t> output = test.initial;
  cl_mem descriptor_buffer = MakeBuffer(
      env, pool, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(descriptor.words), descriptor.words.data());
  cl_mem coordinates_buffer = MakeBuffer(
      env, pool, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      sizeof(coordinates), coordinates.data());
  cl_mem operand_buffer = MakeBuffer(
      env, pool, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      test.operand.size() * sizeof(uint32_t),
      const_cast<uint32_t *>(test.operand.data()));
  cl_mem output_buffer = MakeBuffer(
      env, pool, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      output.size() * sizeof(uint32_t), output.data());
  cl_mem status_buffer = MakeBuffer(
      env, pool, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(status), &status);
  bool ok = descriptor_buffer && coordinates_buffer && operand_buffer &&
            output_buffer && status_buffer &&
            PatchDescriptor(env, descriptor_buffer, output_buffer) &&
            Launch(env, "tensor_s2g_reduce",
                   {descriptor_buffer, coordinates_buffer, operand_buffer,
                    status_buffer},
                   {static_cast<uint32_t>(test.operand.size()), test.mode});
  ok &= Read(env, output_buffer, output.data(),
             output.size() * sizeof(uint32_t));
  ok &= Read(env, status_buffer, &status, sizeof(status));
  size_t mismatches = 0;
  for (size_t i = 0; i < output.size(); ++i) {
    if (output[i] != test.expected[i]) {
      if (mismatches < 8) {
        std::cerr << "FAIL " << test.name << " word=" << i
                  << " got=0x" << std::hex << output[i]
                  << " expected=0x" << test.expected[i] << std::dec << '\n';
      }
      ++mismatches;
    }
  }
  ok &= mismatches == 0 && status == VENTUS_TMA_STATUS_OK;
  if (ok) {
    std::cout << "PASS " << test.name << " words=" << output.size() << '\n';
  } else if (status != VENTUS_TMA_STATUS_OK) {
    std::cerr << "FAIL " << test.name << " status=" << status << '\n';
  }
  return ok ? 0 : 1;
}

int RunReduceCases(OpenClEnv &env, BufferPool &pool) {
  std::vector<ReduceCase> cases;
  auto rank1 = [](const std::string &name, uint8_t dtype, uint32_t mode,
                  uint32_t old_value, uint32_t operand) {
    TensorCase tensor{name, dtype, 1, 0, 0, 0,
                      {8, 0, 0, 0, 0}, {}, {8, 0, 0, 0, 0},
                      {0, 0, 0, 0, 0}, false, true};
    ReduceCase test{name, tensor, mode,
                    std::vector<uint32_t>(8, old_value),
                    std::vector<uint32_t>(8, operand), {}};
    test.expected.resize(8);
    for (size_t i = 0; i < 8; ++i) {
      test.expected[i] = ApplyReduce(mode, dtype, old_value, operand);
    }
    return test;
  };

  cases.push_back(rank1("reduce_add_u32_wrap", VENTUS_TMA_DTYPE_U32,
                        VENTUS_TMA_V2_REDUCE_ADD, 0xffffffffu, 1u));
  cases.push_back(rank1("reduce_add_i32_wrap", VENTUS_TMA_DTYPE_S32,
                        VENTUS_TMA_V2_REDUCE_ADD, 0x7fffffffu, 1u));
  cases.push_back(rank1("reduce_min_u32", VENTUS_TMA_DTYPE_U32,
                        VENTUS_TMA_V2_REDUCE_MIN, 0x80000000u, 0x7fffffffu));
  cases.push_back(rank1("reduce_min_i32", VENTUS_TMA_DTYPE_S32,
                        VENTUS_TMA_V2_REDUCE_MIN, 0x80000000u, 0x7fffffffu));
  cases.push_back(rank1("reduce_max_u32", VENTUS_TMA_DTYPE_U32,
                        VENTUS_TMA_V2_REDUCE_MAX, 0x80000000u, 0x7fffffffu));
  cases.push_back(rank1("reduce_max_i32", VENTUS_TMA_DTYPE_S32,
                        VENTUS_TMA_V2_REDUCE_MAX, 0x80000000u, 0x7fffffffu));
  cases.push_back(rank1("reduce_and_b32", VENTUS_TMA_DTYPE_U32,
                        VENTUS_TMA_V2_REDUCE_AND, 0x33333333u, 0x0ff00ff0u));
  cases.push_back(rank1("reduce_or_b32", VENTUS_TMA_DTYPE_U32,
                        VENTUS_TMA_V2_REDUCE_OR, 0x30303030u, 0x0ff00ff0u));
  cases.push_back(rank1("reduce_xor_b32", VENTUS_TMA_DTYPE_U32,
                        VENTUS_TMA_V2_REDUCE_XOR, 0x33333333u, 0x0ff00ff0u));

  TensorCase partial{"reduce_add_partial_oob", VENTUS_TMA_DTYPE_U32,
                     1, 0, 0, 0, {8, 0, 0, 0, 0}, {},
                     {8, 0, 0, 0, 0}, {4, 0, 0, 0, 0}, false, true};
  ReduceCase partial_case{partial.name, partial, VENTUS_TMA_V2_REDUCE_ADD,
                          std::vector<uint32_t>(8, 10u),
                          std::vector<uint32_t>(8, 1u),
                          std::vector<uint32_t>(8, 10u)};
  for (size_t i = 4; i < 8; ++i) partial_case.expected[i] = 11u;
  cases.push_back(partial_case);

  TensorCase rank2_tensor{"reduce_add_rank2_stride", VENTUS_TMA_DTYPE_U32,
                          2, 0, 0, 0, {4, 2, 0, 0, 0}, {32, 0, 0, 0},
                          {4, 2, 0, 0, 0}, {0, 0, 0, 0, 0}, false, true};
  ReduceCase rank2_case{rank2_tensor.name, rank2_tensor,
                        VENTUS_TMA_V2_REDUCE_ADD,
                        std::vector<uint32_t>(12, 5u),
                        std::vector<uint32_t>(8, 3u),
                        std::vector<uint32_t>(12, 5u)};
  for (size_t i = 0; i < 4; ++i) rank2_case.expected[i] = 8u;
  for (size_t i = 8; i < 12; ++i) rank2_case.expected[i] = 8u;
  cases.push_back(rank2_case);

  int failures = 0;
  for (const ReduceCase &test : cases) failures += RunReduceCase(env, pool, test);

  TensorCase negative{"tensor_reduce_negative_rejected", VENTUS_TMA_DTYPE_U32,
                      1, 0, 0, 0, {8, 0, 0, 0, 0}, {},
                      {8, 0, 0, 0, 0}, {-1, 0, 0, 0, 0}, false, true};
  Descriptor descriptor = Encode(negative);
  std::array<int32_t, 32> coordinates{};
  std::copy(negative.coords.begin(), negative.coords.end(),
            coordinates.begin());
  std::vector<uint32_t> output(8, 10u);
  const std::vector<uint32_t> original = output;
  std::vector<uint32_t> operand(8, 1u);
  uint32_t status = 0xffffffffu;
  cl_mem descriptor_buffer = MakeBuffer(
      env, pool, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(descriptor.words), descriptor.words.data());
  cl_mem coordinates_buffer = MakeBuffer(
      env, pool, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      sizeof(coordinates), coordinates.data());
  cl_mem operand_buffer = MakeBuffer(
      env, pool, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      operand.size() * sizeof(uint32_t), operand.data());
  cl_mem output_buffer = MakeBuffer(
      env, pool, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      output.size() * sizeof(uint32_t), output.data());
  cl_mem status_buffer = MakeBuffer(
      env, pool, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(status), &status);
  bool negative_ok = descriptor_buffer && coordinates_buffer &&
      operand_buffer && output_buffer && status_buffer &&
      PatchDescriptor(env, descriptor_buffer, output_buffer) &&
      Launch(env, "tensor_s2g_reduce",
             {descriptor_buffer, coordinates_buffer, operand_buffer,
              status_buffer},
             {static_cast<uint32_t>(operand.size()),
              uint32_t(VENTUS_TMA_V2_REDUCE_ADD)});
  negative_ok &= Read(env, output_buffer, output.data(),
                      output.size() * sizeof(uint32_t));
  negative_ok &= Read(env, status_buffer, &status, sizeof(status));
  const uint32_t expected_status =
      (static_cast<uint32_t>(Status::kBadCoordinate) << 8) |
      VENTUS_TMA_STATUS_INVALID_DESCRIPTOR;
  negative_ok &= output == original &&
      status == expected_status;
  if (negative_ok) {
    std::cout << "PASS tensor_reduce_negative_rejected status="
              << status << '\n';
  } else {
    std::cerr << "FAIL tensor_reduce_negative_rejected status="
              << status << '\n';
  }
  failures += !negative_ok;
  return failures;
}

int RunInvalid(OpenClEnv &env, BufferPool &pool, const InvalidCase &test,
               const std::vector<uint8_t> &input) {
  Descriptor descriptor = Encode(test.tensor);
  if (test.tensor.name == "unsupported_l2_promotion") {
    descriptor.words[VENTUS_TMA_V2_WORD_CONTROL] |= 1u << 16;
  } else if (test.tensor.name == "invalid_magic") {
    descriptor.words[0] = UINT32_C(0xdeadbeef);
  } else if (test.tensor.name == "invalid_reserved_word27") {
    descriptor.words[27] = 1;
  }
  Request request;
  request.direction = test.direction;
  request.coordinates = test.tensor.coords;
  auto model = DecodeAndPlan(descriptor, request);
  if (model.status == Status::kOk) {
    std::cerr << "FAIL " << test.tensor.name << " cmodel accepted invalid case\n";
    return 1;
  }

  std::array<int32_t, 32> coordinates{};
  std::copy(test.tensor.coords.begin(), test.tensor.coords.end(),
            coordinates.begin());
  uint32_t status = 0;
  cl_mem descriptor_buffer = MakeBuffer(
      env, pool, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(descriptor.words), descriptor.words.data());
  cl_mem coordinates_buffer = MakeBuffer(
      env, pool, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      sizeof(coordinates), coordinates.data());
  cl_mem status_buffer = MakeBuffer(
      env, pool, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(status), &status);
  cl_mem base_buffer = MakeBuffer(
      env, pool, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      input.size(), const_cast<uint8_t *>(input.data()));
  if (!descriptor_buffer || !coordinates_buffer || !status_buffer ||
      !base_buffer || !PatchDescriptor(env, descriptor_buffer, base_buffer)) {
    return 1;
  }
  bool ok = Launch(env, "invalid_tensor",
                   {descriptor_buffer, coordinates_buffer, status_buffer},
                   {test.direction});
  ok &= Read(env, status_buffer, &status, sizeof(status));
  if (status == VENTUS_TMA_STATUS_OK) {
    std::cerr << "FAIL " << test.tensor.name
              << " hardware accepted status=" << status
              << " cmodel=" << ventus::tma::StatusName(model.status) << '\n';
    return 1;
  }
  std::cout << "PASS " << test.tensor.name << " direction="
            << unsigned(test.direction) << " status=" << status
            << " cmodel=" << ventus::tma::StatusName(model.status) << '\n';
  return ok ? 0 : 1;
}

int RunControlCases(OpenClEnv &env, BufferPool &pool,
                    const std::vector<uint8_t> &input) {
  int failures = 0;
  uint32_t status = 0xffffffffu;

  for (uint32_t bytes : {16u, 128u, 1024u, 8192u}) {
    std::vector<uint8_t> output(bytes, kSentinel);
    cl_mem input_buffer = MakeBuffer(
        env, pool, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        bytes, const_cast<uint8_t *>(input.data()));
    cl_mem output_buffer = MakeBuffer(
        env, pool, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        bytes, output.data());
    cl_mem status_buffer = MakeBuffer(
        env, pool, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(status), &status);
    bool ok = input_buffer && output_buffer && status_buffer &&
              Launch(env, "bulk_roundtrip",
                     {input_buffer, output_buffer, status_buffer}, {bytes});
    ok &= Read(env, output_buffer, output.data(), output.size());
    ok &= Read(env, status_buffer, &status, sizeof(status));
    ok &= Compare("bulk_roundtrip_" + std::to_string(bytes), output, input,
                  bytes);
    ok &= status == VENTUS_TMA_STATUS_OK;
    failures += !ok;
  }

  for (uint32_t direction : {0u, 1u}) {
    for (uint32_t mode = 0; mode < 4; ++mode) {
      std::vector<uint8_t> output(32, kSentinel);
      std::vector<uint8_t> expected(32, kSentinel);
      std::vector<uint8_t> ordering(16, kSentinel);
      if (direction == 0u) {
        for (unsigned i = 0; i < 16; ++i)
          expected[i] = static_cast<uint8_t>(0xa0u + i);
      }
      status = 0xffffffffu;
      cl_mem input_buffer = MakeBuffer(
          env, pool, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
          32, const_cast<uint8_t *>(input.data()));
      cl_mem output_buffer = MakeBuffer(
          env, pool, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
          output.size(), output.data());
      cl_mem ordering_buffer = MakeBuffer(
          env, pool, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
          ordering.size(), ordering.data());
      cl_mem status_buffer = MakeBuffer(
          env, pool, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
          sizeof(status), &status);
      bool ok = input_buffer && output_buffer && ordering_buffer &&
                status_buffer &&
                Launch(env, "invalid_bulk",
                       {input_buffer, output_buffer, ordering_buffer,
                        status_buffer},
                       {direction, mode});
      ok &= Read(env, output_buffer, output.data(), output.size());
      ok &= Read(env, status_buffer, &status, sizeof(status));
      ok &= Compare("invalid_bulk_" + std::to_string(direction) + "_" +
                        std::to_string(mode),
                    output, expected, expected.size());
      const uint32_t funct = direction == 0u
                                 ? VENTUS_TMA_V2_FUNCT_BULK_G2S
                                 : VENTUS_TMA_V2_FUNCT_BULK_S2G;
      ok &= (status & 0xffu) == VENTUS_TMA_STATUS_UNSUPPORTED_FEATURE;
      ok &= (status >> 8) == funct;
      failures += !ok;
    }
  }

  {
    TensorCase tensor{"tensormap_invalidate_reload", VENTUS_TMA_DTYPE_U8,
                      1, 0, 0, 0, {128, 0, 0, 0, 0}, {},
                      {128, 0, 0, 0, 0}};
    Descriptor descriptor = Encode(tensor);
    std::array<int32_t, 32> coordinates{};
    std::vector<uint8_t> first(input.begin(), input.begin() + 128);
    std::vector<uint8_t> second(128);
    for (unsigned i = 0; i < second.size(); ++i) {
      second[i] = static_cast<uint8_t>((i * 13u + 0x71u) & 0xffu);
    }
    std::vector<uint8_t> expected = first;
    expected.insert(expected.end(), second.begin(), second.end());
    std::vector<uint8_t> readback(256, 0);
    cl_mem descriptor_buffer = MakeBuffer(
        env, pool, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(descriptor.words), descriptor.words.data());
    cl_mem coordinates_buffer = MakeBuffer(
        env, pool, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(coordinates), coordinates.data());
    cl_mem first_buffer = MakeBuffer(
        env, pool, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        first.size(), first.data());
    cl_mem second_buffer = MakeBuffer(
        env, pool, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        second.size(), second.data());
    cl_mem readback_buffer = MakeBuffer(env, pool, CL_MEM_READ_WRITE,
                                        readback.size());
    cl_mem status_buffer = MakeBuffer(env, pool, CL_MEM_READ_WRITE,
                                      sizeof(status));
    bool ok = descriptor_buffer && coordinates_buffer && first_buffer &&
              second_buffer && readback_buffer && status_buffer &&
              Launch(env, "tensormap_invalidate_reload",
                     {descriptor_buffer, coordinates_buffer, first_buffer,
                      second_buffer, readback_buffer, status_buffer}, {});
    ok &= Read(env, readback_buffer, readback.data(), readback.size());
    ok &= Read(env, status_buffer, &status, sizeof(status));
    ok &= Compare("tensormap_invalidate_reload", readback, expected,
                  expected.size());
    ok &= status == VENTUS_TMA_STATUS_OK;
    failures += !ok;
  }

  {
    std::vector<uint8_t> g2s_readback(32, 0);
    std::vector<uint8_t> s2g_output(32, 0);
    cl_mem input_buffer = MakeBuffer(
        env, pool, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        32, const_cast<uint8_t *>(input.data()));
    cl_mem readback_buffer = MakeBuffer(env, pool, CL_MEM_READ_WRITE, 32);
    cl_mem output_buffer = MakeBuffer(env, pool, CL_MEM_READ_WRITE, 32);
    cl_mem status_buffer = MakeBuffer(env, pool, CL_MEM_READ_WRITE,
                                      sizeof(status));
    bool ok = input_buffer && readback_buffer && output_buffer && status_buffer &&
              Launch(env, "mixed_bidirectional",
                     {input_buffer, readback_buffer, output_buffer,
                      status_buffer}, {});
    ok &= Read(env, readback_buffer, g2s_readback.data(), 32);
    ok &= Read(env, output_buffer, s2g_output.data(), 32);
    ok &= Read(env, status_buffer, &status, sizeof(status));
    std::vector<uint8_t> mixed_expected(32);
    for (unsigned i = 0; i < 32; ++i) mixed_expected[i] = 0xa0u + i;
    ok &= Compare("mixed_bidirectional_g2s", g2s_readback, input, 32);
    ok &= Compare("mixed_bidirectional_s2g", s2g_output, mixed_expected, 32);
    ok &= status == VENTUS_TMA_STATUS_OK;
    failures += !ok;
  }

  {
    TensorCase tensor{"mixed_rank1_u8", VENTUS_TMA_DTYPE_U8, 1, 0, 0, 0,
                      {64, 0, 0, 0, 0}, {}, {32, 0, 0, 0, 0},
                      {0, 0, 0, 0, 0}};
    Descriptor descriptor = Encode(tensor);
    std::array<int32_t, 32> coordinates{};
    std::vector<uint8_t> tensor_readback(32, 0);
    std::vector<uint8_t> bulk_output(32, 0);
    std::vector<uint8_t> bulk_expected(32);
    for (unsigned i = 0; i < 32; ++i) bulk_expected[i] = 0x60u + i;
    cl_mem descriptor_buffer = MakeBuffer(
        env, pool, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(descriptor.words), descriptor.words.data());
    cl_mem coordinates_buffer = MakeBuffer(
        env, pool, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(coordinates), coordinates.data());
    cl_mem input_buffer = MakeBuffer(
        env, pool, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        32, const_cast<uint8_t *>(input.data()));
    cl_mem readback_buffer = MakeBuffer(env, pool, CL_MEM_READ_WRITE, 32);
    cl_mem output_buffer = MakeBuffer(env, pool, CL_MEM_READ_WRITE, 32);
    cl_mem status_buffer = MakeBuffer(env, pool, CL_MEM_READ_WRITE,
                                      sizeof(status));
    bool ok = descriptor_buffer && coordinates_buffer && input_buffer &&
              readback_buffer && output_buffer && status_buffer &&
              PatchDescriptor(env, descriptor_buffer, input_buffer) &&
              Launch(env, "mixed_tensor_g2s_bulk_s2g",
                     {descriptor_buffer, coordinates_buffer, readback_buffer,
                      output_buffer, status_buffer}, {});
    ok &= Read(env, readback_buffer, tensor_readback.data(), 32);
    ok &= Read(env, output_buffer, bulk_output.data(), 32);
    ok &= Read(env, status_buffer, &status, sizeof(status));
    ok &= Compare("mixed_tensor_g2s", tensor_readback, input, 32);
    ok &= Compare("mixed_bulk_s2g", bulk_output, bulk_expected, 32);
    ok &= status == VENTUS_TMA_STATUS_OK;
    failures += !ok;
  }

  {
    TensorCase tensor{"mixed_rank1_u8", VENTUS_TMA_DTYPE_U8, 1, 0, 0, 0,
                      {64, 0, 0, 0, 0}, {}, {32, 0, 0, 0, 0},
                      {0, 0, 0, 0, 0}};
    Descriptor descriptor = Encode(tensor);
    std::array<int32_t, 32> coordinates{};
    std::vector<uint8_t> bulk_readback(32, 0);
    std::vector<uint8_t> tensor_output(32, 0);
    std::vector<uint8_t> tensor_expected(32);
    for (unsigned i = 0; i < 32; ++i) tensor_expected[i] = 0x90u + i;
    cl_mem descriptor_buffer = MakeBuffer(
        env, pool, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(descriptor.words), descriptor.words.data());
    cl_mem coordinates_buffer = MakeBuffer(
        env, pool, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(coordinates), coordinates.data());
    cl_mem input_buffer = MakeBuffer(
        env, pool, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        32, const_cast<uint8_t *>(input.data()));
    cl_mem readback_buffer = MakeBuffer(env, pool, CL_MEM_READ_WRITE, 32);
    cl_mem output_buffer = MakeBuffer(env, pool, CL_MEM_READ_WRITE, 32);
    cl_mem status_buffer = MakeBuffer(env, pool, CL_MEM_READ_WRITE,
                                      sizeof(status));
    bool ok = descriptor_buffer && coordinates_buffer && input_buffer &&
              readback_buffer && output_buffer && status_buffer &&
              PatchDescriptor(env, descriptor_buffer, output_buffer) &&
              Launch(env, "mixed_bulk_g2s_tensor_s2g",
                     {descriptor_buffer, coordinates_buffer, input_buffer,
                      readback_buffer, status_buffer}, {});
    ok &= Read(env, readback_buffer, bulk_readback.data(), 32);
    ok &= Read(env, output_buffer, tensor_output.data(), 32);
    ok &= Read(env, status_buffer, &status, sizeof(status));
    ok &= Compare("mixed_bulk_g2s", bulk_readback, input, 32);
    ok &= Compare("mixed_tensor_s2g", tensor_output, tensor_expected, 32);
    ok &= status == VENTUS_TMA_STATUS_OK;
    failures += !ok;
  }

  {
    std::vector<uint8_t> output(64, 0);
    std::vector<uint8_t> expected(64);
    for (unsigned i = 0; i < 64; ++i) expected[i] = 0x40u + i;
    cl_mem output_buffer = MakeBuffer(env, pool, CL_MEM_READ_WRITE, 64);
    cl_mem status_buffer = MakeBuffer(env, pool, CL_MEM_READ_WRITE,
                                      sizeof(status));
    bool ok = output_buffer && status_buffer &&
              Launch(env, "group_ring_wrap", {output_buffer, status_buffer}, {});
    ok &= Read(env, output_buffer, output.data(), output.size());
    ok &= Read(env, status_buffer, &status, sizeof(status));
    ok &= Compare("group_ring_wrap_wait3210", output, expected, 64);
    ok &= status == VENTUS_TMA_STATUS_OK;
    failures += !ok;
  }

  {
    std::vector<uint8_t> readback(64, 0);
    std::array<uint32_t, 2> statuses{0xffffffffu, 0xffffffffu};
    cl_mem input_buffer = MakeBuffer(
        env, pool, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        64, const_cast<uint8_t *>(input.data()));
    cl_mem readback_buffer = MakeBuffer(env, pool, CL_MEM_READ_WRITE, 64);
    cl_mem status_buffer = MakeBuffer(env, pool, CL_MEM_READ_WRITE,
                                      sizeof(statuses));
    bool ok = input_buffer && readback_buffer && status_buffer &&
              Launch(env, "multiwarp_mbarrier",
                     {input_buffer, readback_buffer, status_buffer}, {}, 64, 64);
    ok &= Read(env, readback_buffer, readback.data(), readback.size());
    ok &= Read(env, status_buffer, statuses.data(), sizeof(statuses));
    ok &= Compare("multiwarp_mbarrier", readback, input, 64);
    ok &= statuses[0] == 0 && statuses[1] == 0;
    failures += !ok;
  }

  {
    std::vector<uint8_t> readback(64, 0);
    uint32_t phase_status = 0xffffffffu;
    cl_mem input_buffer = MakeBuffer(
        env, pool, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        64, const_cast<uint8_t *>(input.data()));
    cl_mem readback_buffer = MakeBuffer(env, pool, CL_MEM_READ_WRITE, 64);
    cl_mem status_buffer = MakeBuffer(env, pool, CL_MEM_READ_WRITE,
                                      sizeof(phase_status));
    bool ok = input_buffer && readback_buffer && status_buffer &&
              Launch(env, "mbarrier_phase_reuse",
                     {input_buffer, readback_buffer, status_buffer}, {});
    ok &= Read(env, readback_buffer, readback.data(), readback.size());
    ok &= Read(env, status_buffer, &phase_status, sizeof(phase_status));
    ok &= Compare("mbarrier_two_phase_reuse", readback, input, 64);
    ok &= phase_status == VENTUS_TMA_STATUS_OK;
    if (!ok) {
      std::cerr << "FAIL mbarrier_two_phase_reuse status=0x" << std::hex
                << phase_status << std::dec << '\n';
    } else {
      std::cout << "PASS mbarrier_two_phase_reuse\n";
    }
    failures += !ok;
  }

  {
    std::array<uint32_t, 4> observed{};
    cl_mem observed_buffer = MakeBuffer(env, pool, CL_MEM_READ_WRITE,
                                        sizeof(observed));
    bool ok = observed_buffer &&
              Launch(env, "status_sticky_clear", {observed_buffer}, {});
    ok &= Read(env, observed_buffer, observed.data(), sizeof(observed));
    const uint32_t group_error =
        (15u << 8) | VENTUS_TMA_STATUS_INVALID_GROUP_OPERATION;
    const uint32_t mbarrier_error = VENTUS_TMA_STATUS_MBARRIER_PROTOCOL;
    ok &= observed[0] == group_error;
    ok &= observed[1] == group_error;
    ok &= observed[2] == mbarrier_error;
    ok &= observed[3] == VENTUS_TMA_STATUS_OK;
    if (!ok) {
      std::cerr << "FAIL status_sticky_clear values=" << std::hex
                << observed[0] << ',' << observed[1] << ',' << observed[2]
                << ',' << observed[3] << std::dec << '\n';
    } else {
      std::cout << "PASS status_sticky_clear first=0xf05 second=0xf05"
                   " after_clear=0x4 final=0\n";
    }
    failures += !ok;
  }

  {
    std::vector<uint8_t> output(256, 0);
    std::array<uint32_t, 4> statuses{};
    cl_mem input_buffer = MakeBuffer(
        env, pool, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        256, const_cast<uint8_t *>(input.data()));
    cl_mem output_buffer = MakeBuffer(env, pool, CL_MEM_READ_WRITE, 256);
    cl_mem status_buffer = MakeBuffer(env, pool, CL_MEM_READ_WRITE,
                                      sizeof(statuses));
    bool ok = input_buffer && output_buffer && status_buffer &&
              Launch(env, "multi_wg_roundtrip",
                     {input_buffer, output_buffer, status_buffer}, {}, 128, 32);
    ok &= Read(env, output_buffer, output.data(), output.size());
    ok &= Read(env, status_buffer, statuses.data(), sizeof(statuses));
    ok &= Compare("multi_wg_roundtrip_4wg", output, input, 256);
    ok &= std::all_of(statuses.begin(), statuses.end(),
                      [](uint32_t value) { return value == 0; });
    failures += !ok;
  }

  for (uint32_t bytes : {16u, 128u, 1024u}) {
    std::vector<uint8_t> output(bytes, kSentinel);
    cl_mem input_buffer = MakeBuffer(
        env, pool, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        bytes, const_cast<uint8_t *>(input.data()));
    cl_mem output_buffer = MakeBuffer(
        env, pool, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        bytes, output.data());
    cl_mem status_buffer = MakeBuffer(
        env, pool, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(status), &status);
    bool ok = input_buffer && output_buffer && status_buffer &&
              Launch(env, "bulk_roundtrip",
                     {input_buffer, output_buffer, status_buffer}, {bytes});
    ok &= Read(env, output_buffer, output.data(), output.size());
    ok &= Read(env, status_buffer, &status, sizeof(status));
    ok &= Compare("post_4wg_bulk_burst_" + std::to_string(bytes),
                  output, input, bytes);
    ok &= status == VENTUS_TMA_STATUS_OK;
    failures += !ok;
  }
  return failures;
}

int Run() {
  OpenClEnv env;
  if (!env.Init()) return 1;
  BufferPool pool;
  std::vector<uint8_t> input(kGlobalCapacity);
  std::vector<uint8_t> seed(kTensorSharedCapacity);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<uint8_t>((i * 37 + 11) & 0xff);
  }
  for (size_t i = 0; i < seed.size(); ++i) {
    seed[i] = static_cast<uint8_t>((i * 29 + 0xa3) & 0xff);
  }

  int failures = RunControlCases(env, pool, input);
  failures += RunBulkReduceCases(env, pool);
  failures += RunReduceCases(env, pool);
  unsigned legal_runs = 0;
  unsigned prefetch_runs = 0;
  for (const TensorCase &test : LegalCases()) {
    if (test.run_g2s) {
      failures += RunTensorDirection(env, pool, test, VENTUS_TMA_G2S,
                                     input, seed);
      if (test.prefetch) {
        std::cout << "PASS descriptor_prefetch_reuse_" << test.name << '\n';
        ++prefetch_runs;
      }
      ++legal_runs;
    }
    if (test.run_s2g) {
      failures += RunTensorDirection(env, pool, test, VENTUS_TMA_S2G,
                                     input, seed);
      ++legal_runs;
    }
  }
  for (const InvalidCase &test : InvalidCases()) {
    failures += RunInvalid(env, pool, test, input);
  }
  std::cout << "TMA_V2_FUNC_SUMMARY legal_runs=" << legal_runs
            << " invalid_runs=" << InvalidCases().size()
            << " prefetch_runs=" << prefetch_runs
            << " control_groups=11 reduce_runs=11 failures=" << failures << '\n';
  return failures ? 1 : 0;
}

}  // namespace

int main() { return Run(); }
