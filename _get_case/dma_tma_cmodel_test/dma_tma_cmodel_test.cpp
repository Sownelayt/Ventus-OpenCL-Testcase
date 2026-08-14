#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "tma_model.h"

using ventus::tma::DecodeAndPlan;
using ventus::tma::Descriptor;
using ventus::tma::Request;
using ventus::tma::Status;

namespace {

struct Case {
  std::string name;
  uint8_t direction;
  uint8_t dtype;
  uint8_t rank;
  uint8_t interleave;
  uint8_t swizzle;
  uint8_t oob;
  std::array<uint32_t, 5> dims;
  std::array<uint32_t, 4> strides;
  std::array<uint32_t, 5> box;
  std::array<int32_t, 5> coords;
  Status expected = Status::kOk;
};

Descriptor Encode(const Case &c) {
  Descriptor d;
  d.words[VENTUS_TMA_V2_WORD_MAGIC] = VENTUS_TMA_V2_MAGIC;
  d.words[VENTUS_TMA_V2_WORD_CONTROL] =
      c.dtype | (c.rank << 5) | (c.interleave << 8) |
      (c.swizzle << 10) | (c.oob << 18);
  d.words[VENTUS_TMA_V2_WORD_GLOBAL_BASE] = 0x10000;
  for (unsigned i = 0; i < 5; ++i) {
    if (i < c.rank) {
      d.words[VENTUS_TMA_V2_WORD_GLOBAL_DIMS + i] = c.dims[i];
      d.words[VENTUS_TMA_V2_WORD_BOX_DIMS + i] = c.box[i];
      d.words[VENTUS_TMA_V2_WORD_ELEMENT_STRIDES + i] = 1;
    }
    if (i + 1 < c.rank)
      d.words[VENTUS_TMA_V2_WORD_GLOBAL_STRIDES + i * 2] = c.strides[i];
  }
  return d;
}

std::vector<Case> Cases() {
  std::vector<Case> cases = {
      {"rank1_u8_plain", VENTUS_TMA_G2S, VENTUS_TMA_DTYPE_U8, 1, 0, 0, 0,
       {128, 0, 0, 0, 0}, {}, {128, 0, 0, 0, 0}, {}},
      {"rank2_fp32_pitch", VENTUS_TMA_S2G, VENTUS_TMA_DTYPE_FP32, 2, 0, 0, 0,
       {32, 8, 0, 0, 0}, {160, 0, 0, 0}, {16, 4, 0, 0, 0},
       {4, 1, 0, 0, 0}},
      {"rank3_u16_swizzle64_oob", VENTUS_TMA_G2S, VENTUS_TMA_DTYPE_U16, 3, 0, 2, 0,
      {32, 8, 3, 0, 0}, {64, 512, 0, 0}, {32, 4, 2, 0, 0},
      {-8, 6, 2, 0, 0}},
      {"swizzle64_rank3_exact", VENTUS_TMA_G2S, VENTUS_TMA_DTYPE_FP32, 3, 0, 2, 0,
       {32, 8, 2, 0, 0}, {128, 1024, 0, 0}, {16, 4, 2, 0, 0},
       {4, 1, 0, 0, 0}},
      {"swizzle128_rank3_exact", VENTUS_TMA_G2S, VENTUS_TMA_DTYPE_FP32, 3, 0, 3, 0,
       {64, 8, 2, 0, 0}, {256, 2048, 0, 0}, {32, 4, 2, 0, 0},
       {8, 1, 0, 0, 0}},
      {"rank3_fp32_interleave16", VENTUS_TMA_G2S, VENTUS_TMA_DTYPE_FP32, 3, 1, 0, 0,
       {4, 8, 2, 0, 0}, {16, 128, 0, 0}, {4, 8, 2, 0, 0}, {}},
      {"rank3_fp32_interleave32_swizzle32", VENTUS_TMA_S2G, VENTUS_TMA_DTYPE_FP32, 3, 2, 1, 0,
       {8, 4, 2, 0, 0}, {32, 128, 0, 0}, {8, 4, 2, 0, 0}, {}},
      {"rank5_bf16", VENTUS_TMA_G2S, VENTUS_TMA_DTYPE_BF16, 5, 0, 1, 1,
       {8, 2, 2, 2, 2}, {16, 32, 64, 128}, {8, 2, 2, 2, 2}, {}},
      {"rank2_fp4", VENTUS_TMA_G2S, VENTUS_TMA_DTYPE_B4X16, 2, 0, 1, 0,
       {64, 2, 0, 0, 0}, {32, 0, 0, 0}, {32, 2, 0, 0, 0}, {}},
      {"invalid_interleave_rank2", VENTUS_TMA_S2G, VENTUS_TMA_DTYPE_FP32, 2, 1, 0, 0,
       {16, 4, 0, 0, 0}, {64, 0, 0, 0}, {8, 4, 0, 0, 0},
       {}, Status::kBadLayout},
      {"invalid_s2g_negative", VENTUS_TMA_S2G, VENTUS_TMA_DTYPE_U8, 1, 0, 0, 0,
       {64, 0, 0, 0, 0}, {}, {16, 0, 0, 0, 0},
       {-1, 0, 0, 0, 0}, Status::kBadCoordinate},
      {"aligned_negative_g2s", VENTUS_TMA_G2S, VENTUS_TMA_DTYPE_U8, 1, 0, 0, 0,
       {64, 0, 0, 0, 0}, {}, {16, 0, 0, 0, 0},
       {-16, 0, 0, 0, 0}},
      {"invalid_dynamic_alignment", VENTUS_TMA_G2S, VENTUS_TMA_DTYPE_U8,
       1, 0, 0, 0, {64, 0, 0, 0, 0}, {}, {16, 0, 0, 0, 0},
       {1, 0, 0, 0, 0}, Status::kBadAlignment},
      {"invalid_b4_coordinate_phase", VENTUS_TMA_G2S,
       VENTUS_TMA_DTYPE_B4X16, 1, 0, 0, 0,
       {64, 0, 0, 0, 0}, {}, {32, 0, 0, 0, 0},
       {16, 0, 0, 0, 0}, Status::kBadAlignment},
      {"invalid_b4_odd_dim0", VENTUS_TMA_G2S,
       VENTUS_TMA_DTYPE_B4X16, 1, 0, 0, 0,
       {65, 0, 0, 0, 0}, {}, {32, 0, 0, 0, 0},
       {}, Status::kBadDimension},
  };

  const std::array<uint8_t, 14> supported_dtypes = {
      VENTUS_TMA_DTYPE_U8, VENTUS_TMA_DTYPE_U16, VENTUS_TMA_DTYPE_U32,
      VENTUS_TMA_DTYPE_S32, VENTUS_TMA_DTYPE_U64, VENTUS_TMA_DTYPE_S64,
      VENTUS_TMA_DTYPE_FP16, VENTUS_TMA_DTYPE_FP32,
      VENTUS_TMA_DTYPE_FP32_FTZ, VENTUS_TMA_DTYPE_FP64,
      VENTUS_TMA_DTYPE_BF16, VENTUS_TMA_DTYPE_TF32,
      VENTUS_TMA_DTYPE_TF32_FTZ, VENTUS_TMA_DTYPE_B4X16};
  for (uint8_t dtype : supported_dtypes) {
    const uint32_t bits = dtype == VENTUS_TMA_DTYPE_B4X16
                              ? 4
                              : (dtype == VENTUS_TMA_DTYPE_U8
                                     ? 8
                                     : (dtype == VENTUS_TMA_DTYPE_U32 ||
                                                dtype == VENTUS_TMA_DTYPE_S32 ||
                                                dtype == VENTUS_TMA_DTYPE_FP32 ||
                                                dtype == VENTUS_TMA_DTYPE_FP32_FTZ ||
                                                dtype == VENTUS_TMA_DTYPE_TF32 ||
                                                dtype == VENTUS_TMA_DTYPE_TF32_FTZ
                                            ? 32
                                            : (dtype == VENTUS_TMA_DTYPE_U64 ||
                                                       dtype == VENTUS_TMA_DTYPE_S64 ||
                                                       dtype == VENTUS_TMA_DTYPE_FP64
                                                   ? 64
                                                   : 16)));
    const uint32_t elements = 128 / bits;
    cases.push_back({"dtype_" + std::to_string(dtype), VENTUS_TMA_G2S,
                     dtype, 1, 0, 0, 0,
                     {elements, 0, 0, 0, 0}, {},
                     {elements, 0, 0, 0, 0}, {}});
  }
  cases.push_back({"rank4_u16", VENTUS_TMA_S2G, VENTUS_TMA_DTYPE_U16,
                   4, 0, 0, 0, {16, 2, 2, 2, 0}, {32, 64, 128, 0},
                   {8, 2, 2, 2, 0}, {}});
  cases.push_back({"b4x16_p64_g2s", VENTUS_TMA_G2S,
                   VENTUS_TMA_DTYPE_B4X16_P64, 1, 0, 3, 0,
                   {128, 0, 0, 0, 0}, {}, {128, 0, 0, 0, 0}, {}});
  cases.push_back({"b6x16_p32_g2s", VENTUS_TMA_G2S,
                   VENTUS_TMA_DTYPE_B6, 1, 0, 3, 0,
                   {128, 0, 0, 0, 0}, {}, {128, 0, 0, 0, 0}, {}});
  cases.push_back({"b6p2x16_s2g", VENTUS_TMA_S2G,
                   VENTUS_TMA_DTYPE_B6, 1, 0, 0, 0,
                   {128, 0, 0, 0, 0}, {}, {128, 0, 0, 0, 0}, {}});
  cases.push_back({"invalid_interleave32_swizzle64", VENTUS_TMA_G2S,
                   VENTUS_TMA_DTYPE_FP32, 3, 2, 2, 0,
                   {16, 4, 2, 0, 0}, {64, 256, 0, 0},
                   {8, 4, 2, 0, 0}, {},
                   Status::kBadLayout});
  cases.push_back({"invalid_fp6_interleave16", VENTUS_TMA_G2S,
                   VENTUS_TMA_DTYPE_B6, 3, 1, 0, 0,
                   {128, 1, 1, 0, 0}, {96, 96, 0, 0},
                   {128, 1, 1, 0, 0}, {},
                   Status::kUnsupportedFeature});
  cases.push_back({"invalid_fp4_interleave16", VENTUS_TMA_G2S,
                   VENTUS_TMA_DTYPE_B4X16, 3, 1, 0, 0,
                   {32, 1, 1, 0, 0}, {32, 32, 0, 0},
                   {32, 1, 1, 0, 0}, {},
                   Status::kUnsupportedFeature});
  cases.push_back({"dim0_2pow32_sentinel", VENTUS_TMA_G2S,
                   VENTUS_TMA_DTYPE_U8, 1, 0, 0, 0,
                   {0, 0, 0, 0, 0}, {}, {16, 0, 0, 0, 0}, {}});
  return cases;
}

std::string HexBytes(const std::array<uint8_t, 16> &bytes) {
  std::ostringstream out;
  for (uint8_t value : bytes) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(value);
  return out.str();
}

int Run(const char *trace_path) {
  std::ofstream trace;
  if (trace_path) {
    trace.open(trace_path);
    trace << "case,status,ordinal,direction,fill,global_atom,shared_atom,global_mask,shared_mask,map,fill_bytes\n";
  }
  unsigned failures = 0;
  const auto cases = Cases();
  const Descriptor abi_golden = Encode(cases[7]);
  const bool abi_golden_pass =
      VENTUS_TMA_V2_WORD_MAGIC == 0 &&
      VENTUS_TMA_V2_WORD_CONTROL == 1 &&
      VENTUS_TMA_V2_WORD_GLOBAL_BASE == 2 &&
      VENTUS_TMA_V2_WORD_GLOBAL_DIMS == 4 &&
      VENTUS_TMA_V2_WORD_GLOBAL_STRIDES == 9 &&
      VENTUS_TMA_V2_WORD_BOX_DIMS == 17 &&
      VENTUS_TMA_V2_WORD_ELEMENT_STRIDES == 22 &&
      abi_golden.words[0] == VENTUS_TMA_V2_MAGIC &&
      abi_golden.words[1] ==
          (VENTUS_TMA_DTYPE_BF16 | (5u << 5) | (1u << 10) |
           (1u << 18)) &&
      abi_golden.words[2] == 0x10000 &&
      abi_golden.words[3] == 0 &&
      abi_golden.words[4] == 8 &&
      abi_golden.words[8] == 2 &&
      abi_golden.words[9] == 16 &&
      abi_golden.words[11] == 32 &&
      abi_golden.words[13] == 64 &&
      abi_golden.words[15] == 128 &&
      abi_golden.words[17] == 8 &&
      abi_golden.words[21] == 2 &&
      abi_golden.words[22] == 1 &&
      abi_golden.words[26] == 1 &&
      abi_golden.words[27] == 0 &&
      abi_golden.words[31] == 0;
  failures += !abi_golden_pass;
  std::cout << "CMODEL_CASE name=descriptor_v3_golden_words result="
            << (abi_golden_pass ? "PASS" : "FAIL") << '\n';
  for (const Case &c : cases) {
    Request request;
    request.direction = c.direction;
    request.shared_base = 0x2000;
    request.coordinates = c.coords;
    auto result = DecodeAndPlan(Encode(c), request);
    const bool pass = result.status == c.expected;
    std::cout << "CMODEL_CASE name=" << c.name
              << " status=" << ventus::tma::StatusName(result.status)
              << " atoms=" << result.atoms.size()
              << " result=" << (pass ? "PASS" : "FAIL") << '\n';
    failures += !pass;
    if (trace) {
      if (result.atoms.empty()) {
        trace << c.name << ',' << ventus::tma::StatusName(result.status)
              << ",-1," << unsigned(c.direction) << ",0,0,0,0,0,,\n";
      }
      for (const auto &atom : result.atoms) {
        trace << c.name << ',' << ventus::tma::StatusName(result.status) << ','
              << atom.ordinal << ',' << unsigned(atom.direction) << ',' << atom.fill
              << ',' << atom.global_atom << ',' << atom.shared_atom << ','
              << atom.global_mask << ',' << atom.shared_mask << ','
              << HexBytes(atom.global_to_shared) << ',' << HexBytes(atom.fill_bytes) << '\n';
      }
    }
  }

  const Case &plain = cases.front();
  Request request;
  request.direction = plain.direction;
  request.shared_base = 0x2000;
  auto l2_descriptor = Encode(plain);
  l2_descriptor.words[VENTUS_TMA_V2_WORD_CONTROL] |= 1u << 16;
  auto l2_result = DecodeAndPlan(l2_descriptor, request);
  failures += l2_result.status != Status::kUnsupportedFeature;
  std::cout << "CMODEL_CASE name=invalid_l2_promotion status="
            << ventus::tma::StatusName(l2_result.status)
            << " result="
            << (l2_result.status == Status::kUnsupportedFeature ? "PASS" : "FAIL")
            << '\n';

  auto overflow_descriptor = Encode(plain);
  overflow_descriptor.words[VENTUS_TMA_V2_WORD_GLOBAL_BASE] = 0xfffffff0u;
  auto overflow_result = DecodeAndPlan(overflow_descriptor, request);
  failures += overflow_result.status != Status::kAddressOverflow;
  std::cout << "CMODEL_CASE name=invalid_address_overflow status="
            << ventus::tma::StatusName(overflow_result.status)
            << " result="
            << (overflow_result.status == Status::kAddressOverflow ? "PASS" : "FAIL")
            << '\n';

  auto unused_stride_descriptor = Encode(cases[1]);
  unused_stride_descriptor.words[
      VENTUS_TMA_V2_WORD_GLOBAL_STRIDES + 2] = 0x100;
  request.direction = cases[1].direction;
  request.coordinates = cases[1].coords;
  auto unused_stride_result = DecodeAndPlan(unused_stride_descriptor, request);
  failures += unused_stride_result.status != Status::kReservedBits;
  std::cout << "CMODEL_CASE name=invalid_unused_stride status="
            << ventus::tma::StatusName(unused_stride_result.status)
            << " result="
            << (unused_stride_result.status == Status::kReservedBits ? "PASS" : "FAIL")
            << '\n';

  auto invalid_magic_descriptor = Encode(plain);
  invalid_magic_descriptor.words[0] = UINT32_C(0xdeadbeef);
  request.direction = plain.direction;
  request.coordinates = plain.coords;
  auto invalid_magic_result = DecodeAndPlan(invalid_magic_descriptor, request);
  failures += invalid_magic_result.status != Status::kBadMagic;
  std::cout << "CMODEL_CASE name=invalid_magic status="
            << ventus::tma::StatusName(invalid_magic_result.status)
            << " result=" << (invalid_magic_result.status == Status::kBadMagic ? "PASS" : "FAIL")
            << '\n';

  auto reserved_descriptor = Encode(plain);
  reserved_descriptor.words[27] = 1;
  auto reserved_result = DecodeAndPlan(reserved_descriptor, request);
  failures += reserved_result.status != Status::kReservedBits;
  std::cout << "CMODEL_CASE name=invalid_reserved_word27 status="
            << ventus::tma::StatusName(reserved_result.status)
            << " result="
            << (reserved_result.status == Status::kReservedBits ? "PASS" : "FAIL")
            << '\n';

  std::cout << "CMODEL_SUMMARY cases=" << cases.size() + 6
            << " failures=" << failures << '\n';
  return failures ? 1 : 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc == 1) return Run(nullptr);
  if (argc == 3 && std::string(argv[1]) == "--emit-trace") return Run(argv[2]);
  std::cerr << "usage: " << argv[0] << " [--emit-trace PATH]\n";
  return 2;
}
