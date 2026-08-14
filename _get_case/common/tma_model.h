#ifndef VENTUS_TMA_MODEL_H
#define VENTUS_TMA_MODEL_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "ventus_tma_v2_spec.h"

namespace ventus::tma {

enum class Status : uint32_t {
  kOk = 0,
  kBadMagic = 1,
  // Value 2 is reserved for the retired descriptor-size error.
  kReservedBits = 3,
  kUnsupportedDType = 4,
  kBadRank = 5,
  kBadLayout = 6,
  kBadAlignment = 7,
  kBadDimension = 8,
  kBadStride = 9,
  kBadCoordinate = 10,
  kAddressOverflow = 11,
  kUnsupportedFeature = 12,
};

struct Descriptor {
  std::array<uint32_t, 32> words{};
};

struct Request {
  uint8_t direction = VENTUS_TMA_G2S;
  uint32_t shared_base = 0;
  std::array<int32_t, 5> coordinates{};
};

struct Command {
  uint8_t direction = 0;
  uint8_t dtype = 0;
  uint8_t dtype_bits = 0;
  uint8_t rank = 0;
  uint8_t interleave = 0;
  uint8_t swizzle = 0;
  uint8_t swizzle_atomicity = 0;
  uint8_t oob_fill = 0;
  uint64_t global_base = 0;
  uint32_t shared_base = 0;
  std::array<uint64_t, 5> global_dims{};
  std::array<uint64_t, 4> global_strides{};
  std::array<uint32_t, 5> box_dims{};
  std::array<uint32_t, 5> element_strides{};
  std::array<int32_t, 5> coordinates{};
};

struct AtomTask {
  uint32_t ordinal = 0;
  uint8_t direction = 0;
  bool fill = false;
  uint32_t global_atom = 0;
  uint32_t shared_atom = 0;
  uint16_t global_mask = 0;
  uint16_t shared_mask = 0;
  std::array<uint8_t, 16> global_to_shared{};
  std::array<uint8_t, 16> fill_bytes{};
};

struct Result {
  Status status = Status::kOk;
  std::string detail;
  Command command{};
  std::vector<AtomTask> atoms;
};

const char *StatusName(Status status);
Result DecodeAndPlan(const Descriptor &descriptor, const Request &request);

}  // namespace ventus::tma

#endif
