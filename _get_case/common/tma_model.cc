#include "tma_model.h"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <sstream>
#include <tuple>

namespace ventus::tma {
namespace {

constexpr uint32_t kMagic = VENTUS_TMA_V2_MAGIC;

struct DTypeInfo {
  uint8_t bits;
  bool floating;
  bool supported;
};

DTypeInfo DType(uint32_t code) {
  switch (code) {
    case VENTUS_TMA_DTYPE_U8:
      return {8, false, true};
    case VENTUS_TMA_DTYPE_U16:
      return {16, false, true};
    case VENTUS_TMA_DTYPE_U32:
    case VENTUS_TMA_DTYPE_S32:
      return {32, false, true};
    case VENTUS_TMA_DTYPE_U64:
    case VENTUS_TMA_DTYPE_S64:
      return {64, false, true};
    case VENTUS_TMA_DTYPE_FP32:
    case VENTUS_TMA_DTYPE_FP32_FTZ:
    case VENTUS_TMA_DTYPE_TF32:
    case VENTUS_TMA_DTYPE_TF32_FTZ:
      return {32, true, true};
    case VENTUS_TMA_DTYPE_FP16:
    case VENTUS_TMA_DTYPE_BF16:
      return {16, true, true};
    case VENTUS_TMA_DTYPE_FP64:
      return {64, true, true};
    case VENTUS_TMA_DTYPE_B4X16:
    case VENTUS_TMA_DTYPE_B4X16_P64:
      return {4, false, true};
    case VENTUS_TMA_DTYPE_B6:
      return {6, false, true};
    default:
      return {0, false, false};
  }
}

uint32_t SpanBytes(uint32_t elements, uint32_t bits) {
  return static_cast<uint32_t>((static_cast<uint64_t>(elements) * bits + 7) / 8);
}

uint32_t SwizzleOffset(uint32_t logical, uint32_t row, uint32_t shared_base,
                       uint32_t mode) {
  if (mode == VENTUS_TMA_SWIZZLE_NONE) return logical;
  const uint32_t chunk_bits = mode;
  const uint32_t span = 16u << chunk_bits;
  const uint32_t mask = (1u << chunk_bits) - 1;
  const uint32_t phase = (shared_base >> 7) & mask;
  return (logical & ~(span - 1)) |
         ((((logical >> 4) & mask) ^ ((row + phase) & mask)) << 4) |
         (logical & 15);
}

uint64_t GlobalOffset(const Command &cmd,
                      const std::array<int64_t, 5> &coord) {
  const uint32_t elem_bytes = cmd.dtype_bits / 8;
  const bool sub_byte = cmd.dtype >= VENTUS_TMA_DTYPE_B4X16;
  if (cmd.interleave == VENTUS_TMA_INTERLEAVE_NONE || sub_byte) {
    uint64_t off = cmd.dtype_bits == 4
                       ? static_cast<uint64_t>(coord[0]) / 2
                       : cmd.dtype_bits == 6
                             ? static_cast<uint64_t>(coord[0]) * 6 / 8
                             : static_cast<uint64_t>(coord[0]) * elem_bytes;
    for (uint32_t d = 1; d < cmd.rank; ++d) {
      off += static_cast<uint64_t>(coord[d]) * cmd.global_strides[d - 1];
    }
    return off;
  }

  const uint32_t slice_bytes =
      cmd.interleave == VENTUS_TMA_INTERLEAVE_16B ? 16 : 32;
  const uint32_t channels_per_slice = slice_bytes / elem_bytes;
  const uint64_t slice = static_cast<uint64_t>(coord[0]) / channels_per_slice;
  const uint64_t in_slice = static_cast<uint64_t>(coord[0]) % channels_per_slice;
  const uint64_t slice_stride =
      static_cast<uint64_t>(cmd.global_strides[cmd.rank - 3]) *
      cmd.global_dims[cmd.rank - 2];
  uint64_t off = in_slice * elem_bytes + slice * slice_stride;
  for (uint32_t d = 1; d < cmd.rank; ++d) {
    off += static_cast<uint64_t>(coord[d]) * cmd.global_strides[d - 1];
  }
  return off;
}

std::array<uint8_t, 8> NaNBytes(uint8_t dtype) {
  switch (dtype) {
    case VENTUS_TMA_DTYPE_FP16:
      return {0x00, 0x7e, 0, 0, 0, 0, 0, 0};
    case VENTUS_TMA_DTYPE_BF16:
      return {0xc0, 0x7f, 0, 0, 0, 0, 0, 0};
    case VENTUS_TMA_DTYPE_FP32:
      return {0x00, 0x00, 0xc0, 0x7f, 0, 0, 0, 0};
    case VENTUS_TMA_DTYPE_FP32_FTZ:
    case VENTUS_TMA_DTYPE_TF32:
    case VENTUS_TMA_DTYPE_TF32_FTZ:
      return {0x00, 0x00, 0xc0, 0x7f, 0, 0, 0, 0};
    case VENTUS_TMA_DTYPE_FP64:
      return {0, 0, 0, 0, 0, 0, 0xf8, 0x7f};
    default:
      return {0, 0, 0, 0, 0, 0, 0, 0};
  }
}

Status Decode(const Descriptor &descriptor, const Request &request,
              Command *cmd, std::string *detail) {
  if (descriptor.words[VENTUS_TMA_V2_WORD_MAGIC] != kMagic) {
    *detail = "word0 magic/version mismatch";
    return Status::kBadMagic;
  }
  for (uint32_t word = 27; word < 32; ++word)
    if (descriptor.words[word] != 0) {
      *detail = "reserved descriptor word is nonzero";
      return Status::kReservedBits;
    }

  const uint32_t control = descriptor.words[VENTUS_TMA_V2_WORD_CONTROL];
  if ((control >> 21) != 0 || (control & (1u << 15)) != 0) {
    *detail = "reserved control bit is nonzero";
    return Status::kReservedBits;
  }
  cmd->dtype = control & 0x1f;
  cmd->rank = (control >> 5) & 0x7;
  cmd->interleave = (control >> 8) & 0x3;
  cmd->swizzle = (control >> 10) & 0x7;
  cmd->swizzle_atomicity = (control >> 13) & 0x3;
  const uint32_t l2_promotion = (control >> 16) & 0x3;
  cmd->oob_fill = (control >> 18) & 0x1;
  const uint32_t access_mode = (control >> 19) & 0x3;
  const DTypeInfo dtype = DType(cmd->dtype);
  cmd->dtype_bits = dtype.bits;
  if (!dtype.supported) {
    *detail = "dtype is outside the v2 payload subset";
    return Status::kUnsupportedDType;
  }
  if (cmd->rank < 1 || cmd->rank > VENTUS_TMA_V2_RANK_MAX) {
    *detail = "rank is outside 1..5";
    return Status::kBadRank;
  }
  if (l2_promotion != 0 || access_mode != 0 ||
      cmd->swizzle_atomicity != VENTUS_TMA_SWIZZLE_ATOM_16B ||
      cmd->swizzle > VENTUS_TMA_SWIZZLE_128B) {
    *detail = "encoded CUDA extension is outside the Ventus subset";
    return Status::kUnsupportedFeature;
  }
  if (cmd->oob_fill > VENTUS_TMA_OOB_NAN ||
      (cmd->oob_fill == VENTUS_TMA_OOB_NAN &&
       (!dtype.floating || request.direction == VENTUS_TMA_S2G))) {
    *detail = "OOB fill mode is unsupported for this direction/dtype";
    return Status::kUnsupportedFeature;
  }
  if (cmd->interleave > VENTUS_TMA_INTERLEAVE_32B) {
    *detail = "layout mode is not standard v2";
    return Status::kBadLayout;
  }
  if (cmd->interleave != VENTUS_TMA_INTERLEAVE_NONE && cmd->rank < 3) {
    *detail = "interleave requires rank >= 3";
    return Status::kBadLayout;
  }
  if (cmd->interleave == VENTUS_TMA_INTERLEAVE_32B &&
      cmd->swizzle != VENTUS_TMA_SWIZZLE_32B) {
    *detail = "32B interleave requires 32B swizzle";
    return Status::kBadLayout;
  }
  cmd->direction = request.direction;
  cmd->global_base =
      static_cast<uint64_t>(descriptor.words[VENTUS_TMA_V2_WORD_GLOBAL_BASE]) |
      (static_cast<uint64_t>(
           descriptor.words[VENTUS_TMA_V2_WORD_GLOBAL_BASE + 1]) << 32);
  cmd->shared_base = request.shared_base;
  cmd->coordinates = request.coordinates;
  const bool sub_byte = cmd->dtype >= VENTUS_TMA_DTYPE_B4X16;
  const bool padded_sub_byte =
      cmd->dtype == VENTUS_TMA_DTYPE_B4X16_P64 ||
      cmd->dtype == VENTUS_TMA_DTYPE_B6;
  const uint32_t global_alignment =
      padded_sub_byte || cmd->interleave == VENTUS_TMA_INTERLEAVE_32B
          ? 32 : 16;
  if (cmd->global_base % global_alignment != 0) {
    *detail = "global base alignment does not match the layout";
    return Status::kBadAlignment;
  }
  if (cmd->shared_base % 128 != 0) {
    *detail = "shared base is not 128B aligned";
    return Status::kBadAlignment;
  }
  if (cmd->global_base > std::numeric_limits<uint32_t>::max()) {
    *detail = "global base exceeds the Ventus virtual-address width";
    return Status::kAddressOverflow;
  }

  for (uint32_t d = 0; d < 5; ++d) {
    const uint32_t raw_dim =
        descriptor.words[VENTUS_TMA_V2_WORD_GLOBAL_DIMS + d];
    cmd->global_dims[d] = raw_dim == 0 && d < cmd->rank
                              ? (uint64_t{1} << 32)
                              : raw_dim;
    cmd->box_dims[d] = descriptor.words[VENTUS_TMA_V2_WORD_BOX_DIMS + d];
    cmd->element_strides[d] =
        descriptor.words[VENTUS_TMA_V2_WORD_ELEMENT_STRIDES + d];
    if (d < 4) {
      const uint32_t word = VENTUS_TMA_V2_WORD_GLOBAL_STRIDES + d * 2;
      cmd->global_strides[d] =
          static_cast<uint64_t>(descriptor.words[word]) |
          (static_cast<uint64_t>(descriptor.words[word + 1]) << 32);
    }
    if (d < cmd->rank) {
      if (cmd->box_dims[d] == 0 || cmd->box_dims[d] > 256) {
        *detail = "active box dimension is outside 1..256";
        return Status::kBadDimension;
      }
      if (cmd->element_strides[d] != 1) {
        *detail = "elementStride other than one is not implemented";
        return Status::kUnsupportedFeature;
      }
    } else if (cmd->global_dims[d] != 0 || cmd->box_dims[d] != 0 ||
               cmd->element_strides[d] != 0 ||
               (d < 4 && cmd->global_strides[d] != 0)) {
      *detail = "inactive dimension field is nonzero";
      return Status::kReservedBits;
    }
  }
  for (uint32_t stride = cmd->rank - 1; stride < 4; ++stride) {
    if (cmd->global_strides[stride] != 0) {
      *detail = "unused global stride field is nonzero";
      return Status::kReservedBits;
    }
  }
  uint64_t required_span = SpanBytes(cmd->global_dims[0], cmd->dtype_bits);
  for (uint32_t d = 1; d < cmd->rank; ++d) {
    const uint64_t stride = cmd->global_strides[d - 1];
    const uint32_t alignment =
        padded_sub_byte || cmd->interleave == VENTUS_TMA_INTERLEAVE_32B
            ? 32 : 16;
    if (stride == 0 || stride % alignment != 0 || stride < required_span) {
      *detail = "global stride is misaligned or overlaps lower dimensions";
      return Status::kBadStride;
    }
    required_span = static_cast<uint64_t>(stride) * cmd->global_dims[d];
  }
  const uint32_t box_row_bytes = SpanBytes(cmd->box_dims[0], cmd->dtype_bits);
  if (box_row_bytes == 0 || box_row_bytes % 16 != 0) {
    *detail = "dim0 box span must be a multiple of 16B";
    return Status::kBadDimension;
  }
  if (cmd->dtype == VENTUS_TMA_DTYPE_B4X16 &&
      (cmd->global_dims[0] & 1) != 0) {
    *detail = "packed B4 dim0 must contain an even number of elements";
    return Status::kBadDimension;
  }
  if (cmd->swizzle != VENTUS_TMA_SWIZZLE_NONE &&
      box_row_bytes > (16u << cmd->swizzle)) {
    *detail = "dim0 box span exceeds selected swizzle span";
    return Status::kBadLayout;
  }
  if (sub_byte && cmd->oob_fill != VENTUS_TMA_OOB_ZERO) {
    *detail = "CUDA sub-byte Tensor Maps only support zero OOB fill";
    return Status::kUnsupportedFeature;
  }
  if (sub_byte && cmd->interleave != VENTUS_TMA_INTERLEAVE_NONE) {
    *detail = "CUDA sub-byte Tensor Maps require interleave NONE";
    return Status::kUnsupportedFeature;
  }
  if (cmd->dtype == VENTUS_TMA_DTYPE_B4X16_P64 &&
      request.direction != VENTUS_TMA_G2S) {
    *detail = "b4x16_p64 is load-only";
    return Status::kUnsupportedFeature;
  }
  if (padded_sub_byte &&
      (cmd->box_dims[0] != 128 || (cmd->global_dims[0] % 128) != 0 ||
       (cmd->coordinates[0] % 128) != 0)) {
    *detail = "padded sub-byte mode requires a 128-element dim0 window";
    return Status::kBadDimension;
  }
  if (cmd->dtype == VENTUS_TMA_DTYPE_B6) {
    const bool legal_fp6_layout =
        cmd->interleave == VENTUS_TMA_INTERLEAVE_NONE &&
        (cmd->swizzle == VENTUS_TMA_SWIZZLE_NONE ||
         cmd->swizzle == VENTUS_TMA_SWIZZLE_128B);
    if (!legal_fp6_layout) {
      *detail = "FP6 interleave/swizzle combination is unsupported";
      return Status::kUnsupportedFeature;
    }
  }
  if (request.direction == VENTUS_TMA_S2G) {
    for (uint32_t d = 0; d < cmd->rank; ++d) {
      if (cmd->coordinates[d] < 0) {
        *detail = "S2G does not accept a negative origin";
        return Status::kBadCoordinate;
      }
    }
  }
  if (cmd->dtype == VENTUS_TMA_DTYPE_B4X16 &&
      (cmd->coordinates[0] & 31) != 0) {
    *detail = "packed B4 coordinate0 must be a multiple of 32 elements";
    return Status::kBadAlignment;
  }

  // CUDA requires the bounding-box start address, not every internal lane,
  // to be 16B aligned. Use signed arithmetic so legal negative G2S
  // coordinates either form a nonnegative aligned origin or fail before any
  // memory task is produced.
  __int128 bounding_origin = cmd->global_base;
  if (cmd->interleave == VENTUS_TMA_INTERLEAVE_NONE || sub_byte) {
    if (cmd->dtype_bits == 4) {
      bounding_origin += static_cast<__int128>(cmd->coordinates[0]) / 2;
    } else if (cmd->dtype_bits == 6) {
      bounding_origin +=
          static_cast<__int128>(cmd->coordinates[0]) * 3 / 4;
    } else {
      bounding_origin += static_cast<__int128>(cmd->coordinates[0]) *
                         (cmd->dtype_bits / 8);
    }
  } else {
    const uint32_t slice_bytes =
        cmd->interleave == VENTUS_TMA_INTERLEAVE_16B ? 16 : 32;
    const uint32_t channels = slice_bytes / (cmd->dtype_bits / 8);
    const int64_t coordinate0 = cmd->coordinates[0];
    const int64_t slice = coordinate0 >= 0
        ? coordinate0 / channels
        : -static_cast<int64_t>(
              (static_cast<uint64_t>(-coordinate0) + channels - 1) /
              channels);
    const uint32_t in_slice =
        static_cast<uint32_t>(coordinate0) & (channels - 1);
    const __int128 slice_stride =
        static_cast<__int128>(cmd->global_strides[cmd->rank - 3]) *
        cmd->global_dims[cmd->rank - 2];
    bounding_origin += static_cast<__int128>(slice) * slice_stride +
                       in_slice * (cmd->dtype_bits / 8);
  }
  for (uint32_t d = 1; d < cmd->rank; ++d) {
    bounding_origin += static_cast<__int128>(cmd->coordinates[d]) *
                       cmd->global_strides[d - 1];
  }
  if (bounding_origin < 0 ||
      bounding_origin > std::numeric_limits<uint32_t>::max()) {
    *detail = "bounding-box origin exceeds the Ventus address width";
    return Status::kAddressOverflow;
  }
  if ((static_cast<uint64_t>(bounding_origin) & 15) != 0) {
    *detail = "bounding-box origin is not 16B aligned";
    return Status::kBadAlignment;
  }
  return Status::kOk;
}

struct AtomKey {
  bool fill;
  uint32_t global_atom;
  uint32_t shared_atom;
  bool operator<(const AtomKey &other) const {
    return std::tie(shared_atom, fill, global_atom) <
           std::tie(other.shared_atom, other.fill, other.global_atom);
  }
};

}  // namespace

const char *StatusName(Status status) {
  switch (status) {
    case Status::kOk: return "ok";
    case Status::kBadMagic: return "bad_magic";
    case Status::kReservedBits: return "reserved_bits";
    case Status::kUnsupportedDType: return "unsupported_dtype";
    case Status::kBadRank: return "bad_rank";
    case Status::kBadLayout: return "bad_layout";
    case Status::kBadAlignment: return "bad_alignment";
    case Status::kBadDimension: return "bad_dimension";
    case Status::kBadStride: return "bad_stride";
    case Status::kBadCoordinate: return "bad_coordinate";
    case Status::kAddressOverflow: return "address_overflow";
    case Status::kUnsupportedFeature: return "unsupported_feature";
  }
  return "unknown";
}

Result DecodeAndPlan(const Descriptor &descriptor, const Request &request) {
  Result result;
  result.status = Decode(descriptor, request, &result.command, &result.detail);
  if (result.status != Status::kOk) return result;
  const Command &cmd = result.command;
  std::map<AtomKey, AtomTask> grouped;
  std::array<uint32_t, 5> index{};
  bool done = false;
  while (!done) {
    std::array<int64_t, 5> coord{};
    uint64_t linear_element = 0;
    uint64_t multiplier = 1;
    uint32_t row = 0;
    uint32_t row_multiplier = 1;
    bool in_bounds = true;
    for (uint32_t d = 0; d < cmd.rank; ++d) {
      coord[d] = static_cast<int64_t>(cmd.coordinates[d]) + index[d];
      in_bounds &= coord[d] >= 0 &&
                   static_cast<uint64_t>(coord[d]) < cmd.global_dims[d];
      linear_element += static_cast<uint64_t>(index[d]) * multiplier;
      multiplier *= cmd.box_dims[d];
      if (d > 0) {
        row += index[d] * row_multiplier;
        row_multiplier *= cmd.box_dims[d];
      }
    }

    const bool padded_sub_byte =
        cmd.dtype == VENTUS_TMA_DTYPE_B4X16_P64 ||
        cmd.dtype == VENTUS_TMA_DTYPE_B6;
    if (padded_sub_byte && index[0] == 0) {
      const uint32_t bytes_per_lane =
          cmd.dtype == VENTUS_TMA_DTYPE_B4X16_P64 ? 8 : 12;
      const uint64_t global_window = in_bounds
          ? cmd.global_base + GlobalOffset(cmd, coord) : 0;
      for (uint32_t lane = 0; lane < 8; ++lane) {
        for (uint32_t byte = 0; byte < bytes_per_lane; ++byte) {
          const uint64_t global64 =
              global_window + lane * bytes_per_lane + byte;
          if (global64 > std::numeric_limits<uint32_t>::max()) {
            result.status = Status::kAddressOverflow;
            result.detail = "window access exceeds the Ventus address width";
            result.atoms.clear();
            return result;
          }
          const uint32_t logical = row * 128 + lane * 16 + byte;
          const uint32_t shared = cmd.shared_base +
              SwizzleOffset(logical, row, cmd.shared_base, cmd.swizzle);
          const uint32_t global = static_cast<uint32_t>(global64);
          const AtomKey key{false, global & ~15u, shared & ~15u};
          AtomTask &atom = grouped[key];
          if (atom.global_mask == 0 && atom.shared_mask == 0)
            atom.global_to_shared.fill(0xff);
          atom.direction = cmd.direction;
          atom.global_atom = key.global_atom;
          atom.shared_atom = key.shared_atom;
          atom.global_mask |= uint16_t{1} << (global & 15);
          atom.shared_mask |= uint16_t{1} << (shared & 15);
          // For FP6 S2G this field names the packed output byte.  The
          // functional executor performs the low-6-bit pack explicitly.
          atom.global_to_shared[global & 15] = shared & 15;
        }
      }
    } else {
      const bool packed_skip = cmd.dtype_bits == 4 && (index[0] & 1u);
      if (!packed_skip && !padded_sub_byte) {
      const uint32_t elem_bytes = cmd.dtype_bits == 4 ? 1 : cmd.dtype_bits / 8;
      const uint32_t byte_in_row =
          cmd.dtype_bits == 4 ? index[0] / 2 : index[0] * elem_bytes;
      const uint64_t logical_byte =
          cmd.swizzle == VENTUS_TMA_SWIZZLE_NONE
              ? (cmd.dtype_bits == 4 ? linear_element / 2
                                     : linear_element * elem_bytes)
              : static_cast<uint64_t>(row) * (16u << cmd.swizzle) +
                    byte_in_row;
      const uint32_t shared_base = cmd.shared_base +
          SwizzleOffset(static_cast<uint32_t>(logical_byte), row,
                        cmd.shared_base, cmd.swizzle);
      if (in_bounds) {
        const uint64_t global = cmd.global_base + GlobalOffset(cmd, coord);
        if (global + elem_bytes - 1 >
            std::numeric_limits<uint32_t>::max()) {
          result.status = Status::kAddressOverflow;
          result.detail = "element access exceeds the Ventus address width";
          result.atoms.clear();
          return result;
        }
        for (uint32_t byte = 0; byte < elem_bytes; ++byte) {
          const uint32_t ga = static_cast<uint32_t>(global + byte);
          const uint32_t sa = shared_base + byte;
          const AtomKey key{false, ga & ~15u, sa & ~15u};
          AtomTask &atom = grouped[key];
          if (atom.global_mask == 0 && atom.shared_mask == 0) {
            atom.global_to_shared.fill(0xff);
          }
          atom.direction = cmd.direction;
          atom.global_atom = key.global_atom;
          atom.shared_atom = key.shared_atom;
          atom.global_mask |= uint16_t{1} << (ga & 15);
          atom.shared_mask |= uint16_t{1} << (sa & 15);
          atom.global_to_shared[ga & 15] = sa & 15;
        }
      } else if (cmd.direction == VENTUS_TMA_G2S) {
        const auto nan = NaNBytes(cmd.dtype);
        for (uint32_t byte = 0; byte < elem_bytes; ++byte) {
          const uint32_t sa = shared_base + byte;
          const AtomKey key{true, 0, sa & ~15u};
          AtomTask &atom = grouped[key];
          if (atom.shared_mask == 0) atom.global_to_shared.fill(0xff);
          atom.direction = cmd.direction;
          atom.fill = true;
          atom.shared_atom = key.shared_atom;
          atom.shared_mask |= uint16_t{1} << (sa & 15);
          atom.fill_bytes[sa & 15] =
              cmd.oob_fill == VENTUS_TMA_OOB_NAN ? nan[byte] : 0;
        }
      }
      }
    }

    uint32_t dim = 0;
    while (dim < cmd.rank) {
      ++index[dim];
      if (index[dim] < cmd.box_dims[dim]) break;
      index[dim] = 0;
      ++dim;
    }
    done = dim == cmd.rank;
  }

  uint32_t ordinal = 0;
  for (auto &[key, atom] : grouped) {
    atom.ordinal = ordinal++;
    result.atoms.push_back(atom);
  }
  return result;
}

}  // namespace ventus::tma
