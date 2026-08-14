/*
 * Tensor TMA advanced-feature diagnostic performance test.
 *
 * Background:
 *   The main movement/pingpong performance tests cover rank2 FP32 linear TMA.
 *   This diagnostic isolates tensor descriptor features with paired baselines:
 *   compare comparable logical shapes while toggling swizzle, interleave,
 *   stride, subbox, OOB suppress, or higher-rank addressing.
 *
 * Implementation:
 *   One OpenCL kernel supports both CP_ASYNC_TENSOR_G2S and
 *   CP_ASYNC_TENSOR_S2G.  The host changes only the tensor descriptor and
 *   direction, validates the final bytes, and reports one timing row for each
 *   direction/case pair.
 *
 * Usage:
 *   ./dma_tma_tensor_feature_perf_test.out
 *   ./dma_tma_tensor_feature_perf_test.out sweep [iterations]
 *   ./dma_tma_tensor_feature_perf_test.out stress [iterations]
 *   ./dma_tma_tensor_feature_perf_test.out single <feature-name> [iterations]
 *
 * Maintenance:
 *   Keep this as diagnostic/profile coverage.  Do not use it as a correctness
 *   substitute for dma_tma_s2g_func_test or dma_tma_g2s_func_test, and do not
 *   add it to directed gate.  Stress mode runs the default sweep plus the
 *   extended stress matrix in both directions, with duplicate names removed.
 */

#define _POSIX_C_SOURCE 200809L

#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

#include "../common/ventus_opencl_test.h"
#include "../common/ventus_tma_v2_spec.h"

#define DESC_WORDS 32u
#define COORD_WORDS 32u
#define MAX_RANK 5u
#define WG_SIZE 32u
#define FEATURE_GLOBAL_BYTES 65536u
#define FEATURE_SHARED_ELEMS 512u
#define FEATURE_SHARED_BYTES (FEATURE_SHARED_ELEMS * sizeof(uint32_t))
#define DEFAULT_ITERATIONS 8u
#define MAX_ITERATIONS 64u

typedef enum {
  FEATURE_DIR_G2S = 0,
  FEATURE_DIR_S2G = 1,
  FEATURE_DIR_COUNT = 2,
} feature_direction_t;

typedef struct {
  const char *name;
  unsigned data_type;
  unsigned rank;
  unsigned interleave_mode;
  unsigned swizzle_mode;
  unsigned global_dim[MAX_RANK];
  unsigned global_strides[MAX_RANK];
  unsigned box_dim[MAX_RANK];
  unsigned coord[MAX_RANK];
} feature_case_t;

typedef struct {
  const feature_case_t *feature;
  feature_direction_t direction;
  unsigned iterations;
  size_t logical_elems;
  size_t written_elems;
  size_t dst_span;
  size_t shared_span;
  size_t unique_global_lines;
  uint64_t ns;
  uint64_t cycles;
  uint64_t cache_requests;
  int cache_requests_valid;
} feature_result_t;

typedef enum {
  RUN_SWEEP,
  RUN_STRESS,
  RUN_SINGLE,
  RUN_SINGLE_DIR,
} run_mode_t;

static const char *direction_name(feature_direction_t direction)
{
  return direction == FEATURE_DIR_G2S ? "g2s" : "s2g";
}

static int parse_direction(const char *text, feature_direction_t *direction)
{
  if (strcmp(text, "g2s") == 0) {
    *direction = FEATURE_DIR_G2S;
    return 0;
  }
  if (strcmp(text, "s2g") == 0) {
    *direction = FEATURE_DIR_S2G;
    return 0;
  }
  return 1;
}

static const feature_case_t feature_cases[] = {
  {
    .name = "linear_rank2_16x16",
    .rank = 2,
    .global_dim = {16, 16, 1, 1, 1},
    .global_strides = {64, 0, 0, 0, 0},
    .box_dim = {16, 16, 1, 1, 1},
  },
  {
    .name = "linear_rank3_8x8x4",
    .rank = 3,
    .global_dim = {8, 8, 4, 1, 1},
    .global_strides = {32, 256, 0, 0, 0},
    .box_dim = {8, 8, 4, 1, 1},
  },
  {
    .name = "base_interleave16_rank3_C8_W3_N2",
    .rank = 3,
    .global_dim = {8, 3, 2, 1, 1},
    .global_strides = {32, 192, 0, 0, 0},
    .box_dim = {8, 3, 2, 1, 1},
  },
  {
    .name = "interleave16_rank3_C8_W3_N2",
    .rank = 3,
    .interleave_mode = 1,
    .global_dim = {8, 3, 2, 1, 1},
    .global_strides = {32, 192, 0, 0, 0},
    .box_dim = {8, 3, 2, 1, 1},
  },
  {
    .name = "base_interleave32_rank3_C16_W2_N2",
    .rank = 3,
    .global_dim = {16, 2, 2, 1, 1},
    .global_strides = {64, 256, 0, 0, 0},
    .box_dim = {16, 2, 2, 1, 1},
  },
  {
    .name = "interleave32_rank3_C16_W2_N2",
    .rank = 3,
    .interleave_mode = 2,
    .global_dim = {16, 2, 2, 1, 1},
    .global_strides = {64, 256, 0, 0, 0},
    .box_dim = {16, 2, 2, 1, 1},
  },
  {
    .name = "interleave16_rank3_C6_W3_N2",
    .rank = 3,
    .interleave_mode = 1,
    .global_dim = {6, 3, 2, 1, 1},
    .global_strides = {16, 96, 0, 0, 0},
    .box_dim = {6, 2, 1, 1, 1},
    .coord = {0, 1, 1, 0, 0},
  },
  {
    .name = "interleave32_rank3_C10_W2",
    .rank = 3,
    .interleave_mode = 2,
    .global_dim = {10, 2, 1, 1, 1},
    .global_strides = {32, 128, 0, 0, 0},
    .box_dim = {10, 2, 1, 1, 1},
  },
  {
    .name = "base_swizzle32_rank2_8x16",
    .rank = 2,
    .global_dim = {8, 16, 1, 1, 1},
    .global_strides = {32, 0, 0, 0, 0},
    .box_dim = {8, 16, 1, 1, 1},
  },
  {
    .name = "swizzle32_rank2_8x16",
    .rank = 2,
    .swizzle_mode = 1,
    .global_dim = {8, 16, 1, 1, 1},
    .global_strides = {32, 0, 0, 0, 0},
    .box_dim = {8, 16, 1, 1, 1},
  },
  {
    .name = "base_swizzle32_rank2_16x16",
    .rank = 2,
    .global_dim = {16, 16, 1, 1, 1},
    .global_strides = {64, 0, 0, 0, 0},
    .box_dim = {16, 16, 1, 1, 1},
  },
  {
    .name = "swizzle32_rank2_16x16",
    .rank = 2,
    .swizzle_mode = 1,
    .global_dim = {16, 16, 1, 1, 1},
    .global_strides = {64, 0, 0, 0, 0},
    .box_dim = {16, 16, 1, 1, 1},
  },
  {
    .name = "base_swizzle64_rank2_16x16",
    .rank = 2,
    .global_dim = {16, 16, 1, 1, 1},
    .global_strides = {64, 0, 0, 0, 0},
    .box_dim = {16, 16, 1, 1, 1},
  },
  {
    .name = "swizzle64_rank2_16x16",
    .rank = 2,
    .swizzle_mode = 2,
    .global_dim = {16, 16, 1, 1, 1},
    .global_strides = {64, 0, 0, 0, 0},
    .box_dim = {16, 16, 1, 1, 1},
  },
  {
    .name = "base_swizzle128_rank2_16x16",
    .rank = 2,
    .global_dim = {16, 16, 1, 1, 1},
    .global_strides = {64, 0, 0, 0, 0},
    .box_dim = {16, 16, 1, 1, 1},
  },
  {
    .name = "swizzle128_rank2_16x16",
    .rank = 2,
    .swizzle_mode = 3,
    .global_dim = {16, 16, 1, 1, 1},
    .global_strides = {64, 0, 0, 0, 0},
    .box_dim = {16, 16, 1, 1, 1},
  },
  {
    .name = "base_swizzle128_rank2_32x8",
    .rank = 2,
    .global_dim = {32, 8, 1, 1, 1},
    .global_strides = {128, 0, 0, 0, 0},
    .box_dim = {32, 8, 1, 1, 1},
  },
  {
    .name = "swizzle128_rank2_32x8",
    .rank = 2,
    .swizzle_mode = 3,
    .global_dim = {32, 8, 1, 1, 1},
    .global_strides = {128, 0, 0, 0, 0},
    .box_dim = {32, 8, 1, 1, 1},
  },
  {
    .name = "base_interleave32_swizzle32_rank3_C8_W8_N4",
    .rank = 3,
    .global_dim = {8, 8, 4, 1, 1},
    .global_strides = {32, 256, 0, 0, 0},
    .box_dim = {8, 8, 4, 1, 1},
  },
  {
    .name = "interleave32_swizzle32_rank3_C8_W8_N4",
    .rank = 3,
    .interleave_mode = 2,
    .swizzle_mode = 1,
    .global_dim = {8, 8, 4, 1, 1},
    .global_strides = {32, 256, 0, 0, 0},
    .box_dim = {8, 8, 4, 1, 1},
  },
  {
    .name = "oob_rank2_16x16_at_8_0",
    .rank = 2,
    .global_dim = {16, 16, 1, 1, 1},
    .global_strides = {64, 0, 0, 0, 0},
    .box_dim = {16, 16, 1, 1, 1},
    .coord = {8, 0, 0, 0, 0},
  },
};

static const feature_case_t feature_stress_cases[] = {
  {
    .name = "base_swizzle32_rank2_4x64",
    .rank = 2,
    .global_dim = {4, 64, 1, 1, 1},
    .global_strides = {16, 0, 0, 0, 0},
    .box_dim = {4, 64, 1, 1, 1},
  },
  {
    .name = "swizzle32_rank2_4x64",
    .rank = 2,
    .swizzle_mode = 1,
    .global_dim = {4, 64, 1, 1, 1},
    .global_strides = {16, 0, 0, 0, 0},
    .box_dim = {4, 64, 1, 1, 1},
  },
  {
    .name = "base_swizzle32_rank2_8x32",
    .rank = 2,
    .global_dim = {8, 32, 1, 1, 1},
    .global_strides = {32, 0, 0, 0, 0},
    .box_dim = {8, 32, 1, 1, 1},
  },
  {
    .name = "swizzle32_rank2_8x32",
    .rank = 2,
    .swizzle_mode = 1,
    .global_dim = {8, 32, 1, 1, 1},
    .global_strides = {32, 0, 0, 0, 0},
    .box_dim = {8, 32, 1, 1, 1},
  },
  {
    .name = "base_swizzle32_rank2_12x16",
    .rank = 2,
    .global_dim = {12, 16, 1, 1, 1},
    .global_strides = {48, 0, 0, 0, 0},
    .box_dim = {12, 16, 1, 1, 1},
  },
  {
    .name = "swizzle32_rank2_12x16",
    .rank = 2,
    .swizzle_mode = 1,
    .global_dim = {12, 16, 1, 1, 1},
    .global_strides = {48, 0, 0, 0, 0},
    .box_dim = {12, 16, 1, 1, 1},
  },
  {
    .name = "base_swizzle32_rank2_16x16",
    .rank = 2,
    .global_dim = {16, 16, 1, 1, 1},
    .global_strides = {64, 0, 0, 0, 0},
    .box_dim = {16, 16, 1, 1, 1},
  },
  {
    .name = "swizzle32_rank2_16x16",
    .rank = 2,
    .swizzle_mode = 1,
    .global_dim = {16, 16, 1, 1, 1},
    .global_strides = {64, 0, 0, 0, 0},
    .box_dim = {16, 16, 1, 1, 1},
  },
  {
    .name = "base_swizzle32_rank2_24x8",
    .rank = 2,
    .global_dim = {24, 8, 1, 1, 1},
    .global_strides = {96, 0, 0, 0, 0},
    .box_dim = {24, 8, 1, 1, 1},
  },
  {
    .name = "swizzle32_rank2_24x8",
    .rank = 2,
    .swizzle_mode = 1,
    .global_dim = {24, 8, 1, 1, 1},
    .global_strides = {96, 0, 0, 0, 0},
    .box_dim = {24, 8, 1, 1, 1},
  },
  {
    .name = "base_swizzle32_rank2_32x8",
    .rank = 2,
    .global_dim = {32, 8, 1, 1, 1},
    .global_strides = {128, 0, 0, 0, 0},
    .box_dim = {32, 8, 1, 1, 1},
  },
  {
    .name = "swizzle32_rank2_32x8",
    .rank = 2,
    .swizzle_mode = 1,
    .global_dim = {32, 8, 1, 1, 1},
    .global_strides = {128, 0, 0, 0, 0},
    .box_dim = {32, 8, 1, 1, 1},
  },
  {
    .name = "base_swizzle32_rank2_48x4",
    .rank = 2,
    .global_dim = {48, 4, 1, 1, 1},
    .global_strides = {192, 0, 0, 0, 0},
    .box_dim = {48, 4, 1, 1, 1},
  },
  {
    .name = "swizzle32_rank2_48x4",
    .rank = 2,
    .swizzle_mode = 1,
    .global_dim = {48, 4, 1, 1, 1},
    .global_strides = {192, 0, 0, 0, 0},
    .box_dim = {48, 4, 1, 1, 1},
  },
  {
    .name = "base_swizzle32_rank2_64x4",
    .rank = 2,
    .global_dim = {64, 4, 1, 1, 1},
    .global_strides = {256, 0, 0, 0, 0},
    .box_dim = {64, 4, 1, 1, 1},
  },
  {
    .name = "swizzle32_rank2_64x4",
    .rank = 2,
    .swizzle_mode = 1,
    .global_dim = {64, 4, 1, 1, 1},
    .global_strides = {256, 0, 0, 0, 0},
    .box_dim = {64, 4, 1, 1, 1},
  },
  {
    .name = "base_swizzle32_rank2_96x2",
    .rank = 2,
    .global_dim = {96, 2, 1, 1, 1},
    .global_strides = {384, 0, 0, 0, 0},
    .box_dim = {96, 2, 1, 1, 1},
  },
  {
    .name = "swizzle32_rank2_96x2",
    .rank = 2,
    .swizzle_mode = 1,
    .global_dim = {96, 2, 1, 1, 1},
    .global_strides = {384, 0, 0, 0, 0},
    .box_dim = {96, 2, 1, 1, 1},
  },
  {
    .name = "base_swizzle64_rank2_8x32",
    .rank = 2,
    .global_dim = {8, 32, 1, 1, 1},
    .global_strides = {32, 0, 0, 0, 0},
    .box_dim = {8, 32, 1, 1, 1},
  },
  {
    .name = "swizzle64_rank2_8x32",
    .rank = 2,
    .swizzle_mode = 2,
    .global_dim = {8, 32, 1, 1, 1},
    .global_strides = {32, 0, 0, 0, 0},
    .box_dim = {8, 32, 1, 1, 1},
  },
  {
    .name = "base_swizzle64_rank2_16x16",
    .rank = 2,
    .global_dim = {16, 16, 1, 1, 1},
    .global_strides = {64, 0, 0, 0, 0},
    .box_dim = {16, 16, 1, 1, 1},
  },
  {
    .name = "swizzle64_rank2_16x16",
    .rank = 2,
    .swizzle_mode = 2,
    .global_dim = {16, 16, 1, 1, 1},
    .global_strides = {64, 0, 0, 0, 0},
    .box_dim = {16, 16, 1, 1, 1},
  },
  {
    .name = "base_swizzle64_rank2_24x8",
    .rank = 2,
    .global_dim = {24, 8, 1, 1, 1},
    .global_strides = {96, 0, 0, 0, 0},
    .box_dim = {24, 8, 1, 1, 1},
  },
  {
    .name = "swizzle64_rank2_24x8",
    .rank = 2,
    .swizzle_mode = 2,
    .global_dim = {24, 8, 1, 1, 1},
    .global_strides = {96, 0, 0, 0, 0},
    .box_dim = {24, 8, 1, 1, 1},
  },
  {
    .name = "base_swizzle64_rank2_32x8",
    .rank = 2,
    .global_dim = {32, 8, 1, 1, 1},
    .global_strides = {128, 0, 0, 0, 0},
    .box_dim = {32, 8, 1, 1, 1},
  },
  {
    .name = "swizzle64_rank2_32x8",
    .rank = 2,
    .swizzle_mode = 2,
    .global_dim = {32, 8, 1, 1, 1},
    .global_strides = {128, 0, 0, 0, 0},
    .box_dim = {32, 8, 1, 1, 1},
  },
  {
    .name = "base_swizzle64_rank2_48x4",
    .rank = 2,
    .global_dim = {48, 4, 1, 1, 1},
    .global_strides = {192, 0, 0, 0, 0},
    .box_dim = {48, 4, 1, 1, 1},
  },
  {
    .name = "swizzle64_rank2_48x4",
    .rank = 2,
    .swizzle_mode = 2,
    .global_dim = {48, 4, 1, 1, 1},
    .global_strides = {192, 0, 0, 0, 0},
    .box_dim = {48, 4, 1, 1, 1},
  },
  {
    .name = "base_swizzle64_rank2_64x4",
    .rank = 2,
    .global_dim = {64, 4, 1, 1, 1},
    .global_strides = {256, 0, 0, 0, 0},
    .box_dim = {64, 4, 1, 1, 1},
  },
  {
    .name = "swizzle64_rank2_64x4",
    .rank = 2,
    .swizzle_mode = 2,
    .global_dim = {64, 4, 1, 1, 1},
    .global_strides = {256, 0, 0, 0, 0},
    .box_dim = {64, 4, 1, 1, 1},
  },
  {
    .name = "base_swizzle128_rank2_16x16",
    .rank = 2,
    .global_dim = {16, 16, 1, 1, 1},
    .global_strides = {64, 0, 0, 0, 0},
    .box_dim = {16, 16, 1, 1, 1},
  },
  {
    .name = "swizzle128_rank2_16x16",
    .rank = 2,
    .swizzle_mode = 3,
    .global_dim = {16, 16, 1, 1, 1},
    .global_strides = {64, 0, 0, 0, 0},
    .box_dim = {16, 16, 1, 1, 1},
  },
  {
    .name = "base_swizzle128_rank2_32x8",
    .rank = 2,
    .global_dim = {32, 8, 1, 1, 1},
    .global_strides = {128, 0, 0, 0, 0},
    .box_dim = {32, 8, 1, 1, 1},
  },
  {
    .name = "swizzle128_rank2_32x8",
    .rank = 2,
    .swizzle_mode = 3,
    .global_dim = {32, 8, 1, 1, 1},
    .global_strides = {128, 0, 0, 0, 0},
    .box_dim = {32, 8, 1, 1, 1},
  },
  {
    .name = "base_swizzle128_rank2_48x4",
    .rank = 2,
    .global_dim = {48, 4, 1, 1, 1},
    .global_strides = {192, 0, 0, 0, 0},
    .box_dim = {48, 4, 1, 1, 1},
  },
  {
    .name = "swizzle128_rank2_48x4",
    .rank = 2,
    .swizzle_mode = 3,
    .global_dim = {48, 4, 1, 1, 1},
    .global_strides = {192, 0, 0, 0, 0},
    .box_dim = {48, 4, 1, 1, 1},
  },
  {
    .name = "base_swizzle128_rank2_64x4",
    .rank = 2,
    .global_dim = {64, 4, 1, 1, 1},
    .global_strides = {256, 0, 0, 0, 0},
    .box_dim = {64, 4, 1, 1, 1},
  },
  {
    .name = "swizzle128_rank2_64x4",
    .rank = 2,
    .swizzle_mode = 3,
    .global_dim = {64, 4, 1, 1, 1},
    .global_strides = {256, 0, 0, 0, 0},
    .box_dim = {64, 4, 1, 1, 1},
  },
  {
    .name = "base_interleave16_rank3_C4_W64_N1",
    .rank = 3,
    .global_dim = {4, 64, 1, 1, 1},
    .global_strides = {16, 1024, 0, 0, 0},
    .box_dim = {4, 64, 1, 1, 1},
  },
  {
    .name = "interleave16_rank3_C4_W64_N1",
    .rank = 3,
    .interleave_mode = 1,
    .global_dim = {4, 64, 1, 1, 1},
    .global_strides = {16, 1024, 0, 0, 0},
    .box_dim = {4, 64, 1, 1, 1},
  },
  {
    .name = "base_interleave16_rank3_C8_W32_N1",
    .rank = 3,
    .global_dim = {8, 32, 1, 1, 1},
    .global_strides = {32, 1024, 0, 0, 0},
    .box_dim = {8, 32, 1, 1, 1},
  },
  {
    .name = "interleave16_rank3_C8_W32_N1",
    .rank = 3,
    .interleave_mode = 1,
    .global_dim = {8, 32, 1, 1, 1},
    .global_strides = {32, 1024, 0, 0, 0},
    .box_dim = {8, 32, 1, 1, 1},
  },
  {
    .name = "base_interleave16_rank3_C12_W16_N1",
    .rank = 3,
    .global_dim = {12, 16, 1, 1, 1},
    .global_strides = {48, 768, 0, 0, 0},
    .box_dim = {12, 16, 1, 1, 1},
  },
  {
    .name = "interleave16_rank3_C12_W16_N1",
    .rank = 3,
    .interleave_mode = 1,
    .global_dim = {12, 16, 1, 1, 1},
    .global_strides = {48, 768, 0, 0, 0},
    .box_dim = {12, 16, 1, 1, 1},
  },
  {
    .name = "base_interleave16_rank3_C16_W16_N1",
    .rank = 3,
    .global_dim = {16, 16, 1, 1, 1},
    .global_strides = {64, 1024, 0, 0, 0},
    .box_dim = {16, 16, 1, 1, 1},
  },
  {
    .name = "interleave16_rank3_C16_W16_N1",
    .rank = 3,
    .interleave_mode = 1,
    .global_dim = {16, 16, 1, 1, 1},
    .global_strides = {64, 1024, 0, 0, 0},
    .box_dim = {16, 16, 1, 1, 1},
  },
  {
    .name = "base_interleave16_rank3_C24_W8_N1",
    .rank = 3,
    .global_dim = {24, 8, 1, 1, 1},
    .global_strides = {96, 768, 0, 0, 0},
    .box_dim = {24, 8, 1, 1, 1},
  },
  {
    .name = "interleave16_rank3_C24_W8_N1",
    .rank = 3,
    .interleave_mode = 1,
    .global_dim = {24, 8, 1, 1, 1},
    .global_strides = {96, 768, 0, 0, 0},
    .box_dim = {24, 8, 1, 1, 1},
  },
  {
    .name = "base_interleave32_rank3_C8_W32_N1",
    .rank = 3,
    .global_dim = {8, 32, 1, 1, 1},
    .global_strides = {32, 1024, 0, 0, 0},
    .box_dim = {8, 32, 1, 1, 1},
  },
  {
    .name = "interleave32_rank3_C8_W32_N1",
    .rank = 3,
    .interleave_mode = 2,
    .global_dim = {8, 32, 1, 1, 1},
    .global_strides = {32, 1024, 0, 0, 0},
    .box_dim = {8, 32, 1, 1, 1},
  },
  {
    .name = "base_interleave32_rank3_C16_W16_N1",
    .rank = 3,
    .global_dim = {16, 16, 1, 1, 1},
    .global_strides = {64, 1024, 0, 0, 0},
    .box_dim = {16, 16, 1, 1, 1},
  },
  {
    .name = "interleave32_rank3_C16_W16_N1",
    .rank = 3,
    .interleave_mode = 2,
    .global_dim = {16, 16, 1, 1, 1},
    .global_strides = {64, 1024, 0, 0, 0},
    .box_dim = {16, 16, 1, 1, 1},
  },
  {
    .name = "base_interleave32_rank3_C24_W8_N1",
    .rank = 3,
    .global_dim = {24, 8, 1, 1, 1},
    .global_strides = {96, 768, 0, 0, 0},
    .box_dim = {24, 8, 1, 1, 1},
  },
  {
    .name = "interleave32_rank3_C24_W8_N1",
    .rank = 3,
    .interleave_mode = 2,
    .global_dim = {24, 8, 1, 1, 1},
    .global_strides = {96, 768, 0, 0, 0},
    .box_dim = {24, 8, 1, 1, 1},
  },
  {
    .name = "base_interleave32_rank3_C32_W8_N1",
    .rank = 3,
    .global_dim = {32, 8, 1, 1, 1},
    .global_strides = {128, 1024, 0, 0, 0},
    .box_dim = {32, 8, 1, 1, 1},
  },
  {
    .name = "interleave32_rank3_C32_W8_N1",
    .rank = 3,
    .interleave_mode = 2,
    .global_dim = {32, 8, 1, 1, 1},
    .global_strides = {128, 1024, 0, 0, 0},
    .box_dim = {32, 8, 1, 1, 1},
  },
  {
    .name = "base_interleave32_rank3_C64_W4_N1",
    .rank = 3,
    .global_dim = {64, 4, 1, 1, 1},
    .global_strides = {256, 1024, 0, 0, 0},
    .box_dim = {64, 4, 1, 1, 1},
  },
  {
    .name = "interleave32_rank3_C64_W4_N1",
    .rank = 3,
    .interleave_mode = 2,
    .global_dim = {64, 4, 1, 1, 1},
    .global_strides = {256, 1024, 0, 0, 0},
    .box_dim = {64, 4, 1, 1, 1},
  },
  {
    .name = "base_interleave32_swizzle32_rank3_C8_W32_N1",
    .rank = 3,
    .global_dim = {8, 32, 1, 1, 1},
    .global_strides = {32, 1024, 0, 0, 0},
    .box_dim = {8, 32, 1, 1, 1},
  },
  {
    .name = "interleave32_swizzle32_rank3_C8_W32_N1",
    .rank = 3,
    .interleave_mode = 2,
    .swizzle_mode = 1,
    .global_dim = {8, 32, 1, 1, 1},
    .global_strides = {32, 1024, 0, 0, 0},
    .box_dim = {8, 32, 1, 1, 1},
  },
  {
    .name = "base_padded_rows_stride128_rank2_16x16",
    .rank = 2,
    .global_dim = {16, 16, 1, 1, 1},
    .global_strides = {64, 0, 0, 0, 0},
    .box_dim = {16, 16, 1, 1, 1},
  },
  {
    .name = "padded_rows_stride128_rank2_16x16",
    .rank = 2,
    .global_dim = {32, 16, 1, 1, 1},
    .global_strides = {128, 0, 0, 0, 0},
    .box_dim = {16, 16, 1, 1, 1},
  },
  {
    .name = "base_subbox_misaligned_rank2_8x8_at_1_1",
    .rank = 2,
    .global_dim = {16, 16, 1, 1, 1},
    .global_strides = {64, 0, 0, 0, 0},
    .box_dim = {8, 8, 1, 1, 1},
  },
  {
    .name = "subbox_misaligned_rank2_8x8_at_1_1",
    .rank = 2,
    .global_dim = {16, 16, 1, 1, 1},
    .global_strides = {64, 0, 0, 0, 0},
    .box_dim = {8, 8, 1, 1, 1},
    .coord = {1, 1, 0, 0, 0},
  },
  {
    .name = "base_oob_dim0_rank2_16x16_at_8_0",
    .rank = 2,
    .global_dim = {16, 16, 1, 1, 1},
    .global_strides = {64, 0, 0, 0, 0},
    .box_dim = {16, 16, 1, 1, 1},
  },
  {
    .name = "oob_dim0_rank2_16x16_at_8_0",
    .rank = 2,
    .global_dim = {16, 16, 1, 1, 1},
    .global_strides = {64, 0, 0, 0, 0},
    .box_dim = {16, 16, 1, 1, 1},
    .coord = {8, 0, 0, 0, 0},
  },
  {
    .name = "base_oob_dim1_rank2_16x16_at_0_8",
    .rank = 2,
    .global_dim = {16, 16, 1, 1, 1},
    .global_strides = {64, 0, 0, 0, 0},
    .box_dim = {16, 16, 1, 1, 1},
  },
  {
    .name = "oob_dim1_rank2_16x16_at_0_8",
    .rank = 2,
    .global_dim = {16, 16, 1, 1, 1},
    .global_strides = {64, 0, 0, 0, 0},
    .box_dim = {16, 16, 1, 1, 1},
    .coord = {0, 8, 0, 0, 0},
  },
  {
    .name = "base_rank3_plain_8x8x4",
    .rank = 2,
    .global_dim = {16, 16, 1, 1, 1},
    .global_strides = {64, 0, 0, 0, 0},
    .box_dim = {16, 16, 1, 1, 1},
  },
  {
    .name = "rank3_plain_8x8x4",
    .rank = 3,
    .global_dim = {8, 8, 4, 1, 1},
    .global_strides = {32, 256, 0, 0, 0},
    .box_dim = {8, 8, 4, 1, 1},
  },
  {
    .name = "base_rank4_plain_4x4x4x4",
    .rank = 2,
    .global_dim = {16, 16, 1, 1, 1},
    .global_strides = {64, 0, 0, 0, 0},
    .box_dim = {16, 16, 1, 1, 1},
  },
  {
    .name = "rank4_plain_4x4x4x4",
    .rank = 4,
    .global_dim = {4, 4, 4, 4, 1},
    .global_strides = {16, 64, 256, 0, 0},
    .box_dim = {4, 4, 4, 4, 1},
  },
  {
    .name = "base_rank5_plain_4x4x4x2x2",
    .rank = 2,
    .global_dim = {16, 16, 1, 1, 1},
    .global_strides = {64, 0, 0, 0, 0},
    .box_dim = {16, 16, 1, 1, 1},
  },
  {
    .name = "rank5_plain_4x4x4x2x2",
    .rank = 5,
    .global_dim = {4, 4, 4, 2, 2},
    .global_strides = {16, 64, 256, 512, 0},
    .box_dim = {4, 4, 4, 2, 2},
  },
};

static uint32_t pattern_word(unsigned idx)
{
  unsigned base = idx << 2;
  unsigned b0 = (base * 7u + 3u) & 0xffu;
  unsigned b1 = ((base + 1u) * 7u + 3u) & 0xffu;
  unsigned b2 = ((base + 2u) * 7u + 3u) & 0xffu;
  unsigned b3 = ((base + 3u) * 7u + 3u) & 0xffu;
  return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

static size_t swizzle_offset(size_t logical_off, size_t row, unsigned mode)
{
  if (mode == 0) return logical_off;
  unsigned chunk_bits = mode;
  size_t span = (size_t)16 << chunk_bits;
  size_t chunk_mask = ((size_t)1 << chunk_bits) - 1u;
  size_t low = logical_off & 0xfu;
  size_t chunk = (logical_off >> 4) & chunk_mask;
  size_t row_low = row & chunk_mask;
  return (logical_off & ~(span - 1u)) | ((chunk ^ row_low) << 4) | low;
}

static unsigned out_dim(const feature_case_t *c, unsigned dim)
{
  return c->box_dim[dim] ? c->box_dim[dim] : 1u;
}

static size_t logical_elems(const feature_case_t *c)
{
  size_t total = 1;
  for (unsigned d = 0; d < c->rank; d++) total *= out_dim(c, d);
  return total;
}

static size_t row_bytes(const feature_case_t *c)
{
  unsigned elems = out_dim(c, 0);
  return (size_t)elems * 4u;
}

static size_t swizzle_span_bytes(const feature_case_t *c)
{
  if (c->swizzle_mode == 0) return 128u;
  return (size_t)16u << c->swizzle_mode;
}

static size_t estimated_row_segments(const feature_case_t *c)
{
  size_t row = row_bytes(c);
  size_t span = swizzle_span_bytes(c);
  if (row == 0 || span == 0) return 0;
  return (row + span - 1u) / span;
}

static size_t tensor_physical_offset(const feature_case_t *c,
                                     const unsigned coord[MAX_RANK])
{
  const unsigned es = 4;
  if (c->interleave_mode == 0 || c->rank < 3) {
    size_t off = (size_t)coord[0] * es;
    for (unsigned d = 1; d < c->rank; d++) {
      off += (size_t)coord[d] * c->global_strides[d - 1];
    }
    return off;
  }

  size_t slice_bytes = c->interleave_mode == 1 ? 16u : 32u;
  size_t channels_per_slice = slice_bytes / es;
  size_t c_slice = coord[0] / channels_per_slice;
  size_t c_in_slice = coord[0] % channels_per_slice;
  size_t c_slice_stride =
    (size_t)c->global_strides[c->rank - 3] * c->global_dim[c->rank - 2];

  size_t off = c_in_slice * es + c_slice * c_slice_stride;
  for (unsigned d = 1; d < c->rank; d++) {
    off += (size_t)coord[d] * c->global_strides[d - 1];
  }
  return off;
}

static size_t tensor_shared_offset(const feature_case_t *c,
                                   const unsigned idx[MAX_RANK],
                                   const unsigned dims[MAX_RANK])
{
  const unsigned es = 4;
  size_t row = 0;
  size_t row_mul = 1;
  for (unsigned d = 1; d < c->rank; d++) {
    row += (size_t)idx[d] * row_mul;
    row_mul *= dims[d];
  }
  if (c->swizzle_mode != VENTUS_TMA_SWIZZLE_NONE) {
    /*
     * CUDA assigns each logical row a physical pitch equal to the selected
     * swizzle span. A short row leaves padding atoms in shared memory; it is
     * not tightly packed before the XOR permutation.
     */
    size_t span = (size_t)16u << c->swizzle_mode;
    size_t off = row * span + (size_t)idx[0] * es;
    return swizzle_offset(off, row, c->swizzle_mode);
  }

  size_t off = 0;
  size_t mul = es;
  for (unsigned d = 0; d < c->rank; d++) {
    off += (size_t)idx[d] * mul;
    mul *= dims[d];
  }
  return off;
}

static void build_desc(uint32_t desc[DESC_WORDS], const feature_case_t *c)
{
  memset(desc, 0, DESC_WORDS * sizeof(uint32_t));
  unsigned data_type = c->data_type ? c->data_type : 7u;
  desc[VENTUS_TMA_V2_WORD_MAGIC] = VENTUS_TMA_V2_MAGIC;
  desc[VENTUS_TMA_V2_WORD_CONTROL] =
      (data_type & 0x1fu) | ((c->rank & 0x7u) << 5) |
      ((c->interleave_mode & 0x3u) << 8) |
      ((c->swizzle_mode & 0x3u) << 10);
  for (unsigned i = 0; i < c->rank; i++) {
    desc[VENTUS_TMA_V2_WORD_GLOBAL_DIMS + i] =
        c->global_dim[i] ? c->global_dim[i] : 1u;
    desc[VENTUS_TMA_V2_WORD_BOX_DIMS + i] =
        c->box_dim[i] ? c->box_dim[i] : 1u;
    desc[VENTUS_TMA_V2_WORD_ELEMENT_STRIDES + i] = 1u;
    if (i + 1u < c->rank)
      desc[VENTUS_TMA_V2_WORD_GLOBAL_STRIDES + i * 2] =
          c->global_strides[i];
  }
}

static int fill_tensor_model(uint8_t *global_model, uint8_t *shared_model,
                             const feature_case_t *c, size_t *written_elems,
                             size_t *dst_span, size_t *shared_span,
                             size_t *unique_global_lines)
{
  unsigned dims[MAX_RANK] = {1, 1, 1, 1, 1};
  size_t total = 1;
  size_t max_global_written = 0;
  size_t max_shared_written = 0;
  size_t writes = 0;
  uint8_t touched_lines[FEATURE_GLOBAL_BYTES / 128u] = {0};
  memset(global_model, 0xcd, FEATURE_GLOBAL_BYTES);
  memset(shared_model, 0, FEATURE_SHARED_BYTES);

  for (unsigned d = 0; d < c->rank; d++) {
    dims[d] = out_dim(c, d);
    total *= dims[d];
  }

  for (size_t lin = 0; lin < total; lin++) {
    unsigned idx[MAX_RANK] = {0, 0, 0, 0, 0};
    unsigned gcoord[MAX_RANK] = {0, 0, 0, 0, 0};
    size_t rem = lin;
    int oob = 0;
    for (unsigned d = 0; d < c->rank; d++) {
      idx[d] = (unsigned)(rem % dims[d]);
      rem /= dims[d];
    }
    for (unsigned d = 0; d < c->rank; d++) {
      gcoord[d] = c->coord[d] + idx[d];
      if (gcoord[d] >= c->global_dim[d]) oob = 1;
    }
    if (oob) continue;

    size_t shared_off = tensor_shared_offset(c, idx, dims);
    size_t dst_off = tensor_physical_offset(c, gcoord);
    uint32_t word = pattern_word((unsigned)(shared_off / 4u));
    if (shared_off + sizeof(word) > FEATURE_SHARED_BYTES) {
      fprintf(stderr, "FAIL %s shared offset overflow: off=%zu bytes=%zu\n",
              c->name, shared_off, (size_t)FEATURE_SHARED_BYTES);
      return 1;
    }
    if (dst_off + sizeof(word) > FEATURE_GLOBAL_BYTES) {
      fprintf(stderr, "FAIL %s global offset overflow: off=%zu bytes=%u\n",
              c->name, dst_off, FEATURE_GLOBAL_BYTES);
      return 1;
    }
    memcpy(global_model + dst_off, &word, sizeof(word));
    memcpy(shared_model + shared_off, &word, sizeof(word));
    writes++;
    touched_lines[dst_off / 128u] = 1u;
    if (dst_off + sizeof(word) > max_global_written) {
      max_global_written = dst_off + sizeof(word);
    }
    if (shared_off + sizeof(word) > max_shared_written) {
      max_shared_written = shared_off + sizeof(word);
    }
  }
  *written_elems = writes;
  *dst_span = max_global_written;
  *shared_span = max_shared_written;
  *unique_global_lines = 0;
  for (size_t line = 0; line < sizeof(touched_lines); ++line) {
    *unique_global_lines += touched_lines[line] != 0;
  }
  return 0;
}

static int first_mismatch(const uint8_t *a, const uint8_t *b, size_t n)
{
  for (size_t i = 0; i < n; i++) {
    if (a[i] != b[i]) return (int)i;
  }
  return -1;
}

static int event_duration_ns(cl_event event, uint64_t *duration_ns)
{
  cl_ulong start = 0, end = 0;
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

static int run_setup_kernel(cl_command_queue queue, cl_kernel setup_kernel,
                            cl_mem desc_buf, cl_mem global_buf)
{
  cl_int err = CL_SUCCESS;
  size_t global = 1;
  err = clSetKernelArg(setup_kernel, 0, sizeof(desc_buf), &desc_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(setup.desc)");
  err = clSetKernelArg(setup_kernel, 1, sizeof(global_buf), &global_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(setup.global)");
  err = clEnqueueNDRangeKernel(queue, setup_kernel, 1, NULL, &global, NULL,
                               0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel(setup)");
  return 0;
FINISH:
  return 1;
}

static cl_int build_program_with_iterations(cl_context context,
                                            cl_device_id device,
                                            unsigned iterations,
                                            cl_program *program)
{
  const char *source_path = getenv("VENTUS_DMA_TMA_TENSOR_FEATURE_SOURCE");
  if (!source_path || !source_path[0]) {
    source_path = "dma_tma_tensor_feature_perf_test.cl";
  }
  size_t source_size = 0;
  char *source = ventus_read_text_file(source_path, &source_size);
  if (!source) return CL_INVALID_PROGRAM;

  cl_int err = CL_SUCCESS;
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
  char options[PATH_MAX + 128];
  snprintf(options, sizeof(options),
           "-I%s/../common -DFEATURE_ITERATIONS=%uu",
           source_dir, iterations);
  err = clBuildProgram(prog, 1, &device, options, NULL, NULL);
  if (err != CL_SUCCESS) {
    ventus_print_build_log(prog, device);
    clReleaseProgram(prog);
    return err;
  }

  *program = prog;
  return CL_SUCCESS;
}

static const feature_case_t *find_feature(const char *name)
{
  for (size_t i = 0; i < sizeof(feature_cases) / sizeof(feature_cases[0]); i++) {
    if (strcmp(feature_cases[i].name, name) == 0) return &feature_cases[i];
  }
  for (size_t i = 0; i < sizeof(feature_stress_cases) / sizeof(feature_stress_cases[0]); i++) {
    if (strcmp(feature_stress_cases[i].name, name) == 0) return &feature_stress_cases[i];
  }
  return NULL;
}

static int run_list_has_name(const feature_case_t *const *cases, size_t count,
                             const char *name)
{
  for (size_t i = 0; i < count; i++) {
    if (strcmp(cases[i]->name, name) == 0) return 1;
  }
  return 0;
}

static int v2_case_legal(const feature_case_t *c, char *reason,
                         size_t reason_size)
{
  unsigned dtype = c->data_type ? c->data_type : VENTUS_TMA_DTYPE_FP32;
  if (c->rank < 1u || c->rank > VENTUS_TMA_V2_RANK_MAX) {
    snprintf(reason, reason_size, "rank-out-of-range");
    return 0;
  }
  if (!((dtype <= VENTUS_TMA_DTYPE_BF16) ||
        dtype == VENTUS_TMA_DTYPE_B4X16)) {
    snprintf(reason, reason_size, "dtype-unsupported");
    return 0;
  }
  if (c->interleave_mode > VENTUS_TMA_INTERLEAVE_32B ||
      c->swizzle_mode > VENTUS_TMA_SWIZZLE_128B) {
    snprintf(reason, reason_size, "layout-code-unsupported");
    return 0;
  }
  if (c->interleave_mode != VENTUS_TMA_INTERLEAVE_NONE && c->rank < 3u) {
    snprintf(reason, reason_size, "interleave-requires-rank3");
    return 0;
  }
  if (c->interleave_mode == VENTUS_TMA_INTERLEAVE_32B &&
      c->swizzle_mode != VENTUS_TMA_SWIZZLE_32B) {
    snprintf(reason, reason_size, "interleave32-requires-swizzle32");
    return 0;
  }
  for (unsigned d = 0; d < c->rank; d++) {
    if (d > 0u) {
      unsigned align = c->interleave_mode == VENTUS_TMA_INTERLEAVE_32B ? 32u : 16u;
      unsigned global_stride = c->global_strides[d - 1u];
      if (global_stride == 0u || (global_stride & (align - 1u)) != 0u) {
        snprintf(reason, reason_size, "global-stride-dim%u", d);
        return 0;
      }
    }
  }
  size_t bytes = row_bytes(c);
  if (bytes == 0u || (bytes & (VENTUS_TMA_V2_ATOM_BYTES - 1u)) != 0u) {
    snprintf(reason, reason_size, "row-bytes-not-16B");
    return 0;
  }
  if (c->swizzle_mode != VENTUS_TMA_SWIZZLE_NONE &&
      bytes > ((size_t)16u << c->swizzle_mode)) {
    snprintf(reason, reason_size, "row-exceeds-swizzle-span");
    return 0;
  }
  reason[0] = '\0';
  return 1;
}

static int build_run_list(run_mode_t mode, const feature_case_t *single_feature,
                          const feature_case_t ***out_cases,
                          size_t *out_count)
{
  const size_t sweep_count = sizeof(feature_cases) / sizeof(feature_cases[0]);
  const size_t stress_count =
    sizeof(feature_stress_cases) / sizeof(feature_stress_cases[0]);
  size_t capacity = mode == RUN_STRESS ? sweep_count + stress_count : sweep_count;
  if (mode == RUN_SINGLE || mode == RUN_SINGLE_DIR) capacity = 1;

  const feature_case_t **cases =
    (const feature_case_t **)calloc(capacity, sizeof(*cases));
  if (!cases) return 1;

  size_t count = 0;
  if (mode == RUN_SINGLE || mode == RUN_SINGLE_DIR) {
    char reason[96];
    if (!v2_case_legal(single_feature, reason, sizeof(reason))) {
      fprintf(stderr, "FEATURE_SKIP name=%s reason=%s\n",
              single_feature->name, reason);
      free(cases);
      return 1;
    }
    cases[count++] = single_feature;
  } else {
    for (size_t i = 0; i < sweep_count; i++) {
      char reason[96];
      if (v2_case_legal(&feature_cases[i], reason, sizeof(reason))) {
        cases[count++] = &feature_cases[i];
      } else {
        printf("FEATURE_SKIP name=%s reason=%s\n", feature_cases[i].name,
               reason);
      }
    }
    if (mode == RUN_STRESS) {
      for (size_t i = 0; i < stress_count; i++) {
        char reason[96];
        if (run_list_has_name(cases, count, feature_stress_cases[i].name)) {
          continue;
        }
        if (v2_case_legal(&feature_stress_cases[i], reason, sizeof(reason))) {
          cases[count++] = &feature_stress_cases[i];
        } else {
          printf("FEATURE_SKIP name=%s reason=%s\n",
                 feature_stress_cases[i].name, reason);
        }
      }
    }
  }

  *out_cases = cases;
  *out_count = count;
  return 0;
}

static int starts_with(const char *text, const char *prefix)
{
  return strncmp(text, prefix, strlen(prefix)) == 0;
}

static void print_host_pair_summary(const feature_result_t *results, size_t count)
{
  printf("FEATURE_PAIR_SUMMARY\n");
  printf("| direction | feature | base | row bytes | feature/base unique lines | feature/base cycles/iter | raw cycle ratio | feature/base cycles/line | normalized cycle ratio | feature/base cache req/line | feature/base logical B/cycle | host ratio |\n");
  printf("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
  for (size_t i = 0; i < count; i++) {
    const char *name = results[i].feature->name;
    if (starts_with(name, "base_")) continue;

    char base_name[160];
    snprintf(base_name, sizeof(base_name), "base_%s", name);
    const feature_result_t *base = NULL;
    for (size_t j = 0; j < count; j++) {
      if (results[j].direction == results[i].direction &&
          strcmp(results[j].feature->name, base_name) == 0) {
        base = &results[j];
        break;
      }
    }
    if (!base || base->ns == 0) continue;

    double feature_ns_iter = (double)results[i].ns / (double)results[i].iterations;
    double base_ns_iter = (double)base->ns / (double)base->iterations;
    double host_ratio = feature_ns_iter / base_ns_iter;
    double feature_cycle_iter = (double)results[i].cycles / (double)results[i].iterations;
    double base_cycle_iter = (double)base->cycles / (double)base->iterations;
    double cycle_ratio = base_cycle_iter > 0.0 ? feature_cycle_iter / base_cycle_iter : 0.0;
    double feature_cycles_line = results[i].unique_global_lines ?
      feature_cycle_iter / (double)results[i].unique_global_lines : 0.0;
    double base_cycles_line = base->unique_global_lines ?
      base_cycle_iter / (double)base->unique_global_lines : 0.0;
    double normalized_cycle_ratio = base_cycles_line > 0.0 ?
      feature_cycles_line / base_cycles_line : 0.0;
    double feature_req_line = results[i].cache_requests_valid &&
        results[i].unique_global_lines ?
      ((double)results[i].cache_requests / results[i].iterations) /
        results[i].unique_global_lines : 0.0;
    double base_req_line = base->cache_requests_valid && base->unique_global_lines ?
      ((double)base->cache_requests / base->iterations) /
        base->unique_global_lines : 0.0;
    double feature_logical_bw = feature_cycle_iter > 0.0 ?
      ((double)results[i].logical_elems * sizeof(uint32_t)) /
        feature_cycle_iter : 0.0;
    double base_logical_bw = base_cycle_iter > 0.0 ?
      ((double)base->logical_elems * sizeof(uint32_t)) / base_cycle_iter : 0.0;
    size_t row = row_bytes(results[i].feature);
    size_t span = swizzle_span_bytes(results[i].feature);
    size_t segments = estimated_row_segments(results[i].feature);
    printf("FEATURE_PAIR direction=%s feature=%s base=%s cycle_ratio=%.3fx normalized_cycle_ratio=%.3fx feature_cycles_iter=%.2f base_cycles_iter=%.2f feature_cycles_per_line=%.4f base_cycles_per_line=%.4f feature_unique_lines=%zu base_unique_lines=%zu feature_cache_req_per_line=%.4f base_cache_req_per_line=%.4f feature_logical_bytes_per_cycle=%.6f base_logical_bytes_per_cycle=%.6f host_ratio=%.3fx feature_ns_iter=%.2f base_ns_iter=%.2f row_bytes=%zu swizzle_span=%zu est_row_segments=%zu\n",
           direction_name(results[i].direction), name, base_name, cycle_ratio,
           normalized_cycle_ratio, feature_cycle_iter, base_cycle_iter,
           feature_cycles_line, base_cycles_line,
           results[i].unique_global_lines, base->unique_global_lines,
           feature_req_line, base_req_line, feature_logical_bw,
           base_logical_bw, host_ratio, feature_ns_iter, base_ns_iter,
           row, span, segments);
    printf("| %s | %s | %s | %zu | %zu/%zu | %.2f/%.2f | %.3fx | %.4f/%.4f | %.3fx | %.4f/%.4f | %.6f/%.6f | %.3fx |\n",
           direction_name(results[i].direction), name, base_name, row,
           results[i].unique_global_lines, base->unique_global_lines,
           feature_cycle_iter, base_cycle_iter, cycle_ratio,
           feature_cycles_line, base_cycles_line, normalized_cycle_ratio,
           feature_req_line, base_req_line, feature_logical_bw,
           base_logical_bw, host_ratio);
  }
}

static int parse_u64_field(const char *line, const char *key, uint64_t *value)
{
  const char *pos = strstr(line, key);
  char *end = NULL;
  if (!pos) return 1;
  pos += strlen(key);
  unsigned long long parsed = strtoull(pos, &end, 0);
  if (end == pos) return 1;
  *value = (uint64_t)parsed;
  return 0;
}

static int parse_size_field(const char *line, const char *key, size_t *value)
{
  uint64_t parsed = 0;
  if (parse_u64_field(line, key, &parsed) != 0) return 1;
  *value = (size_t)parsed;
  return 0;
}

static int parse_uint_field(const char *line, const char *key, unsigned *value)
{
  uint64_t parsed = 0;
  if (parse_u64_field(line, key, &parsed) != 0) return 1;
  *value = (unsigned)parsed;
  return 0;
}

static void print_result_row(const feature_result_t *result)
{
  double ns_per_elem = result->logical_elems ?
    (double)result->ns /
    ((double)result->iterations * (double)result->logical_elems) : 0.0;
  double cycles_per_iter = (double)result->cycles / (double)result->iterations;
  double logical_bytes_per_cycle = cycles_per_iter > 0.0 ?
    ((double)result->logical_elems * sizeof(uint32_t)) / cycles_per_iter : 0.0;
  double cycles_per_unique_line = result->unique_global_lines ?
    cycles_per_iter / (double)result->unique_global_lines : 0.0;
  double cache_requests_per_iter = result->cache_requests_valid ?
    (double)result->cache_requests / (double)result->iterations : 0.0;
  double requests_per_unique_line = result->cache_requests_valid &&
      result->unique_global_lines ?
    cache_requests_per_iter / (double)result->unique_global_lines : 0.0;
  printf("| %s | %s | %u | %u | %u | %zu | %zu | %zu | %zu | %zu | %zu | %zu | %zu | %.2f | %.4f | %.6f | %.2f | %.4f | %.2f | %.2f |\n",
         direction_name(result->direction), result->feature->name,
         result->feature->rank, result->feature->interleave_mode,
         result->feature->swizzle_mode, row_bytes(result->feature),
         swizzle_span_bytes(result->feature),
         estimated_row_segments(result->feature), result->logical_elems,
         result->written_elems, result->dst_span, result->shared_span,
         result->unique_global_lines, cycles_per_iter, cycles_per_unique_line,
         logical_bytes_per_cycle, cache_requests_per_iter,
         requests_per_unique_line,
         (double)result->ns / (double)result->iterations, ns_per_elem);
}

static int parse_child_result_line(const char *line,
                                   const feature_case_t *feature,
                                   feature_direction_t direction,
                                   feature_result_t *result)
{
  if (strncmp(line, "FEATURE_RESULT ", strlen("FEATURE_RESULT ")) != 0) {
    return 1;
  }

  result->feature = feature;
  result->direction = direction;
  if (parse_uint_field(line, "iterations=", &result->iterations) != 0 ||
      parse_size_field(line, "logical_elems=", &result->logical_elems) != 0 ||
      parse_size_field(line, "written_elems=", &result->written_elems) != 0 ||
      parse_size_field(line, "dst_span=", &result->dst_span) != 0 ||
      parse_size_field(line, "shared_span=", &result->shared_span) != 0 ||
      parse_size_field(line, "unique_global_lines=",
                       &result->unique_global_lines) != 0 ||
      parse_u64_field(line, "cycles=", &result->cycles) != 0 ||
      parse_u64_field(line, " ns=", &result->ns) != 0) {
    return 1;
  }
  return 0;
}

static int run_child_feature(const char *self, const feature_case_t *feature,
                             feature_direction_t direction,
                             unsigned iterations, feature_result_t *result)
{
  char command[2048];
  int n = snprintf(command, sizeof(command),
                   "\"%s\" single-dir %s %s %u 2>&1",
                   self, direction_name(direction), feature->name, iterations);
  if (n < 0 || (size_t)n >= sizeof(command)) {
    fprintf(stderr, "child command too long for '%s'\n", feature->name);
    return 1;
  }

  FILE *pipe = popen(command, "r");
  if (!pipe) {
    perror("popen");
    return 1;
  }

  char line[4096];
  int saw_result = 0;
  int saw_g2s_cache_requests = 0;
  int saw_s2g_cache_requests = 0;
  uint64_t g2s_cache_requests = 0;
  uint64_t s2g_cache_requests = 0;
  while (fgets(line, sizeof(line), pipe)) {
    fputs(line, stdout);
    if (strstr(line,
        "[TESTCASE TOTAL] [TMA PERF] G2S cache responses") &&
        parse_u64_field(line, ":", &g2s_cache_requests) == 0) {
      saw_g2s_cache_requests = 1;
    }
    if (strstr(line,
        "[TESTCASE TOTAL] [TMA PERF] S2G cache responses") &&
        parse_u64_field(line, ":", &s2g_cache_requests) == 0) {
      saw_s2g_cache_requests = 1;
    }
    if (parse_child_result_line(line, feature, direction, result) == 0) {
      saw_result = 1;
    }
  }

  int status = pclose(pipe);
  if (!saw_result) {
    fprintf(stderr, "FAIL child %s %s produced no FEATURE_RESULT\n",
            direction_name(direction), feature->name);
    return 1;
  }
  if (status == -1) {
    perror("pclose");
    return 1;
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    fprintf(stderr, "FAIL child %s %s exited with status %d\n",
            direction_name(direction), feature->name, status);
    return 1;
  }
  if (direction == FEATURE_DIR_G2S && saw_g2s_cache_requests) {
    result->cache_requests = g2s_cache_requests;
    result->cache_requests_valid = 1;
  } else if (direction == FEATURE_DIR_S2G && saw_s2g_cache_requests) {
    result->cache_requests = s2g_cache_requests;
    result->cache_requests_valid = 1;
  }
  return 0;
}

static int run_list_child_processes(const char *self,
                                    const feature_case_t *const *run_cases,
                                    size_t run_count, unsigned iterations)
{
  size_t result_count = run_count * FEATURE_DIR_COUNT;
  feature_result_t *results =
    (feature_result_t *)calloc(result_count, sizeof(*results));
  if (!results) {
    fprintf(stderr, "host result alloc failed\n");
    return 1;
  }

  printf("FEATURE_SWEEP iterations=%u global_bytes=%u shared_bytes=%zu wg_size=%u cases=%zu directions=%u process_per_case=1\n",
         iterations, FEATURE_GLOBAL_BYTES, (size_t)FEATURE_SHARED_BYTES,
         WG_SIZE, run_count, FEATURE_DIR_COUNT);
  printf("| direction | feature | rank | interleave | swizzle | row bytes | swizzle span | est row segments | logical elems | written elems | physical span | shared span | unique 128B lines | cycles/iter | cycles/line | logical B/cycle | cache req/iter | cache req/line | ns/iter | ns/elem |\n");
  printf("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");

  int exit_code = 0;
  for (size_t i = 0; i < run_count && exit_code == 0; i++) {
    for (unsigned dir = 0; dir < FEATURE_DIR_COUNT; dir++) {
      size_t idx = i * FEATURE_DIR_COUNT + dir;
      if (run_child_feature(self, run_cases[i], (feature_direction_t)dir,
                            iterations, &results[idx]) != 0) {
        exit_code = 1;
        break;
      }
      print_result_row(&results[idx]);
      fflush(stdout);
    }
  }

  if (exit_code == 0) {
    print_host_pair_summary(results, result_count);
  }
  free(results);
  return exit_code;
}

static int run_feature(cl_context context, cl_command_queue queue,
                       cl_kernel setup_kernel, cl_kernel kernel,
                       const feature_case_t *feature,
                       feature_direction_t direction, unsigned iterations,
                       feature_result_t *result)
{
  cl_int err = CL_SUCCESS;
  cl_mem desc_buf = NULL;
  cl_mem coords_buf = NULL;
  cl_mem global_buf = NULL;
  cl_mem readback_buf = NULL;
  cl_mem metrics_buf = NULL;
  cl_event event = NULL;
  uint32_t desc[DESC_WORDS];
  uint32_t metrics[2] = {0, 0};
  uint32_t coords[COORD_WORDS] = {0};
  uint8_t *global_model = NULL;
  uint8_t *global_got = NULL;
  uint8_t *shared_model = NULL;
  uint8_t *shared_got = NULL;
  uint8_t fill = 0xcd;
  uint64_t ns = 0;
  size_t written = 0;
  size_t span = 0;
  size_t shared_span = 0;
  size_t unique_global_lines = 0;
  int exit_code = 1;
  size_t global = WG_SIZE, local = WG_SIZE;
  unsigned direction_arg = (unsigned)direction;
  uint32_t transaction_bytes = (uint32_t)(logical_elems(feature) * sizeof(uint32_t));
  unsigned effective_iterations =
    direction == FEATURE_DIR_G2S ? 1u : iterations;

  global_model = (uint8_t *)malloc(FEATURE_GLOBAL_BYTES);
  global_got = (uint8_t *)malloc(FEATURE_GLOBAL_BYTES);
  shared_model = (uint8_t *)malloc(FEATURE_SHARED_BYTES);
  shared_got = (uint8_t *)malloc(FEATURE_SHARED_BYTES);
  if (!global_model || !global_got || !shared_model || !shared_got) {
    fprintf(stderr, "FAIL %s %s host alloc\n", direction_name(direction),
            feature->name);
    goto FINISH;
  }

  build_desc(desc, feature);
  for (unsigned i = 0; i < MAX_RANK; i++) coords[i] = feature->coord[i];
  if (fill_tensor_model(global_model, shared_model, feature, &written, &span,
                        &shared_span, &unique_global_lines) != 0) {
    goto FINISH;
  }
  memset(global_got, 0xcd, FEATURE_GLOBAL_BYTES);
  memset(shared_got, 0, FEATURE_SHARED_BYTES);

  desc_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            sizeof(desc), desc, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(desc)");
  coords_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(coords), coords, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(coords)");
  global_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, FEATURE_GLOBAL_BYTES,
                              NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(global)");
  readback_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, FEATURE_SHARED_BYTES,
                                NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(readback)");
  metrics_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                               sizeof(metrics), metrics, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(metrics)");
  if (direction == FEATURE_DIR_G2S) {
    err = clEnqueueWriteBuffer(queue, global_buf, CL_TRUE, 0,
                               FEATURE_GLOBAL_BYTES, global_model, 0, NULL,
                               NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueWriteBuffer(global)");
  } else {
    err = clEnqueueFillBuffer(queue, global_buf, &fill, sizeof(fill), 0,
                              FEATURE_GLOBAL_BYTES, 0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueFillBuffer(global)");
  }
  uint32_t zero = 0;
  err = clEnqueueFillBuffer(queue, readback_buf, &zero, sizeof(zero), 0,
                            FEATURE_SHARED_BYTES, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueFillBuffer(readback)");
  if (run_setup_kernel(queue, setup_kernel, desc_buf, global_buf) != 0) {
    goto FINISH;
  }

  err  = clSetKernelArg(kernel, 0, sizeof(desc_buf), &desc_buf);
  err |= clSetKernelArg(kernel, 1, sizeof(coords_buf), &coords_buf);
  err |= clSetKernelArg(kernel, 2, sizeof(global_buf), &global_buf);
  err |= clSetKernelArg(kernel, 3, sizeof(readback_buf), &readback_buf);
  err |= clSetKernelArg(kernel, 4, sizeof(metrics_buf), &metrics_buf);
  err |= clSetKernelArg(kernel, 5, sizeof(direction_arg), &direction_arg);
  err |= clSetKernelArg(kernel, 6, sizeof(transaction_bytes), &transaction_bytes);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg");

  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                               0, NULL, &event);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel");
  err = clWaitForEvents(1, &event);
  CHECK_OPENCL_ERROR_IN("clWaitForEvents");
  if (event_duration_ns(event, &ns) != 0) {
    fprintf(stderr, "FAIL %s event profiling\n", feature->name);
    goto FINISH;
  }
  err = clEnqueueReadBuffer(queue, metrics_buf, CL_TRUE, 0,
                            sizeof(metrics), metrics, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(metrics)");
  if (metrics[0] != VENTUS_TMA_STATUS_OK) {
    fprintf(stderr, "FAIL %s %s TMA status=%u\n",
            direction_name(direction), feature->name, metrics[0]);
    goto FINISH;
  }
  if (direction == FEATURE_DIR_G2S) {
    err = clEnqueueReadBuffer(queue, readback_buf, CL_TRUE, 0,
                              FEATURE_SHARED_BYTES, shared_got, 0, NULL,
                              NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(readback)");
    int mismatch = first_mismatch(shared_got, shared_model,
                                  FEATURE_SHARED_BYTES);
    if (mismatch >= 0) {
      fprintf(stderr, "FAIL %s %s shared[%d]: got=%02x exp=%02x\n",
              direction_name(direction), feature->name, mismatch,
              shared_got[mismatch], shared_model[mismatch]);
      goto FINISH;
    }
  } else {
    err = clEnqueueReadBuffer(queue, global_buf, CL_TRUE, 0,
                              FEATURE_GLOBAL_BYTES, global_got, 0, NULL,
                              NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(global)");
    int mismatch = first_mismatch(global_got, global_model,
                                  FEATURE_GLOBAL_BYTES);
    if (mismatch >= 0) {
      fprintf(stderr, "FAIL %s %s global[%d]: got=%02x exp=%02x\n",
              direction_name(direction), feature->name, mismatch,
              global_got[mismatch], global_model[mismatch]);
      goto FINISH;
    }
  }

  result->feature = feature;
  result->direction = direction;
  result->iterations = effective_iterations;
  result->logical_elems = logical_elems(feature);
  result->written_elems = written;
  result->dst_span = span;
  result->shared_span = shared_span;
  result->unique_global_lines = unique_global_lines;
  result->ns = ns;
  result->cycles = metrics[1];
  if (result->cycles == 0) {
    fprintf(stderr, "FAIL %s %s movement cycle counter returned zero\n",
            direction_name(direction), feature->name);
    goto FINISH;
  }
  double ns_per_elem = result->logical_elems ?
    (double)ns / ((double)result->iterations * (double)result->logical_elems) : 0.0;
  double cycles_per_iter = (double)result->cycles / (double)result->iterations;
  double logical_bytes_per_cycle = cycles_per_iter > 0.0 ?
    ((double)result->logical_elems * sizeof(uint32_t)) / cycles_per_iter : 0.0;
  double cycles_per_unique_line = unique_global_lines ?
    cycles_per_iter / (double)unique_global_lines : 0.0;
  printf("FEATURE_RESULT direction=%s name=%s rank=%u interleave=%u swizzle=%u row_bytes=%zu swizzle_span=%zu est_row_segments=%zu logical_elems=%zu logical_bytes=%zu written_elems=%zu dst_span=%zu physical_span=%zu shared_span=%zu unique_global_lines=%zu iterations=%u cycles=%" PRIu64 " cycles_per_iter=%.2f cycles_per_unique_line=%.4f logical_bytes_per_cycle=%.6f ns=%" PRIu64 " ns_per_iter=%.2f ns_per_elem=%.2f\n",
         direction_name(direction), feature->name, feature->rank,
         feature->interleave_mode, feature->swizzle_mode, row_bytes(feature),
         swizzle_span_bytes(feature),
         estimated_row_segments(feature), result->logical_elems,
         result->logical_elems * sizeof(uint32_t), result->written_elems,
         result->dst_span, result->dst_span, result->shared_span,
         result->unique_global_lines,
         result->iterations, result->cycles,
         cycles_per_iter, cycles_per_unique_line, logical_bytes_per_cycle,
         ns, (double)ns / (double)result->iterations, ns_per_elem);
  exit_code = 0;

FINISH:
  if (event) clReleaseEvent(event);
  if (metrics_buf) clReleaseMemObject(metrics_buf);
  if (readback_buf) clReleaseMemObject(readback_buf);
  if (global_buf) clReleaseMemObject(global_buf);
  if (coords_buf) clReleaseMemObject(coords_buf);
  /*
   * Keep descriptor buffers alive for the process lifetime.  Releasing and
   * reallocating them can recycle the same device virtual line and accidentally
   * measure descriptor-cache hot-update behavior instead of feature cost.
   */
  free(shared_got);
  free(shared_model);
  free(global_got);
  free(global_model);
  return exit_code;
}

static int parse_iterations(const char *text, unsigned *iterations)
{
  char *end = NULL;
  unsigned long value = strtoul(text, &end, 0);
  if (!text[0] || *end || value == 0 || value > MAX_ITERATIONS) {
    fprintf(stderr, "invalid iterations '%s', expected 1..%u\n",
            text, MAX_ITERATIONS);
    return 1;
  }
  *iterations = (unsigned)value;
  return 0;
}

static void print_usage(const char *prog)
{
  fprintf(stderr,
          "Usage:\n"
          "  %s [sweep [iterations]]\n"
          "  %s stress [iterations]\n"
          "  %s single <feature-name> [iterations]\n"
          "  %s single-dir <g2s|s2g> <feature-name> [iterations]\n"
          "Features:\n",
          prog, prog, prog, prog);
  for (size_t i = 0; i < sizeof(feature_cases) / sizeof(feature_cases[0]); i++) {
    fprintf(stderr, "  %s\n", feature_cases[i].name);
  }
  fprintf(stderr, "Stress features:\n");
  for (size_t i = 0; i < sizeof(feature_stress_cases) / sizeof(feature_stress_cases[0]); i++) {
    fprintf(stderr, "  %s\n", feature_stress_cases[i].name);
  }
}

int main(int argc, char **argv)
{
  cl_int err = CL_SUCCESS;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel setup_kernel = NULL;
  cl_kernel kernel = NULL;
  unsigned iterations = DEFAULT_ITERATIONS;
  run_mode_t mode = RUN_SWEEP;
  const feature_case_t *single_feature = NULL;
  feature_direction_t single_direction = FEATURE_DIR_G2S;
  const feature_case_t **run_cases = NULL;
  size_t run_count = 0;
  feature_result_t *results = NULL;
  size_t result_count = 0;
  int exit_code = 1;

  if (argc >= 2 && strcmp(argv[1], "--list") == 0) {
    print_usage(argv[0]);
    return 0;
  }
  if (argc >= 2 && strcmp(argv[1], "single") == 0) {
    if (argc < 3 || argc > 4) {
      print_usage(argv[0]);
      return 2;
    }
    mode = RUN_SINGLE;
    single_feature = find_feature(argv[2]);
    if (!single_feature) {
      fprintf(stderr, "unknown feature '%s'\n", argv[2]);
      print_usage(argv[0]);
      return 2;
    }
    if (argc == 4 && parse_iterations(argv[3], &iterations) != 0) return 2;
  } else if (argc >= 2 && strcmp(argv[1], "single-dir") == 0) {
    if (argc < 4 || argc > 5) {
      print_usage(argv[0]);
      return 2;
    }
    mode = RUN_SINGLE_DIR;
    if (parse_direction(argv[2], &single_direction) != 0) {
      fprintf(stderr, "unknown direction '%s', expected g2s or s2g\n", argv[2]);
      return 2;
    }
    single_feature = find_feature(argv[3]);
    if (!single_feature) {
      fprintf(stderr, "unknown feature '%s'\n", argv[3]);
      print_usage(argv[0]);
      return 2;
    }
    if (argc == 5 && parse_iterations(argv[4], &iterations) != 0) return 2;
  } else if (argc >= 2 && strcmp(argv[1], "sweep") == 0) {
    if (argc > 3) {
      print_usage(argv[0]);
      return 2;
    }
    if (argc == 3 && parse_iterations(argv[2], &iterations) != 0) return 2;
  } else if (argc >= 2 && strcmp(argv[1], "stress") == 0) {
    if (argc > 3) {
      print_usage(argv[0]);
      return 2;
    }
    mode = RUN_STRESS;
    if (argc == 3 && parse_iterations(argv[2], &iterations) != 0) return 2;
  } else if (argc > 1) {
    print_usage(argv[0]);
    return 2;
  }

  if (build_run_list(mode, single_feature, &run_cases, &run_count) != 0) {
    fprintf(stderr, "host run-list alloc failed\n");
    return 1;
  }

  if (mode == RUN_SWEEP || mode == RUN_STRESS) {
    int rc = run_list_child_processes(argv[0], run_cases, run_count,
                                      iterations);
    free(run_cases);
    return rc;
  }

  err = ventus_get_default_device(&context, &device, &queue, NULL);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device");
  clReleaseCommandQueue(queue);
  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
  CHECK_OPENCL_ERROR_IN("clCreateCommandQueue(profiled)");
  err = build_program_with_iterations(context, device, iterations, &program);
  CHECK_OPENCL_ERROR_IN("build_program_with_iterations");
  setup_kernel = clCreateKernel(program, "dma_tma_tensor_feature_setup_desc_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(setup)");
  kernel = clCreateKernel(program, "dma_tma_tensor_feature_perf_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(feature)");

  if (mode != RUN_SINGLE_DIR) {
    printf("FEATURE_SWEEP iterations=%u global_bytes=%u shared_bytes=%zu wg_size=%u cases=%zu directions=%u\n",
           iterations, FEATURE_GLOBAL_BYTES, (size_t)FEATURE_SHARED_BYTES,
           WG_SIZE, run_count, FEATURE_DIR_COUNT);
    printf("| direction | feature | rank | interleave | swizzle | row bytes | swizzle span | est row segments | logical elems | written elems | physical span | shared span | unique 128B lines | cycles/iter | cycles/line | logical B/cycle | cache req/iter | cache req/line | ns/iter | ns/elem |\n");
    printf("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
  }

  result_count = run_count * FEATURE_DIR_COUNT;
  results = (feature_result_t *)calloc(result_count, sizeof(feature_result_t));
  if (!results) {
    fprintf(stderr, "host result alloc failed\n");
    goto FINISH;
  }

  for (size_t i = 0; i < run_count; i++) {
    const feature_case_t *feature = run_cases[i];
    unsigned dir_begin = mode == RUN_SINGLE_DIR ? (unsigned)single_direction : 0;
    unsigned dir_end = mode == RUN_SINGLE_DIR ? dir_begin + 1 : FEATURE_DIR_COUNT;
    for (unsigned dir = dir_begin; dir < dir_end; dir++) {
      size_t idx = mode == RUN_SINGLE_DIR ? 0 : i * FEATURE_DIR_COUNT + dir;
      if (run_feature(context, queue, setup_kernel, kernel, feature,
                      (feature_direction_t)dir, iterations,
                      &results[idx]) != 0) {
        goto FINISH;
      }
      if (mode != RUN_SINGLE_DIR) {
        print_result_row(&results[idx]);
      }
    }
  }
  if (mode != RUN_SINGLE && mode != RUN_SINGLE_DIR) {
    print_host_pair_summary(results, result_count);
  }

  exit_code = 0;

FINISH:
  free(results);
  free(run_cases);
  if (kernel) clReleaseKernel(kernel);
  if (setup_kernel) clReleaseKernel(setup_kernel);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
  return exit_code;
}
