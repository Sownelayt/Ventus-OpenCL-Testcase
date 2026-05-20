/*
 * TMA (tensor DMA) matrix test.
 *
 * Drives tma_matrix_kernel through a table of (dataType, rank, global shape,
 * box offset, box shape, elementStrides) configurations. For each config:
 *   1. Build a packed source buffer with a known byte-pattern.
 *   2. Construct a 128B tensor-map descriptor plus a 32-word coords block.
 *   3. Launch the kernel, read back dst.
 *   4. On host, compute the expected bytes by replaying the same indexing
 *      formula against the source buffer.
 *   5. Compare — if any element differs, print first mismatch.
 *
 * Data-type codes must match spike/riscv/insns/cp_async_tensor.h:
 *   0 U8, 1 U16, 2 U32, 3 I8, 4 I16, 5 I32,
 *   6 FP32, 7 FP16, 8 BF16,
 *   9 U64, 10 I64, 11 FP64.
 */

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

#include "../common/ventus_opencl_test.h"

#define MAX_RANK       5
#define TENSOR_MAP_WORDS 32
#define COORD_WORDS      32
#define DESC_WORDS       (TENSOR_MAP_WORDS + COORD_WORDS)
#define MAX_SRC_BYTES  (64 * 1024)
#define MAX_BOX_BYTES  1024   /* must match SHARED_BUF_BYTES in the .cl */

typedef struct {
  const char *name;
  unsigned    dataType;             /* see encoding table */
  unsigned    rank;                 /* 1..5 */
  unsigned    globalDim[MAX_RANK];  /* full tensor extents */
  unsigned    globalStrides[MAX_RANK]; /* bytes along each dim; index 0 must
                                          equal elemSize for the innermost
                                          dim unless elementStrides != 1 */
  unsigned    boxOffsetElems[MAX_RANK]; /* box origin within the global
                                           tensor (per-dim element count) */
  unsigned    boxDim[MAX_RANK];     /* box extents */
  unsigned    elementStrides[MAX_RANK]; /* 1 = dense */
  unsigned    interleaveMode;          /* descriptor control interleave: 0/1/2 none/16B/32B */
  unsigned    swizzleMode;            /* descriptor control swizzle: 0/1/2/3 none/32B/64B/128B */
  unsigned    expect_oob;            /* compare logical OOB elements as fill */
  unsigned    oobfill;               /* descriptor control: 0=zero, 1=float all-ones */
  unsigned    rtl_only;              /* Spike lacks OOB and non-dim0 stride */
} tma_config_t;

static unsigned
elem_size(unsigned dataType)
{
  switch (dataType) {
    case 0: case 3: return 1;
    case 1: case 4: case 7: case 8: return 2;
    case 2: case 5: case 6: return 4;
    case 9: case 10: case 11: return 8;
    default: return 0;
  }
}

static const char *
dtype_name(unsigned dataType)
{
  switch (dataType) {
    case 0:  return "U8";
    case 1:  return "U16";
    case 2:  return "U32";
    case 3:  return "I8";
    case 4:  return "I16";
    case 5:  return "I32";
    case 6:  return "FP32";
    case 7:  return "FP16";
    case 8:  return "BF16";
    case 9:  return "U64";
    case 10: return "I64";
    case 11: return "FP64";
    default: return "?";
  }
}

static int
is_integer_dtype(unsigned dataType)
{
  return dataType <= 5 || dataType == 9 || dataType == 10;
}

static void
fill_oob_value(const tma_config_t *c, uint8_t *dst, unsigned bytes)
{
  uint8_t fill = 0;
  if (!is_integer_dtype(c->dataType) && c->oobfill) fill = 0xFF;
  memset(dst, fill, bytes);
}

static size_t
swizzle_offset(size_t logical_off, size_t row, unsigned mode)
{
  if (mode == 0) return logical_off;
  unsigned chunk_bits = mode;  /* 1/2/3 for 32B/64B/128B */
  size_t span = (size_t)16 << chunk_bits;
  size_t chunk_mask = ((size_t)1 << chunk_bits) - 1;
  size_t low = logical_off & 0xf;
  size_t chunk = (logical_off >> 4) & chunk_mask;
  size_t row_low = row & chunk_mask;
  return (logical_off & ~(span - 1)) | ((chunk ^ row_low) << 4) | low;
}

static size_t
tensor_physical_offset(const tma_config_t *c, const unsigned coord[MAX_RANK])
{
  unsigned es = elem_size(c->dataType);

  if (c->interleaveMode == 0 || c->rank < 3) {
    size_t off = (size_t)coord[0] * es;
    for (unsigned d = 1; d < c->rank; d++) {
      off += (size_t)coord[d] * c->globalStrides[d - 1];
    }
    return off;
  }

  size_t slice_bytes = c->interleaveMode == 1 ? 16u : 32u;
  size_t channels_per_slice = es ? slice_bytes / es : 1u;
  if (channels_per_slice == 0) channels_per_slice = 1;

  size_t c_slice = coord[0] / channels_per_slice;
  size_t c_in_slice = coord[0] % channels_per_slice;
  size_t c_slice_stride =
    (size_t)c->globalStrides[c->rank - 3] * c->globalDim[c->rank - 2];

  size_t off = c_in_slice * es + c_slice * c_slice_stride;
  for (unsigned d = 1; d < c->rank; d++) {
    off += (size_t)coord[d] * c->globalStrides[d - 1];
  }
  return off;
}

static int
rtl_directed_enabled(void)
{
  const char *force = getenv("VENTUS_TMA_RUN_RTL_ONLY");
  if (force && (strcmp(force, "1") == 0 ||
                strcmp(force, "true") == 0 ||
                strcmp(force, "TRUE") == 0)) {
    return 1;
  }

  const char *backend = getenv("VENTUS_BACKEND");
  return backend && strstr(backend, "rtl") != NULL;
}

static int
case_is_skipped(const tma_config_t *c)
{
  return c->rtl_only && !rtl_directed_enabled();
}

/* Fill src with a deterministic byte pattern: src[i] = (i * 3 + 7) mod 256.
 * This makes mismatches trivial to spot by index and avoids "all zeros"
 * coincidentally matching uninitialized dst. */
static void
fill_src_pattern(uint8_t *buf, size_t nbytes)
{
  for (size_t i = 0; i < nbytes; i++) {
    buf[i] = (uint8_t)((i * 3 + 7) & 0xFF);
  }
}

static size_t
box_total_bytes(const tma_config_t *c)
{
  size_t total = 1;
  for (unsigned d = 0; d < c->rank; d++) {
    unsigned dim = c->boxDim[d];
    if (d > 0 && c->elementStrides[d] > 1) {
      dim = (dim + c->elementStrides[d] - 1) / c->elementStrides[d];
    }
    total *= dim;
  }
  return total * elem_size(c->dataType);
}

static size_t
box_storage_bytes(const tma_config_t *c)
{
  unsigned es = elem_size(c->dataType);
  if (es == 0) return 0;

  size_t total = 1;
  unsigned out_dim[MAX_RANK] = {1, 1, 1, 1, 1};
  for (unsigned d = 0; d < c->rank; d++) {
    out_dim[d] = c->boxDim[d];
    if (d > 0 && c->elementStrides[d] > 1) {
      out_dim[d] = (c->boxDim[d] + c->elementStrides[d] - 1) /
                   c->elementStrides[d];
    }
    total *= out_dim[d];
  }

  size_t max_end = 0;
  for (size_t lin = 0; lin < total; lin++) {
    unsigned idx[MAX_RANK] = {0};
    size_t rem = lin;
    for (unsigned d = 0; d < c->rank; d++) {
      idx[d] = (unsigned)(rem % out_dim[d]);
      rem   /= out_dim[d];
    }

    size_t dst_off = 0;
    size_t dst_mul = es;
    for (unsigned d = 0; d < c->rank; d++) {
      dst_off += (size_t)idx[d] * dst_mul;
      dst_mul *= out_dim[d];
    }

    size_t row = 0;
    size_t row_mul = 1;
    for (unsigned d = 1; d < c->rank; d++) {
      row += (size_t)idx[d] * row_mul;
      row_mul *= out_dim[d];
    }
    dst_off = swizzle_offset(dst_off, row, c->swizzleMode);

    size_t end = dst_off + es;
    if (end > max_end) max_end = end;
  }

  return max_end;
}

static size_t
global_total_bytes(const tma_config_t *c)
{
  /* Logical tensor footprint = (outermost stride) * (outermost dim) for rank
   * >= 2, or dim[0] * elemSize for rank = 1. For OOB-directed cases, also
   * allocate enough backing bytes for the physical box walk. That keeps a
   * buggy RTL implementation from faulting before the host can detect leaked
   * payload from logically OOB coordinates.
   *
   * dim_byte_stride(d) = elemSize           if d == 0
   *                    = globalStrides[d-1] if d >= 1   */
  unsigned es = elem_size(c->dataType);
  if (c->rank == 0) return es;

  size_t logical_bytes;
  if (c->rank == 1) {
    logical_bytes = (size_t)c->globalDim[0] * es + 64;
  } else {
    unsigned top = c->rank - 1;
    logical_bytes = (size_t)c->globalStrides[top - 1] * c->globalDim[top] + 64;
  }

  unsigned max_coord[MAX_RANK] = {0, 0, 0, 0, 0};
  for (unsigned d = 0; d < c->rank; d++) {
    unsigned out_dim = c->boxDim[d];
    if (d > 0 && c->elementStrides[d] > 1) {
      out_dim = (out_dim + c->elementStrides[d] - 1) /
                c->elementStrides[d];
    }
    max_coord[d] = c->boxOffsetElems[d];
    if (out_dim > 0) {
      max_coord[d] += (out_dim - 1) * c->elementStrides[d];
    }
  }

  size_t physical_walk_bytes = tensor_physical_offset(c, max_coord) + es + 64;
  return physical_walk_bytes > logical_bytes ? physical_walk_bytes : logical_bytes;
}

/* Compute "expected" by replaying the same indexing formula the TMA would
 * do, reading from host `src` and writing bytes to host `expected`. */
static void
compute_expected(const tma_config_t *c, const uint8_t *src,
                 uint8_t *expected, size_t out_bytes)
{
  unsigned es = elem_size(c->dataType);

  size_t total = 1;
  unsigned out_dim[MAX_RANK] = {1, 1, 1, 1, 1};
  for (unsigned d = 0; d < c->rank; d++) {
    out_dim[d] = c->boxDim[d];
    if (d > 0 && c->elementStrides[d] > 1) {
      out_dim[d] = (c->boxDim[d] + c->elementStrides[d] - 1) /
                   c->elementStrides[d];
    }
    total *= out_dim[d];
  }

  memset(expected, 0, out_bytes);

  for (size_t lin = 0; lin < total; lin++) {
    unsigned idx[MAX_RANK] = {0};
    size_t rem = lin;
    for (unsigned d = 0; d < c->rank; d++) {
      idx[d] = (unsigned)(rem % out_dim[d]);
      rem   /= out_dim[d];
    }

    int oob = 0;
    unsigned global_coord[MAX_RANK] = {0, 0, 0, 0, 0};
    for (unsigned d = 0; d < c->rank; d++) {
      global_coord[d] = c->boxOffsetElems[d] +
                        idx[d] * c->elementStrides[d];
      if (c->expect_oob && global_coord[d] >= c->globalDim[d]) oob = 1;
    }
    size_t src_off = tensor_physical_offset(c, global_coord);

    size_t dst_off = 0;
    size_t dst_mul = es;
    for (unsigned d = 0; d < c->rank; d++) {
      dst_off += (size_t)idx[d] * dst_mul;
      dst_mul *= out_dim[d];
    }
    size_t row = 0;
    size_t row_mul = 1;
    for (unsigned d = 1; d < c->rank; d++) {
      row += (size_t)idx[d] * row_mul;
      row_mul *= out_dim[d];
    }
    dst_off = swizzle_offset(dst_off, row, c->swizzleMode);

    if (oob) fill_oob_value(c, expected + dst_off, es);
    else memcpy(expected + dst_off, src + src_off, es);
  }
}

static uint32_t
desc_control(unsigned data_type, unsigned rank,
             unsigned interleave, unsigned swizzle,
             unsigned l2promotion, unsigned oobfill)
{
  return (data_type & 0xfu) |
         ((rank & 0xfu) << 4) |
         ((interleave & 0x3u) << 8) |
         ((swizzle & 0x3u) << 10) |
         ((l2promotion & 0x3u) << 12) |
         ((oobfill & 0x1u) << 14);
}

/* Build the descriptor ABI block that tma_matrix_kernel expects:
 *   desc[0..31]  = 128B tensor-map descriptor
 *   desc[32..63] = dynamic coords[0..4] loaded into VRS2 by the kernel */
static void
build_descriptor(const tma_config_t *c, uint32_t *desc)
{
  memset(desc, 0, DESC_WORDS * sizeof(uint32_t));

  desc[0] = 0x56544d41u;  /* "VTMA" */
  desc[1] = desc_control(c->dataType, c->rank, c->interleaveMode,
                         c->swizzleMode, 0, c->oobfill);
  desc[2] = 0;            /* kernel patches runtime src pointer */
  desc[3] = 128;
  for (unsigned d = 0; d < c->rank; d++) desc[4 + d] = c->globalDim[d];
  for (unsigned d = c->rank; d < 5; d++) desc[4 + d] = 1;

  desc[9] = elem_size(c->dataType);
  for (unsigned d = 1; d < c->rank; d++) desc[9 + d] = c->globalStrides[d - 1];
  for (unsigned d = 0; d < c->rank; d++) desc[14 + d] = c->boxDim[d];
  for (unsigned d = c->rank; d < 5; d++) desc[14 + d] = 1;
  for (unsigned d = 0; d < c->rank; d++) desc[19 + d] = c->elementStrides[d];
  for (unsigned d = c->rank; d < 5; d++) desc[19 + d] = 1;

  for (unsigned d = 0; d < c->rank; d++) {
    desc[TENSOR_MAP_WORDS + d] = c->boxOffsetElems[d];
  }
}

/* ----- Test matrix -----
 *
 * Convention: globalStrides[d] is the byte-stride between successive
 * elements along dim (d+1). globalStrides[d] for d >= rank-1 must be 0
 * (RTL DMA reads the full 5 slots; any non-zero value in an unused slot
 * is treated as a real stride and will walk off the tensor). */
static const tma_config_t g_cases[] = {
  /* 1. 2D FP32 4x4 full (baseline, mirrors tensor_dma_test.cl) */
  {
    .name = "FP32_2D_4x4_full",
    .dataType = 6, .rank = 2,
    .globalDim = {4, 4, 1, 1, 1},
    .globalStrides = {16, 0, 0, 0, 0},
    .boxOffsetElems = {0, 0, 0, 0, 0},
    .boxDim = {4, 4, 1, 1, 1},
    .elementStrides = {1, 1, 1, 1, 1},
  },
  /* 2. 1D FP32 16 elements */
  {
    .name = "FP32_1D_16",
    .dataType = 6, .rank = 1,
    .globalDim = {16, 1, 1, 1, 1},
    .globalStrides = {0, 0, 0, 0, 0},
    .boxOffsetElems = {0, 0, 0, 0, 0},
    .boxDim = {16, 1, 1, 1, 1},
    .elementStrides = {1, 1, 1, 1, 1},
  },
  /* 3. 2D FP32 sub-box (4x4 window inside 8x8 tensor at offset [2,2]) */
  {
    .name = "FP32_2D_subbox_8x8_at_2_2",
    .dataType = 6, .rank = 2,
    .globalDim = {8, 8, 1, 1, 1},
    .globalStrides = {32, 0, 0, 0, 0}, /* 8 FP32 = 32B per row */
    .boxOffsetElems = {2, 2, 0, 0, 0},
    .boxDim = {4, 4, 1, 1, 1},
    .elementStrides = {1, 1, 1, 1, 1},
  },
  /* 4. 2D FP32 padded rows (globalStrides > dim*elemSize) */
  {
    .name = "FP32_2D_padded_rows_4x4_stride64",
    .dataType = 6, .rank = 2,
    .globalDim = {4, 4, 1, 1, 1},
    .globalStrides = {64, 0, 0, 0, 0}, /* 64B per row (4 FP32 + 48B pad) */
    .boxOffsetElems = {0, 0, 0, 0, 0},
    .boxDim = {4, 4, 1, 1, 1},
    .elementStrides = {1, 1, 1, 1, 1},
  },
  /* 5. 3D FP32 2x2x2 full */
  {
    .name = "FP32_3D_2x2x2",
    .dataType = 6, .rank = 3,
    .globalDim = {2, 2, 2, 1, 1},
    .globalStrides = {8, 16, 0, 0, 0}, /* 2 FP32 row=8B, 2x2 slice=16B */
    .boxOffsetElems = {0, 0, 0, 0, 0},
    .boxDim = {2, 2, 2, 1, 1},
    .elementStrides = {1, 1, 1, 1, 1},
  },
  /* 5a. 3D FP32 interleave16, layout like NC/4WC4 for FP32. */
  {
    .name = "FP32_3D_interleave16_C6_W3_N2",
    .dataType = 6, .rank = 3,
    .globalDim = {6, 3, 2, 1, 1},
    .globalStrides = {16, 96, 0, 0, 0},
    .boxOffsetElems = {0, 1, 1, 0, 0},
    .boxDim = {6, 2, 1, 1, 1},
    .elementStrides = {1, 1, 1, 1, 1},
    .interleaveMode = 1,
  },
  /* 5b. 3D FP32 interleave32 + swizzle32 crosses the C-slice boundary. */
  {
    .name = "FP32_3D_interleave32_swizzle32_C10_W2",
    .dataType = 6, .rank = 3,
    .globalDim = {10, 2, 1, 1, 1},
    .globalStrides = {32, 128, 0, 0, 0},
    .boxOffsetElems = {0, 0, 0, 0, 0},
    .boxDim = {10, 2, 1, 1, 1},
    .elementStrides = {1, 1, 1, 1, 1},
    .interleaveMode = 2,
    .swizzleMode = 1,
  },
  /* 6. 2D FP16 8x4 full.
   *
   * Keep this at 64B. Larger 1B/2B boxes exercise the current cache refill
   * limitation instead of the TMA descriptor/data-width path this test owns.
   */
  {
    .name = "FP16_2D_8x4",
    .dataType = 7, .rank = 2,
    .globalDim = {8, 4, 1, 1, 1},
    .globalStrides = {16, 0, 0, 0, 0}, /* 8 FP16 = 16B per row */
    .boxOffsetElems = {0, 0, 0, 0, 0},
    .boxDim = {8, 4, 1, 1, 1},
    .elementStrides = {1, 1, 1, 1, 1},
  },
  /* 7. 2D INT32 4x4 full */
  {
    .name = "I32_2D_4x4",
    .dataType = 5, .rank = 2,
    .globalDim = {4, 4, 1, 1, 1},
    .globalStrides = {16, 0, 0, 0, 0},
    .boxOffsetElems = {0, 0, 0, 0, 0},
    .boxDim = {4, 4, 1, 1, 1},
    .elementStrides = {1, 1, 1, 1, 1},
  },
  /* 8. 2D INT8 16x4 (64 bytes, exercises byte-granular path) */
  {
    .name = "I8_2D_16x4",
    .dataType = 3, .rank = 2,
    .globalDim = {16, 4, 1, 1, 1},
    .globalStrides = {16, 0, 0, 0, 0},
    .boxOffsetElems = {0, 0, 0, 0, 0},
    .boxDim = {16, 4, 1, 1, 1},
    .elementStrides = {1, 1, 1, 1, 1},
  },
  /* 9. 1D INT16 32 elements */
  {
    .name = "I16_1D_32",
    .dataType = 4, .rank = 1,
    .globalDim = {32, 1, 1, 1, 1},
    .globalStrides = {0, 0, 0, 0, 0},
    .boxOffsetElems = {0, 0, 0, 0, 0},
    .boxDim = {32, 1, 1, 1, 1},
    .elementStrides = {1, 1, 1, 1, 1},
  },
  /* 10. 2D FP32 partial box from tensor origin (8x8 tensor, 4x4 box at [0,0]) */
  {
    .name = "FP32_2D_partial_box_8x8_at_0_0",
    .dataType = 6, .rank = 2,
    .globalDim = {8, 8, 1, 1, 1},
    .globalStrides = {32, 0, 0, 0, 0},
    .boxOffsetElems = {0, 0, 0, 0, 0},
    .boxDim = {4, 4, 1, 1, 1},
    .elementStrides = {1, 1, 1, 1, 1},
  },
  /* 12. 2D FP32 elementStride=2 (gather every other column) */
  {
    .name = "FP32_2D_estride2_cols",
    .dataType = 6, .rank = 2,
    .globalDim = {8, 4, 1, 1, 1},
    .globalStrides = {32, 0, 0, 0, 0}, /* 8 FP32 row = 32B */
    .boxOffsetElems = {0, 0, 0, 0, 0},
    .boxDim = {4, 4, 1, 1, 1},          /* 4 cols picked, 4 rows */
    .elementStrides = {2, 1, 1, 1, 1},  /* every other column */
  },
  /* 12a. 2D FP32 32B swizzle: each row is one 32B span. */
  {
    .name = "FP32_2D_swizzle32_rows",
    .dataType = 6, .rank = 2,
    .globalDim = {8, 4, 1, 1, 1},
    .globalStrides = {32, 0, 0, 0, 0},
    .boxOffsetElems = {0, 0, 0, 0, 0},
    .boxDim = {8, 4, 1, 1, 1},
    .elementStrides = {1, 1, 1, 1, 1},
    .swizzleMode = 1,
  },
  /* 12b. 2D FP32 64B swizzle: each row is one 64B span. */
  {
    .name = "FP32_2D_swizzle64_rows",
    .dataType = 6, .rank = 2,
    .globalDim = {16, 2, 1, 1, 1},
    .globalStrides = {64, 0, 0, 0, 0},
    .boxOffsetElems = {0, 0, 0, 0, 0},
    .boxDim = {16, 2, 1, 1, 1},
    .elementStrides = {1, 1, 1, 1, 1},
    .swizzleMode = 2,
  },
  /* 12c. 2D FP32 128B swizzle with row subbox offset. */
  {
    .name = "FP32_2D_swizzle128_subbox_row1",
    .dataType = 6, .rank = 2,
    .globalDim = {32, 3, 1, 1, 1},
    .globalStrides = {128, 0, 0, 0, 0},
    .boxOffsetElems = {0, 1, 0, 0, 0},
    .boxDim = {32, 2, 1, 1, 1},
    .elementStrides = {1, 1, 1, 1, 1},
    .swizzleMode = 3,
  },
  /* 13. 3D FP32 4x4x4 sub-box inside a 6x6x6 tensor at [1,1,1] */
  {
    .name = "FP32_3D_subbox_6x6x6_at_1_1_1",
    .dataType = 6, .rank = 3,
    .globalDim = {6, 6, 6, 1, 1},
    .globalStrides = {24, 144, 0, 0, 0},
    .boxOffsetElems = {1, 1, 1, 0, 0},
    .boxDim = {4, 4, 4, 1, 1},
    .elementStrides = {1, 1, 1, 1, 1},
  },
  /* 14. 3D FP32 non-dim0 elementStride. RTL treats dim>0 boxDim as span. */
  {
    .name = "FP32_3D_estride2_dim1",
    .dataType = 6, .rank = 3,
    .globalDim = {4, 8, 4, 1, 1},
    .globalStrides = {16, 128, 0, 0, 0},
    .boxOffsetElems = {0, 0, 0, 0, 0},
    .boxDim = {4, 8, 4, 1, 1},
    .elementStrides = {1, 2, 1, 1, 1},
    .rtl_only = 1,
  },
  /* 15. 2D FP32 stride on both columns and rows. */
  {
    .name = "FP32_2D_estride2_rows_cols",
    .dataType = 6, .rank = 2,
    .globalDim = {8, 8, 1, 1, 1},
    .globalStrides = {32, 0, 0, 0, 0},
    .boxOffsetElems = {0, 0, 0, 0, 0},
    .boxDim = {4, 8, 1, 1, 1},
    .elementStrides = {2, 2, 1, 1, 1},
    .rtl_only = 1,
  },
  /* 16. 2D FP32 dim0 OOB: columns 8..15 in each row should be zero-filled. */
  {
    .name = "FP32_2D_oob_zero_dim0",
    .dataType = 6, .rank = 2,
    .globalDim = {8, 4, 1, 1, 1},
    .globalStrides = {32, 0, 0, 0, 0},
    .boxOffsetElems = {0, 0, 0, 0, 0},
    .boxDim = {16, 4, 1, 1, 1},
    .elementStrides = {1, 1, 1, 1, 1},
    .expect_oob = 1,
    .oobfill = 0,
    .rtl_only = 1,
  },
  /* 17. 2D FP16 OOB with oobfill=1: floating fill is all-one bits. */
  {
    .name = "FP16_2D_oob_fill",
    .dataType = 7, .rank = 2,
    .globalDim = {8, 4, 1, 1, 1},
    .globalStrides = {16, 0, 0, 0, 0},
    .boxOffsetElems = {0, 0, 0, 0, 0},
    .boxDim = {16, 4, 1, 1, 1},
    .elementStrides = {1, 1, 1, 1, 1},
    .expect_oob = 1,
    .oobfill = 1,
    .rtl_only = 1,
  },
  /* 18. 2D FP32 high-dimension OOB: rows 4..5 should be zero-filled. */
  {
    .name = "FP32_2D_oob_zero_dim1",
    .dataType = 6, .rank = 2,
    .globalDim = {8, 4, 1, 1, 1},
    .globalStrides = {32, 0, 0, 0, 0},
    .boxOffsetElems = {0, 2, 0, 0, 0},
    .boxDim = {8, 4, 1, 1, 1},
    .elementStrides = {1, 1, 1, 1, 1},
    .expect_oob = 1,
    .oobfill = 0,
    .rtl_only = 1,
  },
  /* 19. 3D FP32 high-dimension OOB: the second requested slice is OOB. */
  {
    .name = "FP32_3D_oob_zero_dim2",
    .dataType = 6, .rank = 3,
    .globalDim = {4, 4, 2, 1, 1},
    .globalStrides = {16, 64, 0, 0, 0},
    .boxOffsetElems = {0, 0, 1, 0, 0},
    .boxDim = {4, 4, 2, 1, 1},
    .elementStrides = {1, 1, 1, 1, 1},
    .expect_oob = 1,
    .oobfill = 0,
    .rtl_only = 1,
  },
  /* 20. FP16 subbox with dim0+dim1 OOB and floating all-one fill. */
  {
    .name = "FP16_2D_oob_subbox_dim0_dim1_fill",
    .dataType = 7, .rank = 2,
    .globalDim = {8, 4, 1, 1, 1},
    .globalStrides = {16, 0, 0, 0, 0},
    .boxOffsetElems = {6, 3, 0, 0, 0},
    .boxDim = {4, 3, 1, 1, 1},
    .elementStrides = {1, 1, 1, 1, 1},
    .expect_oob = 1,
    .oobfill = 1,
    .rtl_only = 1,
  },
  /* 21. 2D FP32 row-stride OOB: elementStride[1] must affect validity. */
  {
    .name = "FP32_2D_oob_estride_dim1",
    .dataType = 6, .rank = 2,
    .globalDim = {8, 4, 1, 1, 1},
    .globalStrides = {32, 0, 0, 0, 0},
    .boxOffsetElems = {0, 1, 0, 0, 0},
    .boxDim = {8, 6, 1, 1, 1},
    .elementStrides = {1, 2, 1, 1, 1},
    .expect_oob = 1,
    .oobfill = 0,
    .rtl_only = 1,
  },
};

static const size_t g_num_cases = sizeof(g_cases) / sizeof(g_cases[0]);

static int
run_case_process(const char *self, const char *case_name)
{
  char cmd[2048];
  int n = snprintf(cmd, sizeof(cmd), "\"%s\" \"%s\"", self, case_name);
  if (n < 0 || (size_t)n >= sizeof(cmd)) {
    fprintf(stderr, "case command too long for '%s'\n", case_name);
    return 1;
  }

  int status = system(cmd);
  if (status == -1) {
    perror("system");
    return 1;
  }
  if (WIFEXITED(status)) return WEXITSTATUS(status);
  return 1;
}

static int
run_one(cl_context context, cl_command_queue queue, cl_kernel kernel,
        const tma_config_t *c)
{
  const unsigned wg_size = 32;
  unsigned es = elem_size(c->dataType);
  if (es == 0) {
    printf("  SKIP: dataType %u not mapped\n", c->dataType);
    return 0;
  }

  size_t src_bytes = global_total_bytes(c);
  size_t dst_bytes = box_storage_bytes(c);

  if (src_bytes > MAX_SRC_BYTES) {
    printf("  SKIP: src_bytes=%zu > MAX_SRC_BYTES=%d\n",
           src_bytes, MAX_SRC_BYTES);
    return 0;
  }
  if (dst_bytes > MAX_BOX_BYTES) {
    printf("  SKIP: dst_bytes=%zu > MAX_BOX_BYTES=%d\n",
           dst_bytes, MAX_BOX_BYTES);
    return 0;
  }

  uint8_t *src_host      = (uint8_t *)malloc(src_bytes);
  uint8_t *dst_host      = (uint8_t *)calloc(dst_bytes, 1);
  uint8_t *expected_host = (uint8_t *)malloc(dst_bytes);
  uint32_t desc[DESC_WORDS];
  if (!src_host || !dst_host || !expected_host) {
    printf("  FAIL: host alloc\n");
    free(src_host); free(dst_host); free(expected_host);
    return 1;
  }

  fill_src_pattern(src_host, src_bytes);
  build_descriptor(c, desc);
  compute_expected(c, src_host, expected_host, dst_bytes);

  cl_int err;
  cl_mem desc_buf = clCreateBuffer(context,
    CL_MEM_READ_WRITE, DESC_WORDS * sizeof(uint32_t), NULL, &err);
  if (err != CL_SUCCESS) goto failed;
  err = clEnqueueWriteBuffer(queue, desc_buf, CL_TRUE, 0,
                             DESC_WORDS * sizeof(uint32_t), desc,
                             0, NULL, NULL);
  if (err != CL_SUCCESS) { clReleaseMemObject(desc_buf); goto failed; }

  cl_mem src_buf = clCreateBuffer(context,
    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, src_bytes, src_host, &err);
  if (err != CL_SUCCESS) { clReleaseMemObject(desc_buf); goto failed; }

  cl_mem dst_buf = clCreateBuffer(context,
    CL_MEM_WRITE_ONLY, dst_bytes, NULL, &err);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(desc_buf); clReleaseMemObject(src_buf); goto failed;
  }

  cl_uint dst_bytes_arg = (cl_uint)dst_bytes;
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &desc_buf);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &src_buf);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &dst_buf);
  clSetKernelArg(kernel, 3, sizeof(cl_uint), &dst_bytes_arg);

  size_t gsz = wg_size, lsz = wg_size;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gsz, &lsz,
                               0, NULL, NULL);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(desc_buf); clReleaseMemObject(src_buf);
    clReleaseMemObject(dst_buf); goto failed;
  }
  err = clEnqueueReadBuffer(queue, dst_buf, CL_TRUE, 0, dst_bytes,
                            dst_host, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(desc_buf); clReleaseMemObject(src_buf);
    clReleaseMemObject(dst_buf); goto failed;
  }

  int pass = 1;
  for (size_t i = 0; i < dst_bytes; i++) {
    if (dst_host[i] != expected_host[i]) {
      printf("  FAIL at byte[%zu]: expected 0x%02x got 0x%02x\n",
             i, expected_host[i], dst_host[i]);
      pass = 0;
      /* Show a few neighbors to help debugging, then bail. */
      size_t start = (i >= 4) ? i - 4 : 0;
      size_t end   = (i + 4 < dst_bytes) ? i + 4 : dst_bytes - 1;
      printf("    ctx exp:");
      for (size_t k = start; k <= end; k++) printf(" %02x", expected_host[k]);
      printf("\n    ctx got:");
      for (size_t k = start; k <= end; k++) printf(" %02x", dst_host[k]);
      printf("\n");
      break;
    }
  }

  clReleaseMemObject(desc_buf);
  clReleaseMemObject(src_buf);
  clReleaseMemObject(dst_buf);
  free(src_host); free(dst_host); free(expected_host);
  return pass ? 0 : 1;

failed:
  printf("  FAIL: CL error %d\n", err);
  free(src_host); free(dst_host); free(expected_host);
  return 1;
}

int
main(int argc, char **argv)
{
  if (argc < 2) {
    size_t total_pass = 0, total_fail = 0, total_skip = 0;

    for (size_t i = 0; i < g_num_cases; i++) {
      const tma_config_t *c = &g_cases[i];
      printf("[%2zu/%zu] %-36s dtype=%-5s rank=%u boxBytes=%zu\n",
             i + 1, g_num_cases, c->name,
             dtype_name(c->dataType), c->rank, box_total_bytes(c));

      if (case_is_skipped(c)) {
        printf("      SKIP (RTL-directed; set VENTUS_TMA_RUN_RTL_ONLY=1)\n");
        total_skip++;
        continue;
      }

      int rc = run_case_process(argv[0], c->name);
      if (rc == 0) {
        printf("      PASS\n");
        total_pass++;
      } else {
        printf("      FAIL (exit %d)\n", rc);
        total_fail++;
      }
    }

    printf("\n=== TMA matrix summary ===\n");
    printf("  pass: %zu\n  fail: %zu\n  skip: %zu\n",
           total_pass, total_fail, total_skip);

    if (total_fail == 0 && (total_pass > 0 || total_skip > 0)) {
      printf("OK\n");
      return 0;
    }

    printf("FAILED\n");
    return 1;
  }

  (void)argv;

  cl_int err;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_platform_id platform = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;

  err = ventus_get_default_device(&context, &device, &queue, &platform);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device");

  err = ventus_build_program_from_source(context, device,
                                         "tma_matrix_test.cl", &program);
  CHECK_OPENCL_ERROR_IN("ventus_build_program_from_source");

  kernel = clCreateKernel(program, "tma_matrix_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel");

  size_t filter_count = 0;
  const char *filter = (argc >= 2) ? argv[1] : NULL;

  size_t total_pass = 0, total_fail = 0, total_skip = 0;

  for (size_t i = 0; i < g_num_cases; i++) {
    const tma_config_t *c = &g_cases[i];
    if (filter && !strstr(c->name, filter)) { total_skip++; continue; }
    filter_count++;

    printf("[%2zu/%zu] %-36s dtype=%-5s rank=%u boxBytes=%zu\n",
           i + 1, g_num_cases, c->name,
           dtype_name(c->dataType), c->rank, box_total_bytes(c));

    if (case_is_skipped(c)) {
      printf("      SKIP (RTL-directed; set VENTUS_TMA_RUN_RTL_ONLY=1)\n");
      total_skip++;
      continue;
    }

    int rc = run_one(context, queue, kernel, c);
    if (rc == 0) {
      printf("      PASS\n");
      total_pass++;
    } else {
      total_fail++;
    }
  }

  printf("\n=== TMA matrix summary ===\n");
  printf("  pass: %zu\n  fail: %zu\n  skip: %zu\n",
         total_pass, total_fail, total_skip);

  if (filter && filter_count == 0) {
    printf("  (filter '%s' matched nothing)\n", filter);
  }

  if (total_fail == 0 && (total_pass > 0 || filter_count > 0)) printf("OK\n");
  else printf("FAILED\n");

FINISH:
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);

  return (total_fail == 0 && (total_pass > 0 || filter_count > 0)) ? 0 : 1;
}
