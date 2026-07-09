/*
 * S2G mixed completion/routing functional cases.
 *
 * This file keeps S2G mixed coverage inside the existing S2G functional
 * executable.  The host generates independent byte-level expected data for
 * each kernel and supports S2G_MIXED_CASE_FILTER for quick single-case runs.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/ventus_opencl_test.h"

#define BULK_COPY_BYTES 128
#define TENSOR_SRC_BYTES (8 * 8 * 4)
#define TENSOR_COPY_BYTES 64
#define TENSOR_GLOBAL_BYTES TENSOR_SRC_BYTES
#define DESC_WORDS 32
#define COORD_WORDS 32

#define COMPLEX_OUT_BULK_G2S_OFF 0
#define COMPLEX_OUT_BULK_S2G_OFF 128
#define COMPLEX_OUT_TENSOR_G2S_OFF 256
#define COMPLEX_OUT_TENSOR_S2G_OFF 320
#define COMPLEX_OUT_BYTES (COMPLEX_OUT_TENSOR_S2G_OFF + TENSOR_GLOBAL_BYTES)

#define LONG_OUT_BULK_G2S_OFF 0
#define LONG_OUT_BULK_S2G0_OFF 128
#define LONG_OUT_TENSOR_G2S_OFF 256
#define LONG_OUT_TENSOR_S2G0_OFF 384
#define LONG_OUT_BULK_S2G1_OFF 640
#define LONG_OUT_TENSOR_S2G1_OFF 768
#define LONG_OUT_BYTES (LONG_OUT_TENSOR_S2G1_OFF + TENSOR_GLOBAL_BYTES)
#define LONG_MARKER_WORDS 8

#define ROUTE_SRC_BYTES 512

typedef struct {
  const char *name;
  unsigned local_size;
  unsigned copy_bytes;
  unsigned rounds;
  unsigned src_offset;
} route_case_t;

static uint32_t bulk_pattern_word(unsigned idx)
{
  unsigned base = idx << 2;
  unsigned b0 = (base * 7u + 0x23u) & 0xffu;
  unsigned b1 = ((base + 1u) * 7u + 0x23u) & 0xffu;
  unsigned b2 = ((base + 2u) * 7u + 0x23u) & 0xffu;
  unsigned b3 = ((base + 3u) * 7u + 0x23u) & 0xffu;
  return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

static uint32_t stress_pattern_word(unsigned idx, unsigned seed)
{
  unsigned base = idx << 2;
  unsigned b0 = (base * 13u + seed) & 0xffu;
  unsigned b1 = ((base + 1u) * 13u + seed) & 0xffu;
  unsigned b2 = ((base + 2u) * 13u + seed) & 0xffu;
  unsigned b3 = ((base + 3u) * 13u + seed) & 0xffu;
  return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

static uint32_t tensor_s2g_pattern_word(unsigned idx)
{
  unsigned base = idx << 2;
  unsigned b0 = (base * 7u + 3u) & 0xffu;
  unsigned b1 = ((base + 1u) * 7u + 3u) & 0xffu;
  unsigned b2 = ((base + 2u) * 7u + 3u) & 0xffu;
  unsigned b3 = ((base + 3u) * 7u + 3u) & 0xffu;
  return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

static void fill_src(uint8_t *src, size_t n, unsigned seed)
{
  for (size_t i = 0; i < n; i++) {
    src[i] = (uint8_t)((i * 5u + seed) & 0xffu);
  }
}

static void fill_bulk_pattern(uint8_t *dst, unsigned seed)
{
  for (unsigned i = 0; i < BULK_COPY_BYTES / 4; i++) {
    uint32_t word = stress_pattern_word(i, seed);
    memcpy(dst + i * 4, &word, sizeof(word));
  }
}

static void fill_route_pattern(uint8_t *dst)
{
  for (unsigned i = 0; i < ROUTE_SRC_BYTES / 4; i++) {
    uint32_t word = bulk_pattern_word(i);
    memcpy(dst + i * 4, &word, sizeof(word));
  }
}

static void build_desc(uint32_t *desc)
{
  memset(desc, 0, DESC_WORDS * sizeof(uint32_t));
  desc[0] = 0x56544d41u;
  desc[1] = (6u & 0xfu) | (2u << 4);
  desc[2] = 0;
  desc[3] = 128;
  desc[4] = 8;
  desc[5] = 8;
  desc[6] = desc[7] = desc[8] = 1;
  desc[9] = 4;
  desc[10] = 32;
  desc[14] = 4;
  desc[15] = 4;
  desc[16] = desc[17] = desc[18] = 1;
  for (int i = 0; i < 5; i++) desc[19 + i] = 1;
}

static void compute_tensor_g2s_expected(const uint8_t *src, unsigned coord0,
                                        unsigned coord1, uint8_t *expected)
{
  for (unsigned y = 0; y < 4; y++) {
    for (unsigned x = 0; x < 4; x++) {
      size_t src_off = ((coord1 + y) * 8 + (coord0 + x)) * 4;
      size_t dst_off = (y * 4 + x) * 4;
      memcpy(expected + dst_off, src + src_off, 4);
    }
  }
}

static void fill_tensor_s2g_expected(uint8_t *expected, unsigned coord0,
                                     unsigned coord1)
{
  memset(expected, 0xcd, TENSOR_GLOBAL_BYTES);
  for (unsigned y = 0; y < 4; y++) {
    for (unsigned x = 0; x < 4; x++) {
      uint32_t word = tensor_s2g_pattern_word(y * 4 + x);
      size_t dst_off = ((coord1 + y) * 8 + (coord0 + x)) * 4;
      memcpy(expected + dst_off, &word, sizeof(word));
    }
  }
}

static uint32_t expected_conflict_value(unsigned lid, unsigned rounds)
{
  uint32_t value = 0xA5000000u + lid;
  for (unsigned r = 0; r < rounds; r++) {
    value = value + ((r + 1u) * 17u) + lid;
  }
  return value;
}

static int first_mismatch(const uint8_t *a, const uint8_t *b, size_t n)
{
  for (size_t i = 0; i < n; i++) {
    if (a[i] != b[i]) return (int)i;
  }
  return -1;
}

static int check_bytes(const char *label, const uint8_t *got,
                       const uint8_t *expected, size_t n)
{
  int mismatch = first_mismatch(got, expected, n);
  if (mismatch >= 0) {
    fprintf(stderr, "FAIL %s at byte %d: got=%02x exp=%02x\n",
            label, mismatch, got[mismatch], expected[mismatch]);
    return 1;
  }
  printf("PASS %s bytes=%zu\n", label, n);
  return 0;
}

static int check_marker_pair(const char *label, const uint32_t *marker,
                             uint32_t a, uint32_t b)
{
  if (marker[0] != a || marker[1] != b) {
    fprintf(stderr, "FAIL %s marker: got=%08x,%08x exp=%08x,%08x\n",
            label, marker[0], marker[1], a, b);
    return 1;
  }
  printf("PASS %s marker=%08x,%08x\n", label, marker[0], marker[1]);
  return 0;
}

static int check_complex_marker(const uint32_t *marker)
{
  const uint32_t expected[4] = {
    0xfeed0001u, 0xfeed0002u, 0xfeed0003u, 0xfeed0004u,
  };
  for (size_t i = 0; i < 4; i++) {
    if (marker[i] != expected[i]) {
      fprintf(stderr, "FAIL complex_marker[%zu]: got=%08x exp=%08x\n",
              i, marker[i], expected[i]);
      return 1;
    }
  }
  printf("PASS complex_marker marker=%08x,%08x,%08x,%08x\n",
         marker[0], marker[1], marker[2], marker[3]);
  return 0;
}

static int check_long_marker(const uint32_t *marker)
{
  const uint32_t expected[7] = {
    0xfeed1001u, 0xfeed1002u, 0xfeed1003u, 0xfeed1004u,
    0xfeed1005u, 0xfeed1006u, 0xfeed1007u,
  };
  for (size_t i = 0; i < sizeof(expected) / sizeof(expected[0]); i++) {
    if (marker[i] != expected[i]) {
      fprintf(stderr, "FAIL long_marker[%zu]: got=%08x exp=%08x\n",
              i, marker[i], expected[i]);
      return 1;
    }
  }
  printf("PASS long_marker marker=%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
         marker[0], marker[1], marker[2], marker[3],
         marker[4], marker[5], marker[6]);
  return 0;
}

static int selected(const char *filter, const char *name)
{
  return !filter || !filter[0] || strstr(name, filter);
}

static int run_route_one(cl_context context, cl_command_queue queue,
                         cl_kernel kernel, const route_case_t *c)
{
  cl_int err = CL_SUCCESS;
  cl_mem dst_buf = NULL;
  cl_mem conflict_buf = NULL;
  uint8_t route_src[ROUTE_SRC_BYTES];
  uint8_t *dst = (uint8_t *)malloc(c->copy_bytes);
  uint8_t *expected = (uint8_t *)malloc(c->copy_bytes);
  uint32_t *conflict =
    (uint32_t *)calloc(c->local_size, sizeof(uint32_t));
  int rc = 1;

  if (!dst || !expected || !conflict) {
    fprintf(stderr, "FAIL %s host alloc\n", c->name);
    goto FINISH;
  }
  if (c->src_offset + c->copy_bytes > ROUTE_SRC_BYTES) {
    fprintf(stderr, "FAIL %s source window exceeds route buffer\n", c->name);
    goto FINISH;
  }

  fill_route_pattern(route_src);
  memset(dst, 0xcd, c->copy_bytes);
  memcpy(expected, route_src + c->src_offset, c->copy_bytes);

  dst_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           c->copy_bytes, dst, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(route dst)");
  conflict_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                c->local_size * sizeof(uint32_t), NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(route conflict)");

  cl_uint copy_bytes = c->copy_bytes;
  cl_uint rounds = c->rounds;
  cl_uint src_offset = c->src_offset;
  err  = clSetKernelArg(kernel, 0, sizeof(dst_buf), &dst_buf);
  err |= clSetKernelArg(kernel, 1, sizeof(conflict_buf), &conflict_buf);
  err |= clSetKernelArg(kernel, 2, sizeof(copy_bytes), &copy_bytes);
  err |= clSetKernelArg(kernel, 3, sizeof(rounds), &rounds);
  err |= clSetKernelArg(kernel, 4, sizeof(src_offset), &src_offset);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(route)");

  size_t global = c->local_size, local = c->local_size;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                               0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel(route)");
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(route)");
  err = clEnqueueReadBuffer(queue, dst_buf, CL_TRUE, 0, c->copy_bytes,
                            dst, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(route dst)");
  err = clEnqueueReadBuffer(queue, conflict_buf, CL_TRUE, 0,
                            c->local_size * sizeof(uint32_t), conflict,
                            0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(route conflict)");

  if (check_bytes(c->name, dst, expected, c->copy_bytes) != 0) goto FINISH;
  for (unsigned lid = 0; lid < c->local_size; lid++) {
    uint32_t exp = expected_conflict_value(lid, c->rounds);
    if (conflict[lid] != exp) {
      fprintf(stderr, "FAIL %s conflict lid%u: got=%08x exp=%08x\n",
              c->name, lid, conflict[lid], exp);
      goto FINISH;
    }
  }
  printf("PASS %s conflict lanes=%u\n", c->name, c->local_size);
  rc = 0;

FINISH:
  if (conflict_buf) clReleaseMemObject(conflict_buf);
  if (dst_buf) clReleaseMemObject(dst_buf);
  free(conflict);
  free(expected);
  free(dst);
  return rc;
}

static int run_wait_domain_case(cl_context context, cl_command_queue queue,
                                cl_kernel kernel, const char *name,
                                unsigned s2g_seed, uint32_t marker_a,
                                uint32_t marker_b)
{
  cl_int err = CL_SUCCESS;
  cl_mem g2s_src_buf = NULL;
  cl_mem g2s_out_buf = NULL;
  cl_mem s2g_dst_buf = NULL;
  cl_mem marker_buf = NULL;
  uint8_t g2s_src[BULK_COPY_BYTES];
  uint8_t g2s_out[BULK_COPY_BYTES];
  uint8_t s2g_dst[BULK_COPY_BYTES];
  uint8_t expected_s2g[BULK_COPY_BYTES];
  uint32_t marker[4];
  int rc = 1;

  fill_src(g2s_src, sizeof(g2s_src), 0x31u);
  memset(g2s_out, 0xcd, sizeof(g2s_out));
  memset(s2g_dst, 0xcd, sizeof(s2g_dst));
  fill_bulk_pattern(expected_s2g, s2g_seed);
  for (size_t i = 0; i < 4; i++) marker[i] = 0xcdcdcdcdU;

  g2s_src_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(g2s_src), g2s_src, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(wait g2s_src)");
  g2s_out_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                               sizeof(g2s_out), g2s_out, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(wait g2s_out)");
  s2g_dst_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                               sizeof(s2g_dst), s2g_dst, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(wait s2g_dst)");
  marker_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              sizeof(marker), marker, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(wait marker)");

  cl_uint copy_bytes = BULK_COPY_BYTES;
  err  = clSetKernelArg(kernel, 0, sizeof(g2s_src_buf), &g2s_src_buf);
  err |= clSetKernelArg(kernel, 1, sizeof(g2s_out_buf), &g2s_out_buf);
  err |= clSetKernelArg(kernel, 2, sizeof(s2g_dst_buf), &s2g_dst_buf);
  err |= clSetKernelArg(kernel, 3, sizeof(marker_buf), &marker_buf);
  err |= clSetKernelArg(kernel, 4, sizeof(copy_bytes), &copy_bytes);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(wait domain)");

  size_t global = 32, local = 32;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                               0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel(wait domain)");
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(wait domain)");
  err = clEnqueueReadBuffer(queue, g2s_out_buf, CL_TRUE, 0, sizeof(g2s_out),
                            g2s_out, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(wait g2s_out)");
  err = clEnqueueReadBuffer(queue, s2g_dst_buf, CL_TRUE, 0, sizeof(s2g_dst),
                            s2g_dst, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(wait s2g_dst)");
  err = clEnqueueReadBuffer(queue, marker_buf, CL_TRUE, 0, sizeof(marker),
                            marker, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(wait marker)");

  if (check_bytes(name, g2s_out, g2s_src, sizeof(g2s_src)) != 0) goto FINISH;
  if (check_bytes(name, s2g_dst, expected_s2g, sizeof(expected_s2g)) != 0) {
    goto FINISH;
  }
  if (check_marker_pair(name, marker, marker_a, marker_b) != 0) goto FINISH;
  rc = 0;

FINISH:
  if (marker_buf) clReleaseMemObject(marker_buf);
  if (s2g_dst_buf) clReleaseMemObject(s2g_dst_buf);
  if (g2s_out_buf) clReleaseMemObject(g2s_out_buf);
  if (g2s_src_buf) clReleaseMemObject(g2s_src_buf);
  return rc;
}

static int run_bulk_tensor_same_fence(cl_context context,
                                      cl_command_queue queue,
                                      cl_kernel kernel)
{
  cl_int err = CL_SUCCESS;
  cl_mem desc_buf = NULL;
  cl_mem coords_buf = NULL;
  cl_mem bulk_dst_buf = NULL;
  cl_mem tensor_dst_buf = NULL;
  uint32_t desc[DESC_WORDS];
  uint32_t coords[COORD_WORDS] = {0};
  uint8_t bulk_dst[BULK_COPY_BYTES];
  uint8_t tensor_dst[TENSOR_GLOBAL_BYTES];
  uint8_t expected_bulk[BULK_COPY_BYTES];
  uint8_t expected_tensor[TENSOR_GLOBAL_BYTES];
  int rc = 1;

  build_desc(desc);
  memset(bulk_dst, 0xcd, sizeof(bulk_dst));
  memset(tensor_dst, 0xcd, sizeof(tensor_dst));
  fill_bulk_pattern(expected_bulk, 0x7au);
  fill_tensor_s2g_expected(expected_tensor, 0, 0);

  desc_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            sizeof(desc), desc, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(same fence desc)");
  coords_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(coords), coords, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(same fence coords)");
  bulk_dst_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                sizeof(bulk_dst), bulk_dst, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(same fence bulk)");
  tensor_dst_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  sizeof(tensor_dst), tensor_dst, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(same fence tensor)");

  err  = clSetKernelArg(kernel, 0, sizeof(desc_buf), &desc_buf);
  err |= clSetKernelArg(kernel, 1, sizeof(coords_buf), &coords_buf);
  err |= clSetKernelArg(kernel, 2, sizeof(bulk_dst_buf), &bulk_dst_buf);
  err |= clSetKernelArg(kernel, 3, sizeof(tensor_dst_buf), &tensor_dst_buf);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(same fence)");

  size_t global = 32, local = 32;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                               0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel(same fence)");
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(same fence)");
  err = clEnqueueReadBuffer(queue, bulk_dst_buf, CL_TRUE, 0, sizeof(bulk_dst),
                            bulk_dst, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(same fence bulk)");
  err = clEnqueueReadBuffer(queue, tensor_dst_buf, CL_TRUE, 0,
                            sizeof(tensor_dst), tensor_dst, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(same fence tensor)");

  if (check_bytes("bulk_tensor_s2g_same_fence_bulk",
                  bulk_dst, expected_bulk, sizeof(bulk_dst)) != 0) {
    goto FINISH;
  }
  if (check_bytes("bulk_tensor_s2g_same_fence_tensor",
                  tensor_dst, expected_tensor, sizeof(tensor_dst)) != 0) {
    goto FINISH;
  }
  rc = 0;

FINISH:
  if (tensor_dst_buf) clReleaseMemObject(tensor_dst_buf);
  if (bulk_dst_buf) clReleaseMemObject(bulk_dst_buf);
  if (coords_buf) clReleaseMemObject(coords_buf);
  if (desc_buf) clReleaseMemObject(desc_buf);
  return rc;
}

static int run_bidirectional_descriptor_mix(cl_context context,
                                            cl_command_queue queue,
                                            cl_kernel kernel)
{
  cl_int err = CL_SUCCESS;
  cl_mem tensor_g2s_desc_buf = NULL;
  cl_mem tensor_s2g_desc_buf = NULL;
  cl_mem tensor_g2s_coords_buf = NULL;
  cl_mem tensor_s2g_coords_buf = NULL;
  cl_mem tensor_src_buf = NULL;
  cl_mem bulk_src_buf = NULL;
  cl_mem out_buf = NULL;
  cl_mem marker_buf = NULL;
  uint32_t tensor_g2s_desc[DESC_WORDS];
  uint32_t tensor_s2g_desc[DESC_WORDS];
  uint32_t tensor_g2s_coords[COORD_WORDS] = {0};
  uint32_t tensor_s2g_coords[COORD_WORDS] = {0};
  uint8_t tensor_src[TENSOR_SRC_BYTES];
  uint8_t bulk_src[BULK_COPY_BYTES];
  uint8_t out[COMPLEX_OUT_BYTES];
  uint8_t tensor_g2s_expected[TENSOR_COPY_BYTES];
  uint8_t tensor_s2g_expected[TENSOR_GLOBAL_BYTES];
  uint8_t bulk_s2g_expected[BULK_COPY_BYTES];
  uint32_t marker[4];
  int rc = 1;

  build_desc(tensor_g2s_desc);
  build_desc(tensor_s2g_desc);
  tensor_g2s_coords[0] = 1;
  tensor_g2s_coords[1] = 2;
  tensor_s2g_coords[0] = 2;
  tensor_s2g_coords[1] = 1;
  fill_src(tensor_src, sizeof(tensor_src), 0x67u);
  fill_src(bulk_src, sizeof(bulk_src), 0xb5u);
  compute_tensor_g2s_expected(tensor_src, 1, 2, tensor_g2s_expected);
  fill_tensor_s2g_expected(tensor_s2g_expected, 2, 1);
  fill_bulk_pattern(bulk_s2g_expected, 0x55u);
  memset(out, 0xcd, sizeof(out));
  for (size_t i = 0; i < 4; i++) marker[i] = 0xcdcdcdcdU;

  tensor_g2s_desc_buf = clCreateBuffer(context, CL_MEM_READ_WRITE |
      CL_MEM_COPY_HOST_PTR, sizeof(tensor_g2s_desc), tensor_g2s_desc, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(complex g2s desc)");
  tensor_s2g_desc_buf = clCreateBuffer(context, CL_MEM_READ_WRITE |
      CL_MEM_COPY_HOST_PTR, sizeof(tensor_s2g_desc), tensor_s2g_desc, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(complex s2g desc)");
  tensor_g2s_coords_buf = clCreateBuffer(context, CL_MEM_READ_ONLY |
      CL_MEM_COPY_HOST_PTR, sizeof(tensor_g2s_coords), tensor_g2s_coords, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(complex g2s coords)");
  tensor_s2g_coords_buf = clCreateBuffer(context, CL_MEM_READ_ONLY |
      CL_MEM_COPY_HOST_PTR, sizeof(tensor_s2g_coords), tensor_s2g_coords, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(complex s2g coords)");
  tensor_src_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(tensor_src), tensor_src, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(complex tensor src)");
  bulk_src_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(bulk_src), bulk_src, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(complex bulk src)");
  out_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(out), out, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(complex out)");
  marker_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              sizeof(marker), marker, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(complex marker)");

  cl_uint bulk_bytes = BULK_COPY_BYTES;
  err  = clSetKernelArg(kernel, 0, sizeof(tensor_g2s_desc_buf),
                        &tensor_g2s_desc_buf);
  err |= clSetKernelArg(kernel, 1, sizeof(tensor_g2s_coords_buf),
                        &tensor_g2s_coords_buf);
  err |= clSetKernelArg(kernel, 2, sizeof(tensor_src_buf), &tensor_src_buf);
  err |= clSetKernelArg(kernel, 3, sizeof(tensor_s2g_desc_buf),
                        &tensor_s2g_desc_buf);
  err |= clSetKernelArg(kernel, 4, sizeof(tensor_s2g_coords_buf),
                        &tensor_s2g_coords_buf);
  err |= clSetKernelArg(kernel, 5, sizeof(bulk_src_buf), &bulk_src_buf);
  err |= clSetKernelArg(kernel, 6, sizeof(out_buf), &out_buf);
  err |= clSetKernelArg(kernel, 7, sizeof(marker_buf), &marker_buf);
  err |= clSetKernelArg(kernel, 8, sizeof(bulk_bytes), &bulk_bytes);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(complex)");

  size_t global = 32, local = 32;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                               0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel(complex)");
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(complex)");
  err = clEnqueueReadBuffer(queue, out_buf, CL_TRUE, 0, sizeof(out), out,
                            0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(complex out)");
  err = clEnqueueReadBuffer(queue, marker_buf, CL_TRUE, 0, sizeof(marker),
                            marker, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(complex marker)");

  if (check_bytes("bidirectional_bulk_g2s",
                  out + COMPLEX_OUT_BULK_G2S_OFF, bulk_src,
                  sizeof(bulk_src)) != 0) goto FINISH;
  if (check_bytes("bidirectional_bulk_s2g",
                  out + COMPLEX_OUT_BULK_S2G_OFF, bulk_s2g_expected,
                  sizeof(bulk_s2g_expected)) != 0) goto FINISH;
  if (check_bytes("bidirectional_tensor_g2s",
                  out + COMPLEX_OUT_TENSOR_G2S_OFF, tensor_g2s_expected,
                  sizeof(tensor_g2s_expected)) != 0) goto FINISH;
  if (check_bytes("bidirectional_tensor_s2g",
                  out + COMPLEX_OUT_TENSOR_S2G_OFF, tensor_s2g_expected,
                  sizeof(tensor_s2g_expected)) != 0) goto FINISH;
  if (check_complex_marker(marker) != 0) goto FINISH;
  rc = 0;

FINISH:
  if (marker_buf) clReleaseMemObject(marker_buf);
  if (out_buf) clReleaseMemObject(out_buf);
  if (bulk_src_buf) clReleaseMemObject(bulk_src_buf);
  if (tensor_src_buf) clReleaseMemObject(tensor_src_buf);
  if (tensor_s2g_coords_buf) clReleaseMemObject(tensor_s2g_coords_buf);
  if (tensor_g2s_coords_buf) clReleaseMemObject(tensor_g2s_coords_buf);
  if (tensor_s2g_desc_buf) clReleaseMemObject(tensor_s2g_desc_buf);
  if (tensor_g2s_desc_buf) clReleaseMemObject(tensor_g2s_desc_buf);
  return rc;
}

static int run_mixed_tensor_outstanding_long(cl_context context,
                                             cl_command_queue queue,
                                             cl_kernel kernel)
{
  cl_int err = CL_SUCCESS;
  cl_mem tensor_g2s_desc_buf = NULL;
  cl_mem tensor_s2g0_desc_buf = NULL;
  cl_mem tensor_s2g1_desc_buf = NULL;
  cl_mem tensor_g2s_coords_buf = NULL;
  cl_mem tensor_s2g0_coords_buf = NULL;
  cl_mem tensor_s2g1_coords_buf = NULL;
  cl_mem tensor_src_buf = NULL;
  cl_mem bulk_src_buf = NULL;
  cl_mem out_buf = NULL;
  cl_mem marker_buf = NULL;
  uint32_t tensor_g2s_desc[DESC_WORDS];
  uint32_t tensor_s2g0_desc[DESC_WORDS];
  uint32_t tensor_s2g1_desc[DESC_WORDS];
  uint32_t tensor_g2s_coords[COORD_WORDS] = {0};
  uint32_t tensor_s2g0_coords[COORD_WORDS] = {0};
  uint32_t tensor_s2g1_coords[COORD_WORDS] = {0};
  uint8_t tensor_src[TENSOR_SRC_BYTES];
  uint8_t bulk_src[BULK_COPY_BYTES];
  uint8_t out[LONG_OUT_BYTES];
  uint8_t tensor_g2s_expected[TENSOR_COPY_BYTES];
  uint8_t tensor_s2g0_expected[TENSOR_GLOBAL_BYTES];
  uint8_t tensor_s2g1_expected[TENSOR_GLOBAL_BYTES];
  uint8_t bulk_s2g0_expected[BULK_COPY_BYTES];
  uint8_t bulk_s2g1_expected[BULK_COPY_BYTES];
  uint32_t marker[LONG_MARKER_WORDS];
  int rc = 1;

  build_desc(tensor_g2s_desc);
  build_desc(tensor_s2g0_desc);
  build_desc(tensor_s2g1_desc);
  tensor_g2s_coords[0] = 1;
  tensor_g2s_coords[1] = 1;
  tensor_s2g0_coords[0] = 0;
  tensor_s2g0_coords[1] = 0;
  tensor_s2g1_coords[0] = 4;
  tensor_s2g1_coords[1] = 4;

  fill_src(tensor_src, sizeof(tensor_src), 0x77u);
  fill_src(bulk_src, sizeof(bulk_src), 0xc1u);
  compute_tensor_g2s_expected(tensor_src, 1, 1, tensor_g2s_expected);
  fill_tensor_s2g_expected(tensor_s2g0_expected, 0, 0);
  fill_tensor_s2g_expected(tensor_s2g1_expected, 4, 4);
  fill_bulk_pattern(bulk_s2g0_expected, 0x83u);
  fill_bulk_pattern(bulk_s2g1_expected, 0x91u);
  memset(out, 0xcd, sizeof(out));
  for (size_t i = 0; i < LONG_MARKER_WORDS; i++) marker[i] = 0xcdcdcdcdu;

  tensor_g2s_desc_buf = clCreateBuffer(context, CL_MEM_READ_WRITE |
      CL_MEM_COPY_HOST_PTR, sizeof(tensor_g2s_desc), tensor_g2s_desc, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(long g2s desc)");
  tensor_s2g0_desc_buf = clCreateBuffer(context, CL_MEM_READ_WRITE |
      CL_MEM_COPY_HOST_PTR, sizeof(tensor_s2g0_desc), tensor_s2g0_desc, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(long s2g0 desc)");
  tensor_s2g1_desc_buf = clCreateBuffer(context, CL_MEM_READ_WRITE |
      CL_MEM_COPY_HOST_PTR, sizeof(tensor_s2g1_desc), tensor_s2g1_desc, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(long s2g1 desc)");
  tensor_g2s_coords_buf = clCreateBuffer(context, CL_MEM_READ_ONLY |
      CL_MEM_COPY_HOST_PTR, sizeof(tensor_g2s_coords), tensor_g2s_coords, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(long g2s coords)");
  tensor_s2g0_coords_buf = clCreateBuffer(context, CL_MEM_READ_ONLY |
      CL_MEM_COPY_HOST_PTR, sizeof(tensor_s2g0_coords), tensor_s2g0_coords,
      &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(long s2g0 coords)");
  tensor_s2g1_coords_buf = clCreateBuffer(context, CL_MEM_READ_ONLY |
      CL_MEM_COPY_HOST_PTR, sizeof(tensor_s2g1_coords), tensor_s2g1_coords,
      &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(long s2g1 coords)");
  tensor_src_buf = clCreateBuffer(context, CL_MEM_READ_ONLY |
      CL_MEM_COPY_HOST_PTR, sizeof(tensor_src), tensor_src, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(long tensor src)");
  bulk_src_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(bulk_src), bulk_src, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(long bulk src)");
  out_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(out), out, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(long out)");
  marker_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              sizeof(marker), marker, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(long marker)");

  cl_uint bulk_bytes = BULK_COPY_BYTES;
  err  = clSetKernelArg(kernel, 0, sizeof(tensor_g2s_desc_buf),
                        &tensor_g2s_desc_buf);
  err |= clSetKernelArg(kernel, 1, sizeof(tensor_g2s_coords_buf),
                        &tensor_g2s_coords_buf);
  err |= clSetKernelArg(kernel, 2, sizeof(tensor_src_buf), &tensor_src_buf);
  err |= clSetKernelArg(kernel, 3, sizeof(tensor_s2g0_desc_buf),
                        &tensor_s2g0_desc_buf);
  err |= clSetKernelArg(kernel, 4, sizeof(tensor_s2g0_coords_buf),
                        &tensor_s2g0_coords_buf);
  err |= clSetKernelArg(kernel, 5, sizeof(tensor_s2g1_desc_buf),
                        &tensor_s2g1_desc_buf);
  err |= clSetKernelArg(kernel, 6, sizeof(tensor_s2g1_coords_buf),
                        &tensor_s2g1_coords_buf);
  err |= clSetKernelArg(kernel, 7, sizeof(bulk_src_buf), &bulk_src_buf);
  err |= clSetKernelArg(kernel, 8, sizeof(out_buf), &out_buf);
  err |= clSetKernelArg(kernel, 9, sizeof(marker_buf), &marker_buf);
  err |= clSetKernelArg(kernel, 10, sizeof(bulk_bytes), &bulk_bytes);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(long)");

  size_t global = 32, local = 32;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                               0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel(long)");
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(long)");
  err = clEnqueueReadBuffer(queue, out_buf, CL_TRUE, 0, sizeof(out), out,
                            0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(long out)");
  err = clEnqueueReadBuffer(queue, marker_buf, CL_TRUE, 0, sizeof(marker),
                            marker, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(long marker)");

  if (check_bytes("long_bulk_g2s", out + LONG_OUT_BULK_G2S_OFF,
                  bulk_src, sizeof(bulk_src)) != 0) goto FINISH;
  if (check_bytes("long_bulk_s2g0", out + LONG_OUT_BULK_S2G0_OFF,
                  bulk_s2g0_expected, sizeof(bulk_s2g0_expected)) != 0) {
    goto FINISH;
  }
  if (check_bytes("long_tensor_g2s", out + LONG_OUT_TENSOR_G2S_OFF,
                  tensor_g2s_expected, sizeof(tensor_g2s_expected)) != 0) {
    goto FINISH;
  }
  if (check_bytes("long_tensor_s2g0", out + LONG_OUT_TENSOR_S2G0_OFF,
                  tensor_s2g0_expected, sizeof(tensor_s2g0_expected)) != 0) {
    goto FINISH;
  }
  if (check_bytes("long_bulk_s2g1", out + LONG_OUT_BULK_S2G1_OFF,
                  bulk_s2g1_expected, sizeof(bulk_s2g1_expected)) != 0) {
    goto FINISH;
  }
  if (check_bytes("long_tensor_s2g1", out + LONG_OUT_TENSOR_S2G1_OFF,
                  tensor_s2g1_expected, sizeof(tensor_s2g1_expected)) != 0) {
    goto FINISH;
  }
  if (check_long_marker(marker) != 0) goto FINISH;
  rc = 0;

FINISH:
  if (marker_buf) clReleaseMemObject(marker_buf);
  if (out_buf) clReleaseMemObject(out_buf);
  if (bulk_src_buf) clReleaseMemObject(bulk_src_buf);
  if (tensor_src_buf) clReleaseMemObject(tensor_src_buf);
  if (tensor_s2g1_coords_buf) clReleaseMemObject(tensor_s2g1_coords_buf);
  if (tensor_s2g0_coords_buf) clReleaseMemObject(tensor_s2g0_coords_buf);
  if (tensor_g2s_coords_buf) clReleaseMemObject(tensor_g2s_coords_buf);
  if (tensor_s2g1_desc_buf) clReleaseMemObject(tensor_s2g1_desc_buf);
  if (tensor_s2g0_desc_buf) clReleaseMemObject(tensor_s2g0_desc_buf);
  if (tensor_g2s_desc_buf) clReleaseMemObject(tensor_g2s_desc_buf);
  return rc;
}

static void run_selected_case(const char *name, const char *filter,
                              int (*fn)(void *), void *arg,
                              size_t *pass, size_t *fail, size_t *skip)
{
  if (!selected(filter, name)) {
    (*skip)++;
    return;
  }
  if (fn(arg) == 0) {
    printf("PASS %s\n", name);
    (*pass)++;
  } else {
    printf("FAIL %s\n", name);
    (*fail)++;
  }
}

typedef struct {
  cl_context context;
  cl_command_queue queue;
  cl_kernel kernel;
} kernel_case_arg_t;

static int run_route_bundle(void *arg)
{
  kernel_case_arg_t *a = (kernel_case_arg_t *)arg;
  static const route_case_t cases[] = {
    {"s2g_routing_conflict_64B", 64, 64, 32, 0},
    {"s2g_routing_conflict_crossline", 64, 192, 64, 120},
  };
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
    if (run_route_one(a->context, a->queue, a->kernel, &cases[i]) != 0) {
      return 1;
    }
  }
  return 0;
}

static int run_group_keep1_bundle(void *arg)
{
  kernel_case_arg_t *a = (kernel_case_arg_t *)arg;
  return run_wait_domain_case(a->context, a->queue, a->kernel,
                              "dma_group_keep1_preserves_newer_s2g",
                              0x61u, 0x67010001u, 0x67010002u);
}

static int run_group_wait0_bundle(void *arg)
{
  kernel_case_arg_t *a = (kernel_case_arg_t *)arg;
  return run_wait_domain_case(a->context, a->queue, a->kernel,
                              "dma_group_wait0_drains_g2s_s2g",
                              0x73u, 0x67020001u, 0x67020002u);
}

static int run_same_fence_bundle(void *arg)
{
  kernel_case_arg_t *a = (kernel_case_arg_t *)arg;
  return run_bulk_tensor_same_fence(a->context, a->queue, a->kernel);
}

static int run_complex_bundle(void *arg)
{
  kernel_case_arg_t *a = (kernel_case_arg_t *)arg;
  return run_bidirectional_descriptor_mix(a->context, a->queue, a->kernel);
}

static int run_long_window_bundle(void *arg)
{
  kernel_case_arg_t *a = (kernel_case_arg_t *)arg;
  return run_mixed_tensor_outstanding_long(a->context, a->queue, a->kernel);
}

int s2g_mixed_routing_case_main(int argc, char **argv)
{
  cl_int err = CL_SUCCESS;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel route_kernel = NULL;
  cl_kernel group_keep1_kernel = NULL;
  cl_kernel group_wait0_kernel = NULL;
  cl_kernel same_fence_kernel = NULL;
  cl_kernel complex_kernel = NULL;
  cl_kernel long_window_kernel = NULL;
  size_t pass = 0, fail = 0, skip = 0;
  const char *filter = getenv("S2G_MIXED_CASE_FILTER");
  int rc = 1;

  if ((!filter || !filter[0]) && argc > 1) {
    filter = argv[1];
  }

  err = ventus_get_default_device(&context, &device, &queue, NULL);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device(s2g_mixed)");
  err = ventus_build_program_from_source(context, device,
                                         "s2g_mixed_routing_test.cl",
                                         &program);
  CHECK_OPENCL_ERROR_IN("ventus_build_program_from_source(s2g_mixed)");

  route_kernel = clCreateKernel(program, "s2g_routing_conflict_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(s2g_route)");
  group_keep1_kernel = clCreateKernel(program,
      "dma_group_keep1_preserves_newer_s2g_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(group_keep1)");
  group_wait0_kernel = clCreateKernel(program,
      "dma_group_wait0_drains_g2s_s2g_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(group_wait0)");
  same_fence_kernel = clCreateKernel(program,
      "bulk_tensor_s2g_same_fence_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(same_fence)");
  complex_kernel = clCreateKernel(program,
      "bidirectional_descriptor_mix_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(complex)");
  long_window_kernel = clCreateKernel(program,
      "mixed_tensor_outstanding_long_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(long_window)");

  kernel_case_arg_t route_arg = {context, queue, route_kernel};
  kernel_case_arg_t group_keep1_arg = {context, queue, group_keep1_kernel};
  kernel_case_arg_t group_wait0_arg = {context, queue, group_wait0_kernel};
  kernel_case_arg_t same_fence_arg = {context, queue, same_fence_kernel};
  kernel_case_arg_t complex_arg = {context, queue, complex_kernel};
  kernel_case_arg_t long_window_arg = {context, queue, long_window_kernel};

  run_selected_case("s2g_routing_conflict", filter, run_route_bundle,
                    &route_arg, &pass, &fail, &skip);
  run_selected_case("dma_group_keep1_preserves_newer_s2g", filter,
                    run_group_keep1_bundle, &group_keep1_arg,
                    &pass, &fail, &skip);
  run_selected_case("dma_group_wait0_drains_g2s_s2g", filter,
                    run_group_wait0_bundle, &group_wait0_arg,
                    &pass, &fail, &skip);
  run_selected_case("bulk_tensor_s2g_same_fence", filter,
                    run_same_fence_bundle, &same_fence_arg,
                    &pass, &fail, &skip);
  run_selected_case("bidirectional_descriptor_mix", filter,
                    run_complex_bundle, &complex_arg, &pass, &fail, &skip);
  run_selected_case("mixed_tensor_outstanding_long", filter,
                    run_long_window_bundle, &long_window_arg,
                    &pass, &fail, &skip);

  printf("\n=== S2G mixed routing summary ===\n");
  printf("  pass: %zu\n  fail: %zu\n  skip: %zu\n", pass, fail, skip);
  if (fail == 0 && pass > 0) {
    printf("OK\n");
    rc = 0;
  } else {
    printf("FAILED\n");
  }

FINISH:
  if (long_window_kernel) clReleaseKernel(long_window_kernel);
  if (complex_kernel) clReleaseKernel(complex_kernel);
  if (same_fence_kernel) clReleaseKernel(same_fence_kernel);
  if (group_wait0_kernel) clReleaseKernel(group_wait0_kernel);
  if (group_keep1_kernel) clReleaseKernel(group_keep1_kernel);
  if (route_kernel) clReleaseKernel(route_kernel);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
  return rc;
}
