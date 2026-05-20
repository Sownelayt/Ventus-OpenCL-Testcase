/*
 * Mixed async DMA/TMA fence directed test.
 *
 * Coverage:
 *   - basic bulk G2S + S2G with scalar noise before one fence
 *   - descriptor tensor G2S with PREFETCH_TENSORMAP and scalar/vector noise
 *   - multiple interleaved bulk G2S/S2G issues before one fence
 *   - one kernel mixing bulk G2S, bulk S2G, tensor G2S, tensor S2G,
 *     descriptor prefetch, normal ALU work, local writes, and global markers
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/ventus_opencl_test.h"

#define BULK_COPY_BYTES 128
#define TENSOR_SRC_BYTES (8 * 8 * 4)
#define TENSOR_COPY_BYTES 64
#define TENSOR_DST_BYTES (TENSOR_COPY_BYTES * 2)
#define TENSOR_GLOBAL_BYTES TENSOR_SRC_BYTES
#define COMPLEX_OUT_BULK_G2S_OFF 0
#define COMPLEX_OUT_BULK_S2G_OFF 128
#define COMPLEX_OUT_TENSOR_G2S_OFF 256
#define COMPLEX_OUT_TENSOR_S2G_OFF 320
#define COMPLEX_OUT_BYTES (COMPLEX_OUT_TENSOR_S2G_OFF + TENSOR_GLOBAL_BYTES)
#define DESC_WORDS 32
#define COORD_WORDS 32

static uint32_t
bulk_pattern_word(unsigned idx)
{
  unsigned base = idx << 2;
  unsigned b0 = (base * 7u + 0x23u) & 0xffu;
  unsigned b1 = ((base + 1u) * 7u + 0x23u) & 0xffu;
  unsigned b2 = ((base + 2u) * 7u + 0x23u) & 0xffu;
  unsigned b3 = ((base + 3u) * 7u + 0x23u) & 0xffu;
  return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

static uint32_t
stress_pattern_word(unsigned idx, unsigned seed)
{
  unsigned base = idx << 2;
  unsigned b0 = (base * 13u + seed) & 0xffu;
  unsigned b1 = ((base + 1u) * 13u + seed) & 0xffu;
  unsigned b2 = ((base + 2u) * 13u + seed) & 0xffu;
  unsigned b3 = ((base + 3u) * 13u + seed) & 0xffu;
  return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

static uint32_t
tensor_s2g_pattern_word(unsigned idx)
{
  unsigned base = idx << 2;
  unsigned b0 = (base * 7u + 3u) & 0xffu;
  unsigned b1 = ((base + 1u) * 7u + 3u) & 0xffu;
  unsigned b2 = ((base + 2u) * 7u + 3u) & 0xffu;
  unsigned b3 = ((base + 3u) * 7u + 3u) & 0xffu;
  return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

static void
build_desc(uint32_t *desc)
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

static void
fill_src(uint8_t *src, size_t n, unsigned seed)
{
  for (size_t i = 0; i < n; i++) {
    src[i] = (uint8_t)((i * 5u + seed) & 0xffu);
  }
}

static void
fill_bulk_pattern(uint8_t *dst, unsigned seed)
{
  for (unsigned i = 0; i < BULK_COPY_BYTES / 4; i++) {
    uint32_t word = stress_pattern_word(i, seed);
    memcpy(dst + i * 4, &word, sizeof(word));
  }
}

static void
compute_expected(const uint8_t *src, unsigned coord0, unsigned coord1,
                 uint8_t *expected)
{
  for (unsigned y = 0; y < 4; y++) {
    for (unsigned x = 0; x < 4; x++) {
      size_t src_off = ((coord1 + y) * 8 + (coord0 + x)) * 4;
      size_t dst_off = (y * 4 + x) * 4;
      memcpy(expected + dst_off, src + src_off, 4);
    }
  }
}

static void
fill_tensor_s2g_expected(uint8_t *expected, unsigned coord0, unsigned coord1)
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

static int
first_mismatch(const uint8_t *a, const uint8_t *b, size_t n)
{
  for (size_t i = 0; i < n; i++) {
    if (a[i] != b[i]) return (int)i;
  }
  return -1;
}

static int
check_bytes(const char *label, const uint8_t *got, const uint8_t *expected,
            size_t n)
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

static int
check_marker_xor(const char *label, const uint32_t *marker)
{
  if (marker[0] == 0xcdcdcdcdU || marker[1] == 0xcdcdcdcdU ||
      marker[2] == 0xcdcdcdcdU || marker[3] == 0xcdcdcdcdU) {
    fprintf(stderr, "FAIL %s marker left at sentinel: %08x %08x %08x %08x\n",
            label, marker[0], marker[1], marker[2], marker[3]);
    return 1;
  }
  if (marker[3] != (marker[2] ^ marker[0] ^ marker[1])) {
    fprintf(stderr, "FAIL %s marker xor: got=%08x exp=%08x\n", label,
            marker[3], marker[2] ^ marker[0] ^ marker[1]);
    return 1;
  }
  printf("PASS %s marker=%08x,%08x,%08x,%08x\n", label,
         marker[0], marker[1], marker[2], marker[3]);
  return 0;
}

static int
check_complex_marker(const uint32_t *marker)
{
  const uint32_t expected[4] = {
    0xfeed0001u, 0xfeed0002u, 0xfeed0003u, 0xfeed0004u,
  };
  for (size_t i = 0; i < 4; i++) {
    if (marker[i] != expected[i]) {
      fprintf(stderr,
              "FAIL complex_marker[%zu]: got=%08x exp=%08x\n",
              i, marker[i], expected[i]);
      return 1;
    }
  }
  printf("PASS complex_marker marker=%08x,%08x,%08x,%08x\n",
         marker[0], marker[1], marker[2], marker[3]);
  return 0;
}


static int
run_bulk_case(cl_context context, cl_command_queue queue, cl_kernel kernel)
{
  cl_int err = CL_SUCCESS;
  cl_mem src_buf = NULL;
  cl_mem verify_buf = NULL;
  cl_mem s2g_buf = NULL;
  int exit_code = 1;

  uint8_t src[BULK_COPY_BYTES];
  uint8_t verify[BULK_COPY_BYTES];
  uint8_t s2g[BULK_COPY_BYTES];
  uint8_t expected[BULK_COPY_BYTES];

  for (unsigned i = 0; i < BULK_COPY_BYTES / 4; i++) {
    uint32_t word = bulk_pattern_word(i);
    memcpy(src + i * 4, &word, sizeof(word));
    memcpy(expected + i * 4, &word, sizeof(word));
  }
  memset(verify, 0xcd, sizeof(verify));
  memset(s2g, 0xcd, sizeof(s2g));

  src_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(src), src, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(src)");
  verify_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              sizeof(verify), verify, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(verify)");
  s2g_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(s2g), s2g, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(s2g)");

  unsigned copy_bytes = BULK_COPY_BYTES;
  err  = clSetKernelArg(kernel, 0, sizeof(src_buf), &src_buf);
  err |= clSetKernelArg(kernel, 1, sizeof(verify_buf), &verify_buf);
  err |= clSetKernelArg(kernel, 2, sizeof(s2g_buf), &s2g_buf);
  err |= clSetKernelArg(kernel, 3, sizeof(copy_bytes), &copy_bytes);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(bulk)");

  size_t global = 32, local = 32;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                               0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel(bulk)");
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(bulk)");
  err = clEnqueueReadBuffer(queue, verify_buf, CL_TRUE, 0, sizeof(verify),
                            verify, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(verify)");
  err = clEnqueueReadBuffer(queue, s2g_buf, CL_TRUE, 0, sizeof(s2g),
                            s2g, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(s2g)");

  if (check_bytes("bulk_g2s_after_noise", verify, expected, sizeof(expected)) != 0) {
    goto FINISH;
  }
  if (check_bytes("bulk_s2g_after_noise", s2g, expected, sizeof(expected)) != 0) {
    goto FINISH;
  }

  exit_code = 0;

FINISH:
  if (s2g_buf) clReleaseMemObject(s2g_buf);
  if (verify_buf) clReleaseMemObject(verify_buf);
  if (src_buf) clReleaseMemObject(src_buf);
  return exit_code;
}

static int
run_tensor_case(cl_context context, cl_command_queue queue, cl_kernel kernel)
{
  cl_int err = CL_SUCCESS;
  cl_mem desc_a_buf = NULL;
  cl_mem desc_b_buf = NULL;
  cl_mem coords_a_buf = NULL;
  cl_mem coords_b_buf = NULL;
  cl_mem src_a_buf = NULL;
  cl_mem src_b_buf = NULL;
  cl_mem dst_buf = NULL;
  int exit_code = 1;

  uint32_t desc_a[DESC_WORDS];
  uint32_t desc_b[DESC_WORDS];
  uint32_t coords_a[COORD_WORDS] = {0};
  uint32_t coords_b[COORD_WORDS] = {0};
  uint8_t src_a[TENSOR_SRC_BYTES];
  uint8_t src_b[TENSOR_SRC_BYTES];
  uint8_t expected_a[TENSOR_COPY_BYTES];
  uint8_t expected_b[TENSOR_COPY_BYTES];
  uint8_t expected_dual[TENSOR_DST_BYTES];
  uint8_t got[TENSOR_DST_BYTES];

  fill_src(src_a, sizeof(src_a), 11);
  fill_src(src_b, sizeof(src_b), 0x83);

  desc_a_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
                              sizeof(desc_a), NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(desc_a)");
  desc_b_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
                              sizeof(desc_b), NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(desc_b)");
  coords_a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(coords_a), coords_a, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(coords_a)");
  coords_b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(coords_b), coords_b, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(coords_b)");
  src_a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             sizeof(src_a), src_a, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(src_a)");
  src_b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             sizeof(src_b), src_b, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(src_b)");
  dst_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
                           sizeof(got), NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(dst)");

  const struct {
    const char *name;
    unsigned coord_a0;
    unsigned coord_a1;
    unsigned coord_b0;
    unsigned coord_b1;
  } cases[] = {
    {"origin_mix", 0, 0, 2, 1},
    {"subbox_mix", 2, 1, 1, 3},
  };

  size_t global = 32, local = 32;
  for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
    build_desc(desc_a);
    build_desc(desc_b);
    memset(coords_a, 0, sizeof(coords_a));
    memset(coords_b, 0, sizeof(coords_b));
    coords_a[0] = cases[c].coord_a0;
    coords_a[1] = cases[c].coord_a1;
    coords_b[0] = cases[c].coord_b0;
    coords_b[1] = cases[c].coord_b1;
    compute_expected(src_a, cases[c].coord_a0, cases[c].coord_a1, expected_a);
    compute_expected(src_b, cases[c].coord_b0, cases[c].coord_b1, expected_b);
    memcpy(expected_dual, expected_a, TENSOR_COPY_BYTES);
    memcpy(expected_dual + TENSOR_COPY_BYTES, expected_b, TENSOR_COPY_BYTES);
    memset(got, 0xcd, sizeof(got));

    err = clEnqueueWriteBuffer(queue, desc_a_buf, CL_TRUE, 0, sizeof(desc_a),
                               desc_a, 0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueWriteBuffer(desc_a)");
    err = clEnqueueWriteBuffer(queue, desc_b_buf, CL_TRUE, 0, sizeof(desc_b),
                               desc_b, 0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueWriteBuffer(desc_b)");
    err = clEnqueueWriteBuffer(queue, coords_a_buf, CL_TRUE, 0, sizeof(coords_a),
                               coords_a, 0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueWriteBuffer(coords_a)");
    err = clEnqueueWriteBuffer(queue, coords_b_buf, CL_TRUE, 0, sizeof(coords_b),
                               coords_b, 0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueWriteBuffer(coords_b)");
    err = clEnqueueWriteBuffer(queue, src_a_buf, CL_TRUE, 0, sizeof(src_a),
                               src_a, 0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueWriteBuffer(src_a)");
    err = clEnqueueWriteBuffer(queue, src_b_buf, CL_TRUE, 0, sizeof(src_b),
                               src_b, 0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueWriteBuffer(src_b)");
    err = clEnqueueWriteBuffer(queue, dst_buf, CL_TRUE, 0, sizeof(got),
                               got, 0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueWriteBuffer(dst)");

    unsigned dst_bytes = TENSOR_DST_BYTES;
    err  = clSetKernelArg(kernel, 0, sizeof(desc_a_buf), &desc_a_buf);
    err |= clSetKernelArg(kernel, 1, sizeof(coords_a_buf), &coords_a_buf);
    err |= clSetKernelArg(kernel, 2, sizeof(src_a_buf), &src_a_buf);
    err |= clSetKernelArg(kernel, 3, sizeof(desc_b_buf), &desc_b_buf);
    err |= clSetKernelArg(kernel, 4, sizeof(coords_b_buf), &coords_b_buf);
    err |= clSetKernelArg(kernel, 5, sizeof(src_b_buf), &src_b_buf);
    err |= clSetKernelArg(kernel, 6, sizeof(dst_buf), &dst_buf);
    err |= clSetKernelArg(kernel, 7, sizeof(dst_bytes), &dst_bytes);
    CHECK_OPENCL_ERROR_IN("clSetKernelArg(tensor)");

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                                 0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel(tensor)");
    err = clFinish(queue);
    CHECK_OPENCL_ERROR_IN("clFinish(tensor)");
    err = clEnqueueReadBuffer(queue, dst_buf, CL_TRUE, 0, sizeof(got), got,
                              0, NULL, NULL);
    CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(dst)");

    if (check_bytes(cases[c].name, got, expected_dual, sizeof(expected_dual)) != 0) {
      goto FINISH;
    }
  }

  exit_code = 0;

FINISH:
  if (dst_buf) clReleaseMemObject(dst_buf);
  if (src_b_buf) clReleaseMemObject(src_b_buf);
  if (src_a_buf) clReleaseMemObject(src_a_buf);
  if (coords_b_buf) clReleaseMemObject(coords_b_buf);
  if (coords_a_buf) clReleaseMemObject(coords_a_buf);
  if (desc_b_buf) clReleaseMemObject(desc_b_buf);
  if (desc_a_buf) clReleaseMemObject(desc_a_buf);
  return exit_code;
}

static int
run_bulk_multi_issue_case(cl_context context, cl_command_queue queue,
                          cl_kernel kernel)
{
  cl_int err = CL_SUCCESS;
  cl_mem src_a_buf = NULL, src_b_buf = NULL;
  cl_mem verify_a_buf = NULL, verify_b_buf = NULL;
  cl_mem s2g_a_buf = NULL, s2g_b_buf = NULL;
  cl_mem marker_buf = NULL;
  int exit_code = 1;

  uint8_t src_a[BULK_COPY_BYTES];
  uint8_t src_b[BULK_COPY_BYTES];
  uint8_t verify_a[BULK_COPY_BYTES];
  uint8_t verify_b[BULK_COPY_BYTES];
  uint8_t s2g_a[BULK_COPY_BYTES];
  uint8_t s2g_b[BULK_COPY_BYTES];
  uint8_t expected_s2g_a[BULK_COPY_BYTES];
  uint8_t expected_s2g_b[BULK_COPY_BYTES];
  uint32_t marker[4];

  fill_src(src_a, sizeof(src_a), 0x31);
  fill_src(src_b, sizeof(src_b), 0xa7);
  fill_bulk_pattern(expected_s2g_a, 0x41);
  fill_bulk_pattern(expected_s2g_b, 0x9d);
  memset(verify_a, 0xcd, sizeof(verify_a));
  memset(verify_b, 0xcd, sizeof(verify_b));
  memset(s2g_a, 0xcd, sizeof(s2g_a));
  memset(s2g_b, 0xcd, sizeof(s2g_b));
  for (size_t i = 0; i < 4; i++) marker[i] = 0xcdcdcdcdU;

  src_a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             sizeof(src_a), src_a, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(src_a)");
  src_b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             sizeof(src_b), src_b, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(src_b)");
  verify_a_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                sizeof(verify_a), verify_a, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(verify_a)");
  verify_b_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                sizeof(verify_b), verify_b, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(verify_b)");
  s2g_a_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                             sizeof(s2g_a), s2g_a, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(s2g_a)");
  s2g_b_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                             sizeof(s2g_b), s2g_b, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(s2g_b)");
  marker_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              sizeof(marker), marker, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(marker)");

  unsigned copy_bytes = BULK_COPY_BYTES;
  err  = clSetKernelArg(kernel, 0, sizeof(src_a_buf), &src_a_buf);
  err |= clSetKernelArg(kernel, 1, sizeof(src_b_buf), &src_b_buf);
  err |= clSetKernelArg(kernel, 2, sizeof(verify_a_buf), &verify_a_buf);
  err |= clSetKernelArg(kernel, 3, sizeof(verify_b_buf), &verify_b_buf);
  err |= clSetKernelArg(kernel, 4, sizeof(s2g_a_buf), &s2g_a_buf);
  err |= clSetKernelArg(kernel, 5, sizeof(s2g_b_buf), &s2g_b_buf);
  err |= clSetKernelArg(kernel, 6, sizeof(marker_buf), &marker_buf);
  err |= clSetKernelArg(kernel, 7, sizeof(copy_bytes), &copy_bytes);
  CHECK_OPENCL_ERROR_IN("clSetKernelArg(bulk_multi)");

  size_t global = 32, local = 32;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                               0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueNDRangeKernel(bulk_multi)");
  err = clFinish(queue);
  CHECK_OPENCL_ERROR_IN("clFinish(bulk_multi)");
  err = clEnqueueReadBuffer(queue, verify_a_buf, CL_TRUE, 0, sizeof(verify_a),
                            verify_a, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(verify_a)");
  err = clEnqueueReadBuffer(queue, verify_b_buf, CL_TRUE, 0, sizeof(verify_b),
                            verify_b, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(verify_b)");
  err = clEnqueueReadBuffer(queue, s2g_a_buf, CL_TRUE, 0, sizeof(s2g_a),
                            s2g_a, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(s2g_a)");
  err = clEnqueueReadBuffer(queue, s2g_b_buf, CL_TRUE, 0, sizeof(s2g_b),
                            s2g_b, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(s2g_b)");
  err = clEnqueueReadBuffer(queue, marker_buf, CL_TRUE, 0, sizeof(marker),
                            marker, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(marker)");

  if (check_bytes("bulk_multi_g2s_a", verify_a, src_a, sizeof(src_a)) != 0) goto FINISH;
  if (check_bytes("bulk_multi_g2s_b", verify_b, src_b, sizeof(src_b)) != 0) goto FINISH;
  if (check_bytes("bulk_multi_s2g_a", s2g_a, expected_s2g_a, sizeof(s2g_a)) != 0) goto FINISH;
  if (check_bytes("bulk_multi_s2g_b", s2g_b, expected_s2g_b, sizeof(s2g_b)) != 0) goto FINISH;
  if (check_marker_xor("bulk_multi_marker", marker) != 0) goto FINISH;

  exit_code = 0;

FINISH:
  if (marker_buf) clReleaseMemObject(marker_buf);
  if (s2g_b_buf) clReleaseMemObject(s2g_b_buf);
  if (s2g_a_buf) clReleaseMemObject(s2g_a_buf);
  if (verify_b_buf) clReleaseMemObject(verify_b_buf);
  if (verify_a_buf) clReleaseMemObject(verify_a_buf);
  if (src_b_buf) clReleaseMemObject(src_b_buf);
  if (src_a_buf) clReleaseMemObject(src_a_buf);
  return exit_code;
}

static int
run_tensor_bulk_complex_case(cl_context context, cl_command_queue queue,
                             cl_kernel kernel)
{
  cl_int err = CL_SUCCESS;
  cl_mem tensor_g2s_desc_buf = NULL, tensor_s2g_desc_buf = NULL;
  cl_mem tensor_g2s_coords_buf = NULL, tensor_s2g_coords_buf = NULL;
  cl_mem tensor_src_buf = NULL, bulk_src_buf = NULL;
  cl_mem out_buf = NULL, marker_buf = NULL;
  int exit_code = 1;

  uint32_t tensor_g2s_desc[DESC_WORDS];
  uint32_t tensor_s2g_desc[DESC_WORDS];
  uint32_t tensor_g2s_coords[COORD_WORDS] = {0};
  uint32_t tensor_s2g_coords[COORD_WORDS] = {0};
  uint8_t tensor_src[TENSOR_SRC_BYTES];
  uint8_t out[COMPLEX_OUT_BYTES];
  uint8_t tensor_expected[TENSOR_COPY_BYTES];
  uint8_t tensor_s2g_expected[TENSOR_GLOBAL_BYTES];
  uint8_t bulk_src[BULK_COPY_BYTES];
  uint8_t bulk_s2g_expected[BULK_COPY_BYTES];
  uint32_t marker[4];

  const unsigned tensor_g2s_coord0 = 1;
  const unsigned tensor_g2s_coord1 = 2;
  const unsigned tensor_s2g_coord0 = 2;
  const unsigned tensor_s2g_coord1 = 1;

  build_desc(tensor_g2s_desc);
  build_desc(tensor_s2g_desc);
  tensor_g2s_coords[0] = tensor_g2s_coord0;
  tensor_g2s_coords[1] = tensor_g2s_coord1;
  tensor_s2g_coords[0] = tensor_s2g_coord0;
  tensor_s2g_coords[1] = tensor_s2g_coord1;
  fill_src(tensor_src, sizeof(tensor_src), 0x67);
  compute_expected(tensor_src, tensor_g2s_coord0, tensor_g2s_coord1,
                   tensor_expected);
  fill_tensor_s2g_expected(tensor_s2g_expected, tensor_s2g_coord0,
                           tensor_s2g_coord1);
  memset(out, 0xcd, sizeof(out));
  fill_src(bulk_src, sizeof(bulk_src), 0xb5);
  fill_bulk_pattern(bulk_s2g_expected, 0x55);
  for (size_t i = 0; i < 4; i++) marker[i] = 0xcdcdcdcdU;

  tensor_g2s_desc_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                       sizeof(tensor_g2s_desc), tensor_g2s_desc, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(tensor_g2s_desc)");
  tensor_s2g_desc_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                       sizeof(tensor_s2g_desc), tensor_s2g_desc, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(tensor_s2g_desc)");
  tensor_g2s_coords_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(tensor_g2s_coords), tensor_g2s_coords, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(tensor_g2s_coords)");
  tensor_s2g_coords_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(tensor_s2g_coords), tensor_s2g_coords, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(tensor_s2g_coords)");
  tensor_src_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(tensor_src), tensor_src, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(tensor_src)");
  bulk_src_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(bulk_src), bulk_src, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(bulk_src)");
  out_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(out), out, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(out)");
  marker_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              sizeof(marker), marker, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer(marker)");

  unsigned bulk_bytes = BULK_COPY_BYTES;
  err  = clSetKernelArg(kernel, 0, sizeof(tensor_g2s_desc_buf), &tensor_g2s_desc_buf);
  err |= clSetKernelArg(kernel, 1, sizeof(tensor_g2s_coords_buf), &tensor_g2s_coords_buf);
  err |= clSetKernelArg(kernel, 2, sizeof(tensor_src_buf), &tensor_src_buf);
  err |= clSetKernelArg(kernel, 3, sizeof(tensor_s2g_desc_buf), &tensor_s2g_desc_buf);
  err |= clSetKernelArg(kernel, 4, sizeof(tensor_s2g_coords_buf), &tensor_s2g_coords_buf);
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
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(out)");
  err = clEnqueueReadBuffer(queue, marker_buf, CL_TRUE, 0, sizeof(marker),
                            marker, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("clEnqueueReadBuffer(marker)");

  if (check_bytes("complex_bulk_g2s",
                  out + COMPLEX_OUT_BULK_G2S_OFF, bulk_src,
                  sizeof(bulk_src)) != 0) goto FINISH;
  if (check_bytes("complex_bulk_s2g",
                  out + COMPLEX_OUT_BULK_S2G_OFF, bulk_s2g_expected,
                  sizeof(bulk_s2g_expected)) != 0) goto FINISH;
  if (check_bytes("complex_tensor_g2s",
                  out + COMPLEX_OUT_TENSOR_G2S_OFF, tensor_expected,
                  sizeof(tensor_expected)) != 0) goto FINISH;
  if (check_bytes("complex_tensor_s2g",
                  out + COMPLEX_OUT_TENSOR_S2G_OFF, tensor_s2g_expected,
                  sizeof(tensor_s2g_expected)) != 0) goto FINISH;
  if (check_complex_marker(marker) != 0) goto FINISH;

  exit_code = 0;

FINISH:
  if (marker_buf) clReleaseMemObject(marker_buf);
  if (out_buf) clReleaseMemObject(out_buf);
  if (bulk_src_buf) clReleaseMemObject(bulk_src_buf);
  if (tensor_src_buf) clReleaseMemObject(tensor_src_buf);
  if (tensor_s2g_coords_buf) clReleaseMemObject(tensor_s2g_coords_buf);
  if (tensor_g2s_coords_buf) clReleaseMemObject(tensor_g2s_coords_buf);
  if (tensor_s2g_desc_buf) clReleaseMemObject(tensor_s2g_desc_buf);
  if (tensor_g2s_desc_buf) clReleaseMemObject(tensor_g2s_desc_buf);
  return exit_code;
}


int
main(void)
{
  cl_int err = CL_SUCCESS;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel bulk_kernel = NULL;
  cl_kernel tensor_kernel = NULL;
  cl_kernel bulk_multi_kernel = NULL;
  cl_kernel complex_kernel = NULL;
  int bulk_rc = 1;
  int tensor_rc = 1;
  int bulk_multi_rc = 1;
  int complex_rc = 1;

  err = ventus_get_default_device(&context, &device, &queue, NULL);
  CHECK_OPENCL_ERROR_IN("ventus_get_default_device");
  err = ventus_build_program_from_source(context, device,
                                         "mixed_async_fence_test.cl", &program);
  CHECK_OPENCL_ERROR_IN("ventus_build_program_from_source");
  bulk_kernel = clCreateKernel(program, "bulk_async_fence_mixed_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(bulk)");
  tensor_kernel = clCreateKernel(program, "tensor_async_fence_mixed_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(tensor)");
  bulk_multi_kernel = clCreateKernel(program, "bulk_multi_issue_fence_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(bulk_multi)");
  complex_kernel = clCreateKernel(program, "tensor_bulk_bidirectional_fence_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel(complex)");

  bulk_rc = run_bulk_case(context, queue, bulk_kernel);
  tensor_rc = run_tensor_case(context, queue, tensor_kernel);
  bulk_multi_rc = run_bulk_multi_issue_case(context, queue, bulk_multi_kernel);
  complex_rc = run_tensor_bulk_complex_case(context, queue, complex_kernel);

  printf("\n=== mixed async fence summary ===\n");
  printf("  bulk       : %s\n", bulk_rc == 0 ? "PASS" : "FAIL");
  printf("  tensor     : %s\n", tensor_rc == 0 ? "PASS" : "FAIL");
  printf("  bulk_multi : %s\n", bulk_multi_rc == 0 ? "PASS" : "FAIL");
  printf("  complex    : %s\n", complex_rc == 0 ? "PASS" : "FAIL");
  if (bulk_rc == 0 && tensor_rc == 0 && bulk_multi_rc == 0 &&
      complex_rc == 0) {
    printf("OK\n");
  } else {
    printf("FAILED\n");
  }

FINISH:
  if (complex_kernel) clReleaseKernel(complex_kernel);
  if (bulk_multi_kernel) clReleaseKernel(bulk_multi_kernel);
  if (tensor_kernel) clReleaseKernel(tensor_kernel);
  if (bulk_kernel) clReleaseKernel(bulk_kernel);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
  return (bulk_rc == 0 && tensor_rc == 0 && bulk_multi_rc == 0 &&
          complex_rc == 0) ? 0 : 1;
}
