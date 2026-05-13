#ifndef VENTUS_OPENCL_TEST_H
#define VENTUS_OPENCL_TEST_H

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_OPENCL_ERROR_IN(where)                                           \
  do {                                                                         \
    if (err != CL_SUCCESS) {                                                   \
      fprintf(stderr, "%s failed with OpenCL error %d\n", (where), err);       \
      goto FINISH;                                                             \
    }                                                                          \
  } while (0)

static char *
ventus_read_text_file(const char *path, size_t *size_out)
{
  FILE *fp = fopen(path, "rb");
  if (!fp) {
    perror(path);
    return NULL;
  }

  if (fseek(fp, 0, SEEK_END) != 0) {
    perror("fseek");
    fclose(fp);
    return NULL;
  }

  long len = ftell(fp);
  if (len < 0) {
    perror("ftell");
    fclose(fp);
    return NULL;
  }
  rewind(fp);

  char *buf = (char *)malloc((size_t)len + 1);
  if (!buf) {
    fclose(fp);
    return NULL;
  }

  size_t got = fread(buf, 1, (size_t)len, fp);
  fclose(fp);
  if (got != (size_t)len) {
    free(buf);
    return NULL;
  }

  buf[len] = '\0';
  if (size_out) *size_out = (size_t)len;
  return buf;
}

static void
ventus_print_build_log(cl_program program, cl_device_id device)
{
  size_t log_size = 0;
  cl_int err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                     0, NULL, &log_size);
  if (err != CL_SUCCESS || log_size == 0) return;

  char *log = (char *)malloc(log_size + 1);
  if (!log) return;

  err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              log_size, log, NULL);
  if (err == CL_SUCCESS) {
    log[log_size] = '\0';
    fprintf(stderr, "%s\n", log);
  }
  free(log);
}

static cl_int
ventus_get_default_device(cl_context *context, cl_device_id *device,
                          cl_command_queue *queue, cl_platform_id *platform)
{
  cl_int err;
  cl_platform_id platform_id = NULL;
  cl_device_id device_id = NULL;

  err = clGetPlatformIDs(1, &platform_id, NULL);
  if (err != CL_SUCCESS) return err;

  err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);
  if (err != CL_SUCCESS) return err;

  cl_context ctx = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
  if (err != CL_SUCCESS) return err;

  cl_command_queue q = clCreateCommandQueue(ctx, device_id, 0, &err);
  if (err != CL_SUCCESS) {
    clReleaseContext(ctx);
    return err;
  }

  if (platform) *platform = platform_id;
  *device = device_id;
  *context = ctx;
  *queue = q;
  return CL_SUCCESS;
}

static cl_int
ventus_build_program_from_source(cl_context context, cl_device_id device,
                                 const char *source_path, cl_program *program)
{
  size_t source_size = 0;
  char *source = ventus_read_text_file(source_path, &source_size);
  if (!source) return CL_INVALID_PROGRAM;

  cl_int err;
  const char *sources[] = {source};
  const size_t sizes[] = {source_size};
  cl_program prog = clCreateProgramWithSource(context, 1, sources, sizes, &err);
  free(source);
  if (err != CL_SUCCESS) return err;

  err = clBuildProgram(prog, 1, &device, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    ventus_print_build_log(prog, device);
    clReleaseProgram(prog);
    return err;
  }

  *program = prog;
  return CL_SUCCESS;
}

#endif
