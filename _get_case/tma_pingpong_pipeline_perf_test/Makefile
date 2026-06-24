include ../common/make.config

CC = clang
CFLAGS = -g -O2 -std=c11 -DCL_TARGET_OPENCL_VERSION=120 -I. -I../common -Wno-deprecated-declarations
EXE = tma_pingpong_pipeline_perf_test.out
SRC = tma_pingpong_pipeline_perf_test.c

.PHONY: all clean
all: $(EXE)

$(EXE): $(SRC) tma_pingpong_pipeline_perf_test.cl ../common/ventus_opencl_test.h
	$(CC) $(CFLAGS) $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -Wno-unused-result

clean:
	rm -rf tma_pingpong_pipeline_perf_run_* tma_pingpong_pipeline_perf_logs
	rm -f tma_pingpong_pipeline_perf_report_*.md
	rm -f *.o *~ *.out *.linkinfo object*.cl object*.dump object*.riscv object*.riscv.log object*.vmem *.data *.metadata *.log
