include ../common/make.config

CC = clang
CFLAGS = -g -O2 -std=c11 -DCL_TARGET_OPENCL_VERSION=120 -I. -I../common -Wno-deprecated-declarations
EXE = dma_tma_tensor_feature_perf_test.out
SRC = dma_tma_tensor_feature_perf_test.c
KERNEL = dma_tma_tensor_feature_perf_test.cl

.PHONY: all clean log
all: log $(EXE)

log:
	mkdir -p log

$(EXE): $(SRC) $(KERNEL) ../common/ventus_opencl_test.h ../common/ventus_tma_v2_opencl.h ../common/ventus_tma_v2_spec.h
	$(CC) $(CFLAGS) $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -Wno-unused-result

clean:
	rm -rf log/dma_tma_tensor_feature_perf_*.log
	rm -f *.o *~ *.out *.linkinfo object*.cl object*.dump object*.riscv object*.riscv.log object*.vmem *.data *.metadata *.log
