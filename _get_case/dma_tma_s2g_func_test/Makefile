include ../common/make.config

CC = clang
CFLAGS = -g -O2 -std=c11 -DCL_TARGET_OPENCL_VERSION=120 -I. -I../common -Wno-deprecated-declarations
EXE = dma_tma_s2g_func_test.out
SRC = dma_tma_s2g_func_test.c
CASE_SRCS = s2g_bulk_matrix_case.c s2g_tensor_matrix_case.c
CASE_OBJS = $(CASE_SRCS:.c=.o)
KERNELS = shared_to_global_dma_test.cl tensor_shared_to_global_test.cl

.PHONY: all clean log
all: log $(EXE)

log:
	mkdir -p log

$(EXE): $(SRC) $(CASE_OBJS) $(KERNELS)
	$(CC) $(CFLAGS) $(SRC) $(CASE_OBJS) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -Wno-unused-result

$(CASE_OBJS): %.o: %.c dma_tma_s2g_cases.h ../common/ventus_opencl_test.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o *~ *.out *.linkinfo object*.cl object*.dump object*.riscv object*.riscv.log object*.vmem *.data *.metadata *.log log/*.log log/*.csv log/*.tmp.csv log/*.stats
