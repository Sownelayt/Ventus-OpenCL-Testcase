include ../common/make.config

CC = clang
CFLAGS = -g -O2 -std=c11 -DCL_TARGET_OPENCL_VERSION=120 -I. -I../common -Wno-deprecated-declarations
EXE = dma_tma_g2s_func_test.out
SRC = dma_tma_g2s_func_test.c
CASE_SRCS = g2s_bulk_basic_case.c g2s_copysize_smoke_case.c g2s_bulk_matrix_case.c g2s_tma_smoke_case.c g2s_tma_descriptor_case.c g2s_mixed_async_fence_case.c g2s_routing_conflict_case.c g2s_multi_warp_fence_case.c
CASE_OBJS = $(CASE_SRCS:.c=.o) tma_matrix_test.o
KERNEL = dma_tma_g2s_func_test.cl

.PHONY: all clean log
all: log $(EXE)

log:
	mkdir -p log

$(EXE): $(SRC) $(CASE_OBJS) $(KERNEL)
	$(CC) $(CFLAGS) $(SRC) $(CASE_OBJS) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -Wno-unused-result

$(CASE_SRCS:.c=.o): %.o: %.c dma_tma_g2s_cases.h ../common/ventus_opencl_test.h
	$(CC) $(CFLAGS) -c $< -o $@

tma_matrix_test.o: tma_matrix_test.c $(KERNEL) ../common/ventus_opencl_test.h
	$(CC) $(CFLAGS) -Dmain=tma_matrix_test_suite_main -DTMA_MATRIX_EMBEDDED -c $< -o $@

clean:
	rm -f *.o *~ *.out *.linkinfo object*.cl object*.dump object*.riscv object*.riscv.log object*.vmem *.data *.metadata *.log log/*.log log/*.csv log/*.tmp.csv log/*.stats
