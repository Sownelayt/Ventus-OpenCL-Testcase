include ../common/make.config

CC = clang
CFLAGS = -g -O2 -std=c11 -DCL_TARGET_OPENCL_VERSION=120 -I. -I../common -Wno-deprecated-declarations
EXE = dma_tma_multi_wg_func_test.out
SRC = dma_tma_multi_wg_func_test.c
CASE_SRCS = multi_wg_bulk_g2s_case.c
CASE_OBJS = $(CASE_SRCS:.c=.o)
KERNELS = multi_wg_dma_test.cl multi_wg_dma_s2g_test.cl multi_wg_tma_g2s_test.cl multi_wg_tma_s2g_test.cl multi_wg_tma_s2g_subbox_test.cl multi_wg_tma_s2g_stride_oob_test.cl multi_wg_tma_s2g_descriptor_mix_test.cl multi_wg_tma_s2g_group_page_test.cl

.PHONY: all clean log
all: log $(EXE)

log:
	mkdir -p log

$(EXE): $(SRC) $(CASE_OBJS) $(KERNELS)
	$(CC) $(CFLAGS) $(SRC) $(CASE_OBJS) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -Wno-unused-result

$(CASE_OBJS): %.o: %.c dma_tma_multi_wg_cases.h ../common/ventus_opencl_test.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o *~ *.out *.linkinfo object*.cl object*.dump object*.riscv object*.riscv.log object*.vmem *.data *.metadata *.log log/*.log log/*.csv log/*.tmp.csv log/*.stats
