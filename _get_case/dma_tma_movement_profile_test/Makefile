include ../common/make.config

CC = clang
CFLAGS = -g -O2 -std=c11 -DCL_TARGET_OPENCL_VERSION=120 -I. -I../common -Wno-deprecated-declarations
EXE = dma_tma_movement_profile_test.out
SRC = dma_tma_movement_profile_test.c
KERNEL = dma_tma_movement_profile_test.cl

.PHONY: all clean log
all: log $(EXE)

log:
	mkdir -p log

$(EXE): $(SRC) $(KERNEL) ../common/ventus_opencl_test.h \
	../common/ventus_tma_v2_opencl.h
	$(CC) $(CFLAGS) $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -Wno-unused-result

clean:
	rm -rf log/dma_tma_movement_profile_run_* log/dma_tma_movement_profile_child_*.log log/dma_tma_movement_profile_report_*.md
	rm -rf log/dma_tma_movement_profile_sameskeleton_run_* log/dma_tma_movement_profile_sameskeleton_child_*.log log/dma_tma_movement_profile_sameskeleton_report_*.md
	rm -f *.o *~ *.out *.linkinfo object*.cl object*.dump object*.riscv object*.riscv.log object*.vmem *.data *.metadata *.log
