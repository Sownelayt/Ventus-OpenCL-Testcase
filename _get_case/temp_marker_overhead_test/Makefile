include ../common/make.config

CC = clang
CFLAGS = -g -O2 -std=c11 -DCL_TARGET_OPENCL_VERSION=120 -I. -I../common -Wno-deprecated-declarations
EXE = temp_marker_overhead_test.out
SRC = temp_marker_overhead_test.c
KERNEL = temp_marker_overhead_test.cl

.PHONY: all clean log
all: log $(EXE)

log:
	mkdir -p log

$(EXE): $(SRC) $(KERNEL) ../common/ventus_opencl_test.h
	$(CC) $(CFLAGS) $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -Wno-unused-result

clean:
	rm -rf log
	rm -f *.o *~ *.out *.linkinfo object*.cl object*.dump object*.riscv object*.riscv.log object*.vmem *.data *.metadata *.log
