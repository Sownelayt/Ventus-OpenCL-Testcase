include ../common/make.config

CC = clang
CFLAGS = -g -O2 -std=c11 -DCL_TARGET_OPENCL_VERSION=120 -I. -I../common -Wno-deprecated-declarations
EXE = dma_shared_routing_conflict_test.out
SRC = dma_shared_routing_conflict_test.c

.PHONY: all clean
all: $(EXE)

$(EXE): $(SRC) dma_shared_routing_conflict_test.cl ../common/ventus_opencl_test.h
	$(CC) $(CFLAGS) $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -Wno-unused-result

clean:
	rm -f *.o *~ *.out *.linkinfo object*.cl object*.dump object*.riscv object*.riscv.log object*.vmem *.data *.metadata *.log
