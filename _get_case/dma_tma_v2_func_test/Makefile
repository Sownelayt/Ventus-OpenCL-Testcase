include ../common/make.config

CXX = clang++
CXXFLAGS = -g -O2 -std=c++17 -DCL_TARGET_OPENCL_VERSION=120 -I. -I../common -Wno-deprecated-declarations
EXE = dma_tma_v2_func_test.out
SRC = dma_tma_v2_func_test.cpp
KERNEL = dma_tma_v2_func_test.cl

.PHONY: all clean
all: $(EXE)

$(EXE): $(SRC) $(KERNEL) ../common/ventus_opencl_test.h ../common/tma_model.h \
	../common/tma_model.cc ../common/ventus_tma_v2_spec.h \
	../common/ventus_tma_v2_opencl.h
	$(CXX) $(CXXFLAGS) $(SRC) ../common/tma_model.cc -o $(EXE) \
		-I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -Wno-unused-result

clean:
	rm -f *.o *~ *.out *.linkinfo object*.cl object*.dump object*.riscv \
		object*.riscv.log object*.vmem *.data *.metadata *.log
