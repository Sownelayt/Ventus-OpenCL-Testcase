CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Werror
COMMON := ../common
TARGET := dma_tma_cmodel_test.out

.PHONY: all check golden clean

all: $(TARGET)

$(TARGET): dma_tma_cmodel_test.cpp $(COMMON)/tma_model.cc $(COMMON)/tma_model.h $(COMMON)/ventus_tma_v2_spec.h
	$(CXX) $(CXXFLAGS) -I$(COMMON) dma_tma_cmodel_test.cpp $(COMMON)/tma_model.cc -o $@

check: $(TARGET)
	./$(TARGET)

golden: $(TARGET)
	./$(TARGET) --emit-trace ../../../gpgpu/ventus/tests/resources/tma_v2_atom_trace.csv

clean:
	rm -f $(TARGET)
