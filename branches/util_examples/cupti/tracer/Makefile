CUDA_HOME ?= /usr/local/cuda
CXX ?= g++
NVCC ?= $(CUDA_HOME)/bin/nvcc

CUPTI_INC := $(CUDA_HOME)/extras/CUPTI/include
CUPTI_LIB := $(CUDA_HOME)/extras/CUPTI/lib64
CUDA_INC := $(CUDA_HOME)/include
CUDA_LIB := $(CUDA_HOME)/lib64

CXXFLAGS := -O2 -std=c++17 -fPIC
NVCCFLAGS := -O2 -std=c++17
LDFLAGS := -shared -L$(CUPTI_LIB) -L$(CUDA_LIB) -lcupti -lcudart -ldl -pthread

TARGET := libactivity_tracer.so
EXAMPLE := example_workload

.PHONY: all clean run-example print-config

all: $(TARGET) $(EXAMPLE)

$(TARGET): activity_tracer.cpp
	$(CXX) $(CXXFLAGS) -I$(CUPTI_INC) -I$(CUDA_INC) activity_tracer.cpp -o $(TARGET) $(LDFLAGS)

$(EXAMPLE): example_workload.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

run-example: all
	CUPTI_TRACE_OUTDIR=output/example \
	LD_LIBRARY_PATH=$(CUPTI_LIB):$(CUDA_LIB):$$LD_LIBRARY_PATH \
	CUDA_INJECTION64_PATH=$$(pwd)/$(TARGET) \
	./$(EXAMPLE)

print-config:
	@echo "CUDA_HOME=$(CUDA_HOME)"
	@echo "CUPTI_INC=$(CUPTI_INC)"
	@echo "CUPTI_LIB=$(CUPTI_LIB)"
	@echo "CUDA_LIB=$(CUDA_LIB)"

clean:
	rm -f $(TARGET) $(EXAMPLE)
	rm -rf output
	rm *.so
	rm *.o
	rm example_workload