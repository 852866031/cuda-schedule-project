CUDA_HOME ?= /usr/local/cuda
CXX ?= g++

CUDA_INC := $(CUDA_HOME)/include
CUDA_LIB := $(CUDA_HOME)/lib64

CXXFLAGS := -O2 -std=c++17 -fPIC -I$(CUDA_INC)
LDFLAGS  := -shared -L$(CUDA_LIB) -lcupti -lcudart -lnvperf_host -lnvperf_target -ldl -pthread

TARGET := libauto_profiler.so

SRCS := auto_profiler.cpp nvpw_metrics.cpp tracing.cpp
OBJS := $(SRCS:.cpp=.o)

.PHONY: all clean print-config

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp profiler_common.h nvpw_metrics.h tracing.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

print-config:
	@echo "CUDA_HOME=$(CUDA_HOME)"
	@echo "CUDA_INC=$(CUDA_INC)"
	@echo "CUDA_LIB=$(CUDA_LIB)"

clean:
	rm -f $(TARGET) $(OBJS)
