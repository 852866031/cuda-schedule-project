# Builds the colocator's C++ shared libraries into build/.
# Requires: g++, CUDA toolkit at $(CUDA_HOME) (headers only for the intercept lib).
#
# Targets:
#   make            -> all libs implemented so far
#   make intercept  -> build/libcolocator.so   (LD_PRELOAD interposition layer)
#   make sched      -> build/libsched.so       (scheduler core; Phase 3)
#   make observer   -> build/libobserver.so    (CUPTI observer; Phase 4)

CUDA_HOME ?= /usr/local/cuda
BUILD     := build
CXX       := g++
CXXFLAGS  := -O2 -g -fPIC -Wall -std=c++17 -I$(CUDA_HOME)/include

.PHONY: all intercept sched observer clean

all: intercept sched observer

intercept: $(BUILD)/libcolocator.so

sched: $(BUILD)/libsched.so

observer: $(BUILD)/libobserver.so

$(BUILD)/libobserver.so: src/observer/observer.cpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -shared src/observer/observer.cpp -o $@ \
		-L$(CUDA_HOME)/lib64 -lcupti -Wl,-rpath,$(CUDA_HOME)/lib64 -lpthread

$(BUILD)/libcolocator.so: src/intercept/intercept.cpp src/common/records.h
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -shared src/intercept/intercept.cpp -o $@ -ldl -lpthread

# libsched references globals defined in libcolocator.so; they resolve at load
# time because libcolocator is LD_PRELOADed. -lcudart binds by SONAME to the
# already-loaded copy (torch's), and CUDA calls from the scheduler thread pass
# through the preloaded interposers untouched (unregistered thread).
# -lcuda (driver API, via the toolkit stubs at link time; the real
# libcuda.so.1 from the driver resolves at load) is needed for the green
# context calls behind COLOCATOR_SM_LIMIT.
$(BUILD)/libsched.so: src/sched/scheduler.cpp src/sched/policy.h src/common/records.h
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -shared src/sched/scheduler.cpp -o $@ \
		-L$(CUDA_HOME)/lib64 -lcudart -L$(CUDA_HOME)/lib64/stubs -lcuda -ldl -lpthread

clean:
	rm -rf $(BUILD)
