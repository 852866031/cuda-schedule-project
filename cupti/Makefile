PYTHON ?= $(shell which python)
CUDA_HOME ?= /usr/local/cuda

CUPTI_LIB := $(CUDA_HOME)/extras/CUPTI/lib64
CUDA_LIB := $(CUDA_HOME)/lib64

ROOT_OUT := output
TRACER_DIR := tracer
PROFILER_DIR := profiler
LLM_DIR := llm_app

TRACER_SO := $(TRACER_DIR)/libactivity_tracer.so
PROFILER_SO := $(PROFILER_DIR)/libauto_profiler.so

LLM_METRICS := $(ROOT_OUT)/llm_metrics.json
LLM_TRACE_METRICS := $(ROOT_OUT)/llm_trace_metrics.json
LLM_PROFILE_METRICS := $(ROOT_OUT)/llm_profile_metrics.json
BENCH_CSV := $(ROOT_OUT)/overhead_compare.csv
BENCH_PROFILER_CSV := $(ROOT_OUT)/overhead_compare_profiler.csv

.PHONY: all tracer profiler llm llm-trace llm-profile compare compare-profiler plot clean

all: tracer profiler

tracer:
	$(MAKE) -C $(TRACER_DIR)

profiler:
	$(MAKE) -C $(PROFILER_DIR)

llm:
	mkdir -p $(ROOT_OUT)
	cd $(LLM_DIR) && \
	$(PYTHON) run_llm.py --metrics-output ../$(LLM_METRICS)

llm-trace: tracer
	mkdir -p $(ROOT_OUT)
	cd $(LLM_DIR) && \
	CUPTI_TRACE_OUTDIR=../$(ROOT_OUT) \
	LD_LIBRARY_PATH=$(CUPTI_LIB):$(CUDA_LIB):$$LD_LIBRARY_PATH \
	CUDA_INJECTION64_PATH=$$(pwd)/../$(TRACER_SO) \
	$(PYTHON) run_llm.py --metrics-output ../$(LLM_TRACE_METRICS)

llm-profile: profiler
	mkdir -p $(ROOT_OUT)
	cd $(LLM_DIR) && \
	sudo -E \
	CUPTI_TRACE_OUTDIR=../$(ROOT_OUT) \
	LD_LIBRARY_PATH=$(CUPTI_LIB):$(CUDA_LIB):$$LD_LIBRARY_PATH \
	CUDA_INJECTION64_PATH=$$(pwd)/../$(PROFILER_SO) \
	$(PYTHON) run_llm.py --metrics-output ../$(LLM_PROFILE_METRICS)

plot:
	$(PYTHON) plot_all.py

clean:
	-$(MAKE) -C $(PROFILER_DIR) clean
	-$(MAKE) -C $(TRACER_DIR) clean