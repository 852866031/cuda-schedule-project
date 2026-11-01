PYTHON ?= $(shell which python)
OUT_DIR := outputs

NCU ?= $(shell which ncu)
NCU_SET ?= basic
NCU_PREFIX ?=

BAD_REPORT := $(OUT_DIR)/bad_gather_report
DENSE_REPORT := $(OUT_DIR)/dense_fma_report

.PHONY: run plot ncu-gather ncu-dense ncu clean print-python print-ncu

run:
	@if [ -z "$(PYTHON)" ]; then echo "python not found in PATH"; exit 1; fi
	$(PYTHON) dcgm_run.py

plot:
	@if [ -z "$(PYTHON)" ]; then echo "python not found in PATH"; exit 1; fi
	$(PYTHON) plot_dcgm.py

ncu-gather:
	@if [ -z "$(NCU)" ]; then echo "ncu not found in PATH"; exit 1; fi
	@if [ -z "$(PYTHON)" ]; then echo "python not found in PATH"; exit 1; fi
	mkdir -p $(OUT_DIR)
	sudo $(NCU_PREFIX) $(NCU) --target-processes all \
		--kernel-name bad_gather_kernel \
		--launch-count 1 \
		--set $(NCU_SET) \
		-o $(BAD_REPORT) \
		$(PYTHON) workload_two_gpu.py

ncu-dense:
	@if [ -z "$(NCU)" ]; then echo "ncu not found in PATH"; exit 1; fi
	@if [ -z "$(PYTHON)" ]; then echo "python not found in PATH"; exit 1; fi
	mkdir -p $(OUT_DIR)
	sudo $(NCU_PREFIX) $(NCU) --target-processes all \
		--kernel-name dense_fma_kernel \
		--launch-count 1 \
		--set $(NCU_SET) \
		-o $(DENSE_REPORT) \
		$(PYTHON) workload_two_gpu.py

ncu: ncu-gather ncu-dense

print-python:
	@echo $(PYTHON)

print-ncu:
	@echo $(NCU)

clean:
	rm -rf $(OUT_DIR)/*.ncu-rep $(OUT_DIR)/*.csv $(OUT_DIR)/*.png $(OUT_DIR)/*.log



