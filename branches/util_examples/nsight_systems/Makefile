PYTHON ?= $(shell which python)
NSYS ?= $(shell which nsys)

OUT_DIR := outputs
NSYS_OUT := $(OUT_DIR)/inf_sys

.PHONY: run nsys nvml dcgm plot-nvml plot-dcgm clean

run:
	@if [ -z "$(PYTHON)" ]; then echo "python not found in PATH"; exit 1; fi
	$(PYTHON) inf_sys.py

nvml:
	@if [ -z "$(PYTHON)" ]; then echo "python not found in PATH"; exit 1; fi
	mkdir -p $(OUT_DIR)
	$(PYTHON) nvml_run.py

dcgm:
	@if [ -z "$(PYTHON)" ]; then echo "python not found in PATH"; exit 1; fi
	mkdir -p $(OUT_DIR)
	$(PYTHON) dcgm_run.py

plot-all:
	@if [ -z "$(PYTHON)" ]; then echo "python not found in PATH"; exit 1; fi
	$(PYTHON) plot_all.py
	
nsys:
	@if [ -z "$(NSYS)" ]; then echo "nsys not found in PATH"; exit 1; fi
	@if [ -z "$(PYTHON)" ]; then echo "python not found in PATH"; exit 1; fi
	mkdir -p $(OUT_DIR)
	$(NSYS) profile --trace=cuda,nvtx,osrt --sample=none -o $(NSYS_OUT) $(PYTHON) inf_sys.py

clean:
	rm -rf $(OUT_DIR)