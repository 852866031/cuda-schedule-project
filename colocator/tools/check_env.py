#!/usr/bin/env python
"""Phase 0 environment verifier.

Checks everything the colocator needs and exits non-zero with a clear message
on the first failure. Run inside the `colocator` conda env:

    python colocator/tools/check_env.py
"""

import os
import shutil
import subprocess
import sys

CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")

_failures = []


def check(name, ok, detail=""):
    mark = "ok " if ok else "FAIL"
    print(f"[{mark}] {name}: {detail}")
    if not ok:
        _failures.append(name)


def main():
    # Python + torch
    print(f"python: {sys.version.split()[0]} ({sys.executable})")
    try:
        import torch
        check("torch import", True, f"torch {torch.__version__}, built for CUDA {torch.version.cuda}")
        check("torch CUDA available", torch.cuda.is_available())
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"       gpu{i}: {p.name}, sm_{p.major}{p.minor}, "
                      f"{p.multi_processor_count} SMs, {p.total_memory // (1 << 20)} MiB")
    except ImportError as e:
        check("torch import", False, str(e))

    # Toolchain
    for tool in ("g++", "make", "nvcc"):
        path = shutil.which(tool) or (
            os.path.join(CUDA_HOME, "bin", tool)
            if os.path.exists(os.path.join(CUDA_HOME, "bin", tool)) else None)
        detail = path or "not found"
        if tool == "nvcc" and path:
            out = subprocess.run([path, "--version"], capture_output=True, text=True).stdout
            detail = out.strip().splitlines()[-1]
        check(tool, path is not None, detail)

    # CUPTI (headers + lib, needed by the observer in Phase 4)
    cupti_hdr = [
        os.path.join(CUDA_HOME, "include", "cupti.h"),
        os.path.join(CUDA_HOME, "targets", "x86_64-linux", "include", "cupti.h"),
        os.path.join(CUDA_HOME, "extras", "CUPTI", "include", "cupti.h"),
    ]
    cupti_lib = [
        os.path.join(CUDA_HOME, "lib64", "libcupti.so"),
        os.path.join(CUDA_HOME, "targets", "x86_64-linux", "lib", "libcupti.so"),
        os.path.join(CUDA_HOME, "extras", "CUPTI", "lib64", "libcupti.so"),
    ]
    hdr = next((p for p in cupti_hdr if os.path.exists(p)), None)
    lib = next((p for p in cupti_lib if os.path.exists(p)), None)
    check("CUPTI header", hdr is not None, hdr or f"looked under {CUDA_HOME}")
    check("CUPTI library", lib is not None, lib or f"looked under {CUDA_HOME}")

    if _failures:
        print(f"\nenvironment NOT ready: {', '.join(_failures)}")
        sys.exit(1)
    print("\nenvironment ready")


if __name__ == "__main__":
    main()
