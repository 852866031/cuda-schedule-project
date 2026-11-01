"""Builds the colocator_kernels torch extension (tiled SGEMM).

Usage (inside the `colocator` conda env):

    cd colocator/ext && python setup.py build_ext --inplace

Produces colocator_kernels.*.so next to this file; demo/workloads.py adds this
directory to sys.path.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="colocator_kernels",
    ext_modules=[
        CUDAExtension(
            name="colocator_kernels",
            sources=["csrc/matmul.cu", "csrc/probes.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
