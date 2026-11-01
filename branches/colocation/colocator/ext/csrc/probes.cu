// Resource-probe kernels, ported from the gpu-interfere profiler's probe.cu
// (~/Documents/Projects/gpu-interfere/profiler/code/probe.cu). Each stresses
// ~one shared GPU resource; the demo uses them as colocation workloads.
// Trace machinery from the original is dropped — the colocator's CUPTI
// observer provides timing. All launches go through cudaLaunchKernel on the
// caller's current torch stream, so the colocator intercepts them.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace {

// scheduler/occupancy only: nanosleep loop, no memory or compute
__global__ void k_sleep(long long it) {
    for (long long i = 0; i < it; i++) asm volatile("nanosleep.u32 1000;");
}

// grid-stride streaming copy: DRAM bandwidth (large n) or L2 bandwidth
// (n small enough to be L2-resident)
__global__ void k_copy(const float* __restrict__ in, float* __restrict__ out,
                       long long n, long long it) {
    size_t s = blockIdx.x * blockDim.x + threadIdx.x;
    size_t st = (size_t)gridDim.x * blockDim.x;
    for (long long j = 0; j < it; j++)
        for (size_t i = s; i < (size_t)n; i += st) out[i] = in[i];
}

// per-block copy of a small region: L1-cache-resident working set per SM
__global__ void k_copy_tb(const float* __restrict__ in, float* __restrict__ out,
                          long long npb, long long it, int region_bytes) {
    int fpr = region_bytes / 4;
    int rpt = (npb + fpr - 1) / fpr;
    int b0 = blockIdx.x * rpt * fpr, be = b0 + npb;
    for (long long i = 0; i < it; i++)
        for (int j = b0 + threadIdx.x; j < be; j += blockDim.x) out[j] = in[j];
}

// high-ILP FP32 FMA loop: FMA pipe + issue slots
__global__ void k_fma32(const float* a, const float* b, float* c, long long it) {
    float o1 = a[threadIdx.x], o2 = b[threadIdx.x], x = 0, y = 0, z = 0, w = 0;
    for (long long i = 0; i < it; i++) {
        x = __fmaf_rn(o1, o2, x); y = __fmaf_rn(o1, o2, y);
        z = __fmaf_rn(o1, o2, z); w = __fmaf_rn(o1, o2, w);
    }
    c[threadIdx.x] = x + y + z + w;
}

// high-ILP FP64 FMA loop: FP64 pipe (heavily cut down on GeForce)
__global__ void k_fma64(const double* a, const double* b, double* c, long long it) {
    double o1 = a[threadIdx.x], o2 = b[threadIdx.x], x = 0, y = 0, z = 0, w = 0;
    for (long long i = 0; i < it; i++) {
        x = __fma_rn(o1, o2, x); y = __fma_rn(o1, o2, y);
        z = __fma_rn(o1, o2, z); w = __fma_rn(o1, o2, w);
    }
    c[threadIdx.x] = x + y + z + w;
}

cudaStream_t cur() { return at::cuda::getCurrentCUDAStream(); }

void probe_sleep(int64_t blocks, int64_t threads, int64_t iters) {
    k_sleep<<<blocks, threads, 0, cur()>>>(iters);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void probe_copy(torch::Tensor in, torch::Tensor out, int64_t iters,
                int64_t blocks, int64_t threads) {
    TORCH_CHECK(in.is_cuda() && out.is_cuda() && in.dtype() == torch::kFloat32);
    TORCH_CHECK(in.numel() == out.numel());
    k_copy<<<blocks, threads, 0, cur()>>>(in.data_ptr<float>(), out.data_ptr<float>(),
                                          in.numel(), iters);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void probe_copy_block(torch::Tensor in, torch::Tensor out, int64_t npb,
                      int64_t region_bytes, int64_t iters,
                      int64_t blocks, int64_t threads) {
    TORCH_CHECK(in.is_cuda() && out.is_cuda() && in.dtype() == torch::kFloat32);
    k_copy_tb<<<blocks, threads, 0, cur()>>>(in.data_ptr<float>(), out.data_ptr<float>(),
                                             npb, iters, (int)region_bytes);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void probe_fma32(torch::Tensor a, torch::Tensor b, torch::Tensor c, int64_t iters,
                 int64_t blocks, int64_t threads) {
    k_fma32<<<blocks, threads, 0, cur()>>>(a.data_ptr<float>(), b.data_ptr<float>(),
                                           c.data_ptr<float>(), iters);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void probe_fma64(torch::Tensor a, torch::Tensor b, torch::Tensor c, int64_t iters,
                 int64_t blocks, int64_t threads) {
    k_fma64<<<blocks, threads, 0, cur()>>>(a.data_ptr<double>(), b.data_ptr<double>(),
                                           c.data_ptr<double>(), iters);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// Bound into the same module as matmul via ext/setup.py + a shared TORCH_LIBRARY?
// Simpler: this file provides its own pybind registration hook, called from
// matmul.cu's PYBIND11_MODULE (single-module extension).
void register_probes(pybind11::module_& m) {
    m.def("probe_sleep", &probe_sleep, "nanosleep loop (occupancy only)");
    m.def("probe_copy", &probe_copy, "grid-stride streaming copy (DRAM/L2 bandwidth)");
    m.def("probe_copy_block", &probe_copy_block, "per-block region copy (L1)");
    m.def("probe_fma32", &probe_fma32, "FP32 FMA loop (FMA pipe)");
    m.def("probe_fma64", &probe_fma64, "FP64 FMA loop (FP64 pipe)");
}
