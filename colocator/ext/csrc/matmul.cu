// Tiled shared-memory SGEMM used by the demo workloads.
//
// Exists so the workloads can do real matmuls WITHOUT torch.matmul: stock
// torch.matmul dispatches to cuBLAS, whose kernels are launched internally via
// the CUDA driver API and would bypass the colocator's cudaLaunchKernel
// interposer. A __global__ kernel launched from an extension goes through
// cudaLaunchKernel like any ATen native kernel, so the colocator sees it.
//
// Deliberately simple (one output element per thread, TILE x TILE shared
// tiles): kernel duration scales with problem size, which is what the demo
// needs — short kernels for the latency client, long SM-saturating kernels
// for the throughput client.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace {

constexpr int TILE = 32;

__global__ void sgemm_tiled(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int M, int K, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;  // index into M
    const int col = blockIdx.x * TILE + threadIdx.x;  // index into N

    float acc = 0.0f;
    const int ntiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < ntiles; t++) {
        const int a_col = t * TILE + threadIdx.x;
        const int b_row = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[(long)row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[(long)b_row * N + col] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) C[(long)row * N + col] = acc;
}

torch::Tensor matmul(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32,
                "inputs must be float32");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "inputs must be 2-D");
    TORCH_CHECK(a.size(1) == b.size(0), "shape mismatch: ",
                a.size(1), " vs ", b.size(0));
    auto ac = a.contiguous();
    auto bc = b.contiguous();

    const int M = (int)ac.size(0), K = (int)ac.size(1), N = (int)bc.size(1);
    auto c = torch::empty({M, N}, ac.options());

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sgemm_tiled<<<grid, block, 0, stream>>>(
        ac.data_ptr<float>(), bc.data_ptr<float>(), c.data_ptr<float>(), M, K, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}

}  // namespace

void register_probes(pybind11::module_& m);  // csrc/probes.cu

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul, "tiled fp32 matmul (goes through cudaLaunchKernel)");
    register_probes(m);
}
