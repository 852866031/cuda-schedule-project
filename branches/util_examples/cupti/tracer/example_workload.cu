#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#define CUDA_CALL(call)                                                        \
    do {                                                                       \
        cudaError_t _status = call;                                            \
        if (_status != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                         __FILE__, __LINE__, cudaGetErrorString(_status));     \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

__global__ void dense_fma_kernel(const float* x, float* y, int n, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < n; i += stride) {
        float xi = x[i];
        float yi = y[i];
        #pragma unroll 1
        for (int j = 0; j < iters; ++j) {
            yi = 1.001f * xi + 0.999f * yi;
            xi = 0.999f * yi + 1.001f * xi;
        }
        y[i] = yi;
    }
}

__global__ void bad_gather_kernel(const float* x, const int* idx, float* out,
                                  int n, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < n; i += stride) {
        int cur = idx[i];
        float acc = 0.0f;
        #pragma unroll 1
        for (int j = 0; j < iters; ++j) {
            cur = idx[cur & (n - 1)];
            acc += x[cur];
        }
        out[i] = acc;
    }
}

int main() {
    CUDA_CALL(cudaSetDevice(0));

    const int n = 1 << 22;
    const int dense_iters = 128;
    const int gather_iters = 16;
    const int blocks = 256;
    const int threads = 256;

    std::vector<float> h_x(n, 1.0f), h_y(n, 2.0f);
    std::vector<int> h_idx(n);
    for (int i = 0; i < n; ++i) h_idx[i] = (i * 17 + 13) & (n - 1);

    float *d_x = nullptr, *d_y = nullptr, *d_out = nullptr;
    int* d_idx = nullptr;
    CUDA_CALL(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_y, n * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_idx, n * sizeof(int)));

    CUDA_CALL(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_y, h_y.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_idx, h_idx.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    auto start = std::chrono::steady_clock::now();
    const double duration_s = 3.0;

    while (true) {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start).count();
        if (elapsed >= duration_s) break;

        for (int i = 0; i < 12; ++i) {
            dense_fma_kernel<<<blocks, threads>>>(d_x, d_y, n, dense_iters);
            CUDA_CALL(cudaGetLastError());
        }

        for (int i = 0; i < 6; ++i) {
            bad_gather_kernel<<<blocks, 128>>>(d_x, d_idx, d_out, n, gather_iters);
            CUDA_CALL(cudaGetLastError());
        }

        CUDA_CALL(cudaDeviceSynchronize());
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    CUDA_CALL(cudaFree(d_x));
    CUDA_CALL(cudaFree(d_y));
    CUDA_CALL(cudaFree(d_out));
    CUDA_CALL(cudaFree(d_idx));
    return 0;
}