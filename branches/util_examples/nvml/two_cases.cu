#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <iomanip>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                      << " -> " << cudaGetErrorString(err__) << std::endl;      \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

__global__ void work_kernel(const float* x, float* y, int n, 
        int repeat, float alpha, float beta)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < n; i += stride) {
        float xi = x[i];
        float yi = y[i];
        #pragma unroll 1
        for (int r = 0; r < repeat; ++r) {
            yi = alpha * xi + beta * yi;
            xi = beta * yi + alpha * xi;
        }
        y[i] = yi;
    }
}

void init_arrays(float* d_x, float* d_y, int n)
{
    std::vector<float> h_x(n, 1.0f);
    std::vector<float> h_y(n, 2.0f);
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y.data(), n * sizeof(float), cudaMemcpyHostToDevice));
}

struct Result {
    int gpu_id;
    long long launches_completed;
    double elapsed_s;
    double total_work_units;   // launches * n * repeat
};

void run_gpu(
    int gpu_id,
    int blocks,
    int threads,
    int n,
    int repeat,
    double duration_s,
    const char* label,
    Result* result)
{
    CHECK_CUDA(cudaSetDevice(gpu_id));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    float *d_x = nullptr, *d_y = nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, n * sizeof(float)));

    init_arrays(d_x, d_y, n);

    // Warmup
    work_kernel<<<blocks, threads, 0, stream>>>(d_x, d_y, n, repeat, 1.001f, 0.999f);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::cout << label
              << " gpu=" << gpu_id
              << " blocks=" << blocks
              << " threads=" << threads
              << " n=" << n
              << " repeat=" << repeat
              << std::endl;

    long long launches_completed = 0;

    auto t0 = std::chrono::steady_clock::now();
    while (true) {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - t0).count();
        if (elapsed >= duration_s) break;

        work_kernel<<<blocks, threads, 0, stream>>>(d_x, d_y, n, repeat, 1.001f, 0.999f);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaStreamSynchronize(stream));

        launches_completed++;
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();

    result->gpu_id = gpu_id;
    result->launches_completed = launches_completed;
    result->elapsed_s = elapsed_s;
    result->total_work_units = static_cast<double>(launches_completed) *
                               static_cast<double>(n) *
                               static_cast<double>(repeat);

    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaStreamDestroy(stream));
}

int main()
{
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count < 2) {
        std::cerr << "Need at least 2 CUDA GPUs, found " << device_count << std::endl;
        return 1;
    }

    cudaDeviceProp prop0{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop0, 0));

    // Same workload definition on both GPUs
    int n = 1 << 24;      // reduce if too slow / increase if too fast
    int repeat = 128;
    double duration_s = 10.0;

    // GPU 0: broad spatial parallelism
    int heavy_blocks = prop0.multiProcessorCount * 8;
    int heavy_threads = 256;

    // GPU 1: tiny spatial footprint
    int tiny_blocks = 1;
    int tiny_threads = 32;

    Result r0{}, r1{};

    std::thread t0(run_gpu, 0, heavy_blocks, heavy_threads, n, repeat, duration_s, "HEAVY", &r0);
    std::thread t1(run_gpu, 1, tiny_blocks, tiny_threads, n, repeat, duration_s, "TINY ", &r1);

    t0.join();
    t1.join();

    std::cout << "\n=== Results over ~" << duration_s << " s ===\n";
    std::cout << "GPU 0 launches completed: " << r0.launches_completed << "\n";
    std::cout << "GPU 1 launches completed: " << r1.launches_completed << "\n";

    std::cout << "GPU 0 total work units  : " << std::fixed << std::setprecision(3) << r0.total_work_units << "\n";
    std::cout << "GPU 1 total work units  : " << std::fixed << std::setprecision(3) << r1.total_work_units << "\n";

    double ratio = (r1.total_work_units > 0.0) ? (r0.total_work_units / r1.total_work_units) : 0.0;

    std::cout << "GPU 0 / GPU 1 work ratio: " << ratio << "x\n";

    // Normalize GPU 1 to 100%
    double gpu1_norm = 100.0;
    double gpu0_norm = ratio * 100.0;

    std::cout << "\nNormalized to GPU 1 = 100%:\n";
    std::cout << "GPU 0: " << gpu0_norm << "%\n";
    std::cout << "GPU 1: " << gpu1_norm << "%\n";

    std::cout << "\nDone.\n";
    return 0;
}

