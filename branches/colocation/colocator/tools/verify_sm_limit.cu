// Verifies COLOCATOR_SM_LIMIT end-to-end on the REAL scheduler streams:
// links build/libcolocator.so + build/libsched.so, calls the actual
// col_setup() (which reads COLOCATOR_SM_LIMIT and builds the green-context
// streams exactly as a run does), then launches an %smid census kernel on
// each stream via col_debug_stream() and counts the distinct SMs touched.
//
//   nvcc -arch=sm_120 tools/verify_sm_limit.cu -o build/verify_sm_limit \
//        -Lbuild -lcolocator -lsched -Wl,-rpath,'$ORIGIN'
//   COLOCATOR_SM_LIMIT=50 ./build/verify_sm_limit
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

extern "C" {
int col_setup(int num_clients, const int* priorities);
void* col_debug_stream(int client, int which);
}

__global__ void smid_census(unsigned long long* mask) {
    unsigned smid;
    asm("mov.u32 %0, %%smid;" : "=r"(smid));
    atomicOr(&mask[smid / 64], 1ull << (smid % 64));
    long long s = clock64();                 // linger: blocks from many waves
    while (clock64() - s < 100000) {}        // land on every available SM
}

static int census(cudaStream_t st) {
    unsigned long long* mask;
    cudaMalloc(&mask, 64);
    cudaMemsetAsync(mask, 0, 64, st);
    smid_census<<<2048, 32, 0, st>>>(mask);
    cudaError_t e = cudaGetLastError();
    if (e) { printf("launch FAILED: %s\n", cudaGetErrorString(e)); exit(1); }
    unsigned long long h[8];
    cudaMemcpyAsync(h, mask, 64, cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
    int n = 0;
    for (int i = 0; i < 8; i++) n += __builtin_popcountll(h[i]);
    cudaFree(mask);
    return n;
}

int main() {
    cudaFree(0);  // primary context, as under torch
    if (col_setup(1, nullptr) != 0) { printf("col_setup failed\n"); return 1; }
    cudaStream_t main_s = (cudaStream_t)col_debug_stream(0, 0);
    cudaStream_t gemm_s = (cudaStream_t)col_debug_stream(0, 1);
    printf("main stream census: %3d distinct SMs\n", census(main_s));
    if (gemm_s)
        printf("gemm stream census: %3d distinct SMs  (COLOCATOR_SM_LIMIT=%s)\n",
               census(gemm_s), getenv("COLOCATOR_SM_LIMIT"));
    else
        printf("gemm stream: none (SM limit off)\n");
    return 0;
}
