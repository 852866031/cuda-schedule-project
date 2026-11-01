// Shared definitions between libcolocator.so (interception) and libsched.so
// (scheduler). libcolocator DEFINES the shared globals below; libsched only
// references them — the dynamic linker resolves them at load time because
// libcolocator is LD_PRELOADed (global scope), so no dlsym dance is needed.
//
// Threading/lifetime contract (the reason this stays simple):
//   * A client thread pushes a FuncRecord* that points into its OWN STACK,
//     then spins on record->state. It only returns from the interposer once
//     the scheduler has at least ISSUED the record, so the pointed-to storage
//     (including the args array of a kernel launch) stays valid exactly as
//     long as the scheduler needs it.
//   * Async ops (kernel launch, memcpy/memset async) unblock at ISSUED —
//     launched on the client's colocator stream, not yet complete.
//   * Sync ops (malloc, free, sync memcpy/memset, stream sync) unblock at
//     DONE — executed and synchronized; `ret` carries the result.
#pragma once

#include <dlfcn.h>
#include <pthread.h>
#include <sys/types.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <queue>

#include <cuda_runtime.h>

// Resolve a symbol from an already-loaded shared library. dlsym(RTLD_NEXT)
// is NOT enough for cuBLAS: python dlopens torch's libraries RTLD_LOCAL, so
// the real cublas symbols are invisible to the global RTLD_NEXT search even
// though our preloaded interposers still capture the calls. dlopen(NOLOAD)
// on the soname returns a handle to the already-mapped library instead.
// Aborts on failure — a silent nullptr here would mean losing GEMMs.
inline void* col_resolve(const char* sym, const char* soname) {
    void* p = dlsym(RTLD_NEXT, sym);
    if (!p) {
        void* h = dlopen(soname, RTLD_LAZY | RTLD_NOLOAD);
        if (h) p = dlsym(h, sym);
    }
    if (!p) {
        fprintf(stderr, "[colocator] FATAL: cannot resolve %s (%s not loaded?)\n",
                sym, soname);
        abort();
    }
    return p;
}

// Every CUDA runtime / cuBLAS call the colocator interposes.
enum class OpKind {
    KernelLaunch = 0,   // cudaLaunchKernel
    KernelLaunchExC,    // cudaLaunchKernelExC (counted only; not managed)
    Malloc,             // cudaMalloc
    Free,               // cudaFree
    Memcpy,             // cudaMemcpy (sync)
    MemcpyAsync,        // cudaMemcpyAsync
    Memset,             // cudaMemset (sync)
    MemsetAsync,        // cudaMemsetAsync
    StreamSync,         // cudaStreamSynchronize
    EventRecord,        // cudaEventRecord (redirected to the colocator stream)
    CublasGemmEx,       // cublasGemmEx — the whole library call is queued and
                        // replayed by the scheduler (Orion-style; the kernels
                        // cuBLAS launches internally are NOT interceptable)
    CublasSgemm,        // cublasSgemm_v2 (fp32 path, rare)
    kCount
};

inline const char* op_kind_name(OpKind k) {
    switch (k) {
        case OpKind::KernelLaunch:    return "cudaLaunchKernel";
        case OpKind::KernelLaunchExC: return "cudaLaunchKernelExC";
        case OpKind::Malloc:          return "cudaMalloc";
        case OpKind::Free:            return "cudaFree";
        case OpKind::Memcpy:          return "cudaMemcpy";
        case OpKind::MemcpyAsync:     return "cudaMemcpyAsync";
        case OpKind::Memset:          return "cudaMemset";
        case OpKind::MemsetAsync:     return "cudaMemsetAsync";
        case OpKind::StreamSync:      return "cudaStreamSynchronize";
        case OpKind::EventRecord:     return "cudaEventRecord";
        case OpKind::CublasGemmEx:    return "cublasGemmEx";
        case OpKind::CublasSgemm:     return "cublasSgemm";
        default:                      return "?";
    }
}

// Colocator operating mode, from COLOCATOR_MODE (read once at load).
enum class ColMode { Passthrough = 0, Managed = 1 };

enum RecordState : int {
    REC_QUEUED = 0,  // pushed by client, not yet picked by scheduler
    REC_ISSUED = 1,  // real call made on the colocator stream (async ops stop here)
    REC_DONE   = 2,  // executed AND synchronized (sync ops stop here)
};

// One intercepted operation. Lives on the intercepting client's stack.
struct FuncRecord {
    OpKind kind;
    uint64_t op_id;           // per-client sequence number, from 0
    uint64_t t_intercept_ns;  // CLOCK_MONOTONIC_RAW at interception

    // KernelLaunch
    const void* func = nullptr;
    dim3 grid{}, block{};
    void** args = nullptr;
    size_t shared_mem = 0;

    // Memcpy / Memset / Malloc / Free
    void* dst = nullptr;
    const void* src = nullptr;
    size_t count = 0;               // bytes (memcpy/memset) or alloc size (malloc)
    cudaMemcpyKind copy_kind = cudaMemcpyDefault;
    int memset_value = 0;
    void** devptr_out = nullptr;    // malloc result slot

    cudaStream_t orig_stream = nullptr;  // what the client passed (informational)
    cudaEvent_t event = nullptr;         // EventRecord: the client's event
    unsigned event_flags = 0;            // EventRecord: cudaEventRecordWithFlags flags

    // CublasGemmEx / CublasSgemm. Types kept opaque (void* / int) so this
    // header needs no cuBLAS include; intercept/sched cast at the call site.
    // alpha/beta point at host scalars on the CLIENT's stack — valid for the
    // same reason `args` is: the client spins until the scheduler has made
    // the real call (ISSUED), and cuBLAS reads them synchronously during it.
    struct {
        void* handle = nullptr;              // cublasHandle_t
        int transa = 0, transb = 0;          // cublasOperation_t
        int m = 0, n = 0, k = 0;
        const void* alpha = nullptr;
        const void* A = nullptr; int a_type = 0; int lda = 0;   // cudaDataType
        const void* B = nullptr; int b_type = 0; int ldb = 0;
        const void* beta = nullptr;
        void* C = nullptr; int c_type = 0; int ldc = 0;
        int compute_type = 0;                // cublasComputeType_t
        int algo = 0;                        // cublasGemmAlgo_t
        int status = 0;                      // cublasStatus_t result
    } blas;

    std::atomic<int> state{REC_QUEUED};
    cudaError_t ret = cudaSuccess;  // result, valid once state permits return
};

constexpr int kMaxClients = 8;

// Per-client shared slot. POD-ish; initialized by libcolocator, used by both.
struct ClientSlot {
    std::atomic<pid_t> tid{0};       // registered client thread (0 = unused slot)
    pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
    std::queue<FuncRecord*> q;       // guarded by mtx
    std::atomic<uint64_t> submitted{0};  // ops pushed by the client side
};

extern "C" {
// Defined in libcolocator.so:
extern ClientSlot g_col_clients[kMaxClients];
extern std::atomic<int> g_col_num_clients;
extern std::atomic<int> g_col_mode;  // holds a ColMode value

// Called (via ctypes) from a client thread before it touches CUDA: claims
// slot `idx` for the calling thread. Returns 0 on success.
int col_register_client(int idx);
// Per-client count of ops pushed so far (for submitted == issued checks).
uint64_t col_client_submitted(int idx);
}
