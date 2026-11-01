// libcolocator.so — LD_PRELOAD interposition layer.
//
// Defines the same symbols as the CUDA runtime API so the dynamic linker binds
// PyTorch's calls here instead of libcudart. Each interposer resolves the real
// function once via dlsym(RTLD_NEXT, ...).
//
// Modes (COLOCATOR_MODE, read once at load):
//   passthrough (default) — count every call (and log with COLOCATOR_VERBOSE=1),
//       then forward directly. Phase 1 behavior; also what unregistered
//       threads always get in managed mode.
//   managed — calls from REGISTERED client threads are packed into a
//       FuncRecord on the caller's stack, pushed to that client's queue, and
//       the caller spins until the scheduler (libsched.so) has issued (async
//       ops) or completed (sync ops) it. Calls from any other thread — the
//       Python main thread, the scheduler itself, CUPTI workers — pass
//       through, which is also what keeps the scheduler's own replayed CUDA
//       calls from being re-captured.
//
// See ../common/records.h for the shared-state and lifetime contract.

#include <dlfcn.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cublas_v2.h>  // types/signatures only — resolved at runtime, never linked

#include "../common/records.h"

// ------------------------------------------------------ shared state defs ---

extern "C" {
ClientSlot g_col_clients[kMaxClients];
std::atomic<int> g_col_num_clients{0};
std::atomic<int> g_col_mode{(int)ColMode::Passthrough};
}

namespace {

pid_t my_tid() { return (pid_t)syscall(SYS_gettid); }

uint64_t now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

bool verbose() {
    static const bool v = [] {
        const char* e = getenv("COLOCATOR_VERBOSE");
        return e && e[0] == '1';
    }();
    return v;
}

__attribute__((constructor)) void init_mode() {
    const char* m = getenv("COLOCATOR_MODE");
    if (m && strcmp(m, "managed") == 0) {
        g_col_mode.store((int)ColMode::Managed);
        fprintf(stderr, "[colocator] intercept loaded, mode=managed\n");
    } else if (m && strcmp(m, "passthrough") != 0) {
        fprintf(stderr, "[colocator] warning: unknown COLOCATOR_MODE=%s, using passthrough\n", m);
    }
}

bool managed() { return g_col_mode.load(std::memory_order_relaxed) == (int)ColMode::Managed; }

// Client slot of the calling thread, or -1 if this thread is not a client.
int client_idx() {
    pid_t tid = my_tid();
    int n = g_col_num_clients.load(std::memory_order_acquire);
    for (int i = 0; i < n; i++)
        if (g_col_clients[i].tid.load(std::memory_order_acquire) == tid) return i;
    return -1;
}

// ------------------------------------------------- passthrough accounting ---

constexpr int kMaxThreads = 64;

struct ThreadStats {
    std::atomic<pid_t> tid{0};
    std::atomic<unsigned long long> counts[(int)OpKind::kCount]{};
};

ThreadStats g_stats[kMaxThreads];
std::atomic<unsigned long long> g_dropped{0};

ThreadStats* stats_slot() {
    pid_t tid = my_tid();
    for (int i = 0; i < kMaxThreads; i++) {
        pid_t cur = g_stats[i].tid.load(std::memory_order_acquire);
        if (cur == tid) return &g_stats[i];
        if (cur == 0) {
            pid_t expected = 0;
            if (g_stats[i].tid.compare_exchange_strong(expected, tid))
                return &g_stats[i];
            if (expected == tid) return &g_stats[i];
        }
    }
    return nullptr;
}

void note(OpKind kind) {
    ThreadStats* s = stats_slot();
    if (s) s->counts[(int)kind].fetch_add(1, std::memory_order_relaxed);
    else g_dropped.fetch_add(1, std::memory_order_relaxed);
}

__attribute__((destructor)) void report() {
    unsigned long long totals[(int)OpKind::kCount] = {};
    int nthreads = 0;
    for (int i = 0; i < kMaxThreads; i++) {
        if (g_stats[i].tid.load() == 0) continue;
        nthreads++;
        for (int k = 0; k < (int)OpKind::kCount; k++)
            totals[k] += g_stats[i].counts[k].load();
    }
    fprintf(stderr, "[colocator] ---- intercept summary (passthrough-path calls) ----\n");
    for (int k = 0; k < (int)OpKind::kCount; k++)
        if (totals[k])
            fprintf(stderr, "[colocator]   %-22s %llu\n", op_kind_name((OpKind)k), totals[k]);
    fprintf(stderr, "[colocator]   threads seen: %d\n", nthreads);
    if (managed()) {
        int n = g_col_num_clients.load();
        for (int i = 0; i < n; i++)
            fprintf(stderr, "[colocator]   client %d (tid %d): %llu ops queued\n",
                    i, g_col_clients[i].tid.load(),
                    (unsigned long long)g_col_clients[i].submitted.load());
    }
    if (g_dropped.load())
        fprintf(stderr, "[colocator]   WARNING: %llu calls from >%d threads not counted\n",
                g_dropped.load(), kMaxThreads);
}

// ------------------------------------------------------- managed queueing ---

constexpr uint64_t kWatchdogNs = 30ull * 1000000000ull;

// Push `rec` to client `idx`'s queue and spin until the scheduler moves it to
// at least `wait_state`. Returns rec->ret.
cudaError_t submit_and_wait(int idx, FuncRecord* rec, int wait_state) {
    ClientSlot& c = g_col_clients[idx];
    rec->op_id = c.submitted.fetch_add(1, std::memory_order_relaxed);
    rec->t_intercept_ns = now_ns();

    pthread_mutex_lock(&c.mtx);
    c.q.push(rec);
    pthread_mutex_unlock(&c.mtx);

    if (verbose())
        fprintf(stderr, "[colocator][client %d] queued %s op_id=%llu\n",
                idx, op_kind_name(rec->kind), (unsigned long long)rec->op_id);

    const uint64_t start = rec->t_intercept_ns;
    uint64_t spins = 0;
    while (rec->state.load(std::memory_order_acquire) < wait_state) {
#if defined(__x86_64__)
        __builtin_ia32_pause();
#endif
        // Watchdog (R5): if the scheduler died we abort loudly instead of
        // hanging the client forever or corrupting ordering with a fallback.
        if (((++spins) & 0xFFFFF) == 0 && now_ns() - start > kWatchdogNs) {
            fprintf(stderr, "[colocator] FATAL: client %d waited >30s for %s op_id=%llu — "
                            "scheduler dead or never started\n",
                    idx, op_kind_name(rec->kind), (unsigned long long)rec->op_id);
            abort();
        }
    }
    return rec->ret;
}

// Resolve the real function once per call site (thread-safe static init).
#define REAL(fn_type, name)                                      \
    static fn_type real = (fn_type)dlsym(RTLD_NEXT, name);       \
    if (!real) {                                                 \
        fprintf(stderr, "[colocator] dlsym(%s) failed\n", name); \
        abort();                                                 \
    }

}  // namespace

// ----------------------------------------------------------- control API ----

extern "C" {

int col_register_client(int idx) {
    if (idx < 0 || idx >= kMaxClients) return -1;
    pid_t tid = my_tid();
    g_col_clients[idx].tid.store(tid, std::memory_order_release);
    int n = g_col_num_clients.load();
    while (n <= idx && !g_col_num_clients.compare_exchange_weak(n, idx + 1)) {}
    fprintf(stderr, "[colocator] registered client %d = tid %d\n", idx, tid);
    return 0;
}

uint64_t col_client_submitted(int idx) {
    if (idx < 0 || idx >= kMaxClients) return 0;
    return g_col_clients[idx].submitted.load();
}

// ------------------------------------------------------------ interposers ---

cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
                             void** args, size_t sharedMem, cudaStream_t stream) {
    using Fn = cudaError_t (*)(const void*, dim3, dim3, void**, size_t, cudaStream_t);
    REAL(Fn, "cudaLaunchKernel");
    int idx;
    if (managed() && (idx = client_idx()) >= 0) {
        FuncRecord rec;
        rec.kind = OpKind::KernelLaunch;
        rec.func = func; rec.grid = gridDim; rec.block = blockDim;
        rec.args = args; rec.shared_mem = sharedMem; rec.orig_stream = stream;
        return submit_and_wait(idx, &rec, REC_ISSUED);
    }
    note(OpKind::KernelLaunch);
    if (verbose())
        fprintf(stderr, "[colocator][tid %d] cudaLaunchKernel func=%p grid=(%u,%u,%u) "
                        "block=(%u,%u,%u) shmem=%zu stream=%p\n",
                my_tid(), func, gridDim.x, gridDim.y, gridDim.z,
                blockDim.x, blockDim.y, blockDim.z, sharedMem, (void*)stream);
    return real(func, gridDim, blockDim, args, sharedMem, stream);
}

cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t* config, const void* func,
                                void** args) {
    // Not managed: counted + forwarded so a workload that uses it is noticed
    // (it would run on the client's original stream, outside colocator control).
    using Fn = cudaError_t (*)(const cudaLaunchConfig_t*, const void*, void**);
    REAL(Fn, "cudaLaunchKernelExC");
    note(OpKind::KernelLaunchExC);
    if (managed() && client_idx() >= 0)
        fprintf(stderr, "[colocator] WARNING: client used unmanaged cudaLaunchKernelExC\n");
    return real(config, func, args);
}

cudaError_t cudaMalloc(void** devPtr, size_t size) {
    using Fn = cudaError_t (*)(void**, size_t);
    REAL(Fn, "cudaMalloc");
    int idx;
    if (managed() && (idx = client_idx()) >= 0) {
        FuncRecord rec;
        rec.kind = OpKind::Malloc;
        rec.devptr_out = devPtr; rec.count = size;
        return submit_and_wait(idx, &rec, REC_DONE);
    }
    note(OpKind::Malloc);
    return real(devPtr, size);
}

cudaError_t cudaFree(void* devPtr) {
    using Fn = cudaError_t (*)(void*);
    REAL(Fn, "cudaFree");
    int idx;
    if (managed() && (idx = client_idx()) >= 0) {
        FuncRecord rec;
        rec.kind = OpKind::Free;
        rec.dst = devPtr;
        return submit_and_wait(idx, &rec, REC_DONE);
    }
    note(OpKind::Free);
    return real(devPtr);
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    using Fn = cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind);
    REAL(Fn, "cudaMemcpy");
    int idx;
    if (managed() && (idx = client_idx()) >= 0) {
        FuncRecord rec;
        rec.kind = OpKind::Memcpy;
        rec.dst = dst; rec.src = src; rec.count = count; rec.copy_kind = kind;
        return submit_and_wait(idx, &rec, REC_DONE);  // sync semantics
    }
    note(OpKind::Memcpy);
    return real(dst, src, count, kind);
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream) {
    using Fn = cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
    REAL(Fn, "cudaMemcpyAsync");
    int idx;
    if (managed() && (idx = client_idx()) >= 0) {
        FuncRecord rec;
        rec.kind = OpKind::MemcpyAsync;
        rec.dst = dst; rec.src = src; rec.count = count; rec.copy_kind = kind;
        rec.orig_stream = stream;
        return submit_and_wait(idx, &rec, REC_ISSUED);
    }
    note(OpKind::MemcpyAsync);
    return real(dst, src, count, kind, stream);
}

cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
    using Fn = cudaError_t (*)(void*, int, size_t);
    REAL(Fn, "cudaMemset");
    int idx;
    if (managed() && (idx = client_idx()) >= 0) {
        FuncRecord rec;
        rec.kind = OpKind::Memset;
        rec.dst = devPtr; rec.memset_value = value; rec.count = count;
        return submit_and_wait(idx, &rec, REC_DONE);  // sync semantics
    }
    note(OpKind::Memset);
    return real(devPtr, value, count);
}

cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) {
    using Fn = cudaError_t (*)(void*, int, size_t, cudaStream_t);
    REAL(Fn, "cudaMemsetAsync");
    int idx;
    if (managed() && (idx = client_idx()) >= 0) {
        FuncRecord rec;
        rec.kind = OpKind::MemsetAsync;
        rec.dst = devPtr; rec.memset_value = value; rec.count = count;
        rec.orig_stream = stream;
        return submit_and_wait(idx, &rec, REC_ISSUED);
    }
    note(OpKind::MemsetAsync);
    return real(devPtr, value, count, stream);
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    // Redirected like cudaStreamSynchronize: the event is recorded on the
    // client's COLOCATOR stream (queued, so it lands in stream order behind
    // the client's prior ops). This is what lets clients pipeline with
    // torch.cuda.Event: record -> keep launching -> event.synchronize()
    // (the latter is not interposed — by the time the client calls it, the
    // real record has already been issued, so waiting on the real event is
    // correct in both solo and colocated runs).
    using Fn = cudaError_t (*)(cudaEvent_t, cudaStream_t);
    REAL(Fn, "cudaEventRecord");
    int idx;
    if (managed() && (idx = client_idx()) >= 0) {
        FuncRecord rec;
        rec.kind = OpKind::EventRecord;
        rec.event = event;
        rec.orig_stream = stream;
        return submit_and_wait(idx, &rec, REC_ISSUED);
    }
    note(OpKind::EventRecord);
    if (verbose())
        fprintf(stderr, "[colocator][tid %d] cudaEventRecord event=%p stream=%p\n",
                my_tid(), (void*)event, (void*)stream);
    return real(event, stream);
}

cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream,
                                     unsigned int flags) {
    // torch.cuda.Event.record() uses THIS entry point (not plain
    // cudaEventRecord) — missing it silently records on the client's unused
    // torch stream and the event completes instantly, breaking pipelining.
    using Fn = cudaError_t (*)(cudaEvent_t, cudaStream_t, unsigned int);
    REAL(Fn, "cudaEventRecordWithFlags");
    int idx;
    if (managed() && (idx = client_idx()) >= 0) {
        FuncRecord rec;
        rec.kind = OpKind::EventRecord;
        rec.event = event;
        rec.event_flags = flags;
        rec.orig_stream = stream;
        return submit_and_wait(idx, &rec, REC_ISSUED);
    }
    note(OpKind::EventRecord);
    return real(event, stream, flags);
}

// ---- cuBLAS interposers (Orion-style API-level interception) --------------
//
// The linear layers of LLM workloads go through cuBLAS, whose internal kernel
// launches are invisible to LD_PRELOAD (statically-linked runtime / driver
// API). But the torch → libcublas call itself is a normal dynamic-linker
// call, so we capture it HERE — the whole library call, args and all — queue
// it, and let the scheduler replay it with the stream substituted on the
// handle. cuBLAS's internal launches then land on the colocator stream
// because the scheduler is the one calling. Verified for torch 2.11 llama
// decode: 100% of linear-layer FLOPs funnel through cublasGemmEx (+ a few
// fp32 cublasSgemm_v2); cublasLtMatmul is not on that path (add it here the
// day a workload uses it — the intercept summary will show it as leaked GPU
// kernels in analyze's self-check).
//
// NOT interposed: cublasSetStream_v2. The client may freely set its torch
// stream on the handle between ops (host-side state, no GPU work); the
// scheduler re-sets the stream on the handle immediately before every
// replayed call, and lockstep guarantees the client cannot touch the handle
// while a call of its own is in flight.

cublasStatus_t cublasGemmEx(cublasHandle_t handle,
                            cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k,
                            const void* alpha,
                            const void* A, cudaDataType Atype, int lda,
                            const void* B, cudaDataType Btype, int ldb,
                            const void* beta,
                            void* C, cudaDataType Ctype, int ldc,
                            cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
    using Fn = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t,
                                  int, int, int, const void*, const void*, cudaDataType,
                                  int, const void*, cudaDataType, int, const void*,
                                  void*, cudaDataType, int, cublasComputeType_t,
                                  cublasGemmAlgo_t);
    static Fn real = (Fn)col_resolve("cublasGemmEx", "libcublas.so.12");
    int idx;
    if (managed() && (idx = client_idx()) >= 0) {
        FuncRecord rec;
        rec.kind = OpKind::CublasGemmEx;
        rec.blas.handle = (void*)handle;
        rec.blas.transa = (int)transa; rec.blas.transb = (int)transb;
        rec.blas.m = m; rec.blas.n = n; rec.blas.k = k;
        rec.blas.alpha = alpha; rec.blas.beta = beta;
        rec.blas.A = A; rec.blas.a_type = (int)Atype; rec.blas.lda = lda;
        rec.blas.B = B; rec.blas.b_type = (int)Btype; rec.blas.ldb = ldb;
        rec.blas.C = C; rec.blas.c_type = (int)Ctype; rec.blas.ldc = ldc;
        rec.blas.compute_type = (int)computeType; rec.blas.algo = (int)algo;
        submit_and_wait(idx, &rec, REC_ISSUED);
        return (cublasStatus_t)rec.blas.status;
    }
    note(OpKind::CublasGemmEx);
    return real(handle, transa, transb, m, n, k, alpha, A, Atype, lda,
                B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo);
}

cublasStatus_t cublasSgemm_v2(cublasHandle_t handle,
                              cublasOperation_t transa, cublasOperation_t transb,
                              int m, int n, int k,
                              const float* alpha, const float* A, int lda,
                              const float* B, int ldb,
                              const float* beta, float* C, int ldc) {
    using Fn = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t, cublasOperation_t,
                                  int, int, int, const float*, const float*, int,
                                  const float*, int, const float*, float*, int);
    static Fn real = (Fn)col_resolve("cublasSgemm_v2", "libcublas.so.12");
    int idx;
    if (managed() && (idx = client_idx()) >= 0) {
        FuncRecord rec;
        rec.kind = OpKind::CublasSgemm;
        rec.blas.handle = (void*)handle;
        rec.blas.transa = (int)transa; rec.blas.transb = (int)transb;
        rec.blas.m = m; rec.blas.n = n; rec.blas.k = k;
        rec.blas.alpha = alpha; rec.blas.beta = beta;
        rec.blas.A = A; rec.blas.lda = lda;
        rec.blas.B = B; rec.blas.ldb = ldb;
        rec.blas.C = C; rec.blas.ldc = ldc;
        submit_and_wait(idx, &rec, REC_ISSUED);
        return (cublasStatus_t)rec.blas.status;
    }
    note(OpKind::CublasSgemm);
    return real(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    using Fn = cudaError_t (*)(cudaStream_t);
    REAL(Fn, "cudaStreamSynchronize");
    int idx;
    if (managed() && (idx = client_idx()) >= 0) {
        // Redirect: whatever stream the client thinks it's syncing, what it
        // MEANS is "wait until my work so far is done" — i.e. drain the queue
        // (implicit: we're in lockstep, the queue is empty when we get here
        // except for this record) and sync the client's colocator stream.
        FuncRecord rec;
        rec.kind = OpKind::StreamSync;
        rec.orig_stream = stream;
        return submit_and_wait(idx, &rec, REC_DONE);
    }
    note(OpKind::StreamSync);
    return real(stream);
}

}  // extern "C"
