// libsched.so — the colocator core, driven from Python via ctypes.
//
// Owns one CUDA stream per client (priority configurable) and a busy-wait
// loop that peeks the head of every client queue, asks the Policy which to
// run, and replays that record on the client's colocator stream via the real
// CUDA calls. Every executed op is appended to an in-memory issue log,
// dumped to CSV at teardown.
//
// Shared state (queues, registration) lives in libcolocator.so and is
// resolved by the dynamic linker at load time (see common/records.h).
//
// NOTE on CUDA calls made here: the scheduler thread is not a registered
// client, so its calls go through libcolocator's interposers in passthrough
// mode — i.e. straight to the real runtime.
//
// Sync ops (cudaStreamSynchronize, sync memcpy/memset) are completed by
// POLLING cudaStreamQuery from the loop rather than blocking in
// cudaStreamSynchronize: blocking would stall every other client's queue for
// the duration (measured: 21 ms sync of one client inflated the other
// client's p95 from 0.5 ms to 43 ms). The waiting client just keeps spinning
// on rec->state until the poll flips it to DONE. Malloc/Free stay blocking —
// they are host-synchronous and rare.

#include <dlfcn.h>
#include <sched.h>
#include <time.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <cublas_v2.h>  // types only; symbols resolved at runtime via col_resolve
#include <cuda.h>       // driver API: green contexts for the SM-limit option

#include "../common/records.h"
#include "policy.h"

namespace {

uint64_t now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

// One issue-log entry per executed op (preallocated; no I/O on hot path).
struct IssueEntry {
    int client;
    uint64_t op_id;
    OpKind kind;
    uint64_t t_intercept_ns;
    uint64_t t_issue_ns;      // just before the real CUDA call
    uint64_t t_returned_ns;   // just after the real call returned (launch overhead)
    const void* func;         // kernel func ptr (nullptr for non-kernels)
    unsigned gx, gy, gz, bx, by, bz;
    size_t bytes;             // memcpy/memset/malloc size
    void* stream;             // colocator stream it ran on
};

// Orion-style completion tracking: one cudaEvent recorded after every op we
// enqueue on a stream. Streams complete in order, so advancing a cursor with
// cudaEventQuery gives the EXACT count of ops issued-but-not-yet-finished per
// client at any moment (this is what plain cudaStreamQuery cannot give — it
// only answers "fully drained?"). Ring is far larger than the lockstep
// in-flight bound (≤ one client iteration).
constexpr uint64_t kEventRing = 1024;

// Optional live GPU feed from libobserver.so (resolved via dlsym so libsched
// works without the observer loaded): CUPTI-measured completions per stream,
// lag bounded by the observer's flush period. Events above tell us *that* our
// ops finished; this tells us what the GPU *actually did* (real busy time).
typedef int (*obs_live_enable_fn)(unsigned);
typedef int (*obs_live_track_fn)(void*);
typedef int (*obs_live_get_fn)(int, unsigned long long*, unsigned long long*, unsigned long long*);

struct Sched {
    int num_clients = 0;
    std::vector<cudaStream_t> streams;
    // SM limit (COLOCATOR_SM_LIMIT=<percent>): one green context holding
    // ~percent of the device's SMs; per client a second stream (gemm_streams)
    // created inside it. execute() routes cuBLAS ops there; every stream
    // SWITCH for a client is chained with an event record/wait pair so the
    // client's execution stays LINEAR across its two streams — program order
    // (and hence the single per-client completion-event ring) is preserved.
    int sm_limit_pct = 0;                     // 0 = off
    unsigned sm_capped_count = 0;             // SMs in the capped partition
    std::vector<cudaStream_t> gemm_streams;   // per client; empty when off
    std::vector<cudaStream_t> last_stream;    // per client: stream of last op
    std::vector<cudaEvent_t> chain_ev;        // per client: 2 events, toggled
    std::vector<int> chain_toggle;
    std::vector<uint64_t> issued;
    // Per client: a sync-semantics record whose GPU work has been issued but
    // whose completion (-> REC_DONE) is still being polled. At most one per
    // client, because that client is blocked on it.
    std::vector<FuncRecord*> pending_sync;
    std::unique_ptr<Policy> policy;
    std::vector<IssueEntry> log;
    std::atomic<bool> stop{false};
    std::atomic<bool> running{false};

    // completion tracking (events)
    std::vector<std::vector<cudaEvent_t>> events;  // [client][kEventRing]
    std::vector<uint64_t> ev_issued;               // stream ops with an event
    std::vector<uint64_t> ev_done;                 // completion cursor
    std::vector<uint64_t> max_pending;             // high-water mark (stats)

    // live GPU feed (observer, optional)
    obs_live_get_fn live_get = nullptr;
    std::vector<int> live_slot;
    std::vector<int> live_slot_gemm;  // capped streams (SM limit), -1 if none
};

Sched g_sched;

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "[sched] CUDA error %d (%s) at %s:%d\n", err__,  \
                    cudaGetErrorString(err__), __FILE__, __LINE__);          \
            abort();                                                         \
        }                                                                    \
    } while (0)

// Record a completion event for the stream op just issued for `client`.
void record_completion_event(int client, cudaStream_t stream) {
    uint64_t seq = g_sched.ev_issued[client];
    if (seq - g_sched.ev_done[client] >= kEventRing) return;  // ring full: skip (never hit under lockstep)
    cudaEventRecord(g_sched.events[client][seq % kEventRing], stream);
    g_sched.ev_issued[client] = seq + 1;
}

// Advance each client's completion cursor: how many of its issued stream ops
// the GPU has finished. Non-blocking (cudaEventQuery). Rate-limited to every
// ~30 µs: each query is a CUDA runtime call (≈1 µs, and one CUPTI RUNTIME
// record when the observer is on) — unthrottled, the spin loop fired millions
// of them per second for no extra information.
void advance_completion(bool force = false) {
    static uint64_t last_poll = 0;
    uint64_t now = now_ns();
    if (!force && now - last_poll < 30000) return;
    last_poll = now;
    for (int c = 0; c < g_sched.num_clients; c++) {
        while (g_sched.ev_done[c] < g_sched.ev_issued[c] &&
               cudaEventQuery(g_sched.events[c][g_sched.ev_done[c] % kEventRing]) == cudaSuccess)
            g_sched.ev_done[c]++;
        uint64_t pending = g_sched.ev_issued[c] - g_sched.ev_done[c];
        if (pending > g_sched.max_pending[c]) g_sched.max_pending[c] = pending;
    }
}

#define CHECK_CU(call)                                                        \
    do {                                                                      \
        CUresult r__ = (call);                                                \
        if (r__ != CUDA_SUCCESS) {                                            \
            const char* s__ = nullptr;                                        \
            cuGetErrorString(r__, &s__);                                      \
            fprintf(stderr, "[sched] CU error %s at %s:%d\n",                 \
                    s__ ? s__ : "?", __FILE__, __LINE__);                     \
            abort();                                                          \
        }                                                                     \
    } while (0)

// Stream for this op under the SM-limit option: cuBLAS ops go to the capped
// green-context stream, everything that enqueues work goes to the main
// stream, and ops that only *order* against prior work (event records,
// redirected stream syncs) follow whatever stream the last op used. On a
// switch, chain with an event so the client's two streams stay LINEAR —
// op N+1 cannot start before op N completes, exactly like one stream.
// Two chain events per client, toggled: a pending cudaStreamWaitEvent
// snapshots the event's state at call time, so alternating two events is
// always safe under the scheduler's serial replay.
cudaStream_t route_stream(int client, OpKind kind) {
    if (g_sched.sm_limit_pct <= 0) return g_sched.streams[client];
    cudaStream_t cur = g_sched.last_stream[client];
    cudaStream_t want;
    switch (kind) {
        case OpKind::CublasGemmEx:
        case OpKind::CublasSgemm:
            want = g_sched.gemm_streams[client];
            break;
        case OpKind::EventRecord:
        case OpKind::StreamSync:
            return cur ? cur : g_sched.streams[client];  // order-only: no switch
        default:
            want = g_sched.streams[client];
            break;
    }
    if (cur && cur != want) {
        cudaEvent_t ev = g_sched.chain_ev[client * 2 + g_sched.chain_toggle[client]];
        g_sched.chain_toggle[client] ^= 1;
        CHECK_CUDA(cudaEventRecord(ev, cur));
        CHECK_CUDA(cudaStreamWaitEvent(want, ev, 0));
    }
    g_sched.last_stream[client] = want;
    return want;
}

// Execute one record on `client`'s colocator stream (or its SM-capped gemm
// stream under COLOCATOR_SM_LIMIT). Sets rec->ret and advances rec->state,
// which unblocks the spinning client thread.
void execute(int client, FuncRecord* rec) {
    cudaStream_t stream = route_stream(client, rec->kind);
    bool stream_op = false;  // did this record enqueue work on the stream?
    IssueEntry e{};
    e.client = client;
    e.op_id = rec->op_id;
    e.kind = rec->kind;
    e.t_intercept_ns = rec->t_intercept_ns;
    e.func = nullptr;
    e.bytes = rec->count;
    e.stream = (void*)stream;
    e.t_issue_ns = now_ns();

    // Branches set final_state instead of storing it: the state store is the
    // LAST touch of rec (it's on the client's stack — the client may resume
    // and invalidate it the moment the store lands). -1 = parked in
    // pending_sync (stays alive; poll_pending_syncs() stores DONE later).
    int final_state = -1;
    switch (rec->kind) {
        case OpKind::KernelLaunch:
            rec->ret = cudaLaunchKernel(rec->func, rec->grid, rec->block, rec->args,
                                        rec->shared_mem, stream);
            e.func = rec->func;
            e.gx = rec->grid.x;  e.gy = rec->grid.y;  e.gz = rec->grid.z;
            e.bx = rec->block.x; e.by = rec->block.y; e.bz = rec->block.z;
            stream_op = (rec->ret == cudaSuccess);
            final_state = REC_ISSUED;
            break;

        case OpKind::MemcpyAsync:
            rec->ret = cudaMemcpyAsync(rec->dst, rec->src, rec->count, rec->copy_kind, stream);
            stream_op = (rec->ret == cudaSuccess);
            final_state = REC_ISSUED;
            break;

        case OpKind::MemsetAsync:
            rec->ret = cudaMemsetAsync(rec->dst, rec->memset_value, rec->count, stream);
            stream_op = (rec->ret == cudaSuccess);
            final_state = REC_ISSUED;
            break;

        case OpKind::Memcpy:  // client called the sync variant
            rec->ret = cudaMemcpyAsync(rec->dst, rec->src, rec->count, rec->copy_kind, stream);
            stream_op = (rec->ret == cudaSuccess);
            if (rec->ret != cudaSuccess)
                final_state = REC_DONE;
            else
                g_sched.pending_sync[client] = rec;  // DONE via poll_pending_syncs()
            break;

        case OpKind::Memset:
            rec->ret = cudaMemsetAsync(rec->dst, rec->memset_value, rec->count, stream);
            stream_op = (rec->ret == cudaSuccess);
            if (rec->ret != cudaSuccess)
                final_state = REC_DONE;
            else
                g_sched.pending_sync[client] = rec;
            break;

        case OpKind::Malloc:
            rec->ret = cudaMalloc(rec->devptr_out, rec->count);
            final_state = REC_DONE;
            break;

        case OpKind::Free:
            // Order behind this client's in-flight work before freeing —
            // BOTH streams when the SM limit splits the client's work.
            rec->ret = cudaStreamSynchronize(stream);
            if (rec->ret == cudaSuccess && g_sched.sm_limit_pct > 0)
                rec->ret = cudaStreamSynchronize(g_sched.gemm_streams[client]);
            if (rec->ret == cudaSuccess) rec->ret = cudaFree(rec->dst);
            final_state = REC_DONE;
            break;

        case OpKind::StreamSync:
            // Redirected: complete when the client's colocator stream drains
            // (nothing to issue; DONE comes from poll_pending_syncs()).
            rec->ret = cudaSuccess;
            g_sched.pending_sync[client] = rec;
            break;

        case OpKind::EventRecord:
            // Redirected: record the client's event on its colocator stream,
            // in queue order behind its prior ops. Not counted as stream
            // work (zero duration; keeps in_flight = real work for policies).
            rec->ret = cudaEventRecordWithFlags(rec->event, stream, rec->event_flags);
            final_state = REC_ISSUED;
            break;

        case OpKind::CublasGemmEx: {
            // Replay the whole library call (Orion-style): point the client's
            // handle at its colocator stream, then let cuBLAS launch its
            // kernels from THIS thread — they inherit the stream, and the
            // interposers pass this thread through. Function pointers are
            // resolved via col_resolve, not linked: python loads torch's libs
            // RTLD_LOCAL, so plain dynamic linking could not bind them.
            // (The handle's workspace resets to cuBLAS's default when the
            // stream changes — fine for GEMV-sized ops; revisit if a workload
            // needs torch's oversized workspace for batched GEMMs.)
            using SetStreamFn = cublasStatus_t (*)(cublasHandle_t, cudaStream_t);
            using GemmExFn = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t,
                cublasOperation_t, int, int, int, const void*, const void*,
                cudaDataType, int, const void*, cudaDataType, int, const void*,
                void*, cudaDataType, int, cublasComputeType_t, cublasGemmAlgo_t);
            static SetStreamFn set_stream =
                (SetStreamFn)col_resolve("cublasSetStream_v2", "libcublas.so.12");
            static GemmExFn gemm_ex =
                (GemmExFn)col_resolve("cublasGemmEx", "libcublas.so.12");
            auto& b = rec->blas;
            cublasStatus_t st = set_stream((cublasHandle_t)b.handle, stream);
            if (st == CUBLAS_STATUS_SUCCESS)
                st = gemm_ex((cublasHandle_t)b.handle, (cublasOperation_t)b.transa,
                             (cublasOperation_t)b.transb, b.m, b.n, b.k, b.alpha,
                             b.A, (cudaDataType)b.a_type, b.lda,
                             b.B, (cudaDataType)b.b_type, b.ldb, b.beta,
                             b.C, (cudaDataType)b.c_type, b.ldc,
                             (cublasComputeType_t)b.compute_type,
                             (cublasGemmAlgo_t)b.algo);
            b.status = (int)st;
            rec->ret = (st == CUBLAS_STATUS_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
            stream_op = (st == CUBLAS_STATUS_SUCCESS);
            e.gx = b.m; e.gy = b.n; e.gz = b.k;  // GEMM dims in the grid slots
            final_state = REC_ISSUED;
            break;
        }

        case OpKind::CublasSgemm: {
            using SetStreamFn = cublasStatus_t (*)(cublasHandle_t, cudaStream_t);
            using SgemmFn = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t,
                cublasOperation_t, int, int, int, const float*, const float*, int,
                const float*, int, const float*, float*, int);
            static SetStreamFn set_stream =
                (SetStreamFn)col_resolve("cublasSetStream_v2", "libcublas.so.12");
            static SgemmFn sgemm =
                (SgemmFn)col_resolve("cublasSgemm_v2", "libcublas.so.12");
            auto& b = rec->blas;
            cublasStatus_t st = set_stream((cublasHandle_t)b.handle, stream);
            if (st == CUBLAS_STATUS_SUCCESS)
                st = sgemm((cublasHandle_t)b.handle, (cublasOperation_t)b.transa,
                           (cublasOperation_t)b.transb, b.m, b.n, b.k,
                           (const float*)b.alpha, (const float*)b.A, b.lda,
                           (const float*)b.B, b.ldb, (const float*)b.beta,
                           (float*)b.C, b.ldc);
            b.status = (int)st;
            rec->ret = (st == CUBLAS_STATUS_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
            stream_op = (st == CUBLAS_STATUS_SUCCESS);
            e.gx = b.m; e.gy = b.n; e.gz = b.k;
            final_state = REC_ISSUED;
            break;
        }

        default:
            fprintf(stderr, "[sched] unexpected record kind %d\n", (int)rec->kind);
            abort();
    }
    // Completion event must be enqueued before the client can push its next
    // op; safe here because the state store below hasn't unblocked it yet.
    if (stream_op) record_completion_event(client, stream);
    if (final_state >= 0)
        rec->state.store(final_state, std::memory_order_release);  // rec dead after this
    e.t_returned_ns = now_ns();
    g_sched.issued[client]++;
    g_sched.log.push_back(e);
}

// Complete pending sync-semantics records whose stream(s) have drained.
// Under the SM limit a client's work spans two streams; "my work so far is
// done" means both are empty.
void poll_pending_syncs() {
    for (int i = 0; i < g_sched.num_clients; i++) {
        FuncRecord* rec = g_sched.pending_sync[i];
        if (!rec) continue;
        cudaError_t q = cudaStreamQuery(g_sched.streams[i]);
        if (q == cudaErrorNotReady) continue;
        if (q == cudaSuccess && g_sched.sm_limit_pct > 0) {
            q = cudaStreamQuery(g_sched.gemm_streams[i]);
            if (q == cudaErrorNotReady) continue;
        }
        if (rec->ret == cudaSuccess) rec->ret = q;
        g_sched.pending_sync[i] = nullptr;
        rec->state.store(REC_DONE, std::memory_order_release);  // rec dead after this
    }
}

}  // namespace

extern "C" {

// Create per-client streams (priorities[i]: 0 = default; more negative =
// higher priority, clamped to the device range) and reset scheduler state.
// Call AFTER the CUDA context exists and BEFORE clients start work.
int col_setup(int num_clients, const int* priorities) {
    if (num_clients < 1 || num_clients > kMaxClients) return -1;
    g_sched.num_clients = num_clients;
    g_sched.streams.assign(num_clients, nullptr);
    g_sched.issued.assign(num_clients, 0);
    g_sched.pending_sync.assign(num_clients, nullptr);
    g_sched.ev_issued.assign(num_clients, 0);
    g_sched.ev_done.assign(num_clients, 0);
    g_sched.max_pending.assign(num_clients, 0);
    g_sched.events.assign(num_clients, {});
    g_sched.live_slot.assign(num_clients, -1);
    g_sched.log.clear();
    g_sched.log.reserve(1 << 20);
    g_sched.stop.store(false);

    // Policy selection: COLOCATOR_POLICY=fcfs (default: no throttle) |
    // throttle (in-flight cap from COLOCATOR_THROTTLE, default 2).
    const char* pol = getenv("COLOCATOR_POLICY");
    if (pol && strcmp(pol, "throttle") == 0) {
        const char* thr = getenv("COLOCATOR_THROTTLE");
        g_sched.policy = std::make_unique<ThrottledFcfsPolicy>(thr ? strtoull(thr, nullptr, 10) : 2);
    } else {
        if (pol && strcmp(pol, "fcfs") != 0)
            fprintf(stderr, "[sched] unknown COLOCATOR_POLICY=%s, using fcfs\n", pol);
        g_sched.policy = std::make_unique<FcfsPolicy>();
    }

    int lo, hi;  // numerically: lo = least prio (0), hi = greatest prio (negative)
    CHECK_CUDA(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    std::vector<int> prio(num_clients, 0);
    for (int i = 0; i < num_clients; i++) {
        int p = priorities ? priorities[i] : 0;
        if (p < hi) p = hi;
        if (p > lo) p = lo;
        prio[i] = p;
        CHECK_CUDA(cudaStreamCreateWithPriority(&g_sched.streams[i],
                                                cudaStreamNonBlocking, p));
        fprintf(stderr, "[sched] client %d: stream %p priority %d\n",
                i, (void*)g_sched.streams[i], p);
    }

    // SM limit (COLOCATOR_SM_LIMIT=<percent>): carve a green-context SM
    // partition and give every client a second stream inside it. execute()
    // routes cuBLAS ops there (route_stream, with event chaining on
    // switches). One shared partition: "GEMMs may use at most N% of the
    // GPU"; the remaining SMs stay exclusively available to everything else.
    g_sched.sm_limit_pct = 0;
    g_sched.gemm_streams.assign(num_clients, nullptr);
    g_sched.last_stream.assign(num_clients, nullptr);
    g_sched.chain_toggle.assign(num_clients, 0);
    const char* sml = getenv("COLOCATOR_SM_LIMIT");
    if (sml && strtol(sml, nullptr, 10) > 0) {
        int pct = (int)strtol(sml, nullptr, 10);
        if (pct > 99) pct = 99;
        CUdevice dev;
        CHECK_CU(cuDeviceGet(&dev, 0));
        CUdevResource res{};
        CHECK_CU(cuDeviceGetDevResource(dev, &res, CU_DEV_RESOURCE_TYPE_SM));
        unsigned want = res.sm.smCount * (unsigned)pct / 100;
        if (want < 8) want = 8;  // API minimum granularity
        CUdevResource split{}, rem{};
        unsigned nb = 1;
        CHECK_CU(cuDevSmResourceSplitByCount(&split, &nb, &res, &rem, 0, want));
        CUdevResourceDesc desc;
        CHECK_CU(cuDevResourceGenerateDesc(&desc, &split, 1));
        CUgreenCtx gctx;
        CHECK_CU(cuGreenCtxCreate(&gctx, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM));
        for (int i = 0; i < num_clients; i++) {
            CUstream s;
            CHECK_CU(cuGreenCtxStreamCreate(&s, gctx, CU_STREAM_NON_BLOCKING, prio[i]));
            g_sched.gemm_streams[i] = (cudaStream_t)s;
        }
        g_sched.chain_ev.resize(2 * num_clients);
        for (auto& e : g_sched.chain_ev)
            CHECK_CUDA(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        g_sched.sm_limit_pct = pct;
        g_sched.sm_capped_count = split.sm.smCount;
        fprintf(stderr, "[sched] SM limit %d%%: cuBLAS ops routed to green-context "
                        "streams on %u/%u SMs (event-chained with main streams)\n",
                pct, split.sm.smCount, res.sm.smCount);
    }
    // Completion-event pool (Orion-style): one event per issued stream op.
    for (int i = 0; i < num_clients; i++) {
        g_sched.events[i].resize(kEventRing);
        for (uint64_t j = 0; j < kEventRing; j++)
            CHECK_CUDA(cudaEventCreateWithFlags(&g_sched.events[i][j], cudaEventDisableTiming));
    }

    // Live GPU feed: present only if libobserver.so is loaded (dlsym probe).
    auto live_enable = (obs_live_enable_fn)dlsym(RTLD_DEFAULT, "obs_live_enable");
    auto live_track = (obs_live_track_fn)dlsym(RTLD_DEFAULT, "obs_live_track");
    g_sched.live_get = (obs_live_get_fn)dlsym(RTLD_DEFAULT, "obs_live_get");
    g_sched.live_slot_gemm.assign(num_clients, -1);
    if (live_enable && live_track && g_sched.live_get && live_enable(5) == 0) {
        for (int i = 0; i < num_clients; i++) {
            g_sched.live_slot[i] = live_track((void*)g_sched.streams[i]);
            if (g_sched.sm_limit_pct > 0)
                g_sched.live_slot_gemm[i] = live_track((void*)g_sched.gemm_streams[i]);
        }
        fprintf(stderr, "[sched] observer live feed connected\n");
    } else {
        g_sched.live_get = nullptr;
        fprintf(stderr, "[sched] observer live feed not available (events only)\n");
    }

    fprintf(stderr, "[sched] policy: %s, clients: %d (stream prio range: %d..%d)\n",
            g_sched.policy->name(), num_clients, lo, hi);
    return 0;
}

// Busy-wait scheduling loop. Blocks until col_stop(); call from a dedicated
// thread (ctypes releases the GIL around this call).
void col_run() {
    const int n = g_sched.num_clients;
    std::vector<FuncRecord*> heads(n, nullptr);
    SchedState state;
    state.num_clients = n;
    g_sched.running.store(true);

    while (!g_sched.stop.load(std::memory_order_relaxed)) {
        poll_pending_syncs();
        advance_completion();

        bool any = false;
        for (int i = 0; i < n; i++) {
            ClientSlot& c = g_col_clients[i];
            pthread_mutex_lock(&c.mtx);
            heads[i] = c.q.empty() ? nullptr : c.q.front();
            pthread_mutex_unlock(&c.mtx);
            any |= (heads[i] != nullptr);
        }
        if (!any) {
#if defined(__x86_64__)
            __builtin_ia32_pause();
#endif
            continue;
        }

        state.issued = g_sched.issued;
        state.in_flight.resize(n);
        for (int i = 0; i < n; i++)
            state.in_flight[i] = g_sched.ev_issued[i] - g_sched.ev_done[i];
        int pick = g_sched.policy->pick_next(heads, state);
        if (pick < 0) continue;

        execute(pick, heads[pick]);  // unblocks the client thread

        // Pop AFTER execute: the client spins on rec->state, not queue size,
        // so this ordering is safe and keeps head visible during execution.
        ClientSlot& c = g_col_clients[pick];
        pthread_mutex_lock(&c.mtx);
        c.q.pop();
        pthread_mutex_unlock(&c.mtx);
    }

    g_sched.running.store(false);
    if (g_sched.live_get) {  // let the observer's periodic flush deliver the tail
        timespec ts{0, 20 * 1000 * 1000};
        nanosleep(&ts, nullptr);
    }
    advance_completion(true);
    fprintf(stderr, "[sched] loop exited; issued per client:");
    for (int i = 0; i < n; i++)
        fprintf(stderr, " %llu", (unsigned long long)g_sched.issued[i]);
    fprintf(stderr, "\n");
    for (int i = 0; i < n; i++) {
        fprintf(stderr, "[sched] client %d state model: stream-ops %llu, event-completed %llu, "
                        "still pending %llu, max in-flight seen %llu",
                i,
                (unsigned long long)g_sched.ev_issued[i],
                (unsigned long long)g_sched.ev_done[i],
                (unsigned long long)(g_sched.ev_issued[i] - g_sched.ev_done[i]),
                (unsigned long long)g_sched.max_pending[i]);
        if (g_sched.live_get && g_sched.live_slot[i] >= 0) {
            unsigned long long k = 0, m = 0, busy = 0;
            g_sched.live_get(g_sched.live_slot[i], &k, &m, &busy);
            if (g_sched.live_slot_gemm[i] >= 0) {  // add the capped stream's share
                unsigned long long k2 = 0, m2 = 0, b2 = 0;
                g_sched.live_get(g_sched.live_slot_gemm[i], &k2, &m2, &b2);
                k += k2; m += m2; busy += b2;
            }
            fprintf(stderr, " | gpu measured (live): %llu kernels, %llu copies, busy %.1f ms",
                    k, m, busy / 1e6);
        }
        fprintf(stderr, "\n");
    }
}

void col_stop() { g_sched.stop.store(true); }

// Live queue-state accessors (valid while/after the loop runs).
uint64_t col_in_flight(int client) {
    if (client < 0 || client >= g_sched.num_clients) return 0;
    return g_sched.ev_issued[client] - g_sched.ev_done[client];
}

uint64_t col_max_in_flight(int client) {
    if (client < 0 || client >= g_sched.num_clients) return 0;
    return g_sched.max_pending[client];
}

int col_is_running() { return g_sched.running.load() ? 1 : 0; }

// Debug/verification accessor: the client's streams as created by col_setup
// (which=0 main, which=1 the SM-capped green-context stream, or null).
// Used by tools/verify_sm_limit to run an %smid census on the REAL streams.
void* col_debug_stream(int client, int which) {
    if (client < 0 || client >= g_sched.num_clients) return nullptr;
    if (which == 1)
        return g_sched.sm_limit_pct > 0 ? (void*)g_sched.gemm_streams[client] : nullptr;
    return (void*)g_sched.streams[client];
}

uint64_t col_issued_count(int client) {
    if (client < 0 || client >= g_sched.num_clients) return 0;
    return g_sched.issued[client];
}

// Dump the issue log as CSV. Returns number of entries written, -1 on error.
long col_dump_issue_log(const char* path) {
    FILE* f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "client,op_id,kind,t_intercept_ns,t_issue_ns,t_returned_ns,"
               "func,grid_x,grid_y,grid_z,block_x,block_y,block_z,bytes,stream\n");
    for (const IssueEntry& e : g_sched.log)
        fprintf(f, "%d,%llu,%s,%llu,%llu,%llu,%p,%u,%u,%u,%u,%u,%u,%zu,%p\n",
                e.client, (unsigned long long)e.op_id, op_kind_name(e.kind),
                (unsigned long long)e.t_intercept_ns,
                (unsigned long long)e.t_issue_ns,
                (unsigned long long)e.t_returned_ns,
                e.func, e.gx, e.gy, e.gz, e.bx, e.by, e.bz, e.bytes, e.stream);
    fclose(f);
    return (long)g_sched.log.size();
}

}  // extern "C"
