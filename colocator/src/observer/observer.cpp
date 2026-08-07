// libobserver.so — passive CUPTI-based observer.
//
// Records, via the CUPTI Activity API, when GPU work ACTUALLY ran:
//   * CONCURRENT_KERNEL activity — per-kernel GPU start/end timestamps,
//     stream, grid/block, mangled name, correlation id. (The CONCURRENT
//     variant, unlike KERNEL, does not serialize kernel execution.)
//   * MEMCPY activity — same for DMA copies.
//   * RUNTIME activity — one record per CUDA runtime API call, carrying the
//     calling thread id and the same correlation id as the GPU activity it
//     caused. This is what lets analysis join GPU records back to the
//     scheduler's issue log: the scheduler launches ops one at a time, so its
//     runtime records in start-time order map 1:1 onto issue-log rows.
//
// Non-interference by construction: this library never touches the
// colocator's queues, locks, or streams. Activity buffers are handed to CUPTI
// asynchronously and only drained at obs_dump() (end of run). CUPTI's own
// per-launch overhead (sub-microsecond, in-driver) applies equally to all
// clients. CUPTI's worker thread is not a registered client, so anything it
// does passes through the interposers untouched.
//
// Clock domains: CUPTI timestamps come from cuptiGetTimestamp(); the
// colocator logs CLOCK_MONOTONIC_RAW. obs_init()/obs_dump() sample both
// side-by-side and write the pairs to calib.csv; analysis maps linearly.
//
// Driven from Python via ctypes: obs_init() before CUDA work, obs_dump(dir)
// at teardown.

#include <cupti.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

struct KernelRow {
    uint32_t correlation;
    uint64_t start, end;
    uint32_t device, stream;
    int32_t gx, gy, gz, bx, by, bz;
    std::string name;
};

struct MemcpyRow {
    uint32_t correlation;
    uint64_t start, end;
    uint32_t device, stream;
    uint64_t bytes;
    uint8_t copy_kind;
};

struct RuntimeRow {
    uint32_t correlation;
    uint32_t cbid;
    uint32_t thread_id;   // Linux TID of the caller
    uint64_t start, end;  // host-side duration of the API call
};

struct Calib {  // side-by-side clock samples
    uint64_t cupti_ns, mono_ns;
    const char* tag;
};

std::mutex g_mtx;  // guards the row vectors (bufferCompleted may run on any thread)
std::vector<KernelRow> g_kernels;
std::vector<MemcpyRow> g_memcpys;
std::vector<RuntimeRow> g_runtime;
std::vector<Calib> g_calib;
std::atomic<uint64_t> g_dropped{0};
bool g_inited = false;

// ---- live feed (opt-in) ----------------------------------------------------
// Lets the scheduler cheaply poll "what has the GPU actually finished on this
// stream" while the run is in progress. CUPTI delivers activity records only
// AFTER a kernel completes, so this is a completion feed with a small lag
// (bounded by the periodic flush, obs_live_enable(period_ms)) — it cannot say
// what is on the SMs this instant. Counters are updated by CUPTI's own worker
// thread inside buffer_completed; readers just load atomics, so the observer
// stays passive with respect to the colocator's queues and streams.
constexpr int kMaxTracked = 8;
struct LiveSlot {
    std::atomic<uint32_t> stream_id{0xFFFFFFFFu};
    std::atomic<unsigned long long> kernels{0};   // completed kernels
    std::atomic<unsigned long long> memcpys{0};   // completed copies
    std::atomic<unsigned long long> busy_ns{0};   // sum of measured kernel durations
    std::atomic<unsigned long long> last_end{0};  // CUPTI ns of last completion
};
LiveSlot g_live[kMaxTracked];
std::atomic<int> g_live_n{0};

uint64_t mono_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

void sample_calib(const char* tag) {
    // Bracket the CUPTI read with two monotonic reads and take the midpoint.
    uint64_t m0 = mono_ns();
    uint64_t c = 0;
    cuptiGetTimestamp(&c);
    uint64_t m1 = mono_ns();
    g_calib.push_back({c, m0 + (m1 - m0) / 2, tag});
}

#define CHECK_CUPTI(call)                                                  \
    do {                                                                   \
        CUptiResult r__ = (call);                                          \
        if (r__ != CUPTI_SUCCESS) {                                        \
            const char* s = nullptr;                                       \
            cuptiGetResultString(r__, &s);                                 \
            fprintf(stderr, "[observer] CUPTI error %s at %s:%d\n",        \
                    s ? s : "?", __FILE__, __LINE__);                      \
            abort();                                                       \
        }                                                                  \
    } while (0)

constexpr size_t kBufSize = 8 * 1024 * 1024;

void CUPTIAPI buffer_requested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
    *buffer = (uint8_t*)aligned_alloc(8, kBufSize);
    *size = kBufSize;
    *maxNumRecords = 0;  // as many as fit
}

void CUPTIAPI buffer_completed(CUcontext, uint32_t, uint8_t* buffer,
                               size_t /*size*/, size_t validSize) {
    CUpti_Activity* rec = nullptr;
    std::lock_guard<std::mutex> lock(g_mtx);
    while (true) {
        CUptiResult r = cuptiActivityGetNextRecord(buffer, validSize, &rec);
        if (r == CUPTI_ERROR_MAX_LIMIT_REACHED) break;
        if (r != CUPTI_SUCCESS) break;
        switch (rec->kind) {
            case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
                auto* k = (CUpti_ActivityKernel9*)rec;
                g_kernels.push_back({k->correlationId, k->start, k->end,
                                     k->deviceId, k->streamId,
                                     k->gridX, k->gridY, k->gridZ,
                                     k->blockX, k->blockY, k->blockZ,
                                     k->name ? k->name : "?"});
                for (int i = 0, n = g_live_n.load(); i < n; i++)
                    if (g_live[i].stream_id.load() == k->streamId) {
                        g_live[i].kernels.fetch_add(1);
                        g_live[i].busy_ns.fetch_add(k->end - k->start);
                        g_live[i].last_end.store(k->end);
                    }
                break;
            }
            case CUPTI_ACTIVITY_KIND_MEMCPY: {
                auto* m = (CUpti_ActivityMemcpy6*)rec;
                g_memcpys.push_back({m->correlationId, m->start, m->end,
                                     m->deviceId, m->streamId, m->bytes,
                                     m->copyKind});
                for (int i = 0, n = g_live_n.load(); i < n; i++)
                    if (g_live[i].stream_id.load() == m->streamId) {
                        g_live[i].memcpys.fetch_add(1);
                        g_live[i].last_end.store(m->end);
                    }
                break;
            }
            case CUPTI_ACTIVITY_KIND_RUNTIME: {
                auto* a = (CUpti_ActivityAPI*)rec;
                // Keep only the cbids analysis correlates on (kernel launch,
                // memcpyAsync). The scheduler's event/stream polling emits
                // millions of other RUNTIME records per run — storing them
                // ballooned memory and dump size for zero analytical value.
                if (a->cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 ||
                    a->cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020)
                    g_runtime.push_back({a->correlationId, a->cbid, a->threadId,
                                         a->start, a->end});
                break;
            }
            default:
                break;
        }
    }
    size_t dropped = 0;
    cuptiActivityGetNumDroppedRecords(nullptr, 0, &dropped);
    if (dropped) g_dropped.fetch_add(dropped);
    free(buffer);
}

}  // namespace

extern "C" {

int obs_init() {
    if (g_inited) return 0;
    // Report Linux TIDs in RUNTIME records (default is a CUPTI-internal id),
    // so analysis can match records to the scheduler/client thread ids.
    CHECK_CUPTI(cuptiSetThreadIdType(CUPTI_ACTIVITY_THREAD_ID_TYPE_SYSTEM));
    CHECK_CUPTI(cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed));
    CHECK_CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    CHECK_CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CHECK_CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    sample_calib("init");
    g_inited = true;
    fprintf(stderr, "[observer] CUPTI activity collection enabled\n");
    return 0;
}

// ---- live feed API (called by the scheduler via dlsym) ----

// Periodic flusher thread. cuptiActivityFlushPeriod turned out NOT to deliver
// partially-filled buffers on this CUPTI (2025.1 / CUDA 12.8) — records only
// arrived when an 8 MB buffer filled or at the forced obs_dump flush, so live
// counters could stay at zero for a whole run (verified with a standalone
// probe). Instead the observer runs its own thread calling
// cuptiActivityFlushAll(0) every period: delivery lag is then bounded by the
// period deterministically. The thread is observer-owned and touches nothing
// of the colocator.
std::atomic<bool> g_flusher_on{false};
std::atomic<unsigned> g_flush_period_ms{10};
pthread_t g_flusher;

void* flusher_main(void*) {
    while (g_flusher_on.load()) {
        cuptiActivityFlushAll(0);
        usleep(g_flush_period_ms.load() * 1000);
    }
    return nullptr;
}

// Turn on periodic flushing so completion records reach the live counters
// within ~period_ms instead of waiting for a full 8 MB buffer / obs_dump.
int obs_live_enable(unsigned period_ms) {
    if (!g_inited) return -1;
    g_flush_period_ms.store(period_ms ? period_ms : 10);
    if (!g_flusher_on.exchange(true))
        pthread_create(&g_flusher, nullptr, flusher_main, nullptr);
    fprintf(stderr, "[observer] live feed enabled (flusher thread, every %u ms)\n",
            g_flush_period_ms.load());
    return 0;
}

// Register a CUDA stream for live tracking; returns a slot handle for
// obs_live_get, or -1. Must be called from a thread with a current CUDA
// context (cuptiGetStreamIdEx resolves the stream in the current context).
int obs_live_track(void* cuda_stream) {
    uint32_t sid = 0;
    if (cuptiGetStreamIdEx(nullptr, (CUstream)cuda_stream, 0, &sid) != CUPTI_SUCCESS)
        return -1;
    int i = g_live_n.fetch_add(1);
    if (i >= kMaxTracked) return -1;
    g_live[i].stream_id.store(sid);
    fprintf(stderr, "[observer] live-tracking stream %p (cupti id %u) as slot %d\n",
            cuda_stream, sid, i);
    return i;
}

// Poll a tracked stream's completion counters (measured by CUPTI, lag ≤ flush
// period): completed kernels, completed memcpys, summed real kernel busy ns.
int obs_live_get(int slot, unsigned long long* kernels, unsigned long long* memcpys,
                 unsigned long long* busy_ns) {
    if (slot < 0 || slot >= g_live_n.load()) return -1;
    if (kernels) *kernels = g_live[slot].kernels.load();
    if (memcpys) *memcpys = g_live[slot].memcpys.load();
    if (busy_ns) *busy_ns = g_live[slot].busy_ns.load();
    return 0;
}

// Flush everything CUPTI has and write CSVs into `dir` (must exist):
//   obs_kernels.csv, obs_memcpys.csv, obs_runtime.csv, obs_calib.csv
// Returns number of kernel records written, -1 on error.
long obs_dump(const char* dir) {
    if (!g_inited) return -1;
    sample_calib("dump");
    CHECK_CUPTI(cuptiActivityFlushAll(1));  // forced flush, blocking

    std::lock_guard<std::mutex> lock(g_mtx);
    std::string base(dir);
    {
        FILE* f = fopen((base + "/obs_kernels.csv").c_str(), "w");
        if (!f) return -1;
        fprintf(f, "correlation,start_ns,end_ns,device,stream,"
                   "grid_x,grid_y,grid_z,block_x,block_y,block_z,name\n");
        for (auto& k : g_kernels)
            fprintf(f, "%u,%llu,%llu,%u,%u,%d,%d,%d,%d,%d,%d,\"%s\"\n",
                    k.correlation, (unsigned long long)k.start,
                    (unsigned long long)k.end, k.device, k.stream,
                    k.gx, k.gy, k.gz, k.bx, k.by, k.bz, k.name.c_str());
        fclose(f);
    }
    {
        FILE* f = fopen((base + "/obs_memcpys.csv").c_str(), "w");
        if (!f) return -1;
        fprintf(f, "correlation,start_ns,end_ns,device,stream,bytes,copy_kind\n");
        for (auto& m : g_memcpys)
            fprintf(f, "%u,%llu,%llu,%u,%u,%llu,%u\n",
                    m.correlation, (unsigned long long)m.start,
                    (unsigned long long)m.end, m.device, m.stream,
                    (unsigned long long)m.bytes, m.copy_kind);
        fclose(f);
    }
    {
        FILE* f = fopen((base + "/obs_runtime.csv").c_str(), "w");
        if (!f) return -1;
        fprintf(f, "correlation,cbid,thread_id,start_ns,end_ns\n");
        for (auto& r : g_runtime)
            fprintf(f, "%u,%u,%u,%llu,%llu\n",
                    r.correlation, r.cbid, r.thread_id,
                    (unsigned long long)r.start, (unsigned long long)r.end);
        fclose(f);
    }
    {
        FILE* f = fopen((base + "/obs_calib.csv").c_str(), "w");
        if (!f) return -1;
        fprintf(f, "tag,cupti_ns,mono_ns\n");
        for (auto& c : g_calib)
            fprintf(f, "%s,%llu,%llu\n", c.tag,
                    (unsigned long long)c.cupti_ns, (unsigned long long)c.mono_ns);
        fclose(f);
    }
    if (g_dropped.load())
        fprintf(stderr, "[observer] WARNING: CUPTI dropped %llu records\n",
                (unsigned long long)g_dropped.load());
    fprintf(stderr, "[observer] dumped %zu kernels, %zu memcpys, %zu runtime records\n",
            g_kernels.size(), g_memcpys.size(), g_runtime.size());
    return (long)g_kernels.size();
}

}  // extern "C"
