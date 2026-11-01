// activity_tracer.cpp
//
// Injection-based CUPTI activity tracer.
//
// How injection works
// -------------------
// Setting CUDA_INJECTION64_PATH=/path/to/libactivity_tracer.so causes the
// CUDA runtime to dlopen this library and call InitializeInjection() before
// any CUDA API is exposed to the target application. No changes to the target
// binary are required.
//
// What this tracer does
// ---------------------
// 1. Registers two CUPTI buffer callbacks (bufferRequested / bufferCompleted)
//    and enables the CONCURRENT_KERNEL activity kind so CUPTI fills buffers
//    with one record per kernel execution.
// 2. On each completed buffer, iterates records and accumulates per-kernel
//    launch counts and GPU execution time into two maps:
//      g_kernel_stats_global  – all-time totals since process start
//      g_kernel_stats_recent  – rolling 5-second sliding window
// 3. A background thread (periodicRecentWriter) rewrites the recent-5s CSV
//    every 5 seconds so the file always reflects current activity.
// 4. At process exit (registered via std::atexit), flushes any in-flight
//    CUPTI buffers, stops the writer thread, and writes final CSV files plus
//    a stderr summary of the top-10 kernels by duration and launch count.

#include <cupti.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <cxxabi.h>  // abi::__cxa_demangle — demangle C++ mangled kernel names

// Size of each CUPTI activity buffer we hand to the runtime.
// Larger buffers reduce callback frequency but increase peak memory use.
#define BUF_SIZE (32 * 1024)

// Sliding-window width in nanoseconds (5 seconds).
// Events older than (latest_end - RECENT_WINDOW_NS) are evicted from the
// recent map and deque.
#define RECENT_WINDOW_NS (5ull * 1000ull * 1000ull * 1000ull)

// ---------------------------------------------------------------------------
// Error-checking macro
// ---------------------------------------------------------------------------
// Wraps every CUPTI call. On failure, prints file/line/message and exits.
// We exit rather than returning an error because CUPTI failures are not
// recoverable and silently continuing would produce corrupt data.
#define CUPTI_CALL(call)                                                       \
    do {                                                                       \
        CUptiResult _status = call;                                            \
        if (_status != CUPTI_SUCCESS) {                                        \
            const char* errstr = nullptr;                                      \
            cuptiGetResultString(_status, &errstr);                            \
            std::fprintf(stderr, "CUPTI error at %s:%d: %s\n",                 \
                         __FILE__, __LINE__, errstr ? errstr : "unknown");     \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

// Per-kernel running totals: how many times it launched and cumulative GPU
// execution time in nanoseconds (derived from CUPTI record start/end timestamps).
struct KernelStats {
    uint64_t count = 0;
    uint64_t total_ns = 0;
};

// One entry in the sliding-window deque.
// We need end_ns to decide when the event ages out of the 5-second window,
// and duration_ns so we can subtract it from the recent totals on eviction.
struct KernelEvent {
    std::string name;
    uint64_t end_ns = 0;
    uint64_t duration_ns = 0;
};

// Flattened snapshot of a KernelStats map entry, used for sorting and CSV
// output without holding the mutex during file I/O.
struct Row {
    std::string name;
    uint64_t count;
    uint64_t total_ns;
};

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

static std::unordered_map<std::string, KernelStats> g_kernel_stats_global;
static std::unordered_map<std::string, KernelStats> g_kernel_stats_recent;
static std::deque<KernelEvent> g_recent_events;

// Held only for brief critical sections; file I/O happens after releasing it.
static std::mutex g_stats_mutex;

// Set to true by finalizeTracer() to signal the writer thread to exit cleanly.
static std::atomic<bool> g_stop_writer{false};
// Guards against double-initialization if the runtime somehow calls
// InitializeInjection() more than once.
static std::atomic<bool> g_initialized{false};
// Background thread that periodically rewrites the recent-5s CSV.
static std::thread g_writer_thread;

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

// Demangle a C++ mangled symbol name using the Itanium ABI demangler.
// Returns the original string unchanged if demangling fails (e.g. for C symbols
// or plain function names that are already human-readable).
static std::string demangleName(const char* name) {
    if (!name) return "unknown";

    int status = 0;
    // __cxa_demangle allocates the result with malloc; wrap in unique_ptr so
    // it is freed even if we return early.
    std::unique_ptr<char, decltype(&std::free)> demangled(
        abi::__cxa_demangle(name, nullptr, nullptr, &status),
        &std::free
    );

    if (status == 0 && demangled) {
        return std::string(demangled.get());
    }
    return std::string(name);
}

// ---------------------------------------------------------------------------
// Output path helpers
// ---------------------------------------------------------------------------
// The output directory is controlled by the CUPTI_TRACE_OUTDIR environment
// variable, defaulting to "output" relative to the working directory.

static std::string getOutdir() {
    const char* env = std::getenv("CUPTI_TRACE_OUTDIR");
    if (env && std::strlen(env) > 0) {
        return std::string(env);
    }
    return "output";
}

static std::string getGlobalCsvPath() {
    return getOutdir() + "/kernel_hotspots_global.csv";
}

static std::string getRecentCsvPath() {
    return getOutdir() + "/kernel_hotspots_recent_5s.csv";
}

// Create the output directory if it does not already exist.
static void setupOutputPaths() {
    std::filesystem::create_directories(getOutdir());
}

// ---------------------------------------------------------------------------
// CSV output
// ---------------------------------------------------------------------------

// Snapshot a stats map into a vector of Row objects. Called with the lock held
// (snapshotRows itself does not lock — the caller must).
// Copying into a vector lets us release the lock before doing file I/O.
static std::vector<Row> snapshotRows(
    const std::unordered_map<std::string, KernelStats>& table) {
    std::vector<Row> rows;
    rows.reserve(table.size());
    for (const auto& kv : table) {
        rows.push_back({kv.first, kv.second.count, kv.second.total_ns});
    }
    return rows;
}

// Write rows to a CSV file. Converts raw nanosecond counts to the more
// readable milliseconds (total) and microseconds (average) units.
// Kernel names are double-quoted to handle commas and spaces in mangled names.
static void writeCsvFromRows(const std::vector<Row>& rows, const char* path) {
    std::ofstream ofs(path);
    if (!ofs) {
        std::fprintf(stderr, "[tracer] Failed to open CSV file: %s\n", path);
        return;
    }

    ofs << "kernel_name,launch_count,total_duration_ms,avg_duration_us\n";
    for (const auto& r : rows) {
        double total_ms = static_cast<double>(r.total_ns) / 1e6;
        double avg_us =
            r.count ? static_cast<double>(r.total_ns) / r.count / 1e3 : 0.0;

        ofs << "\"" << r.name << "\","
            << r.count << ","
            << total_ms << ","
            << avg_us << "\n";
    }
}

// Take a snapshot of the global map (under lock) and write it to disk.
static void writeGlobalCsv() {
    std::vector<Row> rows;
    {
        std::lock_guard<std::mutex> lock(g_stats_mutex);
        rows = snapshotRows(g_kernel_stats_global);
    }
    const std::string path = getGlobalCsvPath();
    writeCsvFromRows(rows, path.c_str());
    std::fprintf(stderr, "[tracer] Wrote global CSV to %s\n", path.c_str());
}

// Take a snapshot of the recent map (under lock) and write it to disk.
// Called both by the periodic background thread and at finalization.
static void writeRecentCsv() {
    std::vector<Row> rows;
    {
        std::lock_guard<std::mutex> lock(g_stats_mutex);
        rows = snapshotRows(g_kernel_stats_recent);
    }
    const std::string path = getRecentCsvPath();
    writeCsvFromRows(rows, path.c_str());
    std::fprintf(stderr, "[tracer] Updated recent 5s CSV: %s\n", path.c_str());
}

// ---------------------------------------------------------------------------
// Stderr summary
// ---------------------------------------------------------------------------

// Print the top-10 kernels sorted by two criteria (total GPU time, then launch
// count) to stderr. Called once at finalization for both the global and recent
// snapshots. No lock needed — called with already-snapshotted row vectors.
static void printTopKernelsFromRows(const std::vector<Row>& rows,
                                    const char* title_prefix) {
    // Sort a copy by descending total duration.
    auto by_total = rows;
    std::sort(by_total.begin(), by_total.end(),
              [](const Row& a, const Row& b) {
                  return a.total_ns > b.total_ns;
              });

    // Sort a second copy by descending launch count.
    auto by_count = rows;
    std::sort(by_count.begin(), by_count.end(),
              [](const Row& a, const Row& b) {
                  return a.count > b.count;
              });

    std::fprintf(stderr, "\n=== %s: Top kernels by total duration ===\n", title_prefix);
    for (size_t i = 0; i < std::min<size_t>(10, by_total.size()); ++i) {
        const auto& r = by_total[i];
        double total_ms = static_cast<double>(r.total_ns) / 1e6;
        double avg_us =
            r.count ? static_cast<double>(r.total_ns) / r.count / 1e3 : 0.0;
        std::fprintf(stderr, "%2zu. %-80s count=%" PRIu64
                             " total_ms=%.3f avg_us=%.3f\n",
                     i + 1, r.name.c_str(), r.count, total_ms, avg_us);
    }

    std::fprintf(stderr, "\n=== %s: Top kernels by launch count ===\n", title_prefix);
    for (size_t i = 0; i < std::min<size_t>(10, by_count.size()); ++i) {
        const auto& r = by_count[i];
        double total_ms = static_cast<double>(r.total_ns) / 1e6;
        double avg_us =
            r.count ? static_cast<double>(r.total_ns) / r.count / 1e3 : 0.0;
        std::fprintf(stderr, "%2zu. %-80s count=%" PRIu64
                             " total_ms=%.3f avg_us=%.3f\n",
                     i + 1, r.name.c_str(), r.count, total_ms, avg_us);
    }
}

// ---------------------------------------------------------------------------
// Sliding-window eviction
// ---------------------------------------------------------------------------

// Remove events from the front of g_recent_events whose end timestamp is
// older than (latest_end_ns - RECENT_WINDOW_NS), and subtract their
// contribution from g_kernel_stats_recent.
//
// Must be called with g_stats_mutex held.
//
// Design note: we keep a separate deque instead of re-scanning the map
// because the map only stores aggregated totals — we need per-event duration
// to undo individual contributions as they expire.
static void evictOldRecentEventsLocked(uint64_t latest_end_ns) {
    // Guard against underflow if latest_end_ns < RECENT_WINDOW_NS (very early
    // in the run when timestamps are small).
    const uint64_t cutoff = (latest_end_ns > RECENT_WINDOW_NS)
                                ? (latest_end_ns - RECENT_WINDOW_NS)
                                : 0;

    while (!g_recent_events.empty() && g_recent_events.front().end_ns < cutoff) {
        const KernelEvent& ev = g_recent_events.front();
        auto it = g_kernel_stats_recent.find(ev.name);
        if (it != g_kernel_stats_recent.end()) {
            // Undo this event's contribution to the recent totals.
            if (it->second.count > 0) {
                it->second.count -= 1;
            }
            if (it->second.total_ns >= ev.duration_ns) {
                it->second.total_ns -= ev.duration_ns;
            } else {
                it->second.total_ns = 0;  // guard against rounding drift
            }

            // Remove the entry entirely once no recent launches remain,
            // so the recent map only contains kernels active in the window.
            if (it->second.count == 0) {
                g_kernel_stats_recent.erase(it);
            }
        }
        g_recent_events.pop_front();
    }
}

// ---------------------------------------------------------------------------
// CUPTI record processing
// ---------------------------------------------------------------------------

// Process a single CUpti_Activity record. We only care about
// CONCURRENT_KERNEL records; all other activity kinds are silently skipped.
//
// CUpti_ActivityKernel9 is the version of the kernel activity struct available
// in recent CUPTI releases. It provides the mangled kernel name and GPU-side
// start/end timestamps in nanoseconds (CUPTI device clock domain).
static void handleKernelRecord(const CUpti_Activity* record) {
    if (record->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
        return;
    }

    const auto* k = reinterpret_cast<const CUpti_ActivityKernel9*>(record);

    const char* raw_name = k->name;
    const uint64_t start = k->start;
    const uint64_t end = k->end;

    // Skip malformed records (null name or impossible timestamps).
    if (!raw_name || end < start) {
        return;
    }

    const std::string pretty_name = demangleName(raw_name);
    const uint64_t duration_ns = end - start;

    std::lock_guard<std::mutex> lock(g_stats_mutex);

    // Update all-time global totals.
    {
        auto& s = g_kernel_stats_global[pretty_name];
        s.count += 1;
        s.total_ns += duration_ns;
    }

    // Update sliding-window totals and evict expired events.
    {
        auto& s = g_kernel_stats_recent[pretty_name];
        s.count += 1;
        s.total_ns += duration_ns;

        g_recent_events.push_back({pretty_name, end, duration_ns});
        // Evict any events that have now fallen outside the 5-second window
        // relative to this (the most recent) event's end timestamp.
        evictOldRecentEventsLocked(end);
    }
}

// ---------------------------------------------------------------------------
// CUPTI buffer callbacks
// ---------------------------------------------------------------------------

// Called by CUPTI when it needs a new buffer to write activity records into.
// We allocate a fixed-size heap buffer and set maxNumRecords=0 so CUPTI fills
// the buffer as full as possible before calling bufferCompleted.
static void CUPTIAPI bufferRequested(uint8_t** buffer,
                                     size_t* size,
                                     size_t* maxNumRecords) {
    *size = BUF_SIZE;
    *buffer = reinterpret_cast<uint8_t*>(std::malloc(BUF_SIZE));
    if (!*buffer) {
        std::fprintf(stderr, "[tracer] Failed to allocate CUPTI buffer\n");
        std::exit(EXIT_FAILURE);
    }
    // 0 means "fill the buffer completely"; CUPTI will stop when it runs out
    // of space rather than after a fixed record count.
    *maxNumRecords = 0;
}

// Called by CUPTI when a buffer has been filled (or at flush time).
// Iterates every record in the valid portion of the buffer, dispatches kernel
// records to handleKernelRecord, logs dropped records, then frees the buffer.
//
// This callback runs on the thread that triggered the flush (either the CUDA
// runtime's internal flush or our cuptiActivityFlushAll call at shutdown).
static void CUPTIAPI bufferCompleted(CUcontext ctx,
                                     uint32_t streamId,
                                     uint8_t* buffer,
                                     size_t size,
                                     size_t validSize) {
    (void)size;  // total allocated size; not needed since CUPTI tracks validSize

    CUpti_Activity* record = nullptr;

    // Walk every packed activity record in the buffer.
    while (true) {
        CUptiResult status =
            cuptiActivityGetNextRecord(buffer, validSize, &record);

        if (status == CUPTI_SUCCESS) {
            handleKernelRecord(record);
        } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
            // Normal end-of-buffer sentinel; no more records.
            break;
        } else {
            const char* errstr = nullptr;
            cuptiGetResultString(status, &errstr);
            std::fprintf(stderr,
                         "[tracer] cuptiActivityGetNextRecord error: %s\n",
                         errstr ? errstr : "unknown");
            break;
        }
    }

    // Report how many records CUPTI had to discard because the buffer was
    // full before we could process them. Non-zero values indicate BUF_SIZE
    // should be increased or the target app is launching kernels very rapidly.
    size_t dropped = 0;
    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped > 0) {
        std::fprintf(stderr, "[tracer] Dropped %zu CUPTI activity records\n", dropped);
    }

    std::free(buffer);
}

// ---------------------------------------------------------------------------
// Background writer thread
// ---------------------------------------------------------------------------

// Runs on g_writer_thread. Wakes every 5 seconds to overwrite the recent CSV
// with a fresh snapshot of the current sliding-window data.
// The 5-second sleep matches the window width so the file always reflects the
// most recently completed window when read by external tools.
static void periodicRecentWriter() {
    while (!g_stop_writer.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        // Re-check after waking in case we were signalled to stop during sleep.
        if (g_stop_writer.load()) break;
        writeRecentCsv();
    }
}

// ---------------------------------------------------------------------------
// Finalization (registered with std::atexit)
// ---------------------------------------------------------------------------

// Called when the process exits normally. Sequence:
//   1. Sync the device so in-flight GPU work is complete.
//   2. Flush any CUPTI buffers that have not been delivered yet (flag=1
//      means blocking flush).
//   3. Signal and join the writer thread.
//   4. Write final CSVs and print the stderr summary.
//
// Note: cudaDeviceSynchronize and cuptiActivityFlushAll are best-effort here.
// If the CUDA runtime is already partially torn down (e.g. due to an abnormal
// exit), these calls may fail or be no-ops — that is acceptable.
static void finalizeTracer() {
    if (!g_initialized.load()) {
        return;
    }

    // Best effort only. Do not abort if runtime is already tearing down.
    cudaDeviceSynchronize();
    cuptiActivityFlushAll(1);

    // Stop the background writer thread before taking the final snapshot so
    // the thread does not race with our writeRecentCsv() call below.
    g_stop_writer.store(true);
    if (g_writer_thread.joinable()) {
        g_writer_thread.join();
    }

    // Snapshot under lock, then do all I/O outside the lock.
    std::vector<Row> global_rows, recent_rows;
    {
        std::lock_guard<std::mutex> lock(g_stats_mutex);
        global_rows = snapshotRows(g_kernel_stats_global);
        recent_rows = snapshotRows(g_kernel_stats_recent);
    }

    printTopKernelsFromRows(global_rows, "Global");
    printTopKernelsFromRows(recent_rows, "Recent 5s");
    writeGlobalCsv();
    writeRecentCsv();

    std::fprintf(stderr, "[tracer] CUPTI tracer finalized.\n");
}

// ---------------------------------------------------------------------------
// Injection entry point
// ---------------------------------------------------------------------------

// The CUDA runtime calls this function (by name, via dlsym) immediately after
// loading the library pointed to by CUDA_INJECTION64_PATH. Returning 1
// signals success; returning 0 would cause the runtime to unload the library.
//
// We use g_initialized as a compare-and-swap guard so this function is
// idempotent if somehow called multiple times.
extern "C" int InitializeInjection(void) {
    // exchange returns the old value; if it was already true, someone else
    // initialized first — return immediately without double-registering.
    if (g_initialized.exchange(true)) {
        return 1;
    }

    setupOutputPaths();

    // Register the buffer lifecycle callbacks and enable kernel activity
    // recording. All subsequent kernel launches on any stream will be recorded.
    CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));

    // Start the background thread that keeps the recent-5s CSV up to date.
    g_stop_writer.store(false);
    g_writer_thread = std::thread(periodicRecentWriter);

    // Register the finalization handler. std::atexit handlers run in LIFO
    // order, so if the application registers its own atexit handlers after
    // this point our finalizeTracer will run first (desired: we want to flush
    // CUPTI before the CUDA context is destroyed by other cleanup).
    std::atexit(finalizeTracer);

    const std::string outdir = getOutdir();
    std::fprintf(stderr, "[tracer] CUPTI tracer initialized. outdir=%s\n", outdir.c_str());
    return 1;
}
