// profiler_common.h
//
// Shared types, error-checking macros, configuration constants, and global
// state declarations used across all auto-profiler translation units.
//
// This header is the single point of truth for:
//   - CUPTI / NVPW error handling
//   - Profiler mode state machine
//   - Per-kernel statistics structures
//   - CUPTI Profiler context data layout
//   - Global state that coordinates tracing ↔ profiling transitions
//   - Small inline utility helpers (name demangling, output paths)

#pragma once

// =====================================================================
// System headers
// =====================================================================

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cxxabi.h>       // abi::__cxa_demangle
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// =====================================================================
// CUDA / CUPTI / NVPW headers
// =====================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>
#include <cupti_callbacks.h>
#include <cupti_driver_cbid.h>
#include <cupti_profiler_target.h>
#include <cupti_profiler_host.h>
#include <cupti_target.h>
#include <nvperf_host.h>
#include <nvperf_cuda_host.h>
#include <nvperf_target.h>

// =====================================================================
// Error-checking macros
// =====================================================================

// Fatal CUPTI error: prints context and aborts.
// Used for errors that cannot be recovered from (e.g. activity API setup).
#define CUPTI_CALL(call)                                                       \
    do {                                                                       \
        CUptiResult _s = (call);                                               \
        if (_s != CUPTI_SUCCESS) {                                             \
            const char* errstr = nullptr;                                      \
            cuptiGetResultString(_s, &errstr);                                 \
            std::fprintf(stderr, "[profiler] CUPTI FATAL %s:%d: %s\n",         \
                         __FILE__, __LINE__, errstr ? errstr : "?");           \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// Non-fatal CUPTI error: logs the error and returns false from the
// enclosing function.  Used in profiling setup paths where failure
// should cause a graceful fallback to tracing-only mode.
#define CUPTI_TRY(call)                                                        \
    do {                                                                       \
        CUptiResult _s = (call);                                               \
        if (_s != CUPTI_SUCCESS) {                                             \
            const char* errstr = nullptr;                                      \
            cuptiGetResultString(_s, &errstr);                                 \
            std::fprintf(stderr, "[profiler] CUPTI error %s:%d: %s\n",         \
                         __FILE__, __LINE__, errstr ? errstr : "?");           \
            return false;                                                      \
        }                                                                      \
    } while (0)

// Non-fatal NVPW (Perfworks) error: same semantics as CUPTI_TRY.
#define NVPW_TRY(call)                                                         \
    do {                                                                       \
        NVPA_Status _s = (call);                                               \
        if (_s != NVPA_STATUS_SUCCESS) {                                       \
            std::fprintf(stderr, "[profiler] NVPW error %s:%d: status=%d\n",   \
                         __FILE__, __LINE__, static_cast<int>(_s));            \
            return false;                                                      \
        }                                                                      \
    } while (0)

// =====================================================================
// Configuration constants
// =====================================================================

// Size of each CUPTI activity buffer (bytes).  Larger buffers reduce
// callback frequency at the cost of memory.
#define BUF_SIZE (32 * 1024)

// Default number of seconds to spend in tracing mode before selecting
// the hottest kernel for profiling.  Overridden by CUPTI_PROFILER_TRACE_S.
#define DEFAULT_TRACE_DURATION_S 10

// Maximum seconds to wait for the target kernel to launch once we
// enter TRANSITION_TO_PROFILING.  If exceeded, we revert to tracing.
#define PROFILING_TIMEOUT_S 30

// Default performance counter metrics to collect.
// Can be overridden by setting the INJECTION_METRICS env var.
static const std::vector<std::string> DEFAULT_METRICS = {
    "sm__cycles_elapsed.avg",
    "sm__cycles_active.avg",
    "sm__warps_active.avg",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
};

// =====================================================================
// Mode state machine
// =====================================================================
//
// The profiler cycles through these modes:
//
//   TRACING ──► TRANSITION_TO_PROFILING ──► PROFILING_ACTIVE ──► PROFILING_DONE ──► TRACING
//                         │                        │
//                         └── (timeout) ──► TRACING │
//                                                   └── (failure) ──► TRACING
//
//   SHUTDOWN can be entered from any state when the process exits.

enum class Mode {
    TRACING,                   // Activity-based kernel tracing
    TRANSITION_TO_PROFILING,   // Config ready; waiting for target kernel launch
    PROFILING_ACTIVE,          // Hardware counters being collected
    PROFILING_DONE,            // Collection complete; evaluating results
    SHUTDOWN,                  // Process exiting; cleaning up
};

// =====================================================================
// Per-kernel statistics (populated during tracing mode)
// =====================================================================

// Running totals for a single kernel: how many times it launched and
// cumulative GPU execution time in nanoseconds.
struct KernelStats {
    uint64_t count    = 0;   // Number of kernel launches observed
    uint64_t total_ns = 0;   // Sum of (end - start) across all launches
};

// Flat snapshot of one KernelStats entry, used for sorting and file I/O
// without holding the stats mutex.
struct Row {
    std::string name;
    uint64_t    count;
    uint64_t    total_ns;
};

// =====================================================================
// Profiler context data (for CUPTI Profiler API sessions)
// =====================================================================
//
// Holds all the binary blobs and metadata required to run one profiling
// session.  These are expensive to create (NVPW initialization, config
// image generation) so they are built once and reused across cycles.
// Only the counterDataImage is re-initialized between cycles to clear
// stale counter data.

struct ProfilerCtxData {
    CUcontext                ctx = nullptr;       // CUDA context we profile on
    int                      deviceId = 0;        // CUDA device ordinal
    std::string              deviceName;           // Human-readable GPU name
    std::string              chipName;              // NVPW chip name (e.g. "gb202")

    // Counter availability image: describes which HW counters are
    // accessible on this specific GPU.  Queried once via CUPTI.
    std::vector<uint8_t>     counterAvailabilityImage;

    // Config image: binary blob encoding *which* counters to collect.
    // Generated by NVPW from the list of requested metric names.
    std::vector<uint8_t>     configImage;

    // Counter data prefix: template that CUPTI uses to size and
    // initialize the counter data image.
    std::vector<uint8_t>     counterDataPrefixImage;

    // Counter data image: buffer where CUPTI writes raw counter values
    // during profiling.  Must be re-initialized between cycles.
    std::vector<uint8_t>     counterDataImage;

    // Scratch buffer used internally by CUPTI during counter collection.
    std::vector<uint8_t>     counterDataScratchBuffer;

    // Options struct kept around so we can re-initialize counterDataImage
    // between profiling cycles without recomputing sizes.
    CUpti_Profiler_CounterDataImageOptions counterDataImageOptions = {};

    int maxNumRanges      = 1;    // We profile exactly one kernel per cycle
    int maxRangeNameLength = 256;

    // Number of hardware replay passes required to collect all configured
    // counters.  Queried from NVPW after config image generation.  Used to
    // derive per-pass kernel time from the wall-clock profile duration.
    int numPasses = 0;

    // Wall-clock bracket around the profiled cuLaunchKernel call.
    // profileEnterTime is captured in the CUPTI_API_ENTER callback just
    // before beginProfilingSession; lastProfileWallNs is the delta
    // captured in the matching CUPTI_API_EXIT callback.  In KernelReplay
    // mode cuLaunchKernel blocks for the entire replay sequence, so this
    // delta covers all N passes + save/restore overhead.
    std::chrono::steady_clock::time_point profileEnterTime;
    uint64_t lastProfileWallNs = 0;
};

// =====================================================================
// Global state  (defined in auto_profiler.cpp)
// =====================================================================

// --- Tracing state ---
// Per-kernel stats accumulated from CUPTI activity records.
extern std::unordered_map<std::string, KernelStats> g_kernel_stats;
extern std::mutex g_mutex;   // Protects g_kernel_stats

// --- Mode and synchronization ---
extern std::atomic<Mode> g_mode;
extern std::mutex g_cv_mutex;
extern std::condition_variable g_cv;

// --- Background thread and lifecycle ---
extern std::thread g_state_thread;
extern std::atomic<bool> g_initialized;
extern CUpti_SubscriberHandle g_subscriber;

// --- Profiling state ---
extern std::string              g_target_kernel;   // Demangled name of kernel to profile
extern std::vector<std::string> g_metric_names;    // Metric names to collect
extern ProfilerCtxData          g_profiler_data;   // Profiler context (reused across cycles)
extern int                      g_profiling_cycle; // Monotonic cycle counter
extern int                      g_trace_duration_s;

// =====================================================================
// Inline utility helpers
// =====================================================================

// Demangle a C++ mangled symbol name using the Itanium ABI demangler.
// Returns the original string unchanged if demangling fails.
inline std::string demangleName(const char* name) {
    if (!name) return "unknown";
    int status = 0;
    std::unique_ptr<char, decltype(&std::free)> demangled(
        abi::__cxa_demangle(name, nullptr, nullptr, &status),
        &std::free);
    if (status == 0 && demangled) return std::string(demangled.get());
    return std::string(name);
}

// Return the output directory, controlled by CUPTI_TRACE_OUTDIR.
inline std::string getOutdir() {
    const char* env = std::getenv("CUPTI_TRACE_OUTDIR");
    if (env && std::strlen(env) > 0) return std::string(env);
    return "output";
}

// Create the output directory tree if it does not already exist.
inline void setupOutputPaths() {
    std::filesystem::create_directories(getOutdir());
}
