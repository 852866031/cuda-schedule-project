// auto_profiler.cpp
//
// Injection-based CUPTI auto-profiler — main entry point.
//
// This file contains:
//   - Global state definitions
//   - CUPTI callback handler (context creation + kernel launch interception)
//   - State machine thread (trace → profile → trace cycle)
//   - Process finalization (atexit handler)
//   - InitializeInjection() entry point
//
// How injection works
// -------------------
// Setting CUDA_INJECTION64_PATH=/path/to/libauto_profiler.so causes the
// CUDA runtime to dlopen this library and call InitializeInjection() before
// any CUDA API is exposed to the target application.
//
// What this profiler does
// -----------------------
// 1. TRACING mode:  Records every kernel launch via CUPTI Activity API and
//    builds a per-kernel hotspot table (name → count + total GPU time).
//
// 2. After CUPTI_PROFILER_TRACE_S seconds (default 10), selects the kernel
//    with the highest cumulative GPU time.
//
// 3. PROFILING mode:  Waits for the next launch of that kernel, then uses
//    CUPTI Profiler API with AutoRange + KernelReplay to collect hardware
//    performance counters.  The kernel is replayed transparently for
//    multi-pass counter collection.
//
// 4. Evaluates the collected counters, writes results to a JSON file,
//    and returns to TRACING mode.  The cycle repeats.
//
// Environment variables
// ---------------------
//   CUPTI_PROFILER_TRACE_S   Tracing window in seconds (default: 10)
//   CUPTI_TRACE_OUTDIR       Output directory (default: "output")
//   INJECTION_METRICS        Comma/semicolon-separated metric names
//
// Usage
// -----
//   make
//   sudo CUDA_INJECTION64_PATH=$(pwd)/libauto_profiler.so \
//        LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
//        ./your_cuda_app

#include "profiler_common.h"
#include "nvpw_metrics.h"
#include "tracing.h"

// =====================================================================
// Global state definitions
// =====================================================================
//
// Declared as extern in profiler_common.h, defined here so there is
// exactly one copy in the shared library.

// Per-kernel stats accumulated during tracing mode.
std::unordered_map<std::string, KernelStats> g_kernel_stats;
std::mutex g_mutex;

// Mode state machine and synchronization.
std::atomic<Mode> g_mode{Mode::TRACING};
std::mutex g_cv_mutex;
std::condition_variable g_cv;

// Background state machine thread and lifecycle guard.
std::thread g_state_thread;
std::atomic<bool> g_initialized{false};
CUpti_SubscriberHandle g_subscriber = nullptr;

// Profiling state.
std::string              g_target_kernel;
std::vector<std::string> g_metric_names;
ProfilerCtxData          g_profiler_data;
int                      g_profiling_cycle = 0;
int                      g_trace_duration_s = DEFAULT_TRACE_DURATION_S;

// Thread-local flag used by the callback handler to match the ENTER
// and EXIT callbacks of the same cuLaunchKernel call on the same thread.
// This prevents race conditions when multiple threads launch kernels
// concurrently.
static thread_local bool tl_profiling_this_kernel = false;

// =====================================================================
// CUPTI callback handler
// =====================================================================
//
// Registered once during initialization.  CUPTI invokes this function
// synchronously on the application thread that triggers the event.
//
// We handle two callback domains:
//
//   CUPTI_CB_DOMAIN_RESOURCE — Context creation
//     When the first CUDA context is created, we capture it and query
//     the device properties.  The profiler context initialization
//     (NVPW setup, config images, etc.) happens later in the state
//     machine thread.
//
//   CUPTI_CB_DOMAIN_DRIVER_API — Kernel launches (cuLaunchKernel)
//     When in TRANSITION_TO_PROFILING mode and the target kernel is
//     detected, we:
//       ENTER: re-initialize counter data → begin session → enable profiling
//       EXIT:  end session → signal state machine thread

static void CUPTIAPI callbackHandler(
    void* pUserData,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId callbackId,
    void const* pCallbackData)
{
    // -----------------------------------------------------------------
    // Context creation: capture the first CUDA context we see.
    // -----------------------------------------------------------------
    if (domain == CUPTI_CB_DOMAIN_RESOURCE) {
        if (callbackId == CUPTI_CBID_RESOURCE_CONTEXT_CREATED) {
            const auto* rd = static_cast<const CUpti_ResourceData*>(pCallbackData);
            if (g_profiler_data.ctx == nullptr) {
                g_profiler_data.ctx = rd->context;
                cudaGetDevice(&g_profiler_data.deviceId);

                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, g_profiler_data.deviceId);
                g_profiler_data.deviceName = prop.name;

                std::fprintf(stderr, "[profiler] Captured CUDA context on device %d (%s)\n",
                             g_profiler_data.deviceId, prop.name);
            }
        }
        return;
    }

    // -----------------------------------------------------------------
    // Kernel launch interception
    // -----------------------------------------------------------------
    if (domain != CUPTI_CB_DOMAIN_DRIVER_API) return;
    if (callbackId != CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel) return;

    const auto* cbData = static_cast<const CUpti_CallbackData*>(pCallbackData);
    Mode mode = g_mode.load(std::memory_order_acquire);

    // --- ENTER callback: start profiling if this is the target kernel ---
    if (mode == Mode::TRANSITION_TO_PROFILING &&
        cbData->callbackSite == CUPTI_API_ENTER) {

        std::string name = demangleName(cbData->symbolName);
        if (name == g_target_kernel) {
            // Atomically claim the transition so only one thread wins
            // if multiple threads launch the same kernel concurrently.
            Mode expected = Mode::TRANSITION_TO_PROFILING;
            if (g_mode.compare_exchange_strong(expected, Mode::PROFILING_ACTIVE,
                                               std::memory_order_acq_rel)) {
                // Clear stale data from the previous cycle.
                reinitCounterDataImage(g_profiler_data);

                // Begin the profiling session.  This must happen on a
                // thread with an active CUDA context (guaranteed here
                // since we are inside a cuLaunchKernel callback).
                if (beginProfilingSession(g_profiler_data)) {
                    tl_profiling_this_kernel = true;
                    std::fprintf(stderr, "[profiler] Profiling ENABLED for: %s\n",
                                 g_target_kernel.c_str());
                } else {
                    // Setup failed — revert to tracing.
                    std::fprintf(stderr, "[profiler] Profiling setup failed, reverting.\n");
                    g_mode.store(Mode::PROFILING_DONE, std::memory_order_release);
                    g_cv.notify_all();
                }
            }
        }
    }

    // --- EXIT callback: end profiling after the kernel completes ---
    else if (mode == Mode::PROFILING_ACTIVE &&
             cbData->callbackSite == CUPTI_API_EXIT) {

        if (tl_profiling_this_kernel) {
            tl_profiling_this_kernel = false;

            // End the session.  In KernelReplay mode, all passes have
            // already been replayed within the cuLaunchKernel call.
            endProfilingSession(g_profiler_data);

            std::fprintf(stderr, "[profiler] Profiling COMPLETED for: %s\n",
                         g_target_kernel.c_str());

            // Signal the state machine thread to evaluate results.
            g_mode.store(Mode::PROFILING_DONE, std::memory_order_release);
            g_cv.notify_all();
        }
    }
}

// =====================================================================
// State machine thread
// =====================================================================
//
// Runs continuously in the background, cycling through:
//   1. TRACING — sleep for the configured duration while activity records
//      accumulate in the kernel stats table.
//   2. HOTSPOT ANALYSIS — flush activity buffers, find the kernel with
//      the most GPU time.
//   3. PROFILER SETUP — one-time initialization on first cycle; disable
//      activity tracing to avoid interference with profiling.
//   4. WAIT FOR PROFILING — set mode to TRANSITION_TO_PROFILING and
//      wait for the callback handler to complete the profiling session.
//   5. EVALUATE — decode counter data, write JSON results, print summary.
//   6. RESET — clear stats, re-enable activity tracing, return to step 1.

static void stateMachineLoop() {
    bool profiler_ctx_initialized = false;

    while (true) {

        // =============================================================
        // Phase 1: TRACING — collect kernel stats for N seconds
        // =============================================================

        std::fprintf(stderr, "[profiler] Tracing for %d seconds...\n",
                     g_trace_duration_s);

        // Sleep in 1-second increments so we can respond to shutdown quickly.
        for (int i = 0; i < g_trace_duration_s; ++i) {
            if (g_mode.load() == Mode::SHUTDOWN) return;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        if (g_mode.load() == Mode::SHUTDOWN) return;

        // Flush any pending activity records so the stats are current.
        cuptiActivityFlushAll(1);

        // Write a hotspot CSV snapshot for external tools to consume.
        writeGlobalCsv();

        // =============================================================
        // Phase 2: HOTSPOT ANALYSIS — find the most expensive kernel
        // =============================================================

        std::string hotKernel = findHottestKernel();
        if (hotKernel.empty()) {
            std::fprintf(stderr,
                "[profiler] No kernels observed during trace window; "
                "continuing to trace.\n");
            continue;
        }

        // Snapshot the hot kernel's stats for the JSON output later.
        uint64_t hotCount = 0, hotTotalNs = 0;
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            auto it = g_kernel_stats.find(hotKernel);
            if (it != g_kernel_stats.end()) {
                hotCount = it->second.count;
                hotTotalNs = it->second.total_ns;
            }
        }

        std::fprintf(stderr,
            "[profiler] Hot kernel: %s  (count=%" PRIu64 ", total_ms=%.3f)\n",
            hotKernel.c_str(), hotCount,
            static_cast<double>(hotTotalNs) / 1e6);

        // =============================================================
        // Phase 3: PROFILER SETUP — one-time and per-cycle preparation
        // =============================================================

        // We need a CUDA context to initialize the profiler.
        if (g_profiler_data.ctx == nullptr) {
            std::fprintf(stderr,
                "[profiler] No CUDA context captured yet; "
                "continuing to trace.\n");
            continue;
        }

        // First-time profiler initialization: NVPW, config images, etc.
        if (!profiler_ctx_initialized) {
            // Disable activity tracing during init to avoid interference.
            cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
            cuptiActivityFlushAll(1);

            if (!initializeProfilerContext(g_profiler_data)) {
                std::fprintf(stderr,
                    "[profiler] Profiler init failed; "
                    "falling back to tracing-only mode.\n");
                cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);

                // Stay in tracing-only mode until shutdown.
                while (g_mode.load() != Mode::SHUTDOWN) {
                    std::this_thread::sleep_for(
                        std::chrono::seconds(g_trace_duration_s));
                    cuptiActivityFlushAll(1);
                    writeGlobalCsv();
                }
                return;
            }

            cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
            profiler_ctx_initialized = true;
        }

        // Disable activity tracing during the profiling phase.
        // CUPTI activity records and profiling cannot coexist reliably.
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
        cuptiActivityFlushAll(1);

        // Set the target kernel and enter transition state.
        g_target_kernel = hotKernel;
        g_profiling_cycle++;
        g_mode.store(Mode::TRANSITION_TO_PROFILING, std::memory_order_release);

        std::fprintf(stderr,
            "[profiler] Waiting for next launch of target kernel...\n");

        // =============================================================
        // Phase 4: WAIT — the callback handler does the actual profiling
        // =============================================================

        {
            std::unique_lock<std::mutex> lock(g_cv_mutex);
            bool done = g_cv.wait_for(lock,
                std::chrono::seconds(PROFILING_TIMEOUT_S),
                [] { return g_mode.load() == Mode::PROFILING_DONE ||
                            g_mode.load() == Mode::SHUTDOWN; });

            if (g_mode.load() == Mode::SHUTDOWN) return;

            if (!done) {
                std::fprintf(stderr,
                    "[profiler] Timeout (%ds) waiting for kernel '%s'; "
                    "reverting to tracing.\n",
                    PROFILING_TIMEOUT_S, hotKernel.c_str());
                g_mode.store(Mode::TRACING, std::memory_order_release);
                cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
                continue;
            }
        }

        // =============================================================
        // Phase 5: EVALUATE — decode counter data and write results
        // =============================================================

        std::vector<double> metricValues;
        if (evaluateMetrics(g_profiler_data, g_metric_names, metricValues)) {
            writeProfilingJson(hotKernel, g_profiling_cycle, g_metric_names,
                               metricValues, hotCount, hotTotalNs);

            // Print a summary to stderr.
            std::fprintf(stderr,
                "\n=== Profiling results: cycle %d ===\n", g_profiling_cycle);
            std::fprintf(stderr, "Kernel: %s\n", hotKernel.c_str());
            for (size_t i = 0; i < g_metric_names.size(); ++i) {
                std::fprintf(stderr, "  %-60s = %.6f\n",
                             g_metric_names[i].c_str(), metricValues[i]);
            }
            std::fprintf(stderr, "==================================\n\n");
        } else {
            std::fprintf(stderr,
                "[profiler] Metric evaluation failed for cycle %d.\n",
                g_profiling_cycle);
        }

        // =============================================================
        // Phase 6: RESET — prepare for the next tracing cycle
        // =============================================================

        // Clear stats so the next cycle measures fresh activity.
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_kernel_stats.clear();
        }

        // Re-enable activity tracing and return to TRACING mode.
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
        g_mode.store(Mode::TRACING, std::memory_order_release);
    }
}

// =====================================================================
// Finalization (registered with std::atexit)
// =====================================================================
//
// Called when the target process exits normally.  Ensures the state
// machine thread is stopped, pending activity records are flushed,
// and a final CSV snapshot is written.

static void finalizeProfiler() {
    if (!g_initialized.load()) return;

    // Signal the state machine thread to exit.
    g_mode.store(Mode::SHUTDOWN, std::memory_order_release);
    g_cv.notify_all();

    if (g_state_thread.joinable()) {
        g_state_thread.join();
    }

    // Best-effort flush: sync the device and drain any remaining
    // activity records.  These calls may fail if the CUDA runtime
    // is already partially torn down.
    cudaDeviceSynchronize();
    cuptiActivityFlushAll(1);

    writeGlobalCsv();

    std::fprintf(stderr, "[profiler] Auto-profiler finalized.\n");
}

// =====================================================================
// Injection entry point
// =====================================================================
//
// Called by the CUDA runtime (via dlsym) when CUDA_INJECTION64_PATH is set.
// Sets up:
//   1. Output directory
//   2. Configuration from environment variables
//   3. CUPTI Activity API (for tracing)
//   4. CUPTI Callback API (for kernel launch interception and context tracking)
//   5. Background state machine thread
//   6. atexit finalization handler

extern "C" int InitializeInjection(void) {
    // Guard against double initialization.
    if (g_initialized.exchange(true)) return 1;

    setupOutputPaths();

    // --- Parse configuration from environment ---

    const char* envDuration = std::getenv("CUPTI_PROFILER_TRACE_S");
    if (envDuration) {
        int val = std::atoi(envDuration);
        if (val > 0) g_trace_duration_s = val;
    }

    const char* envMetrics = std::getenv("INJECTION_METRICS");
    if (envMetrics) {
        // Tokenize by comma, semicolon, or space.
        std::string metricsStr(envMetrics);
        std::string token;
        for (size_t i = 0; i <= metricsStr.size(); ++i) {
            char c = (i < metricsStr.size()) ? metricsStr[i] : ',';
            if (c == ',' || c == ';' || c == ' ') {
                if (!token.empty()) {
                    g_metric_names.push_back(token);
                    token.clear();
                }
            } else {
                token += c;
            }
        }
    } else {
        g_metric_names = DEFAULT_METRICS;
    }

    std::fprintf(stderr, "[profiler] Metrics to collect:\n");
    for (const auto& m : g_metric_names)
        std::fprintf(stderr, "[profiler]   %s\n", m.c_str());

    // --- Set up CUPTI Activity API for tracing ---

    CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));

    // --- Set up CUPTI Callback API ---

    // Subscribe to kernel launch callbacks (for profiling interception)
    // and resource callbacks (for CUDA context discovery).
    CUPTI_CALL(cuptiSubscribe(&g_subscriber,
                              (CUpti_CallbackFunc)callbackHandler, nullptr));
    CUPTI_CALL(cuptiEnableCallback(1, g_subscriber,
                                   CUPTI_CB_DOMAIN_DRIVER_API,
                                   CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));
    CUPTI_CALL(cuptiEnableCallback(1, g_subscriber,
                                   CUPTI_CB_DOMAIN_RESOURCE,
                                   CUPTI_CBID_RESOURCE_CONTEXT_CREATED));

    // --- Launch the state machine thread ---

    g_state_thread = std::thread(stateMachineLoop);

    // --- Register finalization ---

    std::atexit(finalizeProfiler);

    std::fprintf(stderr,
        "[profiler] Auto-profiler initialized.  "
        "trace_duration=%ds  outdir=%s\n",
        g_trace_duration_s, getOutdir().c_str());
    return 1;
}
