// nvpw_metrics.h
//
// NVPW (Perfworks) and CUPTI Profiler API helper functions.
//
// This module encapsulates the complex multi-step workflows required to:
//   1. Convert high-level metric names into raw hardware counter requests
//   2. Generate the "config image" that tells CUPTI which counters to collect
//   3. Generate the "counter data prefix" that sizes the results buffer
//   4. Create and manage the counter data image where results are stored
//   5. Evaluate collected counter data back into human-readable metric values
//   6. Initialize the full profiler context for a CUDA device
//   7. Manage profiling sessions (begin / end)
//
// All functions return false on error (using CUPTI_TRY / NVPW_TRY macros)
// so callers can gracefully fall back to tracing-only mode.

#pragma once

#include "profiler_common.h"

// =====================================================================
// NVPW metric configuration
// =====================================================================

// Resolve high-level metric names (e.g. "sm__cycles_active.avg") into
// the low-level raw hardware counter requests that NVPW needs.
//
// Flow:
//   metric name  ──► MetricsEvaluator ──► MetricEvalRequest
//       ──► GetMetricRawDependencies ──► NVPA_RawMetricRequest[]
bool getRawMetricRequests(
    const std::string& chipName,
    const std::vector<std::string>& metricNames,
    std::vector<NVPA_RawMetricRequest>& rawMetricRequests,
    const uint8_t* pCounterAvailabilityImage);

// Build the config image: a binary blob that encodes which hardware
// counters to collect and how to schedule them across passes.
//
// Flow:
//   raw requests ──► RawMetricsConfig ──► BeginPassGroup ──► AddMetrics
//       ──► EndPassGroup ──► GenerateConfigImage ──► GetConfigImage bytes
// outNumPasses (if non-null) receives the total number of hardware
// replay passes required to collect all configured counters
// (numPipelinedPasses + numIsolatedPasses).
bool getConfigImage(
    const std::string& chipName,
    const std::vector<std::string>& metricNames,
    std::vector<uint8_t>& configImage,
    const uint8_t* pCounterAvailabilityImage,
    int* outNumPasses = nullptr);

// Build the counter data prefix image: a template that CUPTI uses to
// determine the size and layout of the counter data image.
//
// Flow:
//   raw requests ──► CounterDataBuilder ──► AddMetrics
//       ──► GetCounterDataPrefix bytes
bool getCounterDataPrefixImage(
    const std::string& chipName,
    const std::vector<std::string>& metricNames,
    std::vector<uint8_t>& counterDataPrefixImage,
    const uint8_t* pCounterAvailabilityImage);

// =====================================================================
// Counter data image lifecycle
// =====================================================================

// Allocate and initialize the counter data image and its scratch buffer.
// Called once during profiler context initialization.
//
// The counter data image is where CUPTI writes raw counter values
// during profiling.  Its size depends on the counter data prefix,
// the maximum number of profiled kernel ranges, and the range name length.
bool createCounterDataImage(ProfilerCtxData& pd);

// Re-initialize the counter data image to clear stale data from the
// previous profiling cycle.  Must be called before each new session.
bool reinitCounterDataImage(ProfilerCtxData& pd);

// =====================================================================
// Metric evaluation
// =====================================================================

// Decode the raw counter data collected during profiling into
// human-readable metric values using the CUPTI Profiler Host API.
//
// Reads from pd.counterDataImage (filled during profiling) and
// writes one double per metric name into `values`.
bool evaluateMetrics(
    const ProfilerCtxData& pd,
    const std::vector<std::string>& metricNames,
    std::vector<double>& values);

// =====================================================================
// Profiler context and session management
// =====================================================================

// One-time initialization of the profiler context for a CUDA device.
// Performs:
//   1. CUPTI Profiler API initialization
//   2. NVPW host initialization
//   3. Chip name discovery  (cuptiDeviceGetChipName)
//   4. Device support check (cuptiProfilerDeviceSupported)
//   5. Counter availability query
//   6. Config image generation     (via getConfigImage)
//   7. Counter data prefix generation (via getCounterDataPrefixImage)
//   8. Counter data image creation  (via createCounterDataImage)
bool initializeProfilerContext(ProfilerCtxData& pd);

// Start a profiling session: BeginSession → SetConfig → EnableProfiling.
//
// After this call, the next kernel launch on the context will be
// profiled.  In AutoRange + KernelReplay mode, CUPTI automatically
// creates a range for each kernel and replays it for multi-pass
// counter collection.
bool beginProfilingSession(ProfilerCtxData& pd);

// End the current profiling session: DisableProfiling → UnsetConfig → EndSession.
//
// After this call, counter data is available in pd.counterDataImage
// for evaluation.
bool endProfilingSession(ProfilerCtxData& pd);
