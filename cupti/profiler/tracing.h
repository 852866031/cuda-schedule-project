// tracing.h
//
// CUPTI Activity API tracing infrastructure.
//
// Provides the tracing-mode functionality of the auto-profiler:
//   - CUPTI buffer lifecycle callbacks (allocate / process / free)
//   - Kernel activity record processing and stats accumulation
//   - CSV hotspot output
//   - Hotspot analysis (find the kernel with the most GPU time)
//   - JSON output for profiling results

#pragma once

#include "profiler_common.h"

// =====================================================================
// CUPTI Activity buffer callbacks
// =====================================================================

// Called by CUPTI when it needs a new buffer to write activity records.
// Allocates a BUF_SIZE heap buffer and sets maxNumRecords=0 so CUPTI
// fills it completely before requesting another.
void CUPTIAPI bufferRequested(uint8_t** buffer,
                              size_t* size,
                              size_t* maxNumRecords);

// Called by CUPTI when a buffer is full or flushed.
// Iterates all packed activity records, dispatches CONCURRENT_KERNEL
// records to handleKernelRecord(), reports dropped records, then frees
// the buffer.
void CUPTIAPI bufferCompleted(CUcontext ctx,
                              uint32_t streamId,
                              uint8_t* buffer,
                              size_t size,
                              size_t validSize);

// =====================================================================
// Hotspot output and analysis
// =====================================================================

// Take a snapshot of g_kernel_stats (under lock) and write it as a CSV
// to <outdir>/kernel_hotspots_global.csv.
void writeGlobalCsv();

// Scan g_kernel_stats and return the demangled name of the kernel with
// the highest total_ns.  Returns an empty string if no kernels have
// been observed.
std::string findHottestKernel();

// =====================================================================
// JSON output for profiling results
// =====================================================================

// Write one profiling cycle's results to <outdir>/profile_cycle_N.json.
// Contains the target kernel name, trace-phase stats, collected metric
// values, and replay-phase timing information:
//   - numPasses:      how many hardware replay passes ran for the kernel
//   - profileWallNs:  total wall-clock duration of the replayed launch
//                     (i.e. cuLaunchKernel blocking time for all passes)
// From these the JSON derives profile_wall_time_us and
// profiled_kernel_time_us (= profile_wall_time_us / num_replay_passes).
void writeProfilingJson(
    const std::string& kernelName,
    int cycle,
    const std::vector<std::string>& metricNames,
    const std::vector<double>& values,
    uint64_t kernelCount,
    uint64_t kernelTotalNs,
    uint64_t profileWallNs,
    int numPasses);
