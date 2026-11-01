// tracing.cpp
//
// CUPTI Activity API tracing implementation.
//
// This module handles the tracing-mode responsibilities of the auto-profiler:
//   - Providing buffer callbacks to CUPTI for activity record storage
//   - Processing CONCURRENT_KERNEL activity records to accumulate per-kernel
//     launch counts and GPU execution time
//   - Writing kernel hotspot CSV snapshots
//   - Identifying the hottest kernel for the next profiling cycle
//   - Writing profiling results as JSON

#include "tracing.h"

// =====================================================================
// Kernel record processing
// =====================================================================

// Process a single CUPTI activity record.  Only CONCURRENT_KERNEL records
// are interesting — they carry the kernel name and GPU-side start/end
// timestamps.  Everything else is silently ignored.
//
// CUpti_ActivityKernel9 fields used:
//   - name:  mangled C++ kernel symbol (demangled for human readability)
//   - start: GPU timestamp when the kernel began executing (nanoseconds)
//   - end:   GPU timestamp when the kernel finished (nanoseconds)
static void handleKernelRecord(const CUpti_Activity* record) {
    if (record->kind != CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) return;

    const auto* k = reinterpret_cast<const CUpti_ActivityKernel9*>(record);

    // Guard against malformed records.
    if (!k->name || k->end < k->start) return;

    const std::string name = demangleName(k->name);
    const uint64_t duration_ns = k->end - k->start;

    // Update global per-kernel statistics under the shared mutex.
    std::lock_guard<std::mutex> lock(g_mutex);
    auto& s = g_kernel_stats[name];
    s.count += 1;
    s.total_ns += duration_ns;
}

// =====================================================================
// CUPTI buffer callbacks
// =====================================================================

void CUPTIAPI bufferRequested(uint8_t** buffer,
                              size_t* size,
                              size_t* maxNumRecords) {
    *size = BUF_SIZE;
    *buffer = reinterpret_cast<uint8_t*>(std::malloc(BUF_SIZE));
    if (!*buffer) {
        std::fprintf(stderr, "[profiler] Activity buffer allocation failed\n");
        std::exit(EXIT_FAILURE);
    }
    // 0 = fill the buffer completely before calling bufferCompleted.
    *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext ctx,
                              uint32_t streamId,
                              uint8_t* buffer,
                              size_t size,
                              size_t validSize) {
    (void)size;  // Total allocated size — not needed; CUPTI tracks validSize.

    // Walk every packed activity record in the valid portion of the buffer.
    CUpti_Activity* record = nullptr;
    while (true) {
        CUptiResult status =
            cuptiActivityGetNextRecord(buffer, validSize, &record);
        if (status == CUPTI_SUCCESS) {
            handleKernelRecord(record);
        } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
            break;   // Normal end-of-buffer sentinel.
        } else {
            break;   // Unexpected error — stop processing this buffer.
        }
    }

    // Check for records CUPTI had to drop because the buffer filled
    // before we could process it.  Non-zero values suggest BUF_SIZE
    // should be increased.
    size_t dropped = 0;
    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped > 0)
        std::fprintf(stderr, "[profiler] Dropped %zu activity records\n", dropped);

    std::free(buffer);
}

// =====================================================================
// CSV output helpers
// =====================================================================

// Copy a stats map into a sorted vector of Row objects.
// Called with the lock held by the caller.
static std::vector<Row> snapshotRows(
    const std::unordered_map<std::string, KernelStats>& table) {
    std::vector<Row> rows;
    rows.reserve(table.size());
    for (const auto& kv : table)
        rows.push_back({kv.first, kv.second.count, kv.second.total_ns});
    return rows;
}

// Write a vector of Row objects to a CSV file.
// Kernel names are double-quoted to handle commas and special characters.
static void writeCsvFromRows(const std::vector<Row>& rows, const char* path) {
    std::ofstream ofs(path);
    if (!ofs) {
        std::fprintf(stderr, "[profiler] Failed to open CSV: %s\n", path);
        return;
    }
    ofs << "kernel_name,launch_count,total_duration_ms,avg_duration_us\n";
    for (const auto& r : rows) {
        double total_ms = static_cast<double>(r.total_ns) / 1e6;
        double avg_us = r.count
            ? static_cast<double>(r.total_ns) / r.count / 1e3
            : 0.0;
        ofs << "\"" << r.name << "\","
            << r.count << "," << total_ms << "," << avg_us << "\n";
    }
}

// =====================================================================
// Public functions
// =====================================================================

void writeGlobalCsv() {
    // Snapshot the stats under lock, then write without holding it.
    std::vector<Row> rows;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        rows = snapshotRows(g_kernel_stats);
    }
    std::string path = getOutdir() + "/kernel_hotspots_global.csv";
    writeCsvFromRows(rows, path.c_str());
    std::fprintf(stderr, "[profiler] Wrote global CSV: %s\n", path.c_str());
}

std::string findHottestKernel() {
    std::lock_guard<std::mutex> lock(g_mutex);

    std::string best_name;
    uint64_t best_ns = 0;
    for (const auto& kv : g_kernel_stats) {
        if (kv.second.total_ns > best_ns) {
            best_ns = kv.second.total_ns;
            best_name = kv.first;
        }
    }
    return best_name;
}

void writeProfilingJson(
    const std::string& kernelName,
    int cycle,
    const std::vector<std::string>& metricNames,
    const std::vector<double>& values,
    uint64_t kernelCount,
    uint64_t kernelTotalNs,
    uint64_t profileWallNs,
    int numPasses)
{
    std::string path = getOutdir() + "/profile_cycle_"
                     + std::to_string(cycle) + ".json";
    std::ofstream ofs(path);
    if (!ofs) {
        std::fprintf(stderr, "[profiler] Failed to open %s\n", path.c_str());
        return;
    }

    double total_ms = static_cast<double>(kernelTotalNs) / 1e6;
    double avg_us = kernelCount
        ? static_cast<double>(kernelTotalNs) / kernelCount / 1e3
        : 0.0;

    // Profile-phase timing (wall clock of the replayed cuLaunchKernel).
    // profiled_kernel_time_us is the amortized per-pass time; treat as an
    // approximation of the "natural" kernel duration under counter
    // collection, since replay passes can have slightly different costs.
    double profile_wall_us = static_cast<double>(profileWallNs) / 1e3;
    double profiled_kernel_us = numPasses > 0
        ? profile_wall_us / numPasses
        : 0.0;

    ofs << "{\n";
    ofs << "  \"cycle\": " << cycle << ",\n";
    ofs << "  \"target_kernel\": \"" << kernelName << "\",\n";
    ofs << "  \"trace_duration_s\": " << g_trace_duration_s << ",\n";
    ofs << "  \"total_launches\": " << kernelCount << ",\n";
    ofs << "  \"total_gpu_time_ms\": " << total_ms << ",\n";
    ofs << "  \"avg_duration_us\": " << avg_us << ",\n";
    ofs << "  \"num_replay_passes\": " << numPasses << ",\n";
    ofs << "  \"profile_wall_time_us\": " << profile_wall_us << ",\n";
    ofs << "  \"profiled_kernel_time_us\": " << profiled_kernel_us << ",\n";
    ofs << "  \"metrics\": {\n";
    for (size_t i = 0; i < metricNames.size(); ++i) {
        ofs << "    \"" << metricNames[i] << "\": " << values[i];
        if (i + 1 < metricNames.size()) ofs << ",";
        ofs << "\n";
    }
    ofs << "  }\n";
    ofs << "}\n";

    std::fprintf(stderr, "[profiler] Wrote profiling results: %s\n", path.c_str());
}
