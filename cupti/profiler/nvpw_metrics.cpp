// nvpw_metrics.cpp
//
// Implementation of NVPW (Perfworks) metric helpers and CUPTI Profiler
// session management.
//
// This file contains all the "heavy" NVPW boilerplate required to:
//   - Convert metric names → raw counter requests → config image
//   - Build counter data prefix and counter data images
//   - Evaluate collected counter data into metric values
//   - Manage profiling sessions (begin / end)
//
// The implementation follows the same patterns used in NVIDIA's
// profiling_injection CUPTI sample and the Metric.cpp / Eval.cpp
// extension helpers shipped with the CUDA toolkit.

#include "nvpw_metrics.h"

// =====================================================================
// getRawMetricRequests
// =====================================================================
//
// High-level metric names like "sm__cycles_active.avg" are composite:
// they may depend on multiple raw hardware counters.  This function
// resolves each metric name into its raw counter dependencies using
// the NVPW MetricsEvaluator.
//
// Steps:
//   1. Create a MetricsEvaluator for the target chip.
//   2. For each metric name:
//      a. Convert the name to an NVPW_MetricEvalRequest.
//      b. Query its raw counter dependencies.
//   3. Build NVPA_RawMetricRequest structs for each raw counter.
//   4. Destroy the evaluator (host-side, no GPU context needed).

bool getRawMetricRequests(
    const std::string& chipName,
    const std::vector<std::string>& metricNames,
    std::vector<NVPA_RawMetricRequest>& rawMetricRequests,
    const uint8_t* pCounterAvailabilityImage)
{
    // --- Step 1: Create a scratch buffer and initialize the evaluator ---

    NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params calcParams = {
        NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
    calcParams.pChipName = chipName.c_str();
    calcParams.pCounterAvailabilityImage = pCounterAvailabilityImage;
    NVPW_TRY(NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(&calcParams));

    std::vector<uint8_t> scratch(calcParams.scratchBufferSize);

    NVPW_CUDA_MetricsEvaluator_Initialize_Params initParams = {
        NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE};
    initParams.scratchBufferSize = scratch.size();
    initParams.pScratchBuffer = scratch.data();
    initParams.pChipName = chipName.c_str();
    initParams.pCounterAvailabilityImage = pCounterAvailabilityImage;
    NVPW_TRY(NVPW_CUDA_MetricsEvaluator_Initialize(&initParams));
    NVPW_MetricsEvaluator* evaluator = initParams.pMetricsEvaluator;

    // --- Step 2-3: Resolve each metric to raw counter names ---

    std::vector<const char*> rawNames;

    for (const auto& metricName : metricNames) {
        // 2a. Convert metric name string → MetricEvalRequest struct.
        NVPW_MetricEvalRequest evalReq;
        NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params cvt = {
            NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE};
        cvt.pMetricsEvaluator = evaluator;
        cvt.pMetricName = metricName.c_str();
        cvt.pMetricEvalRequest = &evalReq;
        cvt.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
        NVPW_TRY(NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(&cvt));

        // 2b. Query raw hardware counter dependencies.
        //     First call: get the count.  Second call: fill the array.
        NVPW_MetricsEvaluator_GetMetricRawDependencies_Params dep = {
            NVPW_MetricsEvaluator_GetMetricRawDependencies_Params_STRUCT_SIZE};
        dep.pMetricsEvaluator = evaluator;
        dep.pMetricEvalRequests = &evalReq;
        dep.numMetricEvalRequests = 1;
        dep.metricEvalRequestStructSize = NVPW_MetricEvalRequest_STRUCT_SIZE;
        dep.metricEvalRequestStrideSize = sizeof(NVPW_MetricEvalRequest);
        NVPW_TRY(NVPW_MetricsEvaluator_GetMetricRawDependencies(&dep));

        std::vector<const char*> deps(dep.numRawDependencies);
        dep.ppRawDependencies = deps.data();
        NVPW_TRY(NVPW_MetricsEvaluator_GetMetricRawDependencies(&dep));

        // 3. Collect all raw counter names.
        for (auto* n : deps) rawNames.push_back(n);
    }

    // Build the raw metric request structs.
    for (auto* name : rawNames) {
        NVPA_RawMetricRequest req = {NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE};
        req.pMetricName = name;
        req.isolated = true;       // Collect in isolation for accurate results
        req.keepInstances = true;  // Keep per-instance data
        rawMetricRequests.push_back(req);
    }

    // --- Step 4: Clean up the evaluator ---

    NVPW_MetricsEvaluator_Destroy_Params destroyParams = {
        NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE};
    destroyParams.pMetricsEvaluator = evaluator;
    NVPW_TRY(NVPW_MetricsEvaluator_Destroy(&destroyParams));

    return true;
}

// =====================================================================
// getConfigImage
// =====================================================================
//
// The config image is a binary blob that tells CUPTI exactly which
// hardware counters to program and how to schedule them across
// profiling passes.  It is generated entirely on the host side by
// NVPW (no GPU context required).
//
// Steps:
//   1. Resolve metric names to raw counter requests.
//   2. Create a RawMetricsConfig for the target chip.
//   3. Set counter availability (optional but recommended).
//   4. Add all raw counter requests within a single pass group.
//   5. Generate the config image bytes.

bool getConfigImage(
    const std::string& chipName,
    const std::vector<std::string>& metricNames,
    std::vector<uint8_t>& configImage,
    const uint8_t* pCounterAvailabilityImage,
    int* outNumPasses)
{
    // Step 1.
    std::vector<NVPA_RawMetricRequest> rawReqs;
    if (!getRawMetricRequests(chipName, metricNames, rawReqs, pCounterAvailabilityImage))
        return false;

    // Step 2: Create a RawMetricsConfig for this chip + activity kind.
    NVPW_CUDA_RawMetricsConfig_Create_V2_Params createParams = {
        NVPW_CUDA_RawMetricsConfig_Create_V2_Params_STRUCT_SIZE};
    createParams.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
    createParams.pChipName = chipName.c_str();
    createParams.pCounterAvailabilityImage = pCounterAvailabilityImage;
    NVPW_TRY(NVPW_CUDA_RawMetricsConfig_Create_V2(&createParams));
    NVPA_RawMetricsConfig* pConfig = createParams.pRawMetricsConfig;

    // Step 3: Inform the config about which counters are actually
    // available on this GPU (filters out unsupported counters).
    if (pCounterAvailabilityImage) {
        NVPW_RawMetricsConfig_SetCounterAvailability_Params setParams = {
            NVPW_RawMetricsConfig_SetCounterAvailability_Params_STRUCT_SIZE};
        setParams.pRawMetricsConfig = pConfig;
        setParams.pCounterAvailabilityImage = pCounterAvailabilityImage;
        NVPW_TRY(NVPW_RawMetricsConfig_SetCounterAvailability(&setParams));
    }

    // Step 4: Add metrics in a single pass group.
    // A pass group defines a set of counters collected together.
    // CUPTI determines how many hardware passes are needed.
    NVPW_RawMetricsConfig_BeginPassGroup_Params beginPG = {
        NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE};
    beginPG.pRawMetricsConfig = pConfig;
    NVPW_TRY(NVPW_RawMetricsConfig_BeginPassGroup(&beginPG));

    NVPW_RawMetricsConfig_AddMetrics_Params addM = {
        NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE};
    addM.pRawMetricsConfig = pConfig;
    addM.pRawMetricRequests = rawReqs.data();
    addM.numMetricRequests = rawReqs.size();
    NVPW_TRY(NVPW_RawMetricsConfig_AddMetrics(&addM));

    NVPW_RawMetricsConfig_EndPassGroup_Params endPG = {
        NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE};
    endPG.pRawMetricsConfig = pConfig;
    NVPW_TRY(NVPW_RawMetricsConfig_EndPassGroup(&endPG));

    // Step 5: Generate the config image.
    // Two-pass pattern: first call gets the size, second copies the data.
    NVPW_RawMetricsConfig_GenerateConfigImage_Params genParams = {
        NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE};
    genParams.pRawMetricsConfig = pConfig;
    NVPW_TRY(NVPW_RawMetricsConfig_GenerateConfigImage(&genParams));

    NVPW_RawMetricsConfig_GetConfigImage_Params getParams = {
        NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE};
    getParams.pRawMetricsConfig = pConfig;
    getParams.bytesAllocated = 0;
    getParams.pBuffer = nullptr;
    NVPW_TRY(NVPW_RawMetricsConfig_GetConfigImage(&getParams));

    configImage.resize(getParams.bytesCopied);
    getParams.bytesAllocated = configImage.size();
    getParams.pBuffer = configImage.data();
    NVPW_TRY(NVPW_RawMetricsConfig_GetConfigImage(&getParams));

    // Query the number of hardware replay passes required.  In KernelReplay
    // mode CUPTI will run the kernel this many times for each profiled
    // launch, each pass programming a different set of counters.
    if (outNumPasses) {
        NVPW_RawMetricsConfig_GetNumPasses_Params numPassesParams = {
            NVPW_RawMetricsConfig_GetNumPasses_Params_STRUCT_SIZE};
        numPassesParams.pRawMetricsConfig = pConfig;
        NVPW_TRY(NVPW_RawMetricsConfig_GetNumPasses(&numPassesParams));
        *outNumPasses = static_cast<int>(
            numPassesParams.numPipelinedPasses +
            numPassesParams.numIsolatedPasses);
    }

    // Clean up.
    NVPW_RawMetricsConfig_Destroy_Params destroyP = {
        NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE};
    destroyP.pRawMetricsConfig = pConfig;
    NVPW_TRY(NVPW_RawMetricsConfig_Destroy(&destroyP));

    return true;
}

// =====================================================================
// getCounterDataPrefixImage
// =====================================================================
//
// The counter data prefix is a template that CUPTI uses to determine
// the layout and size of the counter data image.  It encodes the same
// counter set as the config image but in a format suitable for
// initializing the results buffer.
//
// Steps:
//   1. Resolve metric names to raw counter requests (same as config image).
//   2. Create a CounterDataBuilder for the target chip.
//   3. Add the raw counter requests.
//   4. Extract the prefix bytes (two-pass: size then data).

bool getCounterDataPrefixImage(
    const std::string& chipName,
    const std::vector<std::string>& metricNames,
    std::vector<uint8_t>& counterDataPrefixImage,
    const uint8_t* pCounterAvailabilityImage)
{
    // Step 1.
    std::vector<NVPA_RawMetricRequest> rawReqs;
    if (!getRawMetricRequests(chipName, metricNames, rawReqs, pCounterAvailabilityImage))
        return false;

    // Step 2: Create a CounterDataBuilder.
    NVPW_CUDA_CounterDataBuilder_Create_Params createParams = {
        NVPW_CUDA_CounterDataBuilder_Create_Params_STRUCT_SIZE};
    createParams.pChipName = chipName.c_str();
    createParams.pCounterAvailabilityImage = pCounterAvailabilityImage;
    NVPW_TRY(NVPW_CUDA_CounterDataBuilder_Create(&createParams));

    // Step 3: Add raw counter requests.
    NVPW_CounterDataBuilder_AddMetrics_Params addM = {
        NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE};
    addM.pCounterDataBuilder = createParams.pCounterDataBuilder;
    addM.pRawMetricRequests = rawReqs.data();
    addM.numMetricRequests = rawReqs.size();
    NVPW_TRY(NVPW_CounterDataBuilder_AddMetrics(&addM));

    // Step 4: Extract prefix bytes.
    NVPW_CounterDataBuilder_GetCounterDataPrefix_Params prefixParams = {
        NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE};
    prefixParams.pCounterDataBuilder = createParams.pCounterDataBuilder;
    prefixParams.bytesAllocated = 0;
    prefixParams.pBuffer = nullptr;
    NVPW_TRY(NVPW_CounterDataBuilder_GetCounterDataPrefix(&prefixParams));

    counterDataPrefixImage.resize(prefixParams.bytesCopied);
    prefixParams.bytesAllocated = counterDataPrefixImage.size();
    prefixParams.pBuffer = counterDataPrefixImage.data();
    NVPW_TRY(NVPW_CounterDataBuilder_GetCounterDataPrefix(&prefixParams));

    // Clean up.
    NVPW_CounterDataBuilder_Destroy_Params destroyP = {
        NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE};
    destroyP.pCounterDataBuilder = createParams.pCounterDataBuilder;
    NVPW_TRY(NVPW_CounterDataBuilder_Destroy(&destroyP));

    return true;
}

// =====================================================================
// createCounterDataImage
// =====================================================================
//
// Allocates and initializes the counter data image and its scratch buffer.
// The counter data image is the buffer where CUPTI writes raw hardware
// counter values during profiling.
//
// Steps:
//   1. Configure CounterDataImageOptions from the prefix image.
//   2. Calculate the required image size.
//   3. Allocate and initialize the image.
//   4. Calculate and initialize the scratch buffer.

bool createCounterDataImage(ProfilerCtxData& pd) {
    // Step 1: Set options from prefix image.
    pd.counterDataImageOptions = {};
    pd.counterDataImageOptions.pCounterDataPrefix = pd.counterDataPrefixImage.data();
    pd.counterDataImageOptions.counterDataPrefixSize = pd.counterDataPrefixImage.size();
    pd.counterDataImageOptions.maxNumRanges = pd.maxNumRanges;
    pd.counterDataImageOptions.maxNumRangeTreeNodes = pd.maxNumRanges;
    pd.counterDataImageOptions.maxRangeNameLength = pd.maxRangeNameLength;

    // Step 2: Calculate image size.
    CUpti_Profiler_CounterDataImage_CalculateSize_Params calcParams = {
        CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE};
    calcParams.pOptions = &pd.counterDataImageOptions;
    calcParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    CUPTI_TRY(cuptiProfilerCounterDataImageCalculateSize(&calcParams));

    pd.counterDataImage.resize(calcParams.counterDataImageSize);

    // Step 3: Initialize the image with the computed layout.
    CUpti_Profiler_CounterDataImage_Initialize_Params initParams = {
        CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    initParams.pOptions = &pd.counterDataImageOptions;
    initParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    initParams.counterDataImageSize = pd.counterDataImage.size();
    initParams.pCounterDataImage = pd.counterDataImage.data();
    CUPTI_TRY(cuptiProfilerCounterDataImageInitialize(&initParams));

    // Step 4: Allocate and initialize the scratch buffer.
    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchCalc = {
        CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE};
    scratchCalc.counterDataImageSize = pd.counterDataImage.size();
    scratchCalc.pCounterDataImage = pd.counterDataImage.data();
    CUPTI_TRY(cuptiProfilerCounterDataImageCalculateScratchBufferSize(&scratchCalc));

    pd.counterDataScratchBuffer.resize(scratchCalc.counterDataScratchBufferSize);

    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params scratchInit = {
        CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE};
    scratchInit.counterDataImageSize = pd.counterDataImage.size();
    scratchInit.pCounterDataImage = pd.counterDataImage.data();
    scratchInit.counterDataScratchBufferSize = pd.counterDataScratchBuffer.size();
    scratchInit.pCounterDataScratchBuffer = pd.counterDataScratchBuffer.data();
    CUPTI_TRY(cuptiProfilerCounterDataImageInitializeScratchBuffer(&scratchInit));

    return true;
}

// =====================================================================
// reinitCounterDataImage
// =====================================================================
//
// Clears stale counter data from the previous profiling cycle.
// Reuses the same buffer and options — only resets the contents.
// Must be called before each new profiling session.

bool reinitCounterDataImage(ProfilerCtxData& pd) {
    CUpti_Profiler_CounterDataImage_Initialize_Params initParams = {
        CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    initParams.pOptions = &pd.counterDataImageOptions;
    initParams.sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
    initParams.counterDataImageSize = pd.counterDataImage.size();
    initParams.pCounterDataImage = pd.counterDataImage.data();
    CUPTI_TRY(cuptiProfilerCounterDataImageInitialize(&initParams));
    return true;
}

// =====================================================================
// evaluateMetrics
// =====================================================================
//
// After profiling is complete, the counter data image contains raw
// hardware counter values.  This function uses the CUPTI Profiler Host
// API (CUDA 12.4+) to decode those raw values into the high-level
// metric values that were originally requested.
//
// The Host API is preferred over the older NVPW MetricsEvaluator path
// because it takes metric names directly (no manual MetricEvalRequest
// conversion needed) and handles chip-specific decoding internally.

bool evaluateMetrics(
    const ProfilerCtxData& pd,
    const std::vector<std::string>& metricNames,
    std::vector<double>& values)
{
    // Initialize a Host API object for this chip.
    CUpti_Profiler_Host_Initialize_Params hostInitParams = {
        CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE};
    hostInitParams.profilerType = CUPTI_PROFILER_TYPE_RANGE_PROFILER;
    hostInitParams.pChipName = pd.chipName.c_str();
    hostInitParams.pCounterAvailabilityImage = pd.counterAvailabilityImage.data();
    CUPTI_TRY(cuptiProfilerHostInitialize(&hostInitParams));
    CUpti_Profiler_Host_Object* pHostObject = hostInitParams.pHostObject;

    // Build a C-string array from the metric names.
    std::vector<const char*> names;
    for (const auto& n : metricNames) names.push_back(n.c_str());
    values.resize(metricNames.size());

    // Evaluate all metrics for range index 0 (we profile a single kernel).
    CUpti_Profiler_Host_EvaluateToGpuValues_Params evalParams = {
        CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE};
    evalParams.pHostObject = pHostObject;
    evalParams.pCounterDataImage = pd.counterDataImage.data();
    evalParams.counterDataImageSize = pd.counterDataImage.size();
    evalParams.rangeIndex = 0;
    evalParams.ppMetricNames = names.data();
    evalParams.numMetrics = names.size();
    evalParams.pMetricValues = values.data();
    CUPTI_TRY(cuptiProfilerHostEvaluateToGpuValues(&evalParams));

    // Clean up the Host API object.
    CUpti_Profiler_Host_Deinitialize_Params hostDeinitParams = {
        CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
    hostDeinitParams.pHostObject = pHostObject;
    cuptiProfilerHostDeinitialize(&hostDeinitParams);

    return true;
}

// =====================================================================
// initializeProfilerContext
// =====================================================================
//
// One-time setup that prepares everything needed for profiling sessions.
// This is expensive (NVPW initialization, config image generation) so
// it is done once and the results are reused across profiling cycles.

bool initializeProfilerContext(ProfilerCtxData& pd) {
    // Initialize CUPTI Profiler API (global, idempotent).
    CUpti_Profiler_Initialize_Params profInit = {
        CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    CUPTI_TRY(cuptiProfilerInitialize(&profInit));

    // Initialize NVPW host-side library (global, idempotent).
    NVPW_InitializeHost_Params nvpwInit = {NVPW_InitializeHost_Params_STRUCT_SIZE};
    NVPW_TRY(NVPW_InitializeHost(&nvpwInit));

    // Discover the NVPW chip name for this device (e.g. "gb202").
    CUpti_Device_GetChipName_Params chipParams = {
        CUpti_Device_GetChipName_Params_STRUCT_SIZE};
    chipParams.deviceIndex = pd.deviceId;
    CUPTI_TRY(cuptiDeviceGetChipName(&chipParams));
    pd.chipName = chipParams.pChipName;

    // Verify that this device supports range profiling.
    CUpti_Profiler_DeviceSupported_Params supportParams = {
        CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE};
    supportParams.cuDevice = pd.deviceId;
    supportParams.api = CUPTI_PROFILER_RANGE_PROFILING;
    CUPTI_TRY(cuptiProfilerDeviceSupported(&supportParams));
    if (supportParams.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED) {
        std::fprintf(stderr, "[profiler] Device %d does not support range profiling.\n",
                     pd.deviceId);
        if (supportParams.architecture == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
            std::fprintf(stderr, "[profiler]   Architecture not supported.\n");
        return false;
    }

    // Query counter availability (two-pass: first for size, then fill).
    // This tells NVPW which hardware counters exist on this specific GPU.
    CUpti_Profiler_GetCounterAvailability_Params availParams = {
        CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE};
    availParams.ctx = pd.ctx;
    CUPTI_TRY(cuptiProfilerGetCounterAvailability(&availParams));
    pd.counterAvailabilityImage.resize(availParams.counterAvailabilityImageSize);
    availParams.pCounterAvailabilityImage = pd.counterAvailabilityImage.data();
    CUPTI_TRY(cuptiProfilerGetCounterAvailability(&availParams));

    // Generate config image (which counters to collect).  Also asks NVPW
    // how many hardware passes this counter set will take — in KernelReplay
    // mode this is exactly how many times each profiled kernel is run.
    if (!getConfigImage(pd.chipName, g_metric_names, pd.configImage,
                        pd.counterAvailabilityImage.data(),
                        &pd.numPasses)) {
        std::fprintf(stderr, "[profiler] Failed to create config image.\n");
        return false;
    }
    std::fprintf(stderr, "[profiler] Counter config requires %d replay pass(es)\n",
                 pd.numPasses);

    // Generate counter data prefix image (sizes the results buffer).
    if (!getCounterDataPrefixImage(pd.chipName, g_metric_names,
                                   pd.counterDataPrefixImage,
                                   pd.counterAvailabilityImage.data())) {
        std::fprintf(stderr, "[profiler] Failed to create counter data prefix.\n");
        return false;
    }

    // Create the counter data image (where results will be stored).
    if (!createCounterDataImage(pd)) {
        std::fprintf(stderr, "[profiler] Failed to create counter data image.\n");
        return false;
    }

    std::fprintf(stderr, "[profiler] Profiler context initialized for %s (%s)\n",
                 pd.deviceName.c_str(), pd.chipName.c_str());
    return true;
}

// =====================================================================
// beginProfilingSession
// =====================================================================
//
// Starts a CUPTI profiling session with:
//   - AutoRange:    CUPTI creates a range for each kernel automatically
//   - KernelReplay: CUPTI replays each kernel for multi-pass collection
//
// This must be called on a thread that has the CUDA context active
// (i.e. inside a kernel launch callback).

bool beginProfilingSession(ProfilerCtxData& pd) {
    // Begin session: attach counter data image and configure replay mode.
    CUpti_Profiler_BeginSession_Params beginParams = {
        CUpti_Profiler_BeginSession_Params_STRUCT_SIZE};
    beginParams.counterDataImageSize = pd.counterDataImage.size();
    beginParams.pCounterDataImage = pd.counterDataImage.data();
    beginParams.counterDataScratchBufferSize = pd.counterDataScratchBuffer.size();
    beginParams.pCounterDataScratchBuffer = pd.counterDataScratchBuffer.data();
    beginParams.ctx = pd.ctx;
    beginParams.maxLaunchesPerPass = pd.maxNumRanges;
    beginParams.maxRangesPerPass = pd.maxNumRanges;
    beginParams.pPriv = nullptr;
    beginParams.range = CUPTI_AutoRange;
    beginParams.replayMode = CUPTI_KernelReplay;
    CUPTI_TRY(cuptiProfilerBeginSession(&beginParams));

    // Set config: tell CUPTI which counters to program.
    CUpti_Profiler_SetConfig_Params setConfig = {
        CUpti_Profiler_SetConfig_Params_STRUCT_SIZE};
    setConfig.pConfig = pd.configImage.data();
    setConfig.configSize = pd.configImage.size();
    setConfig.passIndex = 0;            // Start at pass 0
    setConfig.minNestingLevel = 1;      // Minimum nesting for auto-range
    setConfig.numNestingLevels = 1;     // Single level (no nested ranges)
    setConfig.targetNestingLevel = 1;   // Collect at this nesting level
    CUPTI_TRY(cuptiProfilerSetConfig(&setConfig));

    // Enable profiling: subsequent kernel launches on this context will
    // be profiled (and replayed as needed for multi-pass collection).
    CUpti_Profiler_EnableProfiling_Params enableParams = {
        CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE};
    enableParams.ctx = pd.ctx;
    CUPTI_TRY(cuptiProfilerEnableProfiling(&enableParams));

    return true;
}

// =====================================================================
// endProfilingSession
// =====================================================================
//
// Cleanly tears down the profiling session.  After this call,
// pd.counterDataImage contains the collected raw counter values
// ready for evaluation.

bool endProfilingSession(ProfilerCtxData& pd) {
    // Disable profiling: stop collecting counters.
    CUpti_Profiler_DisableProfiling_Params disableParams = {
        CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};
    disableParams.ctx = pd.ctx;
    CUPTI_TRY(cuptiProfilerDisableProfiling(&disableParams));

    // Unset the counter config.
    CUpti_Profiler_UnsetConfig_Params unsetParams = {
        CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE};
    unsetParams.ctx = pd.ctx;
    CUPTI_TRY(cuptiProfilerUnsetConfig(&unsetParams));

    // End the session: releases CUPTI profiling resources for this context.
    CUpti_Profiler_EndSession_Params endParams = {
        CUpti_Profiler_EndSession_Params_STRUCT_SIZE};
    endParams.ctx = pd.ctx;
    CUPTI_TRY(cuptiProfilerEndSession(&endParams));

    return true;
}
