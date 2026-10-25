# Related work: KV-cache memory and transfer techniques vs. the 8B decode-slot problem

Written 2026-10-25. This is a web and source-reading survey only: no code was changed and no GPU was used.

**How to read the claims.** Every number has a source next to it. Most sources are paper abstracts or HTML on arXiv, project READMEs, docs.vllm.ai or docs.lmcache.ai, and GitHub issues and PRs. "Local source" means I read the installed packages in `.venv-matched` (vLLM 0.15.1, LMCache 0.4.4, FlashInfer 0.6.1, torch 2.9.1). Those claims are cited as `file:line` and are true for this box's exact versions. A number marked **not verified** was not checked against a primary source. Any line labelled **Estimate** is my own arithmetic or judgement, not a measured result.

## 0. The workload's arithmetic (used by every estimate below)

- **KV per token.** Llama-3-8B uses 128 KiB per token: 32 layers × 2 × 8 KV heads × 128 × 2 B.
- **KV per request.** One request is 6400 tokens (6144 prefix + 128 suffix + 128 output). That is 400 blocks of 16 tokens, or 0.78 GiB. The prefix alone is 384 blocks (0.75 GiB). The unique part is 16 blocks (32 MiB).
- **Grant at 24 GiB.** The decode budget of 24 GiB gives a KV grant of about 7.77 GiB, which is about 63,650 tokens or about 3,978 blocks. This figure is extrapolated, per `NOTES_KV_SLOTS.md`. It gives ⌊63,650/6400⌋ = **9 slots**.
- **Time a request holds its slot.** Measured in the 9th study at 23 GiB: KV load about 0.40 s, then decode about 4.1 s (`NOTES_KV_SLOTS.md`). Slots needed ≈ λ × hold time, so 2 QPS needs about 8.3 busy slots.
- **Two levers follow.** One is fewer bytes per resident request, which means more slots. The other is a shorter hold time. The synchronous load also stalls every running request (ITL p50 14.8 ms vs. mean 28.4 ms, `NOTES_KV_SLOTS.md`), so a shorter load shortens everyone's hold time, not just the loading request's.

---

## 1. vLLM Automatic Prefix Caching (APC)

**Source.** vLLM design doc "Automatic Prefix Caching", https://docs.vllm.ai/en/latest/design/prefix_caching.html. It is a docs page, not a paper.

**(1) Mechanism.**
- The KV cache is cut into fixed-size blocks. Each full block is hashed from its parent block's hash plus its own token IDs, so identical prefixes produce identical hash chains.
- When a new request's leading blocks hash to blocks already in GPU memory, they are reused by reference count. Blocks that no request uses go on an LRU free queue and stay reusable until they are evicted.
- The doc states that a cached block "can be reused by other requests in the same batch", so concurrent requests share the physical blocks.

**(2) What it saves.** Resident KV bytes: a prefix shared by k concurrent requests is stored once. Load time and bytes: a hit needs no load at all.

**(3) Lossless or lossy.** Lossless. The doc says prefix caching "won't change model outputs".

**(4) Reported numbers.** The design doc gives no performance numbers.

**(5) Availability and fit with this stack.**
- **Default.** APC is built in and on by default in 0.15.1: `enable_prefix_caching: bool = True` at `vllm/config/cache.py:78`. Our decode engine disables it with `--no-enable-prefix-caching`.
- **Interaction with LMCache (local source).** The connector is asked only for tokens beyond the local hit: `get_num_new_matched_tokens(request, num_new_local_computed_tokens)` at `vllm/v1/core/sched/scheduler.py:609`.
- **Loaded blocks become shareable.** With synchronous loading (`load_kv_async=False`, which the LMCache connector always returns, `lmcache_connector.py:249-268`), blocks the connector loads are hashed into the prefix cache immediately. Caching is skipped only when `delay_cache_blocks` is set (`kv_cache_manager.py:362`, `scheduler.py:717`). So the second request of a session can hit the first request's loaded blocks.
- **Runs here?** Yes. The flag is already wired up as `DEC_APC=1` in `scripts/kv_vram/kv_launch.sh:104-105`.

**(6) Estimate for this workload.**
- **Block budget.** With APC, the blocks needed are 384·S + 16·R, where S is the number of distinct concurrent sessions and R the number of concurrent requests. Against about 3,978 blocks at 24 GiB, R_max ≈ (3978 − 384·S)/16: about 104 at S = 6, about 32 at S = 9, and the limit hits at S = 10.
- **What the trace says.** The arrival simulation in `NOTES_KV_SLOTS.md` found 10.3 concurrent requests spanning only 6.8 sessions. Capacity therefore stops being "9 requests" and becomes "about 9–10 distinct sessions".
- **Loads avoided.** Requests for the hottest session (32% of traffic) mostly skip the 0.4 s load and the batch-wide stall while its prefix stays resident.
- **Overall estimate.** This is the largest lossless win available: concurrency ≥ 1.5× at 24 GiB, and queue wait at 2 QPS falling by more than an order of magnitude.
- **Risks.** LRU eviction of a hot prefix by cold ones. Also, the unique 128-token suffix is a partial LMCache chunk: `save_unfull_chunk: false` in `kv_launch.sh`, so it is recomputed on decode, which is already the case today.

## 2. SGLang RadixAttention

**Source.** Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs", NeurIPS 2024 ([proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract.html)), arXiv:2312.07104. Code: https://github.com/sgl-project/sglang.

**(1) Mechanism.** The KV of all requests is kept in a radix tree keyed by token sequences. A new request walks the tree to find its longest cached prefix. Leaves are evicted LRU when memory is needed. The scheduler also orders requests to maximize prefix hits ("cache-aware scheduling"; the radix tree with LRU is described in arXiv v1 per the search result, and the details are in the paper body).

**(2) What it saves.** The same as APC: resident bytes for shared prefixes, and load or prefill time on a hit. Matching works at token granularity rather than block granularity.

**(3) Lossless or lossy.** Lossless (exact KV reuse).

**(4) Reported numbers.** "up to 6.4x higher throughput compared to state-of-the-art inference systems" (arXiv abstract, end-to-end across their structured-program benchmarks).

**(5) Availability.** Open source. It is a different serving framework, and LMCache ships SGLang connectors (`lmcache/v1/gpu_connector/__init__.py` has `SGLangGPUConnector`). Status on sm_120 with this LMCache: **not verified**. Switching frameworks would break comparability with the pinned stack.

**(6) Estimate for this workload.** For this workload, APC in vLLM gives essentially the same sharing: the prefixes are block-aligned (6144 = 384 × 16), so token-level matching gains nothing. Moving to SGLang offers no additional benefit worth the migration.

## 3. LMCache layerwise loading (`use_layerwise`)

**Sources.** LMCache docs, "Layerwise KV Transfer", https://docs.lmcache.ai/kv_cache_optimizations/layerwise.html. LMCache paper arXiv:2510.09665 (§5.2, §6; the arXiv page lists no venue). Local source: LMCache 0.4.4.

**(1) Mechanism.** Instead of copying all 32 layers' KV into the paged GPU blocks before the forward pass, LMCache loads layer i+1 on a separate CUDA stream while layer i computes. The forward pass calls `wait_for_layer_load(layer)` before each attention. Saving can likewise run layer by layer (`save_kv_layer`).

**(2) What it saves.** It overlaps load time with compute. It does **not** reduce resident KV bytes or bytes per token.

**(3) Lossless or lossy.** Lossless.

**(4) Reported numbers.** The docs give no numbers. The LMCache paper gives "no separate measured speedup" for layerwise pipelining. Its loading-bandwidth figure (400 Gbps vs. 88 Gbps for vLLM-native CPU offload, Table 5, 8×H100) is not attributed to layerwise.

**(5) Availability and compatibility (local source, LMCache 0.4.4 + vLLM 0.15.1).**
- **Config.** `use_layerwise` defaults to False (`lmcache/v1/config.py:90`). When true, the vLLM GPU connector becomes `VLLMPagedMemLayerwiseGPUConnector` (`v1/gpu_connector/__init__.py:86-93`).
- **Python fallback.** The copy op `lmc_ops.single_layer_kv_transfer` has a Python fallback in `lmcache/non_cuda_equivalents.py:626`. The fallback this stack runs on (`Using backend: lmcache.non_cuda_equivalents` in `output/logs/kvv_decode.log`) therefore *can* serve the layerwise path. How fast it is on this path is **not verified**.
- **Keys must match.** Layerwise retrieval looks up **per-layer keys** (`key.split_layers(num_layers)`, `cache_engine.py:968`). A non-layerwise prefill engine stores multi-layer chunk keys, so the prefill engine must also run `use_layerwise: true` or the decode engine will see no hits. This is inferred from source and untested.
- **Async plus layerwise.** The retrieval generator states `# TODO(Jiayi): need to make async loading and layerwise compatible` (`v1/storage_backend/storage_manager.py:537`). Treat the two as mutually exclusive in 0.4.4.
- **CUDA graphs.** Current vLLM's `LMCacheConnectorV1.requires_piecewise_for_cudagraph()` returns `use_layerwise`, and its docstring says "LMCache requires PIECEWISE CUDA graph mode when layerwise operations are enabled" ([vLLM API docs](https://docs.vllm.ai/api/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.html)). vLLM 0.15.1 has **no such check** (grep finds no `requires_piecewise` in the installed vLLM). The decode engine runs `cudagraph_mode FULL_AND_PIECEWISE` (`kvv_decode.log`), so a layerwise run would need `cudagraph_mode=PIECEWISE` set by hand to be safe. That costs full-graph decode steps (TPOT effect not verified).
- **Deprecation.** Both doc pages warn that the in-process mode is deprecated in favour of LMCache "MP mode".

**(6) Estimate for this workload.**
- **Small win.** Layerwise can hide at most the forward time of the step that loads, about 15–30 ms (ITL p50 14.8 ms), behind a ~400 ms load. That limit holds because the load is bound by the Python-fallback copy and the TCP store (~2 GB/s), not by PCIe.
- **Expected gain.** ≤ 5% shorter slot hold time. It is not worth the PIECEWISE cost or the both-engines key change.

## 4. LMCache asynchronous loading (`enable_async_loading`)

**Sources.** LMCache docs, "Async Loading", https://docs.lmcache.ai/kv_cache/async_loading.html. vLLM PR #23620, https://github.com/vllm-project/vllm/pull/23620. LMCache paper arXiv:2510.09665 §6 and §8.6.

**(1) Mechanism.**
- The scheduler-side lookup becomes non-blocking. While a request waits in the queue, LMCache workers prefetch its KV chunks from slower tiers (here, the remote TCP store) into the local CPU (L1) tier.
- Until the prefetch is done, `get_num_new_matched_tokens` returns `None` ("try again"), so vLLM schedules other requests. This semantic was added in vLLM PR #23620, merged 2025-09-10.

**(2) What it saves.** It moves the remote-to-host fetch off the critical path. It does **not** make the host-to-GPU copy asynchronous, and it saves no resident bytes.

**(3) Lossless or lossy.** Lossless.

**(4) Reported numbers.** The LMCache paper (§8.6, Figure 13) reports "query asynchronization reduces end-to-end delay by 1.46×". The setup is a timeline from the middle of a longer run, and the hardware for that run is not given in the text read.

**(5) Availability and compatibility (local source).**
- **The flag exists in 0.4.4.** `enable_async_loading` defaults to False (`lmcache/v1/config.py:318`). When true, the scheduler uses `LMCacheAsyncLookupClient` (`v1/lookup_client/factory.py:81`). It is rejected for external lookup clients (`factory.py:62`), and is only active when `not self.enable_pd` (`storage_manager.py:283`). Our stack does not set `enable_pd`.
- **vLLM 0.15.1 supports the "try again" path.** `scheduler.py:614` handles `ext_tokens is None`.
- **The GPU copy stays synchronous.** The vLLM connector wrapper always returns `load_kv_async=False` (`vllm/.../lmcache_connector.py:268`). The H2D copy into the paged blocks still runs synchronously inside the step that admits the request, so the batch-wide stall shrinks only by the share of the 400 ms spent in the TCP fetch.
- **Backends.** The docs require backends to implement `batched_async_contains`. The installed `remote_backend.py:513` and `local_cpu_backend.py` implement it.
- **First LMCache release with the flag:** not verified (present in 0.4.4).

**(6) Estimate for this workload.**
- **It depends on how the 400 ms splits.** If the TCP fetch is most of it, async loading could cut the in-step stall to the L1→GPU copy alone: ≥ 0.78 GiB / 14.5 GB/s ≈ 54 ms at PCIe speed, and more with the Python fallback. That could shorten every request's decode time by an amount comparable to the stall hypothesis in `NOTES_KV_SLOTS.md` (4.1 s → about 2.3 s).
- **If the fallback H2D copy dominates,** the gain is small.
- **Do this first.** Measure the fetch/copy split before relying on async loading. The 6 GiB L1 holds about 7 prefixes, so prefetched data must not be evicted before use.

## 5. CachedAttention / AttentionStore (USENIX ATC'24)

**Source.** Gao et al., "Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention", USENIX ATC 2024, arXiv:2403.19708. Code: not verified.

**(1) Mechanism.**
- KV caches of past conversation turns are kept in a host-DRAM and SSD hierarchy ("AttentionStore") instead of being recomputed.
- Loading is layer-wise and pre-loaded into a read buffer so it overlaps prefill compute. Saving is asynchronous.
- A scheduler-aware policy prefetches KV for queued jobs and evicts KV for jobs far in the future.
- Positional-encoding-decoupled storage allows truncating context overflow without invalidating the KV.

**(2) What it saves.** Prefill recompute time and load latency, by overlap and prefetch. It saves no resident decode bytes.

**(3) Lossless or lossy.** Exact reuse for normal turns. The context-overflow truncation is approximate: Table 1 PPL is within 0.02 of token truncation (e.g., LLaMA-13B WikiText-2 4.91 vs. 4.90), and on MMLU LLaMA-13B scores 52.3% vs. 53.2% (§4.3.5).

**(4) Reported numbers.** Setup: 4× A100-80GB, 128 GB DRAM, 10 TB SSD, PCIe Gen4 (about 26 GB/s effective, §2.4), ShareGPT with 9K sessions, Poisson arrivals at 1 session/s, compared against recomputation.
- **Headline figures.** TTFT −87%, prefill throughput 7.8×, cost −70%. The maxima come from different models: 87% and 7.8× on LLaMA-70B, 70% on LLaMA-13B (§4.2).
- **Ablation (LLaMA-13B, 1 GPU, batch 16, 1K history + 100 new tokens).**
  - Layer-wise pre-loading with no buffer cuts prefill time by 35%.
  - With a 15-layer buffer, loading is "perfectly overlap[ped]" and prefill time drops 61%.
  - Async saving cuts execution time by 13–15% (§4.3.2).

**(5) Availability.** A research system. Its ideas live on in LMCache (layerwise, async/prefetch; §3–4 above). It does not run here as-is.

**(6) Estimate for this workload.** The ideas are already covered by §3–4. Its overlap works because a prefill of 100 new tokens over 1K history has enough compute to hide the load. Our decode-side load step has about 15–30 ms of compute against a 400 ms load, so the overlap gain is small (estimate). Its prefetch-while-queued idea is the useful part, the same as async loading.

## 6. vLLM FP8 KV cache (`--kv-cache-dtype fp8`)

**Sources.**
- vLLM docs, "Quantized KV Cache": https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html
- llm-compressor KV example: https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_kv_cache/
- vLLM issues [#41651](https://github.com/vllm-project/vllm/issues/41651) and [#60262](https://github.com/vllm-project/vllm/issues/60262), PR [#54267](https://github.com/vllm-project/vllm/pull/54267)
- Local vLLM 0.15.1 source

**(1) Mechanism.** K and V are stored as 1-byte FP8 (E4M3 or E5M2) instead of bf16, with one scale per tensor that multiplies back on read. Attention kernels dequantize on load. The FA3 path on Hopper can also compute attention in FP8.

**(2) What it saves.** Bytes per token are halved: 128 KiB → 64 KiB per token, 0.78 → 0.39 GiB per request. Resident bytes, load bytes and store bytes are all halved.

**(3) Lossless or lossy.** Lossy.
- **Scales.** The docs say: "All quantization scales are set to `1.0`" unless they come from the checkpoint or llm-compressor calibration. vLLM 0.15.1 also has `calculate_kv_scales` (dynamic scales; default False, `config/cache.py:106-109`).
- **Accuracy evidence.** Meta-Llama-3-8B-Instruct with FP8 KV, calibrated on ultrachat_200k (512 samples × 2048), scores GSM8K 5-shot 0.7748 flexible / 0.7763 strict, ±0.0115 (llm-compressor example).
- **What is missing.** That example gives **no bf16 baseline**, and its recipe also quantizes weights and activations to FP8, so the KV-only accuracy cost is **not verified**. I found no published KV-only FP8 accuracy number for Llama-3-8B.

**(4) Reported speed or memory numbers.** The docs say only "significantly reduce its memory footprint" and give no numbers.

**(5) Availability on RTX 5090 (sm_120) with vLLM 0.15.1 (local source).**
- **Accepted dtypes.** `auto, bfloat16, fp8, fp8_e4m3, fp8_e5m2, fp8_inc, fp8_ds_mla` (`config/cache.py:24-32`).
- **FlashAttention cannot take FP8 KV here.** It requires FA3 on capability family 90: `flash_attn_supports_fp8()` is `get_flash_attn_version()==3 and is_device_capability_family(90)` (`v1/attention/backends/fa_utils.py:98-102`). Today's decode engine uses FLASH_ATTN (`Using FLASH_ATTN attention backend out of potential backends: ('FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION')`, 236 occurrences in `output/logs`). So with FP8 KV the auto-selector moves to FLASHINFER, the next priority for major ≠ 10 (`platforms/cuda.py:69-80`).
  - FLASHINFER declares fp8, fp8_e4m3 and fp8_e5m2, and capability 7.5–12.1 (`flashinfer.py:282-288, 354-357`).
  - TRITON_ATTN also declares all three FP8 types (`triton_attn.py:262-268`) and passes `k_descale/v_descale` (`:529-530`).
- **Turning on FP8 KV therefore also changes the attention kernel.** Any FP8 vs. bf16 comparison needs a bf16 control on the same backend (FLASHINFER or TRITON_ATTN), or the kernel change is confounded with the dtype change.
- **Known sm_120 problems in later vLLM versions:**
  - **#41651** (vLLM 0.19.2rc1, RTX PRO 6000 sm_120): FlashInfer + FP8 KV + CUDA graphs produced random tokens on 2.7–3.6K-token prompts. The causes were a Triton FP8 scale bug (PR #42580) and per-layer `num_qo_heads` (PR #42650). The issue closed 2026-05-22. The per-layer-heads cause does not apply to Llama-3-8B, which has uniform heads. Whether the Triton scale bug existed in 0.15.1 is not verified; the 0.15.1 Triton code does pass `layer._k_scale/_v_scale`.
  - **#60262** (nightly 0.30.1rc1, sm_120): FP8 KV auto-selects FLASHINFER even when FlashInfer cannot JIT (no nvcc), and crashes. The workaround is `--attention-backend TRITON_ATTN`. The issue is still open, and a commenter reports the Triton route worked on an RTX 5090 (unverified).
  - **PR #54267** (open): with nvcc < 12.9 on sm_120, FlashInfer JIT "drops the arch" and fails ("FlashInfer requires GPUs with sm75 or higher"). That was seen with nvcc 12.8 + torch cu130 on an RTX 5090.
- **This box.** It has nvcc 12.8 (`/usr/local/cuda-12.8`), and `~/.cache/flashinfer/120/cached_ops/` holds JIT-built bf16 `batch_decode`/`batch_prefill` kernels from an earlier FlashInfer. JIT for sm_120 has therefore worked here before. An FP8-KV FlashInfer kernel has **never** been built here.
- **E5M2 on Triton.** The Triton backend views FP8 caches as `current_platform.fp8_dtype()` = `float8_e4m3fn` (`triton_attn.py:388, 473`; `platforms/interface.py:556`). Use `fp8`/`fp8_e4m3`. Whether `fp8_e5m2` on TRITON_ATTN is correct in 0.15.1 is not verified.
- **LMCache with FP8.** The wire protocol carries `float8_e4m3fn` and `float8_e5m2` (`lmcache/v1/protocol.py:47-48`). The Python-fallback copies are dtype-agnostic index copies. Both engines must use the same KV dtype, and `kv_launch.sh` already applies `KV_DTYPE` to both.
- **Not in 0.15.1.** NVFP4 KV for sm_120 is a later, separate PR (#50288).

**(6) Estimate for this workload.**
- **Slots.** 63,650 × 2 / 6400 → **19 slots** instead of 9 at 24 GiB. Per Little's law, the queue at 2–2.5 QPS should essentially disappear.
- **Load and stall.** Load bytes halve, so about 200 ms instead of about 400 ms at the same ~2 GB/s, and the batch stall shrinks with it.
- **Decode steps.** Possibly faster too, since about half the KV bytes are read per step. That is not verified, and kernel-dependent on sm_120.
- **Combined with APC,** the session limit (§1) roughly doubles as well.
- **Accuracy risk.** Uncalibrated scale 1.0 on a bf16 checkpoint. A greedy-output identity check on the fixed prompts is mandatory, as `PLAN_KV_VRAM.md` already plans. The first engine start is the real feasibility test: FlashInfer JIT on nvcc 12.8, and the TRITON_ATTN fallback if it fails.

## 7. KIVI (2-bit asymmetric KV quantization)

**Source.** Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache", ICML 2024, arXiv:2402.02750. Code: https://github.com/jy-yuan/KIVI.

**(1) Mechanism.** Keys are quantized per channel and values per token, to 2 (or 4) bits in groups. The most recent R tokens stay in full precision until a group fills. The method needs no calibration ("tuning-free").

**(2) What it saves.** Bytes per token: about 8× fewer for the quantized part versus fp16, before group metadata and the fp16 residual. Resident and transfer bytes shrink accordingly.

**(3) Lossless or lossy.** Lossy. With group size 32 and residual 128 (§4.1):

| Model | Benchmark | fp16 | KIVI-2 | Source |
|---|---|---|---|---|
| Llama-2-7B | LongBench avg | 44.52 | 44.27 | Table 4 |
| Llama-2-7B | GSM8K | 13.50 | 12.74 | Table 3 |
| Llama-2-13B | GSM8K | 22.67 | 20.77 | Table 3 |
| Llama-3-8B-Instruct | LongBench | 45.21 | 44.37 (KIVI-4: 45.31) | Table 8 |

**(4) Reported numbers.** "2.6× less peak memory" (including weights), "up to 4× larger batch size", "2.35×–3.47× throughput". Setup: Llama-2-7B, a single A100-80GB, ShareGPT-like lengths (161 in / 338 out), batch grown until OOM (§4.2.4).

**(5) Availability.** A HuggingFace custom model class (`LlamaForCausalLM_KIVI`) plus its own CUDA extension in `quant/`. Llama-3 is supported via transformers 4.43. There is **no vLLM integration** (the README never mentions vLLM). It cannot be used in vLLM 0.15.1 without writing a paged attention kernel and a KV-cache dtype. Building on sm_120: not verified.

**(6) Estimate for this workload.** In principle about 4–6× more slots (estimate). In practice it is not usable on this stack: it means a new vLLM attention backend, and LMCache cannot serialize the layout. The output changes as well.

## 8. KVQuant

**Source.** Hooper et al., "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization", NeurIPS 2024, arXiv:2401.18079. Code: https://github.com/SqueezeAILab/KVQuant.

**(1) Mechanism.**
- Keys are quantized per channel *before* RoPE, and RoPE is reapplied after dequantization.
- Values use non-uniform, sensitivity-weighted datatypes ("nuqX", from Fisher-weighted k-means).
- About 1% of outliers per vector are kept sparse in full precision, and the first (attention-sink) token stays in fp16.
- It needs offline calibration: 16 × 2K-token Wikitext-2 samples for the key scales (§4.1).

**(2) What it saves.** Bytes per token at 4, 3 or 2 bits. For LLaMA-7B at 128K context, the KV cache is 17.3 / 13.3 / 9.3 GB for the 4-, 3- and 2-bit variants with 1% outliers (Table 1).

**(3) Lossless or lossy.** Lossy. Wikitext-2 PPL, fp16 → nuq4-1% / nuq3-1% / nuq2-1%:
- Llama-3-8B: 5.54 → 5.56 / 5.64 / 6.04
- Llama-2-7B: 5.12 → 5.13 / 5.17 / 5.41

These figures are from Table 13. Abstract: "< 0.1 perplexity degradation with 3-bit".

**(4) Reported numbers.**
- **Context length.** LLaMA-7B at up to 1M context on one A100-80GB, and up to 10M on 8 GPUs.
- **Kernel speed.** About 1.7× faster than the fp16 matvec (abstract). On an A6000 at batch 1 with LLaMA-2-7B-32K, the Key kernel goes 219.4 → 126.3 µs at 16K (Table 6).

**(5) Availability.** Research code with per-folder environments. The README lists optimized kernels and end-to-end deployment as future work. There is no vLLM integration. Needs calibration. Not runnable on this stack.

**(6) Estimate for this workload.** Same as KIVI: a large theoretical gain (3-bit ≈ 5× more slots), but unusable on vLLM 0.15.1 without kernel work.

## 9. InfiniGen

**Source.** Lee et al., "InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management", OSDI 2024, arXiv:2406.19707. Code: https://github.com/snu-comparch/InfiniGen.

**(1) Mechanism.**
- The KV pool lives in CPU memory.
- At layer i−1, InfiniGen speculates which tokens layer i will attend to, using partial query and key weights (30% of columns, after an offline SVD "skewing"). It then prefetches only those tokens' KV over PCIe: at most 20% per layer (§4, §5.1).

**(2) What it saves.** Resident GPU KV (only the selected subset is resident), PCIe bytes per step, and load time.

**(3) Lossless or lossy.** Lossy.
- On five lm-eval 5-shot tasks it "nearly matches" full cache when the relative KV size is above 10% (Fig. 11).
- OPT-6.7B WikiText-2 PPL: full 11.68. With an 80% memory limit, counter or LRU eviction stays at 11.68, while FIFO rises to 19.64 (Table 2).

**(4) Reported numbers.**
- **Headline.** "up to 3.00×" over prior KV management methods (abstract). I could not trace this to a single experiment in the text.
- **In the body.** Up to 5.28× over FlexGen across sequence lengths (Fig. 16a), 1.34× for OPT-30B.
- **Setup.** FlexGen base, OPT-6.7B/13B/30B and Llama-2-7B/13B, RTX A6000 48 GB, PCIe 3.0 ×16, 96 GB DDR4 (§5.1).

**(5) Availability.** Code on GitHub, built on FlexGen. Not vLLM, not paged. Does not run on this stack.

**(6) Estimate for this workload.** Low benefit. At 6.4K context, the 20%-per-layer fetch is about 0.16 GiB per request per step. Over PCIe at 14.5 GB/s that is about 11 ms per request-step, so a batch of 10 adds about 110 ms per step, which is worse than holding the KV resident (estimate). It is also lossy, and it is not on vLLM.

## 10. ShadowKV

**Source.** Sun et al., "ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference", ICML 2025 (Spotlight, per README), arXiv:2410.21465. Code: https://github.com/bytedance/ShadowKV.

**(1) Mechanism.**
- Keys are kept on the GPU in low-rank pre-RoPE form (rank 160). Values are offloaded to CPU.
- Per 8-token chunk, a mean "landmark" key stays on the GPU, and about 48 outlier chunks stay fully on the GPU.
- Each decode step selects the top-k chunks via landmarks, fetches their values from CPU, and rebuilds their keys from the low-rank factors (§4, §5.1).

**(2) What it saves.** Resident GPU KV: values off-GPU, keys compressed. Bytes moved per step: only a sparse budget, 1.56% on RULER.

**(3) Lossless or lossy.** Lossy. 128K RULER and LongBench, Table 1, full vs. ShadowKV:

| Model | RULER | LongBench |
|---|---|---|
| Llama-3.1-8B | 85.53 vs. 83.57 | 48.96 vs. 48.13 |
| Llama-3-8B-1M | 86.68 vs. 86.88 | 39.86 vs. 39.94 |

**(4) Reported numbers.**
- **Batch.** "up to 6× larger batch sizes" (8 → 48 at 60K context, Llama-3.1-8B).
- **Throughput.** "up to 3.04×" (80.78 → 245.90 tok/s at 122K context).
- **Setup.** One A100 (Table 3).
- **At short context.** At 60K and batch 2, the gain is negligible (89.69 vs. 89.19 tok/s, Table 4). The smallest context tested is 60K.

**(5) Availability.** A standalone PyTorch codebase with custom kernels, flash-attn, FlashInfer (cu121/torch 2.3 wheel) and CUTLASS. It is evaluated on A100 only. Not vLLM. Building on sm_120: not verified.

**(6) Estimate for this workload.** Negligible to negative. It is designed for contexts of 60K and up; at 6.4K the landmark and outlier overhead is a large fraction of the cache. It is lossy and not on vLLM.

## 11. Quest

**Source.** Tang et al., "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference", ICML 2024, arXiv:2406.10774. Code: https://github.com/mit-han-lab/Quest.

**(1) Mechanism.** The KV cache is paged at 16 tokens per page. Each page stores per-channel min/max keys. At each decode step, Quest upper-bounds each page's attention score from the query and those min/max values, then attends only over the top-K pages. The first two layers use full attention (§3.4–3.5, §4.1).

**(2) What it saves.** Bytes *read* per decode step (bandwidth) only. The paper says Quest "retains all of the KV cache" (§2.2), so resident bytes are unchanged.

**(3) Lossless or lossy.** Lossy, but close at modest budgets.
- **Passkey.** LongChat-7B at 10K reaches 99% with a 64-token budget (Table 1).
- **LongBench.** Lossless at 1/5–1/10 sparsity on several datasets (Fig. 7; numbers only in the figure).

**(4) Reported numbers.**
- **Self-attention.** 7.03× faster than FlashInfer at 32K context with a 2048-token budget (Llama-2-7B, RTX 4090, Fig. 9).
- **End to end.** 1.74× (fp16 weights) and 2.23× (4-bit weights) faster single-batch decode latency at 32K/2048 on an Ada 6000 (Fig. 10).

**(5) Availability.** Custom FlashInfer-based kernels (CMake, raft). The README lists "Support GQA models" as unchecked TODO, so Llama-3-8B (GQA) is unsupported as of the README. Not vLLM.

**(6) Estimate for this workload.** None for the slot limit, since resident bytes are unchanged. The per-step attention saving at 6.4K context is small too: our steps are not attention-bandwidth bound at batch ≤ 10 (estimate).

## 12. MagicPIG

**Source.** Chen et al., "MagicPIG: LSH Sampling for Efficient LLM Generation", ICLR 2025 (Spotlight, per README), arXiv:2410.16179. Code: https://github.com/Infini-AI-Lab/MagicPIG.

**(1) Mechanism.**
- Instead of top-k, MagicPIG estimates attention by importance-sampling keys chosen by SimHash LSH collisions with the query.
- The hash codes are computed on the GPU. The hash tables, the KV cache and the sampled attention live on the **CPU**.
- Sink and local-window tokens stay on the GPU, and the CPU and GPU partial outputs are merged (§4.3–4.4).

**(2) What it saves.** Resident GPU KV (moved to the CPU) and per-step compute and bandwidth (about 2–5% of full attention cost).

**(3) Lossless or lossy.** Lossy. Llama-3.1-8B-Instruct, full vs. MagicPIG vs. Quest:
- **RULER 16–96K:** 88.7 vs. 87.4 vs. 82.1 (Table 3)
- **LongBench:** 71.2 vs. 70.7 vs. 70.6 (Table 2)
- **lm-eval average:** 73.7 vs. 70.1 vs. 59.2 (Table 1)

**(4) Reported numbers.**
- **Headline.** "up to 5×" decoding throughput, and "54ms decoding latency on a single RTX 4090 for Llama-3.1-8B-Instruct ... 96k tokens" (abstract).
- **The 4090 result is simulated.** It comes from an L20 capped at 24 GB with CPU bandwidth assumed at 150 GB/s (§5.2).
- **Other measured setups.** L20 with CodeLlama-13B at 16K: about 5.0×. A100 with CodeLlama-34B at 16K: about 1.5×.

**(5) Availability.**
- **CPU.** It requires Intel AVX512 CPUs, and AVX512_BF16 plus GCC ≥ 11 for bf16 (README). This box's Ryzen 9 9950X reports `avx512f` and `avx512_bf16` (`/proc/cpuinfo`), but the README targets Intel and AMD support is not verified.
- **Framework.** HuggingFace-based, not vLLM.

**(6) Estimate for this workload.** Low. The gains appear at long context and big batches. At 6.4K context, a CPU attention pass over 10 × 0.78 GiB per step on dual-channel DDR5 (bandwidth not verified here) would be far slower than GPU-resident decode. It is also lossy.

## 13. NEO (MLSys 2025)

**Source.** Jiang et al., "NEO: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference", MLSys 2025 (per the repo README), arXiv:2411.01142. Code: https://github.com/NEO-MLSys25/NEO.

**(1) Mechanism.**
- For a subset of requests, NEO keeps the *whole* KV cache in CPU memory and runs their decode attention on the CPU. The linear layers of all requests stay on the GPU.
- "Asymmetric pipelining" overlaps a GPU-heavy sub-batch with a CPU-attention-heavy sub-batch.
- A load-aware scheduler picks a GPU-only or a two-batch plan each iteration (§3.1–3.2).

**(2) What it saves.** Resident GPU KV, since CPU-attention requests hold no GPU KV. Nothing is transferred per step for those requests.

**(3) Lossless or lossy.** Lossless: "does not compromise accuracy" (§7). It moves exact computation; it does not approximate it.

**(4) Reported numbers.**
- **Throughput vs. GPU-only.** Up to 7.5× / 26% / 14% higher on T4 / A10G / H100 (abstract), and up to 79.3% on an A10G with a stronger CPU (§5.5). Latency is the same as the baseline.
- **Setup.** SwiftLLM base, LLaMA-2-7B and LLaMA-3.1-8B/70B. Azure code trace on H100/A10G, OSC on T4, input 2000 tokens in §5.3.
- **Limits stated by the authors.** Gains depend on CPU memory bandwidth more than core count. They shrink with long outputs, and with short outputs NEO can be slightly worse than the baseline.

**(5) Availability.**
- **Base.** It is built on SwiftLLM, "a simplified vLLM copy" (§4), not vLLM 0.15.1.
- **Build.** Needs ISPC 1.23, g++ ≥ 13 for the CPU kernel, an AVX2 CPU, and PyTorch 2.4+.
- **Memory.** Needs 30–120 GB of host RAM for the reproduction configs (README).
- **Here.** Not runnable on this stack. Porting it is a large project.

**(6) Estimate for this workload.**
- **Gain.** Modest at best. Each CPU-resident request reads 0.78 GiB per decode step from DDR5, which is tens of ms per request-step (estimate; host bandwidth not measured). A few extra "slow slots" are possible, at a large TPOT cost.
- **RAM.** Host RAM (60 GiB, already used by the LMCache store and pinned L1) limits this further.
- **Fit.** The paper's own H100 result (+14%) is the closest analogue to a large-VRAM GPU.

## 14. FastDecode

**Source.** He & Zhai, "FastDecode: High-Throughput GPU-Efficient LLM Serving using Heterogeneous Pipelines", arXiv:2403.11421. The arXiv comments list no venue, so the venue is not verified.

**(1) Mechanism.**
- The model is split in two. The "R-part" (attention, with its KV cache) runs on CPU workers on *remote* nodes. The "S-part" (all other layers) runs on the GPU.
- Only small per-token activations cross the network.
- A sequence-level load-stabilizing schedule and a performance model balance the two sides (from the PDF text).

**(2) What it saves.** All resident GPU KV moves to the aggregated host memory of the CPU nodes.

**(3) Lossless or lossy.** Lossless: exact computation is relocated.

**(4) Reported numbers.**
- **Headline.** "1.88× − 5.04× the throughput of vLLM" (abstract).
- **Setup.** One NVIDIA A10 24 GB as the S-worker, plus up to 4 extra nodes with dual AMD EPYC CPUs as R-workers, connected over InfiniBand (§6.1).
- **7B model.** Peak above 2k tok/s, which is 4× vLLM.
- **13B model.** 4.12× vLLM, and 2.32×/1.88× at batch 128.

**(5) Availability.** Code: not verified. It needs extra CPU nodes.

**(6) Estimate for this workload.** Not applicable. There is one host with 32 cores and 60 GiB, no remote CPU nodes.

## 15. H2O (Heavy-Hitter Oracle)

**Source.** Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models", NeurIPS 2023 (per repo), arXiv:2306.14048. Code: https://github.com/FMInference/H2O.

**(1) Mechanism.** H2O keeps a fixed KV budget split between "heavy hitters" (tokens with the highest accumulated attention) and the most recent tokens. It greedily evicts the rest during generation (Algorithm 1, §5.1).

**(2) What it saves.** Resident KV bytes, and bytes read per step. The paper claims a 5–10× KV memory reduction (§5.1).

**(3) Lossless or lossy.** Lossy. OPT-30B at a 20% budget, full vs. H2O (Table 2):
- COPA: 85.00 vs. 84.00
- OpenBookQA: 43.20 vs. 43.00
- PiQA: 78.51 vs. 78.45
- Winogrande: 70.24 vs. 69.06

**(4) Reported numbers.**
- **Throughput.** Up to 29× / 29× / 3× over DeepSpeed Zero-Inference / HF Accelerate / FlexGen on OPT-6.7B/30B (abstract). The 29× was measured on a **T4 16 GB** (Tables 3–4, e.g., 6.70 vs. 0.23 tok/s).
- **Latency.** 1.1–1.9× lower than FlexGen on an A100 (Table 5).

**(5) Availability.** FlexGen-based and HF-based code (`h2o_flexgen`, `h2o_hf`). Not vLLM.

**(6) Estimate for this workload.** Not suitable.
- **It breaks sharing.** Eviction is per request and depends on the query, so the shared 6144-token prefix can no longer be stored once and reused (by LMCache or APC).
- **It changes outputs.** That conflicts with the deterministic reference workload.
- **Even ignoring that,** a 20% budget would give about 5× more slots (estimate), at an unmeasured accuracy cost on this task.

## 16. SnapKV

**Source.** Li et al., "SnapKV: LLM Knows What You are Looking for Before Generation", NeurIPS 2024 (poster listed at neurips.cc/virtual/2024/poster/93531, per search), arXiv:2404.14469. Code: https://github.com/FasterDecoding/SnapKV.

**(1) Mechanism.**
- Once, at the end of prefill, SnapKV uses the attention of the last L_obs prompt tokens (the "observation window") to vote for the important prefix positions per head.
- It max-pools the votes to keep neighbouring tokens, keeps the top-k positions plus the window, and discards the rest of the prompt KV (§4).

**(2) What it saves.** Resident prompt-KV bytes, which become a fixed budget (e.g., 1024–4096 tokens).

**(3) Lossless or lossy.** Lossy. Mistral-7B-Instruct-v0.2 on LongBench (Table 1), full vs. SnapKV-1024:
- Qasper: 33.06 vs. 29.51
- GovReport: 32.85 vs. 25.89
- RB-P: 52.88 vs. 51.87

At SnapKV-4096 the scores are 33.36 / 30.74 / 52.65. The abstract claims a "negligible accuracy drop" on Needle-in-a-Haystack.

**(4) Reported numbers.** "3.6x increase in generation speed and 8.2x enhancement in memory efficiency ... 16K tokens" (abstract). Setup: LWM-Text-Chat-1M, one A100-80GB, HF implementation, max KV 2048, 512 generated tokens. The baseline OOMs beyond 16K at batch 2, while SnapKV reaches 131K (§5.1.2).

**(5) Availability.** HF monkey-patches, tested with `transformers==4.37.0` and `flash-attn==2.4.0`. Not vLLM 0.15.1.

**(6) Estimate for this workload.**
- **The compression depends on the suffix.** The observation window is the end of the prompt, i.e., our unique 128-token suffix. So the compressed prefix would differ per request and could not be shared through LMCache/APC.
- **Ceiling.** At a 1024–2048-token budget, a request would be about 1.4–2.2K tokens instead of 6.4K, about 3–4× more slots (estimate). It is lossy, needs a framework change, and conflicts with prefix sharing.

## 17. StreamingLLM (attention sinks)

**Source.** Xiao et al., "Efficient Streaming Language Models with Attention Sinks", ICLR 2024, arXiv:2309.17453. Code: https://github.com/mit-han-lab/streaming-llm.

**(1) Mechanism.** StreamingLLM keeps the first ~4 tokens ("attention sinks") plus a rolling window of recent tokens, and evicts everything in between. Positions are assigned within the cache (§3.2).

**(2) What it saves.** Resident KV: a constant-size cache regardless of stream length.

**(3) Lossless or lossy.** Lossy, and it **forgets** the middle. The README states it does not extend the context window.
- **StreamEval.** With a 4+2044 cache, accuracy falls to 0.00 once the answer is about 2,300 tokens back (Table 7).
- **LongBench.** NarrativeQA scores 11.6 vs. 18.7 for truncation (App. D).
- **Perplexity.** Llama-2-7B PG19 PPL is 9.54 with 4+4092 vs. 3359.95 for window-only attention (Table 2).

**(4) Reported numbers.** "up to 22.2x speedup" in per-token decode latency vs. sliding window with recomputation (A6000, Llama-2-7B/13B, HF; §4.5). Stable LM over 4M tokens.

**(5) Availability.** HF code (transformers 4.33), with a TensorRT-LLM example. vLLM is not mentioned in the README.

**(6) Estimate for this workload.** Not applicable. Our requests need their whole 6144-token prefix; dropping the middle drops the content.

## 18. Grouped-Query Attention (GQA)

**Source.** Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", EMNLP 2023, arXiv:2305.13245.

**(1) Mechanism.** Query heads are split into G groups, and each group shares one K head and one V head. G = 1 is MQA; G = number of heads is MHA. An existing MHA checkpoint can be converted by mean-pooling K/V heads and "uptraining" for about 5% of the original pre-training compute.

**(2) What it saves.** Bytes per token, by a factor of (query heads / KV heads).

**(3) Lossless or lossy.** An architectural change that requires (up)training. Quality on T5-XXL (Table 1):

| Variant | Avg score | Inference time |
|---|---|---|
| MHA | 47.2 | 1.51 |
| GQA-8 (5% uptrained) | 47.1 | 0.28 |
| MQA | 46.6 | 0.24 |

**(4) Reported numbers.** See the table above. The units of the time column (s vs. ms) are inconsistent between the table and Fig. 3.

**(5) Availability.** **Already in use.** Llama-3-8B is GQA with 32 query heads and 8 KV heads. Its 128 KiB/token is already 4× below an MHA equivalent (512 KiB/token; arithmetic).

**(6) Estimate for this workload.** No further gain without changing the model. Going lower (e.g., MQA-style 1–2 KV heads) means retraining or uptraining a different model.

## 19. Multi-head Latent Attention (MLA, DeepSeek-V2)

**Source.** DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model", arXiv:2405.04434 (tech report).

**(1) Mechanism.** K and V are jointly compressed into one low-dimensional latent vector per token (d_c = 512), plus a small decoupled RoPE key (d_h^R = 64). Only these are cached. The per-head K/V are reconstructed through up-projections, which can be absorbed into the query/output matrices (§2.1).

**(2) What it saves.** Bytes per token. The cache is (d_c + d_h^R)·l elements, which the paper equates to GQA with about 2.25 groups (Table 1, §2.1.4).

**(3) Lossless or lossy.** An architectural change that requires training. The paper says MLA performs better than MHA (§2.1; the supporting numbers are in App. D.2, **not verified**).

**(4) Reported numbers.** "reduces the KV cache by 93.3%" and "boosts the maximum generation throughput to 5.76 times" vs. DeepSeek 67B (abstract). The throughput was measured on 8× H800 at over 50K tok/s, *with* FP8 parameters and KV quantized to about 6 bits per element (§3.2.3), so the 5.76× is not MLA alone.

**(5) Availability.** vLLM ships MLA backends (`vllm/v1/attention/backends/mla/`; the sm_120 selection list in `platforms/cuda.py:51-66` includes TRITON_MLA). It only applies to MLA models (DeepSeek-V2/V3 family). Converting Llama-3-8B to MLA would need a retrofit plus fine-tuning; specific retrofit methods are not verified here.

**(6) Estimate for this workload.** Changes the object of study. For Llama-3-8B specifically, the bytes per token would be (512+64)·32 layers·2 B ≈ 36 KiB vs. 128 KiB, about 3.6× more slots. That is pure arithmetic, assuming DeepSeek-V2's dimensions transfer, which is unverified.

## 20. Cross-layer KV sharing: CLA and YOCO

### 20a. Cross-Layer Attention (CLA)

**Source.** Brandon et al., "Reducing Transformer Key-Value Cache Size with Cross-Layer Attention", arXiv:2405.12981. Venue: not verified.

**(1) Mechanism.** Adjacent layers share one set of K/V projections. With CLA2, every second layer computes K/V and the next layer reuses them, so only half the layers cache KV (§3).

**(2) What it saves.** Bytes per token: 2× at CLA2, on top of MQA/GQA.

**(3) Lossless or lossy.** Requires training from scratch (1B and 3B models trained, Table 2). At equal KV bytes, CLA2 beats plain MQA:
- 1B, 2560 B/token: validation PPL 13.89 vs. 14.37 (Table 1)
- 3B: 9.34 at 8192 B/token, vs. 9.52 for an MQA model with twice the cache (Table 5)

**(4) Reported numbers.** A 2× KV reduction "while maintaining nearly the same accuracy as unmodified MQA" (abstract). The work reports no serving speedups.

**(5) Availability / (6) Estimate.** No pretrained Llama-3-8B CLA exists (not verified). It means retraining, so it is out of scope. Arithmetically, CLA2 would give 18 slots instead of 9.

### 20b. YOCO (You Only Cache Once)

**Source.** Sun et al., "You Only Cache Once: Decoder-Decoder Architectures for Language Models", NeurIPS 2024 (per the Microsoft Research listing), arXiv:2405.05254.

**(1) Mechanism.**
- The first half of the layers (the "self-decoder") uses efficient attention with a constant-size cache: gated retention or sliding window.
- It produces one global KV cache that all layers of the second half (the "cross-decoder") reuse through cross-attention.
- Prefill can stop after the self-decoder (§2).

**(2) What it saves.** Bytes per token: about L× fewer than a Transformer (§4.4). Also prefill time.

**(3) Lossless or lossy.** A new architecture trained from scratch. YOCO-3B matches StableLM-3B-4E1T on zero-shot evaluations (average 0.636, Table 4), and gets near-perfect single-needle retrieval at 1M context.

**(4) Reported numbers.** All on H100-80GB (§4.4).
- **Memory.** About 9.4× less inference memory at 1M context for the 3B model (12.4 GB for YOCO), and about 2× less at 32K.
- **Prefill.** 2.87× faster at 32K, and 71.8× at 1M.
- **Throughput.** 9.6× at 512K (43.1 vs. 4.5 tok/s).

**(5) Availability / (6) Estimate.** Different model family. No serving support was checked in vLLM 0.15.1 (not verified). Out of scope for a Llama-3-8B study.

---

## Summary table

"Runs here?" means the RTX 5090 (sm_120) with vLLM 0.15.1 and LMCache 0.4.4, as installed. The effect column is an **estimate** unless it says otherwise.

| Technique | Saves | Lossless? | Needs retraining? | Runs here? | Estimated effect on this workload |
|---|---|---|---|---|---|
| vLLM APC | resident bytes for shared prefixes; loads on hits | yes | no | **yes** (default on; we disable it) | **large**: capacity becomes about 9–10 distinct sessions (R ≫ 9) at 24 GiB; hot-session loads vanish |
| SGLang RadixAttention | same as APC | yes | no | other framework; sm_120 not verified | same as APC; no extra gain (prefix is block-aligned) |
| LMCache `use_layerwise` | overlaps load with that step's compute | yes | no | code present in 0.4.4 (fallback op exists); both engines must enable it; needs manual PIECEWISE graphs on 0.15.1; untested | **small**: ≤ about 15–30 ms per admission out of about 400 ms |
| LMCache `enable_async_loading` | remote→L1 fetch moved off the critical path while queued | yes | no | code present; vLLM 0.15.1 handles `None`; H2D copy stays synchronous; untested | **unknown, potentially medium**: depends on the TCP-fetch share of the 400 ms; measure first |
| CachedAttention (ATC'24) | prefill recompute; load hidden by overlap and prefetch | yes (except overflow truncation) | no | research system; ideas live in LMCache | covered by the two LMCache rows above |
| FP8 KV (`--kv-cache-dtype fp8`) | 2× bytes/token: resident, load, store | **no** (scale 1.0 default; KV-only accuracy for Llama-3-8B not verified) | no (optional calibration) | **probably**: FA can't do FP8 on sm_120, so it auto-switches to FLASHINFER (JIT, nvcc 12.8, risk) or TRITON_ATTN; LMCache protocol carries FP8 | **large**: 9 → 19 slots at 24 GiB; load about 400 → 200 ms; must pass a greedy-text check |
| KIVI (2-bit) | about 4–8× bytes/token | no (small drops, e.g. GSM8K 22.67 → 20.77) | no (tuning-free) | **no** (HF only, own CUDA extension) | large in theory; unusable without a new vLLM backend |
| KVQuant (2–4-bit) | about 4–8× bytes/token | no (L3-8B PPL 5.54 → 5.64 at 3-bit) | no (calibration needed) | **no** (research code) | large in theory; unusable here |
| InfiniGen | resident GPU KV; PCIe bytes | no | no (offline SVD skew) | **no** (FlexGen) | negative at 6.4K context (per-step PCIe refetch) |
| ShadowKV | resident GPU KV (V on CPU, low-rank K) | no (RULER 85.53 → 83.57) | no | **no** (A100-tested, own kernels) | negligible: designed for contexts ≥ 60K |
| Quest | per-step bytes read (not resident) | no (near-lossless at moderate budgets) | no | **no** (no GQA support) | none for slots |
| MagicPIG | resident GPU KV; per-step compute | no (RULER 88.7 → 87.4) | no | **no** (HF; Intel AVX512 target; 9950X has AVX512 but untested) | low at 6.4K context |
| NEO (MLSys'25) | resident GPU KV for CPU-attention requests | yes | no | **no** (SwiftLLM fork) | small: a few slow extra slots; RAM-bound (60 GiB) |
| FastDecode | all GPU KV moves to remote CPU nodes | yes | no | **no** (needs remote CPU nodes) | not applicable |
| H2O | resident bytes (5–10×) | no | no | **no** (FlexGen/HF) | not suitable: breaks prefix sharing, changes outputs |
| SnapKV | resident prompt KV, down to a fixed budget | no (GovReport 32.85 → 25.89 at 1024) | no | **no** (HF patches) | not suitable: compression depends on the suffix, so no sharing |
| StreamingLLM | resident KV, made constant | no (forgets the middle) | no | **no** | not applicable: the prefix content is needed |
| GQA | bytes/token (query heads / KV heads) | architectural | yes (uptraining at 5%) | already in Llama-3-8B (4× vs. MHA) | none further |
| MLA (DeepSeek-V2) | bytes/token (about GQA with 2.25 groups) | architectural | yes | only for MLA models | out of scope (different model) |
| CLA / YOCO | 2× (CLA2) to about L× (YOCO) bytes/token | architectural | yes (from scratch) | no Llama-3-8B variant | out of scope |

**Bottom line (estimate).** Three changes run on this stack without changing the model:
- **APC on the decode engine:** lossless, and attacks the per-session duplication.
- **FP8 KV:** lossy-but-small, halves every byte.
- **LMCache async prefetch:** lossless, may shorten the in-step load stall.

APC and FP8 stack, and both directly raise the integer slot count. Layerwise loading barely helps decode-side admission here. Everything else either needs a different framework or model, targets contexts 10–20× longer than 6.4K, or drops tokens in a way that breaks the shared-prefix design.
