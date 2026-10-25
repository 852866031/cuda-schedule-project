# Prompt for the next session — branch `kvcache-vram_size`

（直接把下面整段贴给新的 Claude session。）

---

你在 `/home/jiaxuan/Documents/Projects/kvcache-vram_size` 工作（2× RTX 5090 32 GB，PCIe Gen4 ×8，无 P2P，32 核，60 GiB RAM）。它是 `/home/jiaxuan/Documents/Projects/vram-limit` 仓库的一个 **git worktree**，分支 `kvcache-vram_size`（从 `vram-limit` 的 bcb0315 建出）。**只在这个目录里工作，不要改动 `vram-limit` 目录**（它是另一个分支的工作区，两个目录共享同一个 git 仓库）。先完整读 `CLAUDE.md`（环境、安全规则、测量约定都是硬性的）和上一个项目的三条记忆：`/home/jiaxuan/.claude/projects/-home-jiaxuan-Documents-Projects-vram-limit/memory/` 下的 `vram-limit-environment-traps.md`、`vram-limit-user-working-style.md`、`gpu-util-measurement-gotchas.md`（对本目录同样适用，可以复制成本项目的记忆）。和我用中文交流，所有交付物（plan、decision log、report、图）用英文。**先给我计划再跑实验**；跑的时候随时检查有没有 stall（heartbeat/stall watchdog、GPU 利用率、engine log 是否还在刷）。每个实验的预测在运行前写进 decision log，运行后补结果。

## 0. 清场（目录和分支已经建好）

1. 已完成：`git worktree add -b kvcache-vram_size ../kvcache-vram_size`；`.venv` 和 `.venv-matched` 是指向 `../vram-limit/` 的**符号链接**（11 GB，不复制；`.gitignore` 已加无斜杠的规则，未 commit）；被 gitignore 的上一研究原始数据已**复制**进来：`output/raw/pfp_*.json`（80 个）、`output/logs/pfp_archive/`。launch 脚本用 `REPO="$(cd ../.. && pwd)"` 找 venv，在新目录下自动指向符号链接，已验证 `.venv-matched` 可导入 vLLM 0.15.1。
   - 注意：两个目录共用 GPU 和端口，**同一时间只能有一个目录在跑栈**；开跑前确认 `vram-limit` 那边没有残留进程（`nvidia-smi`、`ss -ltn` 查 8000/8100/8200/8300）。
2. **保留**（这次要用或要引用）：
   - `CLAUDE.md`、`README.md`（改写成本分支的说明）、`.gitignore`
   - `scripts/common/`（`mem_guard.sh`、`gpu_monitor.py`、`calibrate_pcie.py`、`status.py`）
   - `scripts/inf_pf_prio/`（8B split-LMCache 栈的最新自包含版本：`pfp_launch.sh`、`pfp_sweep.py`、`lmc_proxy.py`、`lmc_server_main.py`、`client.py`、`workload.py`、`gpu_monitor.py`、`host_monitor.py`、`pfp_heartbeat.sh`、`pfp_stop.sh`、`site/`）——以它为基础复制出新的 `scripts/kv_vram/`，去掉 3B 邻居相关部分，只留 8B split（prefill GPU0，decode GPU1）
   - 上一个研究的结果：`reports/report_colocation_prio.md`、`PLAN_PF_PRIO.md`、`DECISIONS_PF_PRIO.md`、`figures/pfp_*.png`、`output/summary_pfp.csv`、`output/inf_pf_prio/`、`output/gpumon/` 里 `pfp_*` 文件、`scripts/plots/plot_pfp_results.py`
   - `reports/report_split_inference.md`（split-LMCache 栈的来历，作背景引用）
3. 其余已跟踪文件用 **`git rm`** 删除（只改本分支，`vram-limit` 分支和目录不受影响）。**绝不要 `rm -rf`** `output/raw/`、`output/logs/`、`.venv*`（未跟踪的本地数据；venv 是符号链接，删链接目标会毁掉另一个目录的环境）。
4. 把下面"已知事实"整理成 `NOTES_KV_SLOTS.md`（英文）作为本研究的起点，commit 并 push（`git push -u origin kvcache-vram_size`）。

## 1. 背景：上一个 session 已经知道的（全部来自现有数据，可在 `output/raw/pfp_*.json` 与 `output/logs/pfp_archive/<cell>/pfp_big_decode.log` 复核）

- 参考负载（不能改）：32 sessions × 6144-token 固定前缀 + 128 unique suffix，128 out，zipf-1.1，open-loop Poisson，seed 0，300 requests；router `--forward-first-token --max-inflight 999`。
- Llama-3-8B KV = 32 层 × 2 × 8 KV heads × 128 × 2 B = **128 KiB/token**；一个请求 6400 token = 400 个 16-token block = **0.78 GiB**。
- decode budget → KV grant：23 GiB → 6.77 GiB，25 GiB → 8.77 GiB（grant ≈ budget − 16.2 GiB）。照此 24/26/28 GiB ≈ 7.77/9.77/11.77 GiB ≈ **9/12/15 个完整请求**（要用 probe 实测，block 取整）。
- 上一个研究里 23 vs 25 GiB（fp8 3B 邻居，8 req/s）：KV 载入 397 vs 413 ms（一样快）；decode 停留时间 4.14 vs 4.89 s（25 GiB 反而更长，batch 更大）；**decode 前排队 5.6 vs 1.3 s**——差别全在排队。
- engine 10 s stats 行：23 GiB 时 Running 从不超过 8（KV usage 91–92%，每请求 ~11.4%），Waiting 一路涨到 27；25 GiB 时 Running 到 11（KV 97%），Waiting 能清空。即调度器只能放整数个完整请求：**c = 8 vs 11**。之前报告里"8.7 slots"的算法（grant ÷ 0.78）忽略了整数取整，需要更正。
- 排队论粗算：2 QPS × 4.14 s = 8.3 > 8（不稳定，队列线性增长）；2 × 4.89 = 9.8 < 11（ρ≈0.89，稳定）。
- vLLM 指标定义（`vllm/v1/metrics/stats.py`）：queued = scheduled − queued；prefill_time = first_token − scheduled（decode 端即 KV 载入）；decode_time = last − first。
- **当前 offload 机制**（读源码确认：`.venv-matched/.../lmcache/integration/vllm/vllm_v1_adapter.py:741` `start_load_kv`）：decode 端调度时一次性给请求分配全部 400 block（block id 跨全部 32 层共享）；LMCache `use_layerwise=False`、`enable_async_loading=False`（默认）→ 在 forward 前**同步**把 32 层全部拷进 paged blocks；decode engine `--no-enable-prefix-caching`（仓库里没记录原因，split 报告只说"decode VRAM 是工作区不是缓存"）。
- **两个疑似浪费（待验证）：**
  1. prefix caching 关闭 → 同 session 并发请求各存一份前缀。用到达序列模拟：并发平均 7.2 个请求只有 5.1 个不同 session；并发 ≥ 9 时 10.3 个请求只有 6.8 个 session；最热 session 占 32% 请求。
  2. 同步载入可能卡住整个 batch：8B 单独时 ITL p50 14.8 ms 但 mean 28.4 ms（fp8b25 p95 155 ms）。推测每次接纳新请求整批停 ~200 ms。若消除，停留时间 ~4.1 → ~2.3 s，2 QPS 所需 slot 8.3 → ~4.7。
- 为什么 decode 不能逐层 streaming：每个 decode step 每层都对全部历史 token 做 attention，需要全部层的全部 KV；按 PCIe ~13 GB/s（未实测，用 `scripts/common/calibrate_pcie.py` 测）每请求每步要传 0.78 GiB ≈ 60 ms，而整步只要 ~15 ms。逐层流水只能加速"载入"（LMCache `use_layerwise`、CachedAttention），不减少驻留显存。另：实测 0.78 GiB 载入 ~400 ms ≈ 2 GB/s，远低于 PCIe（LMCache 编译 op 不加载、走 Python fallback + TCP store）。

## 2. 本研究：只研究 decode（8B split 栈，GPU1 上只有 8B decode，没有任何邻居）

写 `PLAN_KV_VRAM.md`（pre-registration，预测写死，只注 outcome）和 `DECISIONS_KV_VRAM.md`。研究对象是 decode engine 的 KV 空间、在跑请求数、显存分配大小三者的关系。prefill（GPU0，util 0.80）只是 KV 来源。

### 交付 1：KV slot 与性能
- **逐请求时间线 instrumentation（E1）**：给 8B decode engine 加只记录不改行为的补丁（放 `scripts/kv_vram/site/`，经 PYTHONPATH sitecustomize 加载，参照 `inf_pf_prio/site/`）。每个请求记：到达 decode engine 时刻、到达时空闲 KV block 数、被调度时刻、KV 载入完成（第一个 decode token）时刻、生成完时刻；每个 scheduler step 记：running、waiting、free blocks、step 时长、该 step 是否在做 KV 载入。先在一个 smoke cell 上验证补丁不改变 TPOT/throughput。
- **扫描 A**：固定 VRAM，变 request rate（例如 24 GiB，QPS 1–4 步长 0.5，你来定，要覆盖 knee 两侧），测 throughput、TPOT、ITL、decode queue time（`vllm:request_queue_time_seconds`）、failed、以及 E1 数据。
- **扫描 B**：固定 request rate（选 A 中最敏感的那个），VRAM = 24 / 26 / 28 GiB，测同样指标。若时间允许，做 rate × VRAM 的完整网格。
- 先用 probe 确认每个 budget 的 KV grant 和"能同时放下的完整请求数"（GPU 两次 probe 间验证为空）。
- **图**：
  - 3D 图（或等价的热力图 + 等高线，选更易读的那个，但至少出一张 3D）：两轴 VRAM size × request rate，高度/颜色为 queue time（另做 throughput、TPOT）。
  - **自己设计一张图，精准反映"剩余 KV 空间"与性能的关系**，要认真想，让不知道实验细节的人也能看懂。例如：时间轴上叠画 free blocks（以"还能放几个请求"为单位）与排队数、标出每个请求到达/被放入/完成的时刻，挑一个具体请求做时间线注释；或者以"到达时空闲槽位数"为横轴、该请求的等待时间为纵轴的散点。要求标题写明测量的指标，baseline 醒目，图例分组，标签不重叠，**画完一定要看 PNG**。
  - 理论对照：M/G/c（c = 实测整数槽位，服务时间用实测停留时间分布）预测 queue time vs 实测。
- 报告里按时间线举具体例子（"请求 #N 在 24 GiB 下 t=xx s 到达，此时 9 槽全满、空闲 block 只有 yy 个，等到请求 #M 在 t=zz s 完成才被放入；同一请求在 28 GiB 下到达时有 3 个空槽，立刻载入"）。

### 交付 2：系统路线图 / 示意图（3–5 组）
1. 一个请求在 split-LMCache 栈里的完整路线（router → prefill GPU0 → LMCache L1/store → decode GPU1），哪一步在哪个设备、什么时间。
2. 排队是怎么形成的：槽位 = 整数个完整请求，到达 Poisson，c 个服务台；配合实测时间线。
3. LMCache 在 decode 中怎么搬 KV：256-token chunk、L1 pinned pool、store、`start_load_kv` 同步拷入 paged blocks；以及 LMCache / vLLM 的显存与 host 内存布局（paged block × 32 层张量，chunk 格式）。
4. 为什么 decode 不能逐层载入，或隔几层提前载入（prefetch depth d）：**要做实验验证**——(a) 用 `calibrate_pcie.py` 实测 pinned H2D 带宽；(b) 写一个独立 micro-benchmark（`.venv`，不用 vLLM）：模拟一个 decode step，KV 常驻 HBM vs 每步逐层从 pinned host 流入（prefetch 深度 d = 0,1,2,4,8），扫 batch 大小，画 step 时长 vs batch 的性能图；(c) 若可行，对比 LMCache `use_layerwise=True` 对"载入阶段"的效果。示意图 + 性能图一起解释。

### 交付 3：相关工作（每种一张示意图，对懂 KV cache 和推理基本原理的非专家讲清楚）
至少覆盖：前缀共享（vLLM APC、SGLang RadixAttention）；载入与计算重叠（LMCache layerwise/async loading、CachedAttention/AttentionStore）；KV 量化（vLLM `--kv-cache-dtype fp8`、KIVI、KVQuant）；offload + 稀疏取回（InfiniGen、ShadowKV、Quest、MagicPIG）；CPU 侧 attention（NEO MLSys'25、FastDecode）；token 丢弃（H2O、SnapKV、StreamingLLM）；结构性（GQA、MLA、跨层 KV 共享）。每种写清：省的是什么（驻留量 / 停留时间 / 每 token 字节）、是否无损、对我们这个 workload 预计的收益、引用来源（查证，不要凭记忆写数字）。

### 交付 4：可行性小测试
对可能可行的方法各做一个小 smoke（可以用其他 vLLM 版本建**新 venv**，绝不要改动 `.venv-matched`；或写独立实验）：
- decode 端打开 prefix caching（一个 flag，无损；注意第二个同 session 请求只有在第一个的 block 载入并缓存后才能命中）
- LMCache `enable_async_loading` / `use_layerwise`（先确认与 vLLM 0.15.1 的 connector 兼容）
- `--kv-cache-dtype fp8`（prefill 与 decode 两端必须同 dtype；确认 Blackwell sm_120 attention backend 支持与 LMCache 0.4.4 能否存取 fp8 KV；有轻微精度差，记录输出是否变化）
- 可选：ShadowKV / NEO 的独立复现可行性（只评估能否在本机跑，不强求）
每个 smoke 记录：能否启动、是否 hang、KV grant / 槽位数变化、单点 TPOT/queue。

### 交付 5：对可行方法重复交付 1
用与交付 1 完全相同的负载和 cell 重跑扫描 A/B，和 baseline 同图对比（baseline 醒目），解释每种方法改变的是 c（槽位数）、停留时间、还是两者。

## 3. 必须遵守（摘自 CLAUDE.md，违反会冻机或丢数据）
- vLLM 只用 `.venv-matched`（0.15.1），每次带 `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`，`PYTHONHASHSEED=0`，`OMP_NUM_THREADS` 设好（单 8B 栈 8 即可）。
- 只通过 launch 脚本起栈（预算检查 + mem_guard + heartbeat），一次只跑一套栈；pinned 内存超配会硬冻机。
- config 名、telemetry 文件名、raw 文件名永不冲突（加后缀）；raw 不覆盖。
- swap-in > 0.5 GB 的 cell 重跑，报告 swap 少的那次。
- 一个 cell 只有零 failed 才算 ok；TTFT 只在 `--forward-first-token` 下报告。
- 每次改动后 commit 并 push 到 `kvcache-vram_size`。
