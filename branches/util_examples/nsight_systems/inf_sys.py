#!/usr/bin/env python3
import multiprocessing as mp
import queue
import threading
import time
from dataclasses import dataclass

import torch


@dataclass
class Request:
    req_id: int
    created_t: float
    payload: torch.Tensor


def make_requests(num_requests: int, dim: int):
    reqs = []
    for i in range(num_requests):
        payload = torch.randn(dim, dtype=torch.float32)
        reqs.append(Request(i, time.perf_counter(), payload))
    return reqs


def summarize(name: str, latencies, total_runtime, completed):
    if completed == 0:
        print(f"{name}: completed=0")
        return
    avg_latency_ms = sum(latencies) / len(latencies) * 1000.0
    throughput = completed / total_runtime
    print(
        f"{name}: completed={completed}, "
        f"avg_latency_ms={avg_latency_ms:.2f}, "
        f"throughput_req_per_s={throughput:.2f}"
    )


def efficient_pipeline_worker():
    torch.cuda.set_device(0)
    device = "cuda:0"

    torch.manual_seed(0)

    # Simulated inference model
    hidden = 4096
    model = torch.nn.Sequential(
        torch.nn.Linear(hidden, hidden * 2),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden * 2, hidden),
    ).to(device).half().eval()

    # Prepare requests
    num_requests = 128
    batch_size = 16
    requests = make_requests(num_requests, hidden)

    latencies = []
    completed = 0
    start_t = time.perf_counter()

    with torch.cuda.nvtx.range("GPU0 setup"):
        pass

    idx = 0
    while idx < len(requests):
        batch_reqs = requests[idx: idx + batch_size]
        idx += batch_size

        with torch.cuda.nvtx.range("GPU0 request batch"):
            with torch.cuda.nvtx.range("GPU0 CPU preprocess"):
                batch_cpu = torch.stack([r.payload for r in batch_reqs], dim=0)
                time.sleep(0.003)

            with torch.cuda.nvtx.range("GPU0 H2D"):
                batch_gpu = batch_cpu.to(device, dtype=torch.float16, non_blocking=True)

            with torch.cuda.nvtx.range("GPU0 inference compute"):
                with torch.no_grad():
                    out = model(batch_gpu)
                    out = torch.nn.functional.gelu(out)
                    out = out @ out.transpose(0, 1)

            with torch.cuda.nvtx.range("GPU0 D2H"):
                result = out.cpu()

            torch.cuda.synchronize()

        done_t = time.perf_counter()
        for r in batch_reqs:
            latencies.append(done_t - r.created_t)
            completed += 1

    total_runtime = time.perf_counter() - start_t
    summarize("GPU0 efficient pipeline", latencies, total_runtime, completed)


def inefficient_pipeline_worker():
    torch.cuda.set_device(1)
    device = "cuda:1"

    torch.manual_seed(0)

    hidden = 4096
    model = torch.nn.Sequential(
        torch.nn.Linear(hidden, hidden * 2),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden * 2, hidden),
    ).to(device).half().eval()

    num_requests = 128
    requests = make_requests(num_requests, hidden)

    latencies = []
    completed = 0
    start_t = time.perf_counter()

    with torch.cuda.nvtx.range("GPU1 setup"):
        pass

    for r in requests:
        with torch.cuda.nvtx.range(f"GPU1 request {r.req_id}"):
            with torch.cuda.nvtx.range("GPU1 CPU preprocess"):
                x_cpu = r.payload.unsqueeze(0)
                time.sleep(0.003)

            with torch.cuda.nvtx.range("GPU1 H2D"):
                x_gpu = x_cpu.to(device, dtype=torch.float16, non_blocking=True)
            torch.cuda.synchronize()

            # Deliberately fragmented compute pipeline
            with torch.cuda.nvtx.range("GPU1 fragmented inference"):
                with torch.no_grad():
                    y = x_gpu
                    for _ in range(8):
                        y = model[0](y)
                        torch.cuda.synchronize()

                        y = torch.relu(y)
                        torch.cuda.synchronize()

                        y = model[2](y)
                        torch.cuda.synchronize()

                        y = y + 0.01 * x_gpu.repeat(1, 2)[:, :y.shape[1]]
                        torch.cuda.synchronize()

            with torch.cuda.nvtx.range("GPU1 D2H"):
                result = y.cpu()
            torch.cuda.synchronize()

        done_t = time.perf_counter()
        latencies.append(done_t - r.created_t)
        completed += 1

    total_runtime = time.perf_counter() - start_t
    summarize("GPU1 inefficient pipeline", latencies, total_runtime, completed)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    if torch.cuda.device_count() < 2:
        raise RuntimeError("Need at least 2 GPUs")

    mp.set_start_method("spawn", force=True)

    p0 = mp.Process(target=efficient_pipeline_worker)
    p1 = mp.Process(target=inefficient_pipeline_worker)

    p0.start()
    p1.start()

    p0.join()
    p1.join()

    if p0.exitcode != 0:
        raise RuntimeError(f"GPU0 worker failed with exit code {p0.exitcode}")
    if p1.exitcode != 0:
        raise RuntimeError(f"GPU1 worker failed with exit code {p1.exitcode}")


if __name__ == "__main__":
    main()