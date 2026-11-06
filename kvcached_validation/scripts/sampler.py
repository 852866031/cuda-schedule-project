#!/usr/bin/env python
"""Sample, at a fixed interval, (1) physical GPU memory in use via NVML, (2) kvcached's own
accounting of the instance (limit / used bytes from its /dev/shm segment, read in-process the
same way `kvctl list` does), and (3) vLLM's Prometheus gauges (running, waiting, KV usage,
preemptions). One CSV row per sample. Run for --duration seconds or until SIGTERM."""
import argparse, csv, signal, sys, time, threading, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--interval", type=float, default=0.1)
ap.add_argument("--duration", type=float, default=1e9)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--ipc", default="vllm_a")
ap.add_argument("--metrics-url", default="http://localhost:8000/metrics")
ap.add_argument("--metrics-interval", type=float, default=0.5)
args = ap.parse_args()

import pynvml
pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)

try:
    from kvcached.cli.utils import get_kv_cache_limit
except Exception as e:  # kvcached absent in a plain-only environment
    get_kv_cache_limit = None
    print("kvcached CLI utils unavailable:", e, file=sys.stderr)

WANT = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
        "vllm:num_preemptions_total", "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
        "vllm:external_prefix_cache_queries_total", "vllm:external_prefix_cache_hits_total",
        "vllm:request_queue_time_seconds_sum", "vllm:request_queue_time_seconds_count",
        "vllm:request_success_total")
metrics = {k: "" for k in WANT}
metrics_lock = threading.Lock()
stop = threading.Event()

def scrape():
    while not stop.is_set():
        try:
            body = urllib.request.urlopen(args.metrics_url, timeout=0.4).read().decode()
            vals = {}
            for line in body.splitlines():
                for k in WANT:
                    if line.startswith(k + "{") or line.startswith(k + " "):
                        vals[k] = line.rsplit(" ", 1)[-1]
            with metrics_lock:
                for k in WANT:
                    metrics[k] = vals.get(k, "")
        except Exception:
            with metrics_lock:
                for k in WANT: metrics[k] = ""
        stop.wait(args.metrics_interval)

threading.Thread(target=scrape, daemon=True).start()

pcie = {"tx": "", "rx": ""}
def pcie_loop():
    while not stop.is_set():
        try:
            pcie["tx"] = pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_TX_BYTES)   # KB/s, 20 ms window
            pcie["rx"] = pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_RX_BYTES)
        except Exception:
            pcie["tx"] = pcie["rx"] = ""
        stop.wait(0.01)
threading.Thread(target=pcie_loop, daemon=True).start()
signal.signal(signal.SIGTERM, lambda *a: stop.set())
signal.signal(signal.SIGINT, lambda *a: stop.set())

with open(args.out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t", "nvml_used", "nvml_free", "kvc_limit", "kvc_used", "kvc_prealloc",
                "running", "waiting", "kv_usage", "preemptions", "pc_queries", "pc_hits", "ext_queries", "ext_hits", "queue_sum", "queue_count", "success", "pcie_tx_kbs", "pcie_rx_kbs"])
    t_end = time.time() + args.duration
    nxt = time.time()
    while not stop.is_set() and time.time() < t_end:
        t = time.time()
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        lim = used = pre = ""
        if get_kv_cache_limit is not None:
            try:
                info = get_kv_cache_limit(args.ipc)
                if info is not None:
                    lim, used, pre = info.total_size, info.used_size, info.prealloc_size
            except Exception:
                pass
        with metrics_lock:
            m = [metrics[k] for k in WANT]
        w.writerow([f"{t:.3f}", mem.used, mem.free, lim, used, pre, *m, pcie["tx"], pcie["rx"]])
        f.flush()
        nxt += args.interval
        d = nxt - time.time()
        if d > 0: stop.wait(d)
        else: nxt = time.time()
