#!/usr/bin/env python3
"""Launch / probe / tear down a vLLM server for one sweep point.

Each sweep point is a fresh server: `--gpu-memory-utilization` and the offload pool are
startup-time settings, and a fresh process also guarantees an empty cache, so runs can't
contaminate each other.
"""

import json
import os
import sys
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG_DIR = REPO / "output" / "logs"

SYSTEM_LIBSTDCXX = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
MODEL = "NousResearch/Meta-Llama-3-8B-Instruct"
PORT = 8100

# Metrics worth capturing per run. The external_* pair is the DRAM tier: queries/hits there
# are exactly the requests offloading rescued from a recompute.
METRIC_KEYS = [
    "vllm:prefix_cache_queries", "vllm:prefix_cache_hits",
    "vllm:external_prefix_cache_queries", "vllm:external_prefix_cache_hits",
    "vllm:num_preemptions", "vllm:kv_offload_total_bytes", "vllm:kv_offload_total_time",
]


class VLLMServer:
    def __init__(self, gpu_mem_util, kv_offload_gib=None, backend="native",
                 cpu_offload_gb=0.0, model=MODEL, port=PORT, max_model_len=8192,
                 run_name="run", gpu=0, extra_args=None):
        self.gpu_mem_util = gpu_mem_util
        self.kv_offload_gib = kv_offload_gib
        self.backend = backend
        self.cpu_offload_gb = cpu_offload_gb
        self.model = model
        self.port = port
        self.max_model_len = max_model_len
        self.run_name = run_name
        self.gpu = gpu
        self.extra_args = extra_args or []
        self.proc = None
        self.log_path = LOG_DIR / f"{run_name}.log"
        self.startup_info = {}

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}"

    def command(self):
        # Launch with *this* interpreter: the sweep runs from a venv that shadows the base
        # conda env's too-old transformers (see README). A bare `vllm` would resolve to the
        # base env and fail at import.
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", self.model,
            "--port", str(self.port),
            "--gpu-memory-utilization", f"{self.gpu_mem_util:.4f}",
            "--max-model-len", str(self.max_model_len),
            "--seed", "0",
            "--disable-log-requests",
        ]
        if self.kv_offload_gib:
            cmd += ["--kv-offloading-size", str(self.kv_offload_gib),
                    "--kv-offloading-backend", self.backend]
        # OffloadingConnector refuses to run with the hybrid KV cache manager. Llama-3 is
        # uniform full attention so HMA buys nothing here -- but it is disabled for *every*
        # arm, offload or not, so both arms share one allocator and stay comparable.
        cmd += ["--disable-hybrid-kv-cache-manager"]
        if self.cpu_offload_gb:
            cmd += ["--cpu-offload-gb", str(self.cpu_offload_gb)]
        return cmd + self.extra_args

    def start(self, timeout=900):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.gpu), VLLM_LOGGING_LEVEL="INFO")
        # This conda env ships libstdc++ 6.0.29 (GLIBCXX_3.4.29) but vllm._C.abi3.so needs
        # GLIBCXX_3.4.32. Preload the system libstdc++ (3.4.33) rather than mutating the
        # user's conda env. Without this, `vllm serve` dies at import.
        if os.path.exists(SYSTEM_LIBSTDCXX):
            env["LD_PRELOAD"] = SYSTEM_LIBSTDCXX + (
                ":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
        cmd = self.command()
        self.log_file = open(self.log_path, "w")
        self.log_file.write(f"$ {' '.join(cmd)}\n\n")
        self.log_file.flush()
        # New process group so a hung engine subprocess tree can be killed as a unit.
        self.proc = subprocess.Popen(cmd, stdout=self.log_file, stderr=subprocess.STDOUT,
                                     env=env, start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"server exited early (rc={self.proc.returncode}); see {self.log_path}\n"
                    + self._log_tail())
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=2) as r:
                    if r.status == 200:
                        self.startup_info = self.parse_log()
                        self.startup_info["startup_s"] = round(time.time() - t0, 1)
                        return self.startup_info
            except Exception:
                time.sleep(2)
        self.stop()
        raise TimeoutError(f"server not healthy after {timeout}s; see {self.log_path}")

    def _log_tail(self, n=40):
        try:
            return "".join(self.log_path.read_text(errors="replace").splitlines(True)[-n:])
        except Exception:
            return ""

    def parse_log(self):
        """Pull the actual (not assumed) KV cache sizing out of the startup log."""
        text = self.log_path.read_text(errors="replace")
        info = {}
        m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", text)
        if m:
            info["gpu_kv_tokens"] = int(m.group(1).replace(",", ""))
            info["gpu_kv_gib"] = round(info["gpu_kv_tokens"] * 131072 / (1 << 30), 3)
        m = re.search(r"Available KV cache memory:\s*([\d.]+)\s*GiB", text)
        if m:
            info["available_kv_gib"] = float(m.group(1))
        m = re.search(r"model weights take\s*([\d.]+)\s*GiB", text)
        if m:
            info["weights_gib"] = float(m.group(1))
        m = re.search(r"non-torch memory takes\s*([\d.]+)\s*GiB", text)
        if m:
            info["non_torch_gib"] = float(m.group(1))
        m = re.search(r"PyTorch activation peak memory takes\s*([\d.]+)\s*GiB", text)
        if m:
            info["activation_peak_gib"] = float(m.group(1))
        m = re.search(r"([\d.]+)\s*GiB\s*memory for CUDA graphs", text)
        if m:
            info["cudagraph_gib"] = float(m.group(1))
        m = re.search(r"Maximum concurrency for ([\d,]+) tokens per request:\s*([\d.]+)x", text)
        if m:
            info["max_concurrency"] = float(m.group(2))
        return info

    def metrics(self):
        """Scrape /metrics into a flat dict. Labelled series are summed across labels."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/metrics", timeout=10) as r:
                text = r.read().decode()
        except Exception as e:
            return {"error": str(e)}

        out = {}
        for line in text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            name = line.split("{")[0].split(" ")[0]
            base = name.rsplit("_total", 1)[0] if name.endswith("_total") else name
            if base not in METRIC_KEYS and name not in METRIC_KEYS:
                continue
            try:
                value = float(line.rsplit(" ", 1)[1])
            except (ValueError, IndexError):
                continue
            key = base
            # Keep the offload direction (cpu_to_gpu = load, gpu_to_cpu = store) separate;
            # they are different physical operations and cost different amounts.
            m = re.search(r'transfer_type="([^"]+)"', line)
            if m:
                key = f"{base}:{m.group(1)}"
            out[key] = out.get(key, 0.0) + value
        return out

    def stop(self):
        if self.proc is None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
            self.proc.wait(timeout=60)
        except Exception:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                self.proc.wait(timeout=30)
            except Exception:
                pass
        try:
            self.log_file.close()
        except Exception:
            pass
        self.proc = None
        time.sleep(5)  # let the driver release VRAM before the next run starts

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


def gpu_free_gib(gpu=0):
    q = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used,memory.total",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    used, total = [float(x) for x in q.stdout.strip().split(",")]
    return {"used_gib": round(used / 1024, 3), "total_gib": round(total / 1024, 3)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Start one server and print its sizing, then exit.")
    ap.add_argument("--util", type=float, default=0.90)
    ap.add_argument("--kv-offload-gib", type=float, default=None)
    ap.add_argument("--hold", action="store_true", help="keep serving until Ctrl-C")
    ap.add_argument("--extra", nargs="*", default=[], help="extra args passed to vllm serve")
    args = ap.parse_args()

    s = VLLMServer(args.util, args.kv_offload_gib, run_name="manual", extra_args=args.extra)
    info = s.start()
    print(json.dumps(info, indent=2))
    print(json.dumps(s.metrics(), indent=2))
    if args.hold:
        print(f"serving at {s.base_url} -- Ctrl-C to stop")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    s.stop()
