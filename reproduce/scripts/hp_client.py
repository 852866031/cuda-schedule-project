"""High-priority client: simulates online inference; reports per-iteration latency."""
import torch
import time
import statistics
import sys

n_iters = int(sys.argv[1]) if len(sys.argv) > 1 else 200

x = torch.randn(64, 512, device="cuda")
w1 = torch.randn(512, 1024, device="cuda")
w2 = torch.randn(1024, 512, device="cuda")

# warmup (also lets Tally profile/transform the kernels)
for _ in range(20):
    y = torch.relu(x @ w1) @ w2
torch.cuda.synchronize()

lat = []
for i in range(n_iters):
    t0 = time.perf_counter()
    y = torch.relu(x @ w1) @ w2
    torch.cuda.synchronize()
    lat.append((time.perf_counter() - t0) * 1000)
    time.sleep(0.01)  # think time between requests -> creates idle gaps

lat.sort()
p50 = statistics.median(lat)
p99 = lat[int(len(lat) * 0.99) - 1]
print(f"HP RESULTS: iters={n_iters} p50={p50:.3f}ms p99={p99:.3f}ms avg={statistics.mean(lat):.3f}ms")
