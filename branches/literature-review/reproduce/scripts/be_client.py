"""Best-effort client: continuously runs large matmuls; reports throughput."""
import torch
import time
import sys

duration = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0

a = torch.randn(4096, 4096, device="cuda")
b = torch.randn(4096, 4096, device="cuda")

# warmup (also lets Tally profile/transform the kernels)
for _ in range(5):
    c = a @ b
torch.cuda.synchronize()

count = 0
t_start = time.perf_counter()
while time.perf_counter() - t_start < duration:
    c = a @ b
    torch.cuda.synchronize()
    count += 1
elapsed = time.perf_counter() - t_start
print(f"BE RESULTS: matmuls={count} elapsed={elapsed:.1f}s throughput={count/elapsed:.2f} it/s")
