#!/usr/bin/env python3
"""What torch.compile changes for one layer of the tenant (Qwen2 RMSNorm, one micro-batch 2 x 512 x 896), on the CPU.
  CUDA_VISIBLE_DEVICES= ../kvcached_validation/.venv-main/bin/python compile_demo.py eager      # every operator the eager layer issues, with the size of the tensor it writes
  CUDA_VISIBLE_DEVICES= ../kvcached_validation/.venv-main/bin/python compile_demo.py compiled   # the traced forward/backward graphs and the generated code (stderr)
Run eager first: compiled compares its input gradient with the eager one."""
import os, sys, torch
assert not torch.cuda.is_available()
from torch.utils._python_dispatch import TorchDispatchMode
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
torch.manual_seed(0)
norm = Qwen2RMSNorm(896).to(torch.bfloat16)
x = torch.randn(2, 512, 896).to(torch.bfloat16).requires_grad_(True)

class Log(TorchDispatchMode):
    def __init__(self): super().__init__(); self.rows = []
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if isinstance(out, torch.Tensor): self.rows.append((func.__name__, tuple(out.shape), str(out.dtype).replace("torch.", ""), out.numel() * out.element_size()))
        return out

if sys.argv[1] == "eager":
    with Log() as L:
        y = norm(x); L.rows.append(("---- backward ----", (), "", 0)); y.float().sum().backward()
    tot = 0
    for n, sh, dt, b in L.rows:
        print(f"{n:28s} {str(sh):18s} {dt:9s} {b/1e6:7.2f} MB"); tot += b
    print("ops:", len(L.rows) - 1, " bytes written to new tensors: %.1f MB" % (tot / 1e6))
    torch.save(x.grad.float(), os.path.join(os.path.dirname(__file__), "g_eager.pt"))
else:
    import torch._logging
    torch._logging.set_logs(aot_graphs=True, output_code=True)
    cn = torch.compile(norm)
    y = cn(x); y.float().sum().backward()
    g = torch.load(os.path.join(os.path.dirname(__file__), "g_eager.pt"))
    print("MAXDIFF vs eager grad:", (x.grad.float() - g).abs().max().item(), "grad max", g.abs().max().item(), file=sys.stderr)
