#!/usr/bin/env python3
"""Stream a transformer layer's weights from pinned DRAM, correctly through backward.

Why this exists: accelerate's `device_map={"model.layers.N": "cpu"}` is built for inference.
It leaves parameters on the `meta` device and materialises them per forward, which breaks the
backward pass ("expected device meta but got cuda:0"). Training needs the weights present
again when gradients flow, so residency has to be managed around the *whole* step.

Two modes, which is the point of the experiment:

  prefetch=False  fetch-on-demand. The copy is enqueued on the compute stream immediately
                  before the layer that needs it, so transfer and compute serialise. This is
                  the pessimistic bound.
  prefetch=True   while layer N computes, the next `prefetch_depth` layers' weights copy on a
                  side stream. Depth 1 gives each copy one layer's compute window to finish in;
                  deeper lookahead borrows compute time from layers further back, at the cost of
                  holding `depth` staged buffers (~0.406 GiB each) instead of one.

Lifecycle with gradient checkpointing on:

    initial forward   fetch -> compute -> release      (no graph is kept in the segment)
    backward          fetch -> recompute -> backward -> release

so each offloaded layer crosses PCIe exactly twice per step. The `in_backward` flag is what
makes that correct: during backward, torch re-runs the segment's forward, and releasing at the
end of *that* forward would free the weights before the gradients that need them.
"""

import torch


class LayerStreamer:
    """Holds one layer's frozen weights in pinned host memory and pages them around use."""

    def __init__(self, layer, device="cuda"):
        self.layer = layer
        self.device = device
        self.cpu = {}
        self.staged = {}          # name -> GPU buffer copied on the side stream
        self.event = None         # completion of that copy
        self.resident = False
        self.bytes = 0
        for name, p in layer.named_parameters(recurse=True):
            # Stream only the frozen base weights. The LoRA adapters are trainable and tiny
            # (208 MB for the whole model); releasing them would hand the optimiser
            # zero-sized tensors on the next step.
            if p.requires_grad:
                continue
            host = torch.empty_like(p.data, device="cpu", pin_memory=True)
            host.copy_(p.data)
            self.cpu[name] = host
            self.bytes += host.numel() * host.element_size()
        self._release()

    def _params(self):
        return dict(self.layer.named_parameters(recurse=True))

    def stage(self, stream):
        """Start an async copy on `stream`. Safe to call repeatedly."""
        if self.resident or self.staged:
            return
        with torch.cuda.stream(stream):
            for name, host in self.cpu.items():
                self.staged[name] = host.to(self.device, non_blocking=True)
        self.event = torch.cuda.Event()
        self.event.record(stream)

    def bind(self):
        """Make the weights usable on the compute stream, waiting on a staged copy if any."""
        if self.resident:
            return
        compute = torch.cuda.current_stream()
        if self.staged:
            compute.wait_event(self.event)
            params = self._params()
            for name, buf in self.staged.items():
                # The buffer was allocated on the copy stream; tell the caching allocator it
                # is now in use on the compute stream so it is not recycled underneath us.
                buf.record_stream(compute)
                params[name].data = buf
            self.staged.clear()
        else:
            params = self._params()
            for name, host in self.cpu.items():
                params[name].data = host.to(self.device, non_blocking=True)
        self.resident = True

    def _release(self):
        params = self._params()
        for name in self.cpu:
            p = params[name]
            p.data = torch.empty(0, device=self.device, dtype=p.dtype)
        self.resident = False
        self.staged.clear()

    def release(self):
        self._release()


class OffloadManager:
    """Installs streamers on the last `n_offload` layers, optionally with depth-1 prefetch."""

    def __init__(self, layers, n_offload, device="cuda", prefetch=False, depth=1,
                 pattern="tail"):
        self.in_backward = False
        self.prefetch = prefetch
        self.depth = max(1, depth)
        self.pattern = pattern
        self.streamers = []
        self.handles = []
        self.fetches = 0
        self.copy_stream = torch.cuda.Stream() if prefetch else None

        # Which layers to offload matters as much as how many. "tail" bunches every transfer
        # into the last stretch of the step, where there is little compute left to hide behind.
        # "interleave" spreads them so each streamed layer has a resident neighbour's compute
        # to overlap with.
        all_layers = list(layers)
        if pattern == "interleave" and 0 < n_offload < len(all_layers):
            stride = len(all_layers) / n_offload
            picked = sorted({min(int(i * stride), len(all_layers) - 1) for i in range(n_offload)})
            managed = [all_layers[i] for i in picked]
        else:
            managed = all_layers[len(all_layers) - n_offload:]
        for idx, layer in enumerate(managed):
            s = LayerStreamer(layer, device)
            self.streamers.append(s)

            def pre_hook(module, args, i=idx):
                cur = self.streamers[i]
                cur.bind()
                self.fetches += 1
                if self.prefetch:
                    # Forward runs ascending, backward's recompute runs descending: stage
                    # whichever layers are about to be needed, `depth` of them.
                    step = -1 if self.in_backward else 1
                    for d in range(1, self.depth + 1):
                        nxt = i + step * d
                        if 0 <= nxt < len(self.streamers):
                            self.streamers[nxt].stage(self.copy_stream)

            def fwd_hook(module, args, output, i=idx):
                # Keep weights alive through the recomputed-forward + backward window.
                if not self.in_backward:
                    self.streamers[i].release()

            def bwd_hook(module, grad_in, grad_out, i=idx):
                self.streamers[i].release()

            self.handles.append(layer.register_forward_pre_hook(pre_hook))
            self.handles.append(layer.register_forward_hook(fwd_hook))
            self.handles.append(layer.register_full_backward_hook(bwd_hook))

    @property
    def offloaded_bytes(self):
        return sum(s.bytes for s in self.streamers)

    def pre_forward(self):
        """Warm the leading layers so the first binds are not cold synchronous copies."""
        if self.prefetch:
            for d in range(min(self.depth, len(self.streamers))):
                self.streamers[d].stage(self.copy_stream)

    def backward(self, loss):
        """Run backward with the flag set, so recomputed forwards do not free too early."""
        self.in_backward = True
        if self.prefetch:
            for d in range(min(self.depth, len(self.streamers))):
                self.streamers[len(self.streamers) - 1 - d].stage(self.copy_stream)
        try:
            loss.backward()
        finally:
            self.in_backward = False
            for s in self.streamers:
                s.release()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
