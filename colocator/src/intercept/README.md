# src/intercept/ — LD_PRELOAD interposition layer (`libcolocator.so`)

One file, `intercept.cpp`. It defines the exact public symbols of the CUDA
runtime calls we manage, so `LD_PRELOAD=libcolocator.so` makes PyTorch call
us instead of libcudart. Real functions are resolved once per call site with
`dlsym(RTLD_NEXT, ...)`.

## Interposed symbols (v1)

`cudaLaunchKernel`, `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaMemcpyAsync`,
`cudaMemset`, `cudaMemsetAsync`, `cudaStreamSynchronize`, `cudaEventRecord`,
`cudaEventRecordWithFlags` (the entry point `torch.cuda.Event.record()`
actually uses — interposing only the plain one silently breaks event-based
pipelining), plus `cudaLaunchKernelExC` (counted + warned only). Everything
else reaches libcudart directly.

## cuBLAS interposition (Orion-style, for LLM workloads)

Also interposed: **`cublasGemmEx`** and **`cublasSgemm_v2`** — the calls
torch 2.11's linear layers make (verified: 100% of llama-decode linears
funnel through `cublasGemmEx`; `cublasLtMatmul` is not on that path).
cuBLAS's *internal* kernel launches cannot be interposed (it statically
links the CUDA runtime, and some paths use the driver API), so the whole
**library call** is captured instead — args packed into the `FuncRecord`'s
`blas` block — and the scheduler replays it after pointing the client's
handle at its colocator stream (`cublasSetStream_v2`). The internal kernels
then land on the right stream because the scheduler is the one calling.

Two subtleties:
- Real cuBLAS symbols resolve via `col_resolve()` (records.h):
  `dlsym(RTLD_NEXT)` fails because python loads torch's libs `RTLD_LOCAL`,
  so resolution falls back to `dlopen("libcublas.so.12", RTLD_NOLOAD)`.
- `cublasSetStream_v2` is deliberately NOT interposed: the client may set
  its (meaningless) torch stream on the handle freely — host-side state
  only — and the scheduler re-sets the stream before every replayed call;
  lockstep guarantees no concurrent handle use.
- `alpha`/`beta` are pointers to host scalars on the client's stack — alive
  for the duration of the real call for the same reason kernel `args` are
  (the client spins until ISSUED).

## Two paths per interposer

1. **Managed** (`COLOCATOR_MODE=managed` *and* calling thread registered via
   `col_register_client(idx)`): pack a `FuncRecord` on the caller's stack,
   stamp `t_intercept` (CLOCK_MONOTONIC_RAW), push to the client's queue,
   spin until the scheduler flips the state — `ISSUED` for async ops,
   `DONE` for sync ops (see `../common/records.h`). A 30 s watchdog aborts
   loudly if the scheduler never picks the op (R5 in PROPOSAL.md).
2. **Passthrough** (everything else — passthrough mode, unregistered threads,
   the scheduler's own replayed calls): bump a per-thread per-kind counter and
   forward. `COLOCATOR_VERBOSE=1` logs each call; an exit-time destructor
   prints the summary table.

## Notable choices

- `cudaStreamSynchronize` is *redirected*, not forwarded: the stream the
  client passes is meaningless once its work runs on a colocator stream, so
  the record completes when the client's **colocator** stream drains. This is
  what makes `.cpu()` / `.item()` correct.
- `cudaEventRecord(WithFlags)` is redirected the same way: the event is
  recorded on the client's colocator stream, in queue order behind its prior
  ops. `cudaEventSynchronize`/`Query` need no interposition — by the time the
  client can call them, the real record has been issued, so waiting on the
  real event is correct. This enables depth-N pipelining in workloads.
- Client identity = Linux TID, registered explicitly from Python before the
  thread touches CUDA (simpler than Orion's pre-filled TID table + autograd
  thread adoption; our demo clients are inference-only single threads).
