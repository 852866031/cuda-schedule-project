"""Patches for vLLM's P2pNcclConnector, loaded into every vLLM process.

CPython imports `sitecustomize` automatically at startup, so putting this
directory on PYTHONPATH reaches the API server *and* the EngineCore it spawns.
That matters: the bugs below live in the engine process, which is started with
`spawn` and therefore does not inherit a monkeypatch applied in the parent.

Pinned to vllm==0.15.1. Three changes:

  0. `InputProcessor.assign_request_id` is made a no-op for connector-routed requests
     (always on). Upstream rewrites every incoming id as

         request.request_id = f"{request.external_req_id}-{random_uuid():.8}"

     with no way to opt out. P2pNcclConnector keys its transfers on that *internal* id,
     and the prefill and decode instances draw independent random suffixes -- so the
     producer stores `...-0-9bf8be13#model.layers.0.self_attn.attn` while the consumer
     waits for `...-0-86d6dc44#...`. The keys can never match and no KV is ever loaded:
     the connector is simply broken in this version, and it fails by hanging rather than
     erroring. Skipping the suffix when the id carries `___decode_addr_` restores the
     agreement, and leaves every other request on upstream behaviour. Uniqueness still
     holds because the router mints a uuid4 per request.

  1. A deadline on the consumer's wait (always on, `P2P_RECV_TIMEOUT_S`, default
     600). Upstream does

         while tensor_id not in self.recv_store:
             self.recv_store_cv.wait()

     with no timeout and no logging, so one missing or mis-keyed tensor wedges
     the decode engine permanently and silently -- not just that request: the
     engine's step loop never returns, so the whole instance stops, including
     its /metrics. This gives up after the deadline, logs the id it wanted next
     to the ids it actually holds, and returns None so the engine survives.
     Once a request times out on one layer, its remaining layers fail fast
     rather than serially burning another 31 deadlines.

     A 🔴RECV TIMEOUT line means the KV never arrived: that request decoded
     from uninitialised cache, so the config's numbers are invalid.

  3. The consumer's spill to the pinned host pool runs on a dedicated CUDA stream
     (always on). Upstream's `TensorMemoryPool.store_tensor` issues its D2H `copy_` on
     the listener thread's current stream -- the DEFAULT stream, so every spill queues
     behind whatever decode step is in flight (~17 ms at batch 8), and because the
     listener is one thread, layer n's spill delays layer n+1's ZMQ ack to the producer.
     Measured: 26.6 ms of the producer's 33.2 ms per-layer send was waiting for that ack
     when the decode engine was busy, throttling prefill to half its idle rate. A private
     stream decouples the copy from decode compute. Safe on both sides: the NCCL recv
     synchronizes recv_stream before store_tensor is called (data complete), and the
     blocking copy_ finishes before the tensor is released (no cross-stream reuse).

  4. Backpressure on the producer's send queue (always on, `P2P_SEND_QUEUE_CAP_GB`,
     default 4). Upstream's PUT_ASYNC queue is unbounded in VRAM: save_kv_layer enqueues
     a fresh GPU copy of each layer's KV (extract is advanced indexing), and the single
     sender thread drains at ~12.5 ms/layer. With prefix caching on the producer, a step
     can complete a dozen nearly-free cache-hit prefills at once and enqueue ~11 GiB of
     GPU tensors in one forward -- which is exactly how the prefill node OOMed on a
     774 MiB MLP buffer with 29 GiB allocated against a 25 GiB target. save now blocks
     while the queued bytes exceed the cap; the engine step stretches instead of the
     allocator dying. No deadlock: the sender thread never takes this lock to drain.

  2b. `P2P_TIMING=1` decomposes the producer's per-layer send into "moving bytes" and
     "everything else", which is mostly waiting for the consumer to acknowledge. send_sync
     is synchronous per layer -- ZMQ PUT, wait for ack, ncclSend, stream.synchronize --
     so a request costs 32 round-trips, and a consumer whose listener thread is starved
     (GIL, or a D2H queued behind decode kernels on the default stream) shows up here as
     ack time rather than transfer time. Timing `send_sync` and `send` separately splits
     the two without replicating either.

  2. `P2P_TRACE=1` logs every tensor id the producer sends and every id the
     consumer stores, which is what makes a key mismatch visible at all.
     One line per layer per request -- diagnostics only, never for a sweep.
"""

import os
import sys
import time
from importlib.machinery import PathFinder

_ENGINE = "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine"
_OFFLOAD_WORKER = "vllm.v1.kv_offload.worker.cpu_gpu"
_POOL = "vllm.distributed.kv_transfer.kv_connector.v1.p2p.tensor_memory_pool"
_INPUT_PROCESSOR = "vllm.v1.engine.input_processor"

# The router puts this in every id it mints; see disagg_p2p_proxy.py.
_ROUTED = "___decode_addr_"


def _patch_input_processor(mod):
    cls = mod.InputProcessor
    random_uuid = mod.random_uuid

    @staticmethod
    def assign_request_id(request):
        if request.external_req_id is not None:
            raise ValueError(
                "The external_req_id field should not be set on EngineCoreRequests"
                " passed to vLLM; use the request_id field."
            )
        request.external_req_id = request.request_id
        if _ROUTED in request.request_id:
            # Leave it byte-identical, or the peer instance cannot key on it.
            return
        request.request_id = f"{request.external_req_id}-{random_uuid():.8}"

    cls.assign_request_id = assign_request_id


def _patch_offload_worker(mod):
    """Times DRAM->GPU retrievals in the offloading tier (always on when the module
    loads). vLLM 0.15.1 exposes no offload latency metric, and the handler's transfers
    are async on pooled streams -- the honest retrieve time is submission to completion,
    including queueing behind earlier transfers. One log line per 8 completed loads:

        📦8 loads: avg 71.2 ms, avg 768.0 MiB, 10.8 GB/s effective

    GPU->CPU saves are deliberately not timed: they overlap compute and nothing waits
    on them; loads are what a request's prefill blocks on."""
    import time

    handler_cls = mod.SingleDirectionOffloadingHandler
    orig_transfer = handler_cls.transfer_async
    orig_finished = handler_cls.get_finished
    logger = mod.logger

    def transfer_async(self, job_id, transfer_spec):
        ok = orig_transfer(self, job_id, transfer_spec)
        if ok and not self.gpu_to_cpu:
            starts = getattr(self, "_load_starts", None)
            if starts is None:
                starts = self._load_starts = {}
            src_spec, _ = transfer_spec
            nbytes = int(src_spec.block_ids.size) * int(sum(self.block_size_in_bytes))
            starts[job_id] = (time.perf_counter(), nbytes)
        return ok

    def get_finished(self):
        results = orig_finished(self)
        if self.gpu_to_cpu or not results:
            return results
        starts = getattr(self, "_load_starts", None)
        if not starts:
            return results
        now = time.perf_counter()
        for job_id, _ok in results:
            entry = starts.pop(job_id, None)
            if entry is None:
                continue
            t0, nbytes = entry
            self._load_n = n = getattr(self, "_load_n", 0) + 1
            self._load_s = sec = getattr(self, "_load_s", 0.0) + (now - t0)
            self._load_b = tot = getattr(self, "_load_b", 0) + nbytes
            if n % 8 == 0:
                logger.info(
                    "📦%d loads: avg %.1f ms, avg %.1f MiB, %.1f GB/s effective",
                    n, sec / n * 1000, tot / n / 2**20, tot / sec / 1e9,
                )
        return results

    handler_cls.transfer_async = transfer_async
    handler_cls.get_finished = get_finished
    logger.info("🩹p2p_patch: offload DRAM->GPU load timing on")


def _patch_pool(mod):
    import torch

    pool_cls = mod.TensorMemoryPool
    orig_store = pool_cls.store_tensor

    def store_tensor(self, tensor):
        stream = getattr(self, "_spill_stream", None)
        if stream is None:
            stream = self._spill_stream = torch.cuda.Stream(tensor.device)
        with torch.cuda.stream(stream):
            return orig_store(self, tensor)

    pool_cls.store_tensor = store_tensor
    mod.logger.info("🩹p2p_patch: store_tensor spills on a dedicated stream")


def _patch_engine(mod):
    engine = mod.P2pNcclEngine
    logger = mod.logger
    timeout_s = float(os.environ.get("P2P_RECV_TIMEOUT_S", "600"))
    trace = os.environ.get("P2P_TRACE", "") == "1"

    orig_recv_tensor = engine.recv_tensor

    def recv_tensor(self, tensor_id, remote_address=None):
        # GET keeps upstream's path; it does not use the condition variable.
        if self.send_type not in ("PUT", "PUT_ASYNC"):
            return orig_recv_tensor(self, tensor_id, remote_address)

        request_id = tensor_id.split("#")[0]
        timed_out = getattr(self, "_timed_out_requests", None)
        if timed_out is None:
            timed_out = self._timed_out_requests = set()
        if request_id in timed_out:
            return None

        start_time = time.time()
        deadline = start_time + timeout_s
        with self.recv_store_cv:
            while tensor_id not in self.recv_store:
                remaining = deadline - time.time()
                if remaining <= 0:
                    timed_out.add(request_id)
                    logger.error(
                        "🔴RECV TIMEOUT after %.0fs, tensor_id:%s, from:%s, "
                        "rank:%d -- recv_store holds %d ids, e.g. %s",
                        timeout_s, tensor_id, remote_address, self.rank,
                        len(self.recv_store), sorted(self.recv_store)[:3],
                    )
                    return None
                self.recv_store_cv.wait(timeout=min(remaining, 5.0))
            tensor = self.recv_store[tensor_id]

        if tensor is not None:
            if isinstance(tensor, tuple):
                addr, dtype, shape = tensor
                tensor = self.pool.load_tensor(addr, dtype, shape, self.device)
            else:
                self.buffer_size -= tensor.element_size() * tensor.numel()
        else:
            logger.warning(
                "🔴[PUT]Recv From %s, tensor_id:%s, duration:%.3fms, rank:%d",
                remote_address, tensor_id,
                (time.time() - start_time) * 1000, self.rank,
            )
        return tensor

    engine.recv_tensor = recv_tensor

    cap_bytes = float(os.environ.get("P2P_SEND_QUEUE_CAP_GB", "4")) * 1024**3
    orig_send_tensor = engine.send_tensor
    orig_send_sync_bp = engine.send_sync

    def send_tensor(self, tensor_id, tensor, remote_address=None):
        if self.send_type == "PUT_ASYNC" and remote_address is not None:
            nbytes = tensor.element_size() * tensor.numel()
            with self.send_queue_cv:
                while getattr(self, "_queued_bytes", 0) + nbytes > cap_bytes and                         getattr(self, "_queued_bytes", 0) > 0:
                    self.send_queue_cv.wait(timeout=1.0)
                self._queued_bytes = getattr(self, "_queued_bytes", 0) + nbytes
        return orig_send_tensor(self, tensor_id, tensor, remote_address)

    def send_sync_bp(self, item):
        try:
            return orig_send_sync_bp(self, item)
        finally:
            if self.send_type == "PUT_ASYNC":
                nbytes = item.tensor.element_size() * item.tensor.numel()
                with self.send_queue_cv:
                    self._queued_bytes = max(
                        0, getattr(self, "_queued_bytes", 0) - nbytes)
                    self.send_queue_cv.notify_all()

    engine.send_tensor = send_tensor
    engine.send_sync = send_sync_bp

    if os.environ.get("P2P_TIMING", "") == "1":
        orig_send_sync = engine.send_sync
        orig_send = engine.send

        def send(self, comm, tensor, dst, stream=None):
            t0 = time.perf_counter()
            try:
                return orig_send(self, comm, tensor, dst, stream)
            finally:
                self._nccl_s = getattr(self, "_nccl_s", 0.0) + time.perf_counter() - t0

        def send_sync(self, item):
            t0 = time.perf_counter()
            before = getattr(self, "_nccl_s", 0.0)
            try:
                return orig_send_sync(self, item)
            finally:
                total = time.perf_counter() - t0
                self._tot_s = getattr(self, "_tot_s", 0.0) + total
                self._n_sends = n = getattr(self, "_n_sends", 0) + 1
                if n % 512 == 0:
                    nccl = self._nccl_s
                    logger.info(
                        "⏱️%d sends: %.2f ms/layer total = %.2f ms bytes + %.2f ms ack",
                        n, self._tot_s / n * 1000, nccl / n * 1000,
                        (self._tot_s - nccl) / n * 1000,
                    )
                _ = before

        engine.send = send
        engine.send_sync = send_sync

    if trace:
        # Both are one-liners called at exactly the point a tensor id is
        # committed on each side -- cheaper to wrap than the 90-line loop and
        # the 40-line send path they sit inside.
        orig_sent = engine.have_sent_tensor_id
        orig_received = engine.have_received_tensor_id

        def have_sent_tensor_id(self, tensor_id):
            logger.info("📤TRACE sent tensor_id:%s", tensor_id)
            return orig_sent(self, tensor_id)

        def have_received_tensor_id(self, tensor_id):
            logger.info("📥TRACE stored tensor_id:%s", tensor_id)
            return orig_received(self, tensor_id)

        engine.have_sent_tensor_id = have_sent_tensor_id
        engine.have_received_tensor_id = have_received_tensor_id

    logger.info(
        "🩹p2p_patch applied: request ids kept verbatim for routed requests, "
        "recv deadline %.0fs, trace %s",
        timeout_s, "on" if trace else "off",
    )


_PATCHES = {_ENGINE: _patch_engine, _POOL: _patch_pool,
            _OFFLOAD_WORKER: _patch_offload_worker,
            _INPUT_PROCESSOR: _patch_input_processor}


class _PatchOnImport:
    """Delegates to the normal finder, then patches the module once it loads."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _PATCHES:
            return None
        sys.meta_path.remove(self)
        try:
            spec = PathFinder.find_spec(fullname, path, target)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        orig_exec_module = spec.loader.exec_module

        patch = _PATCHES[fullname]

        def exec_module(module):
            orig_exec_module(module)
            patch(module)

        spec.loader.exec_module = exec_module
        return spec


if not any(isinstance(f, _PatchOnImport) for f in sys.meta_path):
    sys.meta_path.insert(0, _PatchOnImport())
