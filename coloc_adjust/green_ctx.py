"""CUDA green contexts from Python (ctypes on libcuda, no compiler): carve N SMs out of the device and
get a stream whose work runs only on those SMs. torch kernels launched under
`torch.cuda.stream(torch.cuda.ExternalStream(ptr))` go to that stream.

    g = GreenCtx(sm_count=32)          # on the current device (CUDA_VISIBLE_DEVICES-relative index 0)
    with torch.cuda.stream(g.torch_stream()): ...
    g.close()

Green contexts cannot be resized: to change the share, create another one and switch streams
between steps (synchronize first). Requires driver >= 12.4 (cuGreenCtxCreate).
"""
import ctypes, ctypes.util
import torch

_cuda = ctypes.CDLL("libcuda.so.1")
CU_DEV_RESOURCE_TYPE_SM = 1
CU_GREEN_CTX_DEFAULT_STREAM = 1
CU_STREAM_NON_BLOCKING = 1
_RES_SIZE = 144          # sizeof(CUdevResource): type(4) + padding(92) + union(48)

def _chk(r, what):
    if r != 0:
        s = ctypes.c_char_p(); _cuda.cuGetErrorString(r, ctypes.byref(s))
        raise RuntimeError(f"{what}: CUDA error {r} {s.value!r}")

class GreenCtx:
    def __init__(self, sm_count: int, device: int = 0, remaining: bool = False):
        """sm_count SMs; with remaining=True use the SMs *left over* after an sm_count split
        (so two processes can hold disjoint partitions: one takes the first sm_count, the other
        the remainder)."""
        torch.cuda.init(); torch.cuda.synchronize(device)
        dev = ctypes.c_int(); _chk(_cuda.cuDeviceGet(ctypes.byref(dev), device), "cuDeviceGet")
        self.dev = dev
        full = ctypes.create_string_buffer(_RES_SIZE)
        _chk(_cuda.cuDeviceGetDevResource(dev, full, CU_DEV_RESOURCE_TYPE_SM), "cuDeviceGetDevResource")
        self.device_sms = ctypes.c_uint.from_buffer_copy(full, 96).value   # union starts after type+padding
        n = ctypes.c_uint(1)
        result = ctypes.create_string_buffer(_RES_SIZE); remaining_buf = ctypes.create_string_buffer(_RES_SIZE)
        _chk(_cuda.cuDevSmResourceSplitByCount(result, ctypes.byref(n), full, remaining_buf, 0, ctypes.c_uint(sm_count)),
             "cuDevSmResourceSplitByCount")
        chosen = remaining_buf if remaining else result
        self.sm_count = ctypes.c_uint.from_buffer_copy(chosen, 96).value
        desc = ctypes.c_void_p()
        _chk(_cuda.cuDevResourceGenerateDesc(ctypes.byref(desc), chosen, 1), "cuDevResourceGenerateDesc")
        gctx = ctypes.c_void_p()
        _chk(_cuda.cuGreenCtxCreate(ctypes.byref(gctx), desc, dev, CU_GREEN_CTX_DEFAULT_STREAM), "cuGreenCtxCreate")
        self.gctx = gctx
        st = ctypes.c_void_p()
        _chk(_cuda.cuGreenCtxStreamCreate(ctypes.byref(st), gctx, CU_STREAM_NON_BLOCKING, 0), "cuGreenCtxStreamCreate")
        self.stream_ptr = st.value
        self._torch_stream = torch.cuda.ExternalStream(self.stream_ptr, device=device)

    def torch_stream(self):
        return self._torch_stream

    def close(self):
        if self.gctx:
            torch.cuda.synchronize()
            _cuda.cuStreamDestroy_v2(ctypes.c_void_p(self.stream_ptr))
            _cuda.cuGreenCtxDestroy(self.gctx); self.gctx = None

if __name__ == "__main__":
    import time, sys
    x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16); y = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    def bench(stream, label):
        with torch.cuda.stream(stream):
            for _ in range(3): x @ y
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(20): x @ y
            torch.cuda.synchronize(); dt = (time.time() - t0) / 20
        print(f"{label}: {dt*1e3:.2f} ms per 8k matmul, {2*8192**3/dt/1e12:.0f} TFLOP/s")
    bench(torch.cuda.current_stream(), "full device")
    for n in (int(a) for a in sys.argv[1:] or ["16", "32", "64", "128"]):
        g = GreenCtx(n); bench(g.torch_stream(), f"green ctx {g.sm_count} of {g.device_sms} SMs"); g.close()
