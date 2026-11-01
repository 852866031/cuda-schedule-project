#!/usr/bin/env python3
import argparse
import multiprocessing as mp
import time

import cupy as cp
import torch


KERNEL_CODE = r'''
extern "C" __global__
void dense_fma_kernel(const float* x,
                      float* y,
                      int n,
                      int inner_iters) {
    extern __shared__ float smem[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    if (threadIdx.x < 32) {
        smem[threadIdx.x] = (float)threadIdx.x;
    }
    __syncthreads();

    for (int i = tid; i < n; i += stride) {
        float xi = x[i];
        float yi = y[i];
        float acc = smem[threadIdx.x & 31];

        #pragma unroll 1
        for (int j = 0; j < inner_iters; ++j) {
            yi = 1.001f * xi + 0.999f * yi + 1e-6f * acc;
            xi = 0.999f * yi + 1.001f * xi + 1e-6f * acc;
        }

        y[i] = yi;
    }
}

extern "C" __global__
void bad_gather_kernel(const float* x,
                       const int* idx,
                       float* out,
                       int n,
                       int inner_iters) {
    extern __shared__ float smem[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    if (threadIdx.x < 32) {
        smem[threadIdx.x] = (float)threadIdx.x;
    }
    __syncthreads();

    for (int i = tid; i < n; i += stride) {
        int cur = idx[i];
        float acc = smem[threadIdx.x & 31];

        #pragma unroll 1
        for (int j = 0; j < inner_iters; ++j) {
            cur = idx[cur & (n - 1)];
            acc += x[cur];
        }
        out[i] = acc;
    }
}
'''


def build_parser():
    p = argparse.ArgumentParser()

    # Global
    p.add_argument("--duration-s", type=float, default=1.0)
    p.add_argument("--n", type=int, default=1 << 22)
    p.add_argument("--cpu-sleep-ms", type=float, default=20.0)

    # GPU 0: dense custom kernel
    p.add_argument("--g0-threads", type=int, default=256)
    p.add_argument("--g0-blocks-mult", type=float, default=4.0)
    p.add_argument("--g0-inner-iters", type=int, default=128)
    p.add_argument("--g0-launches-per-iter", type=int, default=64)
    p.add_argument("--g0-shmem-kb", type=int, default=1)

    # GPU 1: fragmented bad gather
    p.add_argument("--g1-threads", type=int, default=128)
    p.add_argument("--g1-blocks-mult", type=float, default=4.0)
    p.add_argument("--g1-inner-iters", type=int, default=16)
    p.add_argument("--g1-launches-per-iter", type=int, default=6)
    p.add_argument("--g1-pointwise-repeats", type=int, default=1)
    p.add_argument("--g1-shmem-kb", type=int, default=1)

    return p

def gpu0_dense_worker(args):
    torch.cuda.set_device(0)
    device = "cuda:0"

    with cp.cuda.Device(0):
        dense_fma_kernel = cp.RawKernel(KERNEL_CODE, "dense_fma_kernel")

    props = torch.cuda.get_device_properties(0)
    blocks = max(1, int(props.multi_processor_count * args.g0_blocks_mult))
    threads = args.g0_threads
    shmem_bytes = args.g0_shmem_kb * 1024

    n = args.n
    duration_s = args.duration_s
    cpu_sleep_s = args.cpu_sleep_ms / 1000.0

    with torch.cuda.nvtx.range("GPU0 setup"):
        x_t = torch.randn(n, device=device, dtype=torch.float32)
        y_t = torch.randn(n, device=device, dtype=torch.float32)

        x_cp = cp.asarray(x_t)
        y_cp = cp.asarray(y_t)

    it = 0
    start = time.perf_counter()

    while time.perf_counter() - start < duration_s:
        with torch.cuda.nvtx.range(f"GPU0 iteration {it}"):
            with torch.cuda.nvtx.range("GPU0 CPU preprocess"):
                time.sleep(cpu_sleep_s)

            with torch.cuda.nvtx.range("GPU0 dense custom kernel"):
                for _ in range(args.g0_launches_per_iter):
                    dense_fma_kernel(
                        (blocks,),
                        (threads,),
                        (x_cp, y_cp, n, args.g0_inner_iters),
                        shared_mem=shmem_bytes,
                    )

            torch.cuda.synchronize()
        it += 1

    print(
        f"GPU0 done. iters={it}, blocks={blocks}, threads={threads}, "
        f"inner_iters={args.g0_inner_iters}, launches={args.g0_launches_per_iter}, "
        f"shmem_kb={args.g0_shmem_kb}"
    )


def gpu1_fragmented_worker(args):
    torch.cuda.set_device(1)
    device = "cuda:1"

    with cp.cuda.Device(1):
        bad_gather_kernel = cp.RawKernel(KERNEL_CODE, "bad_gather_kernel")

    props = torch.cuda.get_device_properties(1)
    blocks = max(1, int(props.multi_processor_count * args.g1_blocks_mult))
    threads = args.g1_threads
    shmem_bytes = args.g1_shmem_kb * 1024

    n = args.n
    duration_s = args.duration_s
    cpu_sleep_s = args.cpu_sleep_ms / 1000.0

    with torch.cuda.nvtx.range("GPU1 setup"):
        x_t = torch.randn(n, device=device, dtype=torch.float32)
        idx_t = torch.randint(0, n, (n,), device=device, dtype=torch.int32)
        out_t = torch.empty_like(x_t)

        x_cp = cp.asarray(x_t)
        idx_cp = cp.asarray(idx_t)
        out_cp = cp.asarray(out_t)

    it = 0
    start = time.perf_counter()

    while time.perf_counter() - start < duration_s:
        with torch.cuda.nvtx.range(f"GPU1 iteration {it}"):
            with torch.cuda.nvtx.range("GPU1 CPU preprocess"):
                time.sleep(cpu_sleep_s)

            with torch.cuda.nvtx.range("GPU1 fragmented bad gather"):
                for _ in range(args.g1_launches_per_iter):
                    bad_gather_kernel(
                        (blocks,),
                        (threads,),
                        (x_cp, idx_cp, out_cp, n, args.g1_inner_iters),
                        shared_mem=shmem_bytes,
                    )

            with torch.cuda.nvtx.range("GPU1 small pointwise ops"):
                for _ in range(args.g1_pointwise_repeats):
                    out_t = torch.sin(out_t) + 0.01 * x_t

            torch.cuda.synchronize()
        it += 1

    print(
        f"GPU1 done. iters={it}, blocks={blocks}, threads={threads}, "
        f"inner_iters={args.g1_inner_iters}, launches={args.g1_launches_per_iter}, "
        f"pointwise={args.g1_pointwise_repeats}, shmem_kb={args.g1_shmem_kb}"
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    args = build_parser().parse_args()
    mp.set_start_method("spawn", force=True)

    p0 = mp.Process(target=gpu0_dense_worker, args=(args,))
    p1 = mp.Process(target=gpu1_fragmented_worker, args=(args,))

    p0.start()
    p1.start()

    p0.join()
    p1.join()

    if p0.exitcode != 0:
        raise RuntimeError(f"GPU0 worker failed with exit code {p0.exitcode}")
    if p1.exitcode != 0:
        raise RuntimeError(f"GPU1 worker failed with exit code {p1.exitcode}")

    print("All done.")


if __name__ == "__main__":
    main()