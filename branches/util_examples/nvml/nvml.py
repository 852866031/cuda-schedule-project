import time
import threading
import torch
import torch.nn as nn
from pynvml import *

gpu_utils = []
mem_utils = []
stop_flag = False

def monitor(gpu_index=0, interval=0.2):
    handle = nvmlDeviceGetHandleByIndex(gpu_index)
    while not stop_flag:
        util = nvmlDeviceGetUtilizationRates(handle)
        gpu_utils.append(util.gpu)
        mem_utils.append(util.memory)
        time.sleep(interval)

def main():
    global stop_flag
    nvmlInit()
    device = "cuda:0"
    model = nn.Sequential(
        nn.Linear(4096, 8192),
        nn.ReLU(),
        nn.Linear(8192, 4096),
    ).to(device).half().eval()
    x = torch.randn(2048, 4096, device=device, dtype=torch.float16)
    t = threading.Thread(target=monitor, args=(0, 0.2), daemon=True)
    t.start()
    start = time.perf_counter()
    with torch.no_grad():
        while time.perf_counter() - start < 10:
            y = model(x)
    torch.cuda.synchronize()
    stop_flag = True
    t.join()
    avg_gpu = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0
    avg_mem = sum(mem_utils) / len(mem_utils) if mem_utils else 0
    print(f"Average GPU util: {avg_gpu:.2f}%")
    print(f"Average memory util: {avg_mem:.2f}%")
    nvmlShutdown()

if __name__ == "__main__":
    main()