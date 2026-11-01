#!/usr/bin/env python3
import os
import time
import math
import multiprocessing as mp


def set_affinity(cpu_id: int) -> None:
    """
    Pin current process to one logical CPU.
    Linux only.
    """
    try:
        os.sched_setaffinity(0, {cpu_id})
    except AttributeError:
        raise RuntimeError("os.sched_setaffinity is not available on this system.")
    except PermissionError as e:
        raise RuntimeError(f"Failed to set CPU affinity for CPU {cpu_id}: {e}")


def burn_cpu_for_interval(busy_time: float) -> None:
    """
    Busy-loop for approximately busy_time seconds.
    Use some arithmetic so the loop is not optimized away in practice.
    """
    end = time.perf_counter() + busy_time
    x = 0.0
    while time.perf_counter() < end:
        x += math.sqrt(12345.6789)
        if x > 1e12:
            x = 0.0


def worker(cpu_id: int, duty_cycle: float, duration: float, period: float = 0.1) -> None:
    """
    duty_cycle: fraction of each period spent busy, e.g. 1.0 for ~100%, 0.5 for ~50%
    duration: total runtime in seconds
    period: control period in seconds
    """
    set_affinity(cpu_id)

    start = time.perf_counter()
    busy_time = period * duty_cycle
    idle_time = period - busy_time

    while (time.perf_counter() - start) < duration:
        if busy_time > 0:
            burn_cpu_for_interval(busy_time)
        if idle_time > 0:
            time.sleep(idle_time)


def main() -> None:
    duration = 10.0

    # Adjust these if you want different logical CPU IDs.
    full_load_cpus = list(range(0, 8))     # CPUs 0-7 -> ~100%
    medium_load_cpus = list(range(8, 16))  # CPUs 8-15 -> ~50%

    procs = []

    for cpu in full_load_cpus:
        p = mp.Process(target=worker, args=(cpu, 1.0, duration))
        p.start()
        procs.append(p)

    for cpu in medium_load_cpus:
        p = mp.Process(target=worker, args=(cpu, 0.5, duration))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("Done.")


if __name__ == "__main__":
    main()