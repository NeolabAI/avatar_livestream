#!/usr/bin/env python
"""Create synthetic compute load on two GPUs and report utilization balance."""

from __future__ import annotations

import argparse
import csv
import subprocess
import threading
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run synthetic matrix-multiply workload on 2 GPUs and monitor balance."
    )
    parser.add_argument("--duration", type=int, default=30, help="Workload time in seconds.")
    parser.add_argument(
        "--matrix-size",
        type=int,
        default=8192,
        help="Square matrix size for each matmul (higher = heavier load).",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Tensor dtype used in the workload.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between nvidia-smi samples.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional path to save sampled GPU metrics as CSV.",
    )
    return parser.parse_args()


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def query_nvidia_smi() -> Dict[int, Dict[str, float]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(cmd, text=True).strip().splitlines()
    result: Dict[int, Dict[str, float]] = {}
    for line in output:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        idx = int(parts[0])
        result[idx] = {
            "gpu_util": _to_float(parts[1]),
            "mem_util": _to_float(parts[2]),
            "mem_used_mib": _to_float(parts[3]),
            "power_w": _to_float(parts[4]),
        }
    return result


def gpu_worker(
    gpu_idx: int,
    end_time: float,
    matrix_size: int,
    dtype_name: str,
    result_map: Dict[int, Dict[str, float]],
    result_lock: threading.Lock,
) -> None:
    import torch

    device = torch.device(f"cuda:{gpu_idx}")
    dtype = torch.float16 if dtype_name == "float16" else torch.float32

    torch.cuda.set_device(device)
    a = torch.randn((matrix_size, matrix_size), device=device, dtype=dtype)
    b = torch.randn((matrix_size, matrix_size), device=device, dtype=dtype)

    iterations = 0
    start = time.perf_counter()

    while time.perf_counter() < end_time:
        c = torch.matmul(a, b)
        a = torch.relu(c)
        b = b.mul_(0.999).add_(0.001)
        if iterations % 4 == 0:
            torch.cuda.synchronize(device)
        iterations += 1

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    with result_lock:
        result_map[gpu_idx] = {
            "gpu_idx": float(gpu_idx),
            "iterations": float(iterations),
            "elapsed_sec": elapsed,
            "iter_per_sec": iterations / max(elapsed, 1e-6),
        }


def build_conclusion(avg_util: Dict[int, float], gpu_ids: List[int]) -> str:
    u0 = avg_util.get(gpu_ids[0], 0.0)
    u1 = avg_util.get(gpu_ids[1], 0.0)
    if u0 >= 80 and u1 >= 80:
        return "Ca 2 GPU dang duoc tan dung cao (gan toi da)."
    if max(u0, u1) >= 80 and min(u0, u1) < 35:
        return "Tai dang lech ro, mot GPU ganh phan lon workload."
    if u0 >= 50 and u1 >= 50:
        return "Ca 2 GPU deu duoc su dung, nhung chua toi da."
    return "Muc tai GPU thap/khong on dinh, can tang matrix-size hoac duration de test ro hon."


def main() -> int:
    args = parse_args()

    import torch

    if not torch.cuda.is_available():
        print("Khong tim thay CUDA GPU.")
        return 1

    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        print(f"Chi phat hien {gpu_count} GPU. Bai test nay can it nhat 2 GPU.")
        return 1

    gpu_ids = [0, 1]
    print(f"Su dung GPU: {gpu_ids[0]} ({torch.cuda.get_device_name(gpu_ids[0])})")
    print(f"Su dung GPU: {gpu_ids[1]} ({torch.cuda.get_device_name(gpu_ids[1])})")
    print(
        f"Khoi tao workload: duration={args.duration}s, matrix_size={args.matrix_size}, dtype={args.dtype}"
    )

    start = time.perf_counter()
    deadline = start + args.duration
    worker_results: Dict[int, Dict[str, float]] = {}
    result_lock = threading.Lock()

    workers = [
        threading.Thread(
            target=gpu_worker,
            args=(gpu_id, deadline, args.matrix_size, args.dtype, worker_results, result_lock),
            daemon=True,
        )
        for gpu_id in gpu_ids
    ]

    for worker in workers:
        worker.start()

    samples: List[Dict[str, float]] = []

    print("\nMonitoring GPU utilization...")
    while time.perf_counter() < deadline:
        loop_start = time.perf_counter()
        stats = query_nvidia_smi()
        t = loop_start - start
        row = {"t_sec": t}
        parts = [f"t+{t:5.1f}s"]
        for gpu_id in gpu_ids:
            gpu_stats = stats.get(gpu_id, {})
            util = gpu_stats.get("gpu_util", 0.0)
            mem = gpu_stats.get("mem_used_mib", 0.0)
            pwr = gpu_stats.get("power_w", 0.0)
            row[f"gpu{gpu_id}_util"] = util
            row[f"gpu{gpu_id}_mem_used_mib"] = mem
            row[f"gpu{gpu_id}_power_w"] = pwr
            parts.append(f"GPU{gpu_id}: util={util:5.1f}% mem={mem:7.0f}MiB pwr={pwr:6.1f}W")
        samples.append(row)
        print(" | ".join(parts))
        sleep_for = max(0.0, args.interval - (time.perf_counter() - loop_start))
        time.sleep(sleep_for)

    for worker in workers:
        worker.join()

    worker_rows = [worker_results[gpu_id] for gpu_id in gpu_ids if gpu_id in worker_results]
    worker_rows.sort(key=lambda x: x["gpu_idx"])

    avg_util = {}
    for gpu_id in gpu_ids:
        util_values = [s.get(f"gpu{gpu_id}_util", 0.0) for s in samples]
        avg_util[gpu_id] = mean(util_values) if util_values else 0.0

    print("\n=== Tong ket ===")
    for gpu_id in gpu_ids:
        print(f"GPU{gpu_id} avg util: {avg_util[gpu_id]:.1f}%")
    for res in worker_rows:
        print(
            f"GPU{int(res['gpu_idx'])} worker: {int(res['iterations'])} iterations "
            f"({res['iter_per_sec']:.2f} iter/s)"
        )
    print(f"Ket luan: {build_conclusion(avg_util, gpu_ids)}")

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "t_sec",
            "gpu0_util",
            "gpu0_mem_used_mib",
            "gpu0_power_w",
            "gpu1_util",
            "gpu1_mem_used_mib",
            "gpu1_power_w",
        ]
        with args.csv_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(samples)
        print(f"Da luu log: {args.csv_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
