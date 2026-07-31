#!/usr/bin/env python
"""Compare single-GPU vs split-GPU stress using LiveTalking model components."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether splitting LiveTalking heavy model load across 2 GPUs reduces stress."
    )
    parser.add_argument("--model", choices=["musetalk", "wav2lip"], default="musetalk")
    parser.add_argument("--duration", type=int, default=20, help="Seconds per scenario.")
    parser.add_argument("--interval", type=float, default=1.0, help="nvidia-smi sample interval.")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size in workers.")
    parser.add_argument("--modelres", type=int, default=256, help="Model resolution for wav2lip.")
    parser.add_argument(
        "--wav2lip-ckpt",
        type=str,
        default="./models/wav2lip.pth",
        help="Checkpoint path used when --model=wav2lip.",
    )
    parser.add_argument(
        "--mode",
        choices=["both", "single_gpu", "split_gpu"],
        default="both",
        help="Run both scenarios (recommended) or one specific scenario.",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=5,
        help="Seconds to wait between scenarios.",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("logs"),
        help="Folder to save scenario sampling CSV files.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--result-json", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ready-file", type=Path, default=None, help=argparse.SUPPRESS)
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


def run_worker(args: argparse.Namespace) -> int:
    repo_root = str(Path(__file__).resolve().parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    utils_pkg_path = str(Path(repo_root) / "utils")
    forced_utils_pkg = types.ModuleType("utils")
    forced_utils_pkg.__path__ = [utils_pkg_path]
    sys.modules["utils"] = forced_utils_pkg

    import torch

    if not torch.cuda.is_available():
        print("Worker error: CUDA is not available.")
        return 1

    worker_name = os.environ.get("LT_WORKER_NAME", "worker")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    torch.set_grad_enabled(False)

    if args.model == "musetalk":
        import avatars.musetalk_avatar as mod

        vae, unet, pe, timesteps, _ = mod.load_model()
        device = unet.device
        dtype = unet.model.dtype

        latent_batch = torch.ones((args.batch_size, 8, 32, 32), device=device, dtype=dtype)
        whisper_batch = torch.ones((args.batch_size, 50, 384), device=device, dtype=dtype)
        audio_feature_batch = pe(whisper_batch)

        def one_step() -> None:
            pred_latents = unet.model(
                latent_batch,
                timesteps,
                encoder_hidden_states=audio_feature_batch,
            ).sample
            _ = vae.decode_latents(pred_latents)

    else:
        import avatars.wav2lip_avatar as mod

        model = mod.load_model(args.wav2lip_ckpt)
        device = next(model.parameters()).device
        img_batch = torch.ones(
            (args.batch_size, 6, args.modelres, args.modelres), device=device, dtype=torch.float32
        )
        mel_batch = torch.ones((args.batch_size, 1, 80, 16), device=device, dtype=torch.float32)

        def one_step() -> None:
            _ = model(mel_batch, img_batch)

    if args.ready_file:
        args.ready_file.write_text("ready", encoding="utf-8")

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    deadline = t0 + args.duration
    iterations = 0

    while time.perf_counter() < deadline:
        one_step()
        if iterations % 3 == 0:
            torch.cuda.synchronize(device)
        iterations += 1

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    result = {
        "worker": worker_name,
        "model": args.model,
        "visible_devices": visible,
        "device_name": torch.cuda.get_device_name(0),
        "iterations": iterations,
        "elapsed_sec": elapsed,
        "iter_per_sec": iterations / max(elapsed, 1e-6),
    }
    if args.result_json:
        args.result_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"[{worker_name}] done: visible={visible}, iterations={iterations}, "
        f"iter_per_sec={result['iter_per_sec']:.2f}"
    )
    return 0


def launch_worker(
    scenario: str,
    worker_idx: int,
    physical_gpu: int,
    args: argparse.Namespace,
    ready_file: Path,
    result_json: Path,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--model",
        args.model,
        "--duration",
        str(args.duration),
        "--batch-size",
        str(args.batch_size),
        "--modelres",
        str(args.modelres),
        "--wav2lip-ckpt",
        args.wav2lip_ckpt,
        "--ready-file",
        str(ready_file),
        "--result-json",
        str(result_json),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    env["LT_WORKER_NAME"] = f"{scenario}_w{worker_idx}"
    repo_root = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(cmd, env=env)


def collect_samples(
    processes: List[subprocess.Popen],
    gpu_ids: List[int],
    duration: int,
    interval: float,
) -> List[Dict[str, float]]:
    samples: List[Dict[str, float]] = []
    start = time.perf_counter()
    deadline = start + duration
    while time.perf_counter() < deadline:
        loop_t = time.perf_counter()
        stats = query_nvidia_smi()
        row: Dict[str, float] = {"t_sec": loop_t - start}
        parts = [f"t+{row['t_sec']:5.1f}s"]
        for gpu_id in gpu_ids:
            s = stats.get(gpu_id, {})
            util = s.get("gpu_util", 0.0)
            mem = s.get("mem_used_mib", 0.0)
            pwr = s.get("power_w", 0.0)
            row[f"gpu{gpu_id}_util"] = util
            row[f"gpu{gpu_id}_mem_used_mib"] = mem
            row[f"gpu{gpu_id}_power_w"] = pwr
            parts.append(f"GPU{gpu_id}: util={util:5.1f}% mem={mem:7.0f}MiB pwr={pwr:6.1f}W")
        print(" | ".join(parts))
        samples.append(row)
        if all(p.poll() is not None for p in processes):
            break
        sleep_for = max(0.0, interval - (time.perf_counter() - loop_t))
        time.sleep(sleep_for)
    for p in processes:
        p.wait(timeout=60)
    return samples


def summarize_samples(samples: List[Dict[str, float]], gpu_ids: List[int]) -> Dict[int, Dict[str, float]]:
    summary: Dict[int, Dict[str, float]] = {}
    for gpu_id in gpu_ids:
        util_vals = [r.get(f"gpu{gpu_id}_util", 0.0) for r in samples]
        pwr_vals = [r.get(f"gpu{gpu_id}_power_w", 0.0) for r in samples]
        mem_vals = [r.get(f"gpu{gpu_id}_mem_used_mib", 0.0) for r in samples]
        summary[gpu_id] = {
            "avg_util": mean(util_vals) if util_vals else 0.0,
            "max_util": max(util_vals) if util_vals else 0.0,
            "avg_power_w": mean(pwr_vals) if pwr_vals else 0.0,
            "avg_mem_mib": mean(mem_vals) if mem_vals else 0.0,
        }
    return summary


def save_samples_csv(samples: List[Dict[str, float]], csv_path: Path, gpu_ids: List[int]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["t_sec"]
    for gpu_id in gpu_ids:
        fields.extend([f"gpu{gpu_id}_util", f"gpu{gpu_id}_mem_used_mib", f"gpu{gpu_id}_power_w"])
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(samples)


def run_scenario(
    scenario_name: str,
    assignments: Tuple[int, int],
    args: argparse.Namespace,
    temp_dir: Path,
    gpu_ids: List[int],
) -> Dict[str, object]:
    print(f"\n=== Scenario: {scenario_name} (worker->physical_gpu: {assignments}) ===")
    result_paths = [temp_dir / f"{scenario_name}_worker{i}.json" for i in range(2)]
    ready_paths = [temp_dir / f"{scenario_name}_worker{i}.ready" for i in range(2)]
    procs = [
        launch_worker(scenario_name, i, assignments[i], args, ready_paths[i], result_paths[i])
        for i in range(2)
    ]
    ready_deadline = time.perf_counter() + 240
    while time.perf_counter() < ready_deadline:
        if all(p.exists() for p in ready_paths):
            break
        if any(p.poll() is not None for p in procs):
            break
        time.sleep(0.2)
    if all(p.exists() for p in ready_paths):
        print("Workers are ready. Start measuring compute window.")
    else:
        print("Warning: workers not fully ready before measurement window.")
    samples = collect_samples(procs, gpu_ids=gpu_ids, duration=args.duration, interval=args.interval)
    worker_results = []
    for i, rp in enumerate(result_paths):
        if rp.exists():
            worker_results.append(json.loads(rp.read_text(encoding="utf-8")))
        else:
            worker_results.append({"worker": f"{scenario_name}_w{i}", "error": "missing_result"})
    sample_summary = summarize_samples(samples, gpu_ids)
    csv_path = args.csv_dir / f"gpu_dist_{scenario_name}_{args.model}.csv"
    save_samples_csv(samples, csv_path, gpu_ids)
    total_iter_per_sec = sum(float(w.get("iter_per_sec", 0.0)) for w in worker_results)
    return {
        "name": scenario_name,
        "assignments": assignments,
        "samples": samples,
        "summary": sample_summary,
        "workers": worker_results,
        "total_iter_per_sec": total_iter_per_sec,
        "csv_path": str(csv_path),
    }


def print_scenario_result(data: Dict[str, object], gpu_ids: List[int]) -> None:
    print(f"\n--- {data['name']} result ---")
    summary = data["summary"]
    for gpu_id in gpu_ids:
        row = summary[gpu_id]
        print(
            f"GPU{gpu_id}: avg_util={row['avg_util']:.1f}% max_util={row['max_util']:.1f}% "
            f"avg_power={row['avg_power_w']:.1f}W avg_mem={row['avg_mem_mib']:.0f}MiB"
        )
    for w in data["workers"]:
        if "error" in w:
            print(f"{w['worker']}: error={w['error']}")
        else:
            print(
                f"{w['worker']}: visible={w['visible_devices']} "
                f"iter_per_sec={w['iter_per_sec']:.2f}"
            )
    print(f"Total throughput: {data['total_iter_per_sec']:.2f} iter/s")
    print(f"CSV: {data['csv_path']}")


def main() -> int:
    args = parse_args()
    if args.worker:
        return run_worker(args)

    try:
        stats = query_nvidia_smi()
    except Exception as exc:
        print(f"Cannot query nvidia-smi: {exc}")
        return 1
    gpu_ids = sorted(stats.keys())
    if len(gpu_ids) < 2:
        print(f"Need at least 2 GPUs. Detected: {gpu_ids}")
        return 1

    selected_gpu_ids = gpu_ids[:2]
    print(f"Detected GPUs: {selected_gpu_ids}")
    print(
        f"Model={args.model}, duration={args.duration}s/scenario, "
        f"batch_size={args.batch_size}, mode={args.mode}"
    )

    scenario_defs: List[Tuple[str, Tuple[int, int]]] = []
    if args.mode in ("both", "single_gpu"):
        scenario_defs.append(("single_gpu", (selected_gpu_ids[0], selected_gpu_ids[0])))
    if args.mode in ("both", "split_gpu"):
        scenario_defs.append(("split_gpu", (selected_gpu_ids[0], selected_gpu_ids[1])))

    with tempfile.TemporaryDirectory(prefix="livetalking_gpu_dist_") as td:
        temp_dir = Path(td)
        results: List[Dict[str, object]] = []
        for i, (name, assign) in enumerate(scenario_defs):
            result = run_scenario(name, assign, args, temp_dir, selected_gpu_ids)
            results.append(result)
            print_scenario_result(result, selected_gpu_ids)
            if i < len(scenario_defs) - 1 and args.cooldown > 0:
                print(f"\nCooldown {args.cooldown}s before next scenario...")
                time.sleep(args.cooldown)

    if len(results) == 2:
        base = next(r for r in results if r["name"] == "single_gpu")
        split = next(r for r in results if r["name"] == "split_gpu")
        base_s = base["summary"]
        split_s = split["summary"]
        gpu0 = selected_gpu_ids[0]
        gpu1 = selected_gpu_ids[1]
        gpu0_relief = base_s[gpu0]["avg_util"] - split_s[gpu0]["avg_util"]
        split_gpu1_gain = split_s[gpu1]["avg_util"] - base_s[gpu1]["avg_util"]
        throughput_gain = (
            (split["total_iter_per_sec"] - base["total_iter_per_sec"])
            / max(base["total_iter_per_sec"], 1e-6)
            * 100.0
        )
        print("\n=== Comparison summary ===")
        print(f"GPU{gpu0} relief (avg util): {gpu0_relief:+.1f} percentage points")
        print(f"GPU{gpu1} usage increase (avg util): {split_gpu1_gain:+.1f} percentage points")
        print(f"Total throughput change: {throughput_gain:+.1f}%")
        if gpu0_relief > 10 and split_gpu1_gain > 20:
            print("Conclusion: Splitting workload across 2 GPUs clearly reduces single-GPU stress.")
        else:
            print("Conclusion: Split load effect is limited with current test settings.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
