import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from cfar import (
    RUN_NAME,
    build_coordinate_grid,
    build_params,
    ca_noise_map,
    compute_chamfer_distance_2d,
    estimate_flops_per_frame,
    radar_energy_from_cube,
    ru_maxrss_mb,
    scene_frame_from_metadata,
)
from radelft.loaders.rad_cube_loader import RADCUBE_DATASET
from radelft.utils.compute_metrics import compute_pd_pfa


METHOD = {
    "name": "CACFAR_train5_guard5",
    "kind": "ca",
    "train": 5,
    "guard": 5,
    "k_fraction": np.nan,
}

_DATASET = None
_COORDINATE_GRID = None


def _limit_worker_threads():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass


def init_worker(dataset_path, train_val_scenes, test_scenes):
    global _DATASET, _COORDINATE_GRID
    _limit_worker_threads()
    params = build_params(dataset_path, train_val_scenes, test_scenes)
    _DATASET = RADCUBE_DATASET(mode="test", params=params)
    _COORDINATE_GRID = build_coordinate_grid(params)


def chunk_indices(indices, num_chunks):
    if num_chunks <= 1:
        return [indices]
    chunks = np.array_split(np.asarray(indices), num_chunks)
    return [chunk.astype(int).tolist() for chunk in chunks if len(chunk) > 0]


def process_indices(indices, alphas):
    x_grid, y_grid = _COORDINATE_GRID
    pd_sum = np.zeros(len(alphas), dtype=np.float64)
    pfa_sum = np.zeros(len(alphas), dtype=np.float64)
    chamfer_sum = np.zeros(len(alphas), dtype=np.float64)
    chamfer_count = np.zeros(len(alphas), dtype=np.int64)
    frame_count = 0
    num_training_cells = None
    estimated_flops = None

    for dataset_index in indices:
        input_cube, gt_cube, metadata = _DATASET[dataset_index]
        energy = radar_energy_from_cube(input_cube)
        gt_np = np.squeeze(gt_cube).astype(np.float32, copy=False)

        if energy.shape != gt_np.shape:
            scene, frame = scene_frame_from_metadata(metadata, dataset_index)
            raise ValueError(
                f"Prediction/GT shape mismatch at scene {scene} frame {frame}: "
                f"energy={energy.shape}, gt={gt_np.shape}"
            )

        noise, num_training_cells = ca_noise_map(energy, METHOD["train"], METHOD["guard"])
        estimated_flops = estimate_flops_per_frame(energy.shape, METHOD)

        gt_mask = gt_np > 0.5
        gt_points = np.column_stack((x_grid[gt_mask], y_grid[gt_mask]))
        ratio = energy / np.maximum(noise, 1e-12)

        for alpha_index, alpha in enumerate(alphas):
            pred_np = ratio > alpha
            pd_val, pfa_val = compute_pd_pfa(gt_np, pred_np.astype(np.float32, copy=False))
            pred_points = np.column_stack((x_grid[pred_np], y_grid[pred_np]))
            chamfer_val = compute_chamfer_distance_2d(gt_points, pred_points)

            pd_sum[alpha_index] += pd_val
            pfa_sum[alpha_index] += pfa_val
            if not np.isnan(chamfer_val):
                chamfer_sum[alpha_index] += chamfer_val
                chamfer_count[alpha_index] += 1

        frame_count += 1

    return {
        "pd_sum": pd_sum,
        "pfa_sum": pfa_sum,
        "chamfer_sum": chamfer_sum,
        "chamfer_count": chamfer_count,
        "frame_count": frame_count,
        "num_training_cells": num_training_cells,
        "estimated_flops": estimated_flops,
        "peak_memory_mb": ru_maxrss_mb(),
    }


def write_run_config(args, output_dir):
    config_path = Path(output_dir) / "cfar_alpha_sweep_config.txt"
    with open(config_path, "w") as f:
        for key, value in sorted(vars(args).items()):
            f.write(f"{key}: {value}\n")
    return config_path


def parse_args():
    parser = argparse.ArgumentParser(description="Alpha sweep for CACFAR train5 guard5")
    parser.add_argument("--dataset_path", type=str, default="/scratch/shujianjia/dataset/")
    parser.add_argument("--output_dir", type=str, default="./results/cfar_baseline")
    parser.add_argument("--csv_name", type=str, default="cacfar_t5g5_alpha_sweep_summary.csv")
    parser.add_argument("--train_val_scenes", type=int, nargs="+", default=[1, 3, 4, 5, 7])
    parser.add_argument("--test_scenes", type=int, nargs="+", default=[2, 6])
    parser.add_argument("--alpha_min", type=float, default=1.0)
    parser.add_argument("--alpha_max", type=float, default=3.5)
    parser.add_argument("--num_alphas", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=max(1, min(os.cpu_count() or 1, 4)))
    parser.add_argument("--log_every", type=int, default=250)
    parser.add_argument("--max_frames", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = write_run_config(args, output_dir)

    alphas = np.linspace(args.alpha_min, args.alpha_max, args.num_alphas, dtype=np.float64)
    _limit_worker_threads()
    params = build_params(args.dataset_path, args.train_val_scenes, args.test_scenes)
    dataset = RADCUBE_DATASET(mode="test", params=params)
    total_frames = len(dataset)
    frame_count = min(total_frames, args.max_frames) if args.max_frames else total_frames
    indices = list(range(frame_count))

    print("=" * 70, flush=True)
    print("启动 CACFAR alpha sweep", flush=True)
    print(f" Method: {METHOD['name']}", flush=True)
    print(f" Test scenes: {args.test_scenes}", flush=True)
    print(f" Frames: {frame_count}/{total_frames}", flush=True)
    print(f" Alpha range: {args.alpha_min} to {args.alpha_max} ({args.num_alphas} points)", flush=True)
    print(f" Workers: {args.num_workers}", flush=True)
    print(f" Output dir: {output_dir}", flush=True)
    print(f" Config: {config_path}", flush=True)
    print("=" * 70, flush=True)

    worker_count = max(1, min(args.num_workers, len(indices)))
    chunks = chunk_indices(indices, worker_count)
    partials = []
    completed_frames = 0

    if worker_count == 1:
        init_worker(args.dataset_path, args.train_val_scenes, args.test_scenes)
        for chunk in chunks:
            result = process_indices(chunk, alphas)
            partials.append(result)
            completed_frames += result["frame_count"]
            if args.log_every and completed_frames % args.log_every == 0:
                print(f"[alpha_sweep] processed {completed_frames}/{frame_count} frames", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=init_worker,
            initargs=(args.dataset_path, args.train_val_scenes, args.test_scenes),
        ) as executor:
            futures = [executor.submit(process_indices, chunk, alphas) for chunk in chunks]
            for future in as_completed(futures):
                result = future.result()
                partials.append(result)
                completed_frames += result["frame_count"]
                if args.log_every:
                    print(f"[alpha_sweep] processed {completed_frames}/{frame_count} frames", flush=True)

    pd_sum = sum(part["pd_sum"] for part in partials)
    pfa_sum = sum(part["pfa_sum"] for part in partials)
    chamfer_sum = sum(part["chamfer_sum"] for part in partials)
    chamfer_count = sum(part["chamfer_count"] for part in partials)
    processed_frames = sum(part["frame_count"] for part in partials)
    peak_memory_mb = max(part["peak_memory_mb"] for part in partials)
    num_training_cells = next(part["num_training_cells"] for part in partials if part["num_training_cells"] is not None)
    estimated_flops = next(part["estimated_flops"] for part in partials if part["estimated_flops"] is not None)

    summary = pd.DataFrame(
        {
            "Run": RUN_NAME,
            "Method": METHOD["name"],
            "ThresholdFactor": alphas,
            "Pd": pd_sum / processed_frames,
            "Pfa": pfa_sum / processed_frames,
            "Chamfer_Dist": np.divide(
                chamfer_sum,
                chamfer_count,
                out=np.full(len(alphas), np.nan, dtype=np.float64),
                where=chamfer_count > 0,
            ),
            "TrainCellsPerSide": METHOD["train"],
            "GuardCellsPerSide": METHOD["guard"],
            "NumTrainingCells": num_training_cells,
            "OSKFraction": METHOD["k_fraction"],
            "OSKIndex": np.nan,
            "Estimated_FLOPs_PerFrame": estimated_flops,
            "PeakMemoryMB": peak_memory_mb,
            "Frames": processed_frames,
        }
    )

    csv_path = output_dir / args.csv_name
    summary.to_csv(csv_path, index=False)

    print("\nCACFAR alpha sweep 完成", flush=True)
    print(f"Summary CSV: {csv_path}", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
