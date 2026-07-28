import argparse
import math
import os
import resource
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


current_dir = Path(__file__).resolve().parent
radelft_dir = current_dir / "radelft"
sys.path.insert(0, str(radelft_dir))
sys.path.insert(0, str(current_dir))

from radelft.data_preparation import data_preparation
from radelft.loaders.rad_cube_loader import RADCUBE_DATASET
from radelft.utils.compute_metrics import compute_pd_pfa


try:
    from numpy.lib.stride_tricks import sliding_window_view
except ImportError as exc:
    raise ImportError("cfar.py requires numpy.lib.stride_tricks.sliding_window_view") from exc


RUN_NAME = "cfar_baseline"
DEFAULT_THRESHOLDS = (1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5)
METHODS = (
    {
        "name": "OSCFAR_train5_guard5_k0.75",
        "kind": "os",
        "train": 5,
        "guard": 5,
        "k_fraction": 0.75,
    },
    {
        "name": "OSCFAR_train5_guard0_k0.75",
        "kind": "os",
        "train": 5,
        "guard": 0,
        "k_fraction": 0.75,
    },
    {
        "name": "OSCFAR_train9_guard0_k0.75",
        "kind": "os",
        "train": 9,
        "guard": 0,
        "k_fraction": 0.75,
    },
    {
        "name": "CACFAR_train5_guard0",
        "kind": "ca",
        "train": 5,
        "guard": 0,
        "k_fraction": np.nan,
    },
    {
        "name": "CACFAR_train5_guard5",
        "kind": "ca",
        "train": 5,
        "guard": 5,
        "k_fraction": np.nan,
    },
    {
        "name": "CACFAR_train9_guard0",
        "kind": "ca",
        "train": 9,
        "guard": 0,
        "k_fraction": np.nan,
    },
)

_DATASET = None
_PARAMS = None
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


def build_params(dataset_path, train_val_scenes, test_scenes):
    params = data_preparation.get_default_params()
    params["dataset_path"] = dataset_path
    params["train_val_scenes"] = train_val_scenes
    params["test_scenes"] = test_scenes
    params["bev"] = True
    return params


def init_worker(dataset_path, train_val_scenes, test_scenes):
    global _DATASET, _PARAMS, _COORDINATE_GRID
    _limit_worker_threads()
    _PARAMS = build_params(dataset_path, train_val_scenes, test_scenes)
    _DATASET = RADCUBE_DATASET(mode="test", params=_PARAMS)
    _COORDINATE_GRID = build_coordinate_grid(_PARAMS)


def build_coordinate_grid(params):
    range_axis = np.asarray(params["range_axis"])
    azimuth_axis = np.asarray(params["azimuth_axis"])
    theta, radius = np.meshgrid(azimuth_axis, range_axis)
    x = radius * np.sin(theta)
    y = radius * np.cos(theta)
    return x, y


def ru_maxrss_mb():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return value / (1024 ** 2)
    return value / 1024.0


def as_range_doppler_azimuth(input_cube):
    if input_cube.ndim != 3:
        raise ValueError(f"Expected a 3D radar cube, got shape {input_cube.shape}")
    if input_cube.shape[0] == 128:
        return np.transpose(input_cube, (1, 0, 2))
    return input_cube


def radar_energy_from_cube(input_cube):
    radar_cube = as_range_doppler_azimuth(input_cube)
    energy = np.max(radar_cube, axis=1)
    return energy[:-12, 8:-8].astype(np.float32, copy=False)


def box_sum_centered(image, radius):
    if radius == 0:
        return image.astype(np.float32, copy=False)

    padded = np.pad(image, radius, mode="edge")
    integral = np.pad(
        padded.cumsum(axis=0, dtype=np.float64).cumsum(axis=1, dtype=np.float64),
        ((1, 0), (1, 0)),
        mode="constant",
    )
    size = 2 * radius + 1
    sums = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return sums.astype(np.float32, copy=False)


def ca_noise_map(energy, train_cells, guard_cells):
    outer_radius = train_cells + guard_cells
    outer_sum = box_sum_centered(energy, outer_radius)
    inner_sum = box_sum_centered(energy, guard_cells)
    outer_area = (2 * outer_radius + 1) ** 2
    inner_area = (2 * guard_cells + 1) ** 2
    num_training_cells = outer_area - inner_area
    return (outer_sum - inner_sum) / float(num_training_cells), num_training_cells


def training_mask(train_cells, guard_cells):
    outer_radius = train_cells + guard_cells
    width = 2 * outer_radius + 1
    mask = np.ones((width, width), dtype=bool)
    center = outer_radius
    mask[
        center - guard_cells : center + guard_cells + 1,
        center - guard_cells : center + guard_cells + 1,
    ] = False
    return mask


def os_noise_map(energy, train_cells, guard_cells, k_fraction):
    outer_radius = train_cells + guard_cells
    mask = training_mask(train_cells, guard_cells)
    num_training_cells = int(mask.sum())
    k_index = int(math.ceil(k_fraction * num_training_cells))
    kth_zero_based = max(0, min(num_training_cells - 1, k_index - 1))

    padded = np.pad(energy, outer_radius, mode="edge")
    windows = sliding_window_view(padded, (2 * outer_radius + 1, 2 * outer_radius + 1))
    training_values = windows[..., mask]
    kth_values = np.partition(training_values, kth_zero_based, axis=-1)[..., kth_zero_based]
    return kth_values.astype(np.float32, copy=False), num_training_cells, k_index


def estimate_flops_per_frame(map_shape, method):
    num_pixels = int(np.prod(map_shape))
    train = method["train"]
    guard = method["guard"]
    outer_radius = train + guard
    outer_area = (2 * outer_radius + 1) ** 2
    inner_area = (2 * guard + 1) ** 2
    num_training = outer_area - inner_area

    # One comparison per Doppler bin for max-energy extraction, plus a rough
    # per-cell estimate for the CFAR statistic and threshold comparison.
    doppler_max_ops = num_pixels * 127
    if method["kind"] == "ca":
        cfar_ops = num_pixels * (num_training + 4)
    else:
        cfar_ops = num_pixels * (num_training * math.log2(max(num_training, 2)) + 1)
    return float(doppler_max_ops + cfar_ops)


def compute_chamfer_distance_2d(gt_points, pred_points):
    if len(gt_points) == 0 or len(pred_points) == 0:
        return np.nan

    try:
        from scipy.spatial import cKDTree

        tree_gt = cKDTree(gt_points)
        tree_pred = cKDTree(pred_points)
        dist_pred_to_gt, _ = tree_gt.query(pred_points)
        dist_gt_to_pred, _ = tree_pred.query(gt_points)
        return float(np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred))
    except Exception:
        pass

    try:
        from sklearn.neighbors import KDTree

        tree_gt = KDTree(gt_points, metric="euclidean")
        tree_pred = KDTree(pred_points, metric="euclidean")
        dist_pred_to_gt, _ = tree_gt.query(pred_points, k=1)
        dist_gt_to_pred, _ = tree_pred.query(gt_points, k=1)
        return float(np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred))
    except Exception:
        return chamfer_distance_chunked(gt_points, pred_points)


def nearest_mean_distance(query_points, reference_points, chunk_size=4096):
    total = 0.0
    count = 0
    ref = reference_points.astype(np.float32, copy=False)
    for start in range(0, len(query_points), chunk_size):
        query = query_points[start : start + chunk_size].astype(np.float32, copy=False)
        diff = query[:, None, :] - ref[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        total += float(np.sqrt(np.min(dist2, axis=1)).sum())
        count += len(query)
    return total / max(count, 1)


def chamfer_distance_chunked(gt_points, pred_points):
    return nearest_mean_distance(pred_points, gt_points) + nearest_mean_distance(gt_points, pred_points)


def scene_frame_from_metadata(metadata, fallback_index):
    path = str(metadata.get("power_path", ""))
    scene = metadata.get("scene", None)
    frame = metadata.get("frame", None)

    if scene is None and "Scene" in path:
        try:
            scene = int(path.split("Scene", 1)[1].split(os.sep, 1)[0])
        except Exception:
            scene = f"unk_{fallback_index}"
    if frame is None and "Pow_Frame_" in path:
        try:
            frame = int(path.rsplit("Pow_Frame_", 1)[1].split(".", 1)[0])
        except Exception:
            frame = fallback_index

    return scene if scene is not None else f"unk_{fallback_index}", frame if frame is not None else fallback_index


def evaluate_predictions(method, threshold, noise, energy, gt_np, gt_points, x_grid, y_grid, scene, frame):
    pred_np = energy > (threshold * noise)
    pd_val, pfa_val = compute_pd_pfa(gt_np, pred_np.astype(np.float32, copy=False))
    pred_points = np.column_stack((x_grid[pred_np], y_grid[pred_np]))
    cd_val = compute_chamfer_distance_2d(gt_points, pred_points)

    return {
        "Run": RUN_NAME,
        "Method": method["name"],
        "ThresholdFactor": threshold,
        "Scene": scene,
        "Frame": frame,
        "TrainCellsPerSide": method["train"],
        "GuardCellsPerSide": method["guard"],
        "Pd": pd_val,
        "Pfa": pfa_val,
        "Chamfer_Dist": cd_val,
        "Estimated_FLOPs_PerFrame": estimate_flops_per_frame(energy.shape, method),
        "PeakMemoryMB": ru_maxrss_mb(),
    }


def process_indices(indices, thresholds):
    records = []
    x_grid, y_grid = _COORDINATE_GRID

    for dataset_index in indices:
        input_cube, gt_cube, metadata = _DATASET[dataset_index]
        energy = radar_energy_from_cube(input_cube)
        gt_np = np.squeeze(gt_cube).astype(np.float32, copy=False)

        if energy.shape != gt_np.shape:
            raise ValueError(
                f"Prediction/GT shape mismatch at dataset index {dataset_index}: "
                f"energy={energy.shape}, gt={gt_np.shape}"
            )

        scene, frame = scene_frame_from_metadata(metadata, dataset_index)
        gt_mask = gt_np > 0.5
        gt_points = np.column_stack((x_grid[gt_mask], y_grid[gt_mask]))

        for method in METHODS:
            if method["kind"] == "ca":
                noise, num_training_cells = ca_noise_map(energy, method["train"], method["guard"])
                k_index = np.nan
            else:
                noise, num_training_cells, k_index = os_noise_map(
                    energy, method["train"], method["guard"], method["k_fraction"]
                )

            for threshold in thresholds:
                record = evaluate_predictions(
                    method, threshold, noise, energy, gt_np, gt_points, x_grid, y_grid, scene, frame
                )
                record["NumTrainingCells"] = num_training_cells
                record["OSKFraction"] = method["k_fraction"]
                record["OSKIndex"] = k_index
                records.append(record)

    return records


def chunk_indices(indices, num_chunks):
    if num_chunks <= 1:
        return [indices]
    chunks = np.array_split(np.asarray(indices), num_chunks)
    return [chunk.astype(int).tolist() for chunk in chunks if len(chunk) > 0]


def write_run_config(args, output_dir):
    config_path = Path(output_dir) / "cfar_run_config.txt"
    with open(config_path, "w") as f:
        for key, value in sorted(vars(args).items()):
            f.write(f"{key}: {value}\n")
    return config_path


def parse_args():
    parser = argparse.ArgumentParser(description="CPU CFAR baseline for RaDelft BEV maps")
    parser.add_argument("--dataset_path", type=str, default="/scratch/shujianjia/dataset/")
    parser.add_argument("--output_dir", type=str, default="./results/cfar_baseline")
    parser.add_argument("--csv_name", type=str, default="cfar_metrics.csv")
    parser.add_argument("--train_val_scenes", type=int, nargs="+", default=[1, 3, 4, 5, 7])
    parser.add_argument("--test_scenes", type=int, nargs="+", default=[2, 6])
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--num_workers", type=int, default=max(1, min(os.cpu_count() or 1, 4)))
    parser.add_argument("--log_every", type=int, default=250)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--max_frames", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = write_run_config(args, output_dir)

    _limit_worker_threads()
    params = build_params(args.dataset_path, args.train_val_scenes, args.test_scenes)
    dataset = RADCUBE_DATASET(mode="test", params=params)
    total_frames = len(dataset)
    frame_count = min(total_frames, args.max_frames) if args.max_frames else total_frames
    indices = list(range(frame_count))

    print("=" * 70, flush=True)
    print("启动 CFAR baseline", flush=True)
    print(f" Test scenes: {args.test_scenes}", flush=True)
    print(f" Frames: {frame_count}/{total_frames}", flush=True)
    print(f" Threshold factors: {args.thresholds}", flush=True)
    print(f" Workers: {args.num_workers}", flush=True)
    print(f" Output dir: {output_dir}", flush=True)
    print(f" Config: {config_path}", flush=True)
    print("=" * 70, flush=True)

    worker_count = max(1, min(args.num_workers, len(indices)))
    chunks = chunk_indices(indices, worker_count)
    records = []
    completed_frames = 0

    if worker_count == 1:
        init_worker(args.dataset_path, args.train_val_scenes, args.test_scenes)
        for chunk in chunks:
            records.extend(process_indices(chunk, args.thresholds))
            completed_frames += len(chunk)
            if args.log_every and completed_frames % args.log_every == 0:
                print(f"[{RUN_NAME}] processed {completed_frames}/{frame_count} frames", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=init_worker,
            initargs=(args.dataset_path, args.train_val_scenes, args.test_scenes),
        ) as executor:
            futures = [executor.submit(process_indices, chunk, args.thresholds) for chunk in chunks]
            for future in as_completed(futures):
                chunk_records = future.result()
                records.extend(chunk_records)
                completed_frames += len(chunk_records) // (len(METHODS) * len(args.thresholds))
                if args.log_every:
                    print(f"[{RUN_NAME}] processed {completed_frames}/{frame_count} frames", flush=True)

    df_metrics = pd.DataFrame(records)
    sort_cols = ["Scene", "Frame", "Method", "ThresholdFactor"]
    df_metrics = df_metrics.sort_values(sort_cols).reset_index(drop=True)

    summary = df_metrics.groupby(
        ["Run", "Method", "ThresholdFactor"],
        as_index=False,
        sort=False,
    ).agg(
        Pd=("Pd", "mean"),
        Pfa=("Pfa", "mean"),
        Chamfer_Dist=("Chamfer_Dist", "mean"),
        TrainCellsPerSide=("TrainCellsPerSide", "first"),
        GuardCellsPerSide=("GuardCellsPerSide", "first"),
        NumTrainingCells=("NumTrainingCells", "first"),
        OSKFraction=("OSKFraction", "first"),
        OSKIndex=("OSKIndex", "first"),
        Estimated_FLOPs_PerFrame=("Estimated_FLOPs_PerFrame", "first"),
        PeakMemoryMB=("PeakMemoryMB", "max"),
    )

    csv_path = output_dir / args.csv_name
    summary_csv_path = output_dir / f"summary_{args.csv_name}"
    df_metrics.to_csv(csv_path, index=False)
    summary.to_csv(summary_csv_path, index=False)

    print("\nCFAR baseline 完成", flush=True)
    print(f"详细逐帧指标: {csv_path}", flush=True)
    print(f"按方法汇总指标: {summary_csv_path}", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
