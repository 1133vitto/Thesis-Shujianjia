import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["image.composite_image"] = True

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


current_dir = Path(__file__).resolve().parent
radelft_dir = current_dir / "radelft"
sys.path.insert(0, str(radelft_dir))
sys.path.insert(0, str(current_dir))

from cfar import (
    METHODS,
    build_coordinate_grid,
    build_params,
    ca_noise_map,
    os_noise_map,
    radar_energy_from_cube,
)
from radelft.loaders.rad_cube_loader import RADCUBE_DATASET


TARGET_METHODS = (
    "CACFAR_train5_guard5",
    "CACFAR_train9_guard0",
    "OSCFAR_train5_guard5_k0.75",
    "OSCFAR_train9_guard0_k0.75",
)
COMPARISON_SOURCE_METHOD = "CACFAR_train5_guard5"
OUTPUT_FORMAT = "pdf"


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize selected CFAR baseline frames")
    parser.add_argument("--metrics_csv", type=str, default="./results/cfar_baseline/cfar_metrics.csv")
    parser.add_argument("--output_dir", type=str, default="./results/cfar_baseline/visualizations_pdf")
    parser.add_argument("--dataset_path", type=str, default="/scratch/shujianjia/dataset/")
    parser.add_argument("--train_val_scenes", type=int, nargs="+", default=[1, 3, 4, 5, 7])
    parser.add_argument("--test_scenes", type=int, nargs="+", default=[2, 6])
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--threshold", type=float, default=1.5)
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def extract_scene_frame(power_path):
    path = str(power_path)
    scene = int(path.split("Scene", 1)[1].split(os.sep, 1)[0])
    frame = int(path.rsplit("Pow_Frame_", 1)[1].split(".", 1)[0])
    return scene, frame


def build_dataset_index(dataset):
    mapping = {}
    for dataset_index, metadata in dataset.data_dict.items():
        scene, frame = extract_scene_frame(metadata["power_path"])
        mapping[(scene, frame)] = dataset_index
    return mapping


def method_by_name():
    return {method["name"]: method for method in METHODS}


def select_frames(metrics, method_name, threshold, top_k):
    subset = metrics[
        (metrics["Method"] == method_name)
        & np.isclose(metrics["ThresholdFactor"].astype(float), threshold)
    ].dropna(subset=["Chamfer_Dist"])

    best = subset.sort_values("Chamfer_Dist", ascending=True).head(top_k).copy()
    best["Selection"] = "best"
    best["Rank"] = np.arange(1, len(best) + 1)

    worst = subset.sort_values("Chamfer_Dist", ascending=False).head(top_k).copy()
    worst["Selection"] = "worst"
    worst["Rank"] = np.arange(1, len(worst) + 1)
    return best, worst


def load_frame(dataset, dataset_index, x_grid, y_grid):
    input_cube, gt_cube, metadata = dataset[dataset_index]
    energy = radar_energy_from_cube(input_cube)
    gt = np.squeeze(gt_cube).astype(np.float32, copy=False)
    gt_mask = gt > 0.5
    gt_points = np.column_stack((x_grid[gt_mask], y_grid[gt_mask]))
    scene, frame = extract_scene_frame(metadata["power_path"])
    return {
        "scene": scene,
        "frame": frame,
        "metadata": metadata,
        "energy": energy,
        "gt": gt,
        "gt_points": gt_points,
    }


def compute_cfar_mask(energy, method, threshold):
    if method["kind"] == "ca":
        noise, _ = ca_noise_map(energy, method["train"], method["guard"])
    else:
        noise, _, _ = os_noise_map(energy, method["train"], method["guard"], method["k_fraction"])
    return energy > (threshold * noise)


def pred_points_from_mask(mask, x_grid, y_grid):
    return np.column_stack((x_grid[mask], y_grid[mask]))


def camera_image(metadata):
    image = plt.imread(metadata["cam_path"])
    if image.shape[0] > 650:
        image = image[500:-150, :, :]
    return np.fliplr(image)


def title_metrics(row):
    pd_val = row["Pd"] if isinstance(row, (dict, pd.Series)) else row.Pd
    pfa_val = row["Pfa"] if isinstance(row, (dict, pd.Series)) else row.Pfa
    cd_val = row["Chamfer_Dist"] if isinstance(row, (dict, pd.Series)) else row.Chamfer_Dist
    return (
        f"Pd={pd_val:.4f} | Pfa={pfa_val:.6f} | "
        f"Chamfer={cd_val:.4f}"
    )


def setup_point_axis(ax, x_grid, y_grid):
    ax.set_aspect("equal")
    ax.set_xlim(-30, 30)
    ax.set_ylim(0, float(np.max(y_grid)))
    ax.set_xlabel("x - Lateral (m)")
    ax.set_ylabel("y - Forward (m)")
    ax.grid(True, linestyle=":", alpha=0.5)


def flipped_x_grid(x_grid):
    return -x_grid


def flipped_points(points):
    if len(points) == 0:
        return points
    plotted = points.copy()
    plotted[:, 0] = -plotted[:, 0]
    return plotted


def plot_radar_energy(ax, energy, x_grid, y_grid):
    radar_energy_db = 10 * np.log10(energy + 1e-9) + 39.54
    mesh = ax.pcolormesh(flipped_x_grid(x_grid), y_grid, radar_energy_db, cmap="jet", shading="gouraud")
    mesh.set_rasterized(True)
    ax.set_title("Radar Energy: Doppler Max (dB)")
    setup_point_axis(ax, x_grid, y_grid)
    return mesh


def plot_single(row, frame_data, method, threshold, x_grid, y_grid, save_path, dpi):
    mask = compute_cfar_mask(frame_data["energy"], method, threshold)
    pred_points = flipped_points(pred_points_from_mask(mask, x_grid, y_grid))
    gt_points = flipped_points(frame_data["gt_points"])

    layout = [["camera", "camera", "camera"], ["radar", "cfar", "gt"]]
    fig, axd = plt.subplot_mosaic(
        layout,
        figsize=(17, 9),
        layout="constrained",
        gridspec_kw={"height_ratios": [0.72, 1.0]},
    )
    fig.suptitle(
        f"Scene {int(row['Scene'])} Frame {int(row['Frame'])} | "
        f"{row['Method']} | factor={threshold:g}\n{title_metrics(row)}",
        fontsize=14,
    )

    axd["camera"].imshow(camera_image(frame_data["metadata"]))
    axd["camera"].set_title("Camera")
    axd["camera"].axis("off")

    mesh = plot_radar_energy(axd["radar"], frame_data["energy"], x_grid, y_grid)
    fig.colorbar(mesh, ax=axd["radar"], fraction=0.046, pad=0.04)

    axd["cfar"].scatter(
        pred_points[:, 0],
        pred_points[:, 1],
        s=3,
        c="tab:blue",
        edgecolors="none",
        rasterized=True,
    )
    axd["cfar"].set_title(f"CFAR Point Cloud (n={len(pred_points)})")
    setup_point_axis(axd["cfar"], x_grid, y_grid)

    axd["gt"].scatter(
        gt_points[:, 0],
        gt_points[:, 1],
        s=3,
        c="tab:green",
        edgecolors="none",
        rasterized=True,
    )
    axd["gt"].set_title(f"LiDAR GT Point Cloud (n={len(gt_points)})")
    setup_point_axis(axd["gt"], x_grid, y_grid)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(rows_by_method, frame_data, methods_by_name, threshold, x_grid, y_grid, save_path, dpi):
    layout = [
        ["camera", "camera", "radar", "gt"],
        ["CACFAR_train5_guard5", "CACFAR_train9_guard0", "OSCFAR_train5_guard5_k0.75", "OSCFAR_train9_guard0_k0.75"],
    ]
    fig, axd = plt.subplot_mosaic(
        layout,
        figsize=(22, 9),
        layout="constrained",
        gridspec_kw={"height_ratios": [0.78, 1.0]},
    )
    fig.suptitle(
        f"CFAR Comparison | Scene {frame_data['scene']} Frame {frame_data['frame']} | factor={threshold:g}",
        fontsize=16,
    )

    axd["camera"].imshow(camera_image(frame_data["metadata"]))
    axd["camera"].set_title("Camera")
    axd["camera"].axis("off")

    mesh = plot_radar_energy(axd["radar"], frame_data["energy"], x_grid, y_grid)
    fig.colorbar(mesh, ax=axd["radar"], fraction=0.046, pad=0.04)

    gt_points = flipped_points(frame_data["gt_points"])
    axd["gt"].scatter(
        gt_points[:, 0],
        gt_points[:, 1],
        s=3,
        c="tab:green",
        edgecolors="none",
        rasterized=True,
    )
    axd["gt"].set_title(f"LiDAR GT (n={len(gt_points)})")
    setup_point_axis(axd["gt"], x_grid, y_grid)

    for method_name in TARGET_METHODS:
        row = rows_by_method[method_name]
        method = methods_by_name[method_name]
        mask = compute_cfar_mask(frame_data["energy"], method, threshold)
        pred_points = flipped_points(pred_points_from_mask(mask, x_grid, y_grid))
        ax = axd[method_name]
        ax.scatter(
            pred_points[:, 0],
            pred_points[:, 1],
            s=3,
            c="tab:blue",
            edgecolors="none",
            rasterized=True,
        )
        ax.set_title(f"{method_name}\n{title_metrics(row)}")
        setup_point_axis(ax, x_grid, y_grid)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def row_filename(row, prefix):
    return (
        f"scene{int(row['Scene'])}_frame{int(row['Frame'])}_"
        f"{prefix}{int(row['Rank']):02d}_{safe_name(row['Method'])}_"
        f"pd{row['Pd']:.4f}_pfa{row['Pfa']:.6f}_cd{row['Chamfer_Dist']:.4f}.{OUTPUT_FORMAT}"
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    metrics = pd.read_csv(args.metrics_csv)

    params = build_params(args.dataset_path, args.train_val_scenes, args.test_scenes)
    dataset = RADCUBE_DATASET(mode="test", params=params)
    dataset_index = build_dataset_index(dataset)
    x_grid, y_grid = build_coordinate_grid(params)
    methods_by_name = method_by_name()

    selected = {}
    for method_name in TARGET_METHODS:
        best, worst = select_frames(metrics, method_name, args.threshold, args.top_k)
        selected[method_name] = {"best": best, "worst": worst}

    frame_cache = {}

    def get_frame(scene, frame):
        key = (int(scene), int(frame))
        if key not in frame_cache:
            frame_cache[key] = load_frame(dataset, dataset_index[key], x_grid, y_grid)
        return frame_cache[key]

    saved_single = 0
    for method_name, groups in selected.items():
        method = methods_by_name[method_name]
        for selection, rows in groups.items():
            for _, row in rows.iterrows():
                frame_data = get_frame(row["Scene"], row["Frame"])
                save_path = (
                    output_dir
                    / "single"
                    / safe_name(method_name)
                    / selection
                    / row_filename(row, selection)
                )
                plot_single(row, frame_data, method, args.threshold, x_grid, y_grid, save_path, args.dpi)
                saved_single += 1

    source_rows = pd.concat(
        [
            selected[COMPARISON_SOURCE_METHOD]["best"],
            selected[COMPARISON_SOURCE_METHOD]["worst"],
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["Scene", "Frame"], keep="first")

    comparison_metrics = metrics[
        np.isclose(metrics["ThresholdFactor"].astype(float), args.threshold)
        & metrics["Method"].isin(TARGET_METHODS)
    ].copy()
    comparison_lookup = {
        (int(row.Scene), int(row.Frame), row.Method): row
        for row in comparison_metrics.itertuples(index=False)
    }

    saved_comparison = 0
    comparison_dir = output_dir / "comparison" / f"from_{safe_name(COMPARISON_SOURCE_METHOD)}_factor{args.threshold:g}"
    for _, source_row in source_rows.iterrows():
        scene = int(source_row["Scene"])
        frame = int(source_row["Frame"])
        rows_by_method = {
            method_name: comparison_lookup[(scene, frame, method_name)]
            for method_name in TARGET_METHODS
        }
        frame_data = get_frame(scene, frame)
        save_path = comparison_dir / f"scene{scene}_frame{frame}_comparison.{OUTPUT_FORMAT}"
        plot_comparison(rows_by_method, frame_data, methods_by_name, args.threshold, x_grid, y_grid, save_path, args.dpi)
        saved_comparison += 1

    print(f"Saved single-method {OUTPUT_FORMAT.upper()}s: {saved_single}")
    print(f"Saved comparison {OUTPUT_FORMAT.upper()}s: {saved_comparison}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
