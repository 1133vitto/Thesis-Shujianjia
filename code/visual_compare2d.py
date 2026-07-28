import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm import tqdm


current_dir = Path(__file__).resolve().parent
radelft_dir = current_dir / "radelft"
sys.path.insert(0, str(radelft_dir))
sys.path.insert(0, str(current_dir))

from cfar import build_coordinate_grid, build_params, ca_noise_map, compute_chamfer_distance_2d
from model import CustomUNet, CustomUNet2Layer, CustomUNet3Plus, CustomUNetPlusPlus, MaxPower2DModel
from train2d import RaDelftWrapper
from mpl_toolkits.axes_grid1 import make_axes_locatable


MODEL_REGISTRY = {
    "CustomUNet": CustomUNet,
    "CustomUNetPlusPlus": CustomUNetPlusPlus,
    "CustomUNet2Layer": CustomUNet2Layer,
    "CustomUNet3Plus": CustomUNet3Plus,
}


def parse_args():
    parser = argparse.ArgumentParser(description="2D inference comparison visualization")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/visual_compare2d")
    parser.add_argument("--dataset_path", type=str, default="/scratch/shujianjia/dataset/")
    parser.add_argument("--train_val_scenes", type=int, nargs="+", default=[1, 3, 4, 5, 7])
    parser.add_argument("--test_scenes", type=int, nargs="+", default=[2, 6])
    parser.add_argument("--model", type=str, default="CustomUNet2Layer", choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--frames", type=int, nargs="+", default=None)
    parser.add_argument("--global_indices", type=int, nargs="+", default=None,
                        help="0-based indices in the concatenated test dataset order")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--cfar_threshold", type=float, default=1.5)
    parser.add_argument("--cfar_folder", type=str, default="radar_ososos")
    parser.add_argument("--ignacio_folder", type=str, default="network2d_original")
    parser.add_argument("--output_format", type=str, default="png", choices=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def extract_scene_frame(metadata, fallback_index):
    power_path = str(metadata.get("power_path", ""))
    scene = f"unk_{fallback_index}"
    frame = fallback_index

    if "Scene" in power_path:
        scene = int(power_path.split("Scene", 1)[1].split(os.sep, 1)[0])
    if "Pow_Frame_" in power_path:
        frame = int(power_path.rsplit("Pow_Frame_", 1)[1].split(".", 1)[0])

    return scene, frame


def load_model(args):
    model = MaxPower2DModel(model=MODEL_REGISTRY[args.model], in_channels=2).to(args.device)
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def crop_model_output(outputs):
    occupancy_prob = outputs["occupancy_prob"]
    if occupancy_prob.dim() == 2:
        occupancy_prob = occupancy_prob.unsqueeze(0)
    occupancy_prob = occupancy_prob[:, :-12, 8:-8]

    ra_energy = outputs["ra_energy"]
    if ra_energy.dim() == 4:
        ra_energy = ra_energy[:, 0, :-12, 8:-8]
    else:
        ra_energy = ra_energy[:, :-12, 8:-8]

    return occupancy_prob, ra_energy


def proposed_mask(model, radar_cube, alpha):
    outputs = model(radar_cube)
    occupancy_prob, radar_energy = crop_model_output(outputs)

    background_prob = 1.0 - occupancy_prob.unsqueeze(1)
    radar_energy_4d = radar_energy.unsqueeze(1)

    kernel_size = 5
    pad = kernel_size // 2
    bgenergy = background_prob * radar_energy_4d
    bgenergy = F.pad(bgenergy, (pad, pad, pad, pad), mode="replicate")
    local_bg_noise = F.avg_pool2d(bgenergy, kernel_size=kernel_size, stride=1, padding=0)
    return (radar_energy_4d > (alpha * local_bg_noise)).squeeze().detach().cpu().numpy()


def cfar_mask(energy, threshold):
    noise, _ = ca_noise_map(energy, train_cells=5, guard_cells=5)
    return energy > (threshold * noise)


def points_from_mask(mask, x_grid, y_grid):
    return np.column_stack((x_grid[mask], y_grid[mask]))


def load_ignacio_points(metadata, folder_name):
    cfar_path = Path(str(metadata["cfar_path"]))
    scene_dir = cfar_path.parents[2]
    point_path = scene_dir / "rosDS" / folder_name / cfar_path.name
    if not point_path.is_file():
        return point_path, np.empty((0, 2), dtype=np.float32)

    points = np.load(point_path)
    points = points.reshape((-1, points.shape[-1]))
    bev_points = np.column_stack((points[:, 1], points[:, 0]))
    return point_path, bev_points.astype(np.float32, copy=False)


def camera_image(metadata):
    image = plt.imread(metadata["cam_path"])
    if image.shape[0] > 650:
        image = image[500:-150, :, :]
    return np.fliplr(image)


def radar_energy_from_tensor(radar_cube):
    energy = torch.max(radar_cube, dim=2).values
    return energy[:, :-12, 8:-8].squeeze(0).detach().cpu().numpy()


def flip_points_for_plot(points):
    if len(points) == 0:
        return points
    plotted = points.copy()
    plotted[:, 0] = -plotted[:, 0]
    return plotted


def plot_points(ax, points, color, title, x_grid, y_grid):
    points = flip_points_for_plot(points)
    if len(points) > 0:
        ax.scatter(points[:, 0], points[:, 1], s=8, c=color, marker="o", edgecolors="none", rasterized=True)
    ax.set_title(title, fontsize=20)
    setup_point_axis(ax, x_grid, y_grid)


def setup_point_axis(ax, x_grid, y_grid):
    ax.set_aspect("equal")
    ax.set_anchor("C")
    ax.set_xlim(-30, 30)
    ax.set_ylim(0, float(np.max(y_grid)))
    ax.set_xlabel("x - Lateral (m)", fontsize=15)
    ax.set_ylabel("y - Forward (m)", fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, linestyle=":", alpha=0.5)


def format_cd(value):
    return "nan" if np.isnan(value) else f"{value:.4f} m"


def save_visualization(frame_data, save_path, dpi, output_format):
    layout = [
        ["camera", "camera", "camera", "radar_energy"],
        ["cfar_pred", "Proposed method", "Ignacionetwork", "gt_pc"],
    ]
    fig, axd = plt.subplot_mosaic(
        layout,
        figsize=(20, 9),
        layout="constrained",
        gridspec_kw={"height_ratios": [0.95, 1.0]},
    )
    fig.suptitle(f"Scene {frame_data['scene']} | Frame {frame_data['frame']}", fontsize=24)

    axd["camera"].imshow(camera_image(frame_data["metadata"]), aspect="auto")
    axd["camera"].set_title("Camera Reference", fontsize=24)
    axd["camera"].axis("off")


    radar_energy_db = 10 * np.log10(frame_data["energy"] + 1e-9) + 39.54
    mesh = axd["radar_energy"].pcolormesh(
        -frame_data["x_grid"],
        frame_data["y_grid"],
        radar_energy_db,
        cmap="jet",
        shading="gouraud",
    )
    mesh.set_rasterized(True)
    axd["radar_energy"].set_title("Radar Energy (dB)", fontsize=24)
    setup_point_axis(axd["radar_energy"], frame_data["x_grid"], frame_data["y_grid"])
    # cax = inset_axes(axd["radar_energy"], width="3%", height="62%", loc="center right", borderpad=1.0)
    # divider = make_axes_locatable(axd["radar_energy"])
    # cax = divider.append_axes("right", size="3%", pad=0.1)
    # cbar = fig.colorbar(mesh, cax=cax)
    cbar = fig.colorbar(mesh, ax=axd['radar_energy'], fraction=0.035, pad=0.05)
    cbar.ax.tick_params(labelsize=13)

    plot_points(
        axd["cfar_pred"],
        frame_data["cfar_points"],
        "tab:purple",
        f"CA-CFAR, CD={format_cd(frame_data['cfar_cd'])}",
        frame_data["x_grid"],
        frame_data["y_grid"],
    )
    plot_points(
        axd["Proposed method"],
        frame_data["proposed_points"],
        "tab:red",
        f"Proposed method, CD={format_cd(frame_data['proposed_cd'])}",
        frame_data["x_grid"],
        frame_data["y_grid"],
    )
    plot_points(
        axd["Ignacionetwork"],
        frame_data["ignacio_points"],
        "tab:blue",
        f"DL baseline, CD={format_cd(frame_data['ignacio_cd'])}",
        frame_data["x_grid"],
        frame_data["y_grid"],
    )
    plot_points(
        axd["gt_pc"],
        frame_data["gt_points"],
        "tab:green",
        f"LiDAR GT (n={len(frame_data['gt_points'])})",
        frame_data["x_grid"],
        frame_data["y_grid"],
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format=output_format)
    plt.close(fig)


def selected_indices(dataset, requested_frames, requested_global_indices, max_frames):
    if requested_frames is not None and requested_global_indices is not None:
        raise ValueError("Use either --frames for per-scene Pow_Frame numbers or --global_indices for 0-based dataset indices, not both.")

    if requested_global_indices is not None:
        total = len(dataset)
        invalid = [idx for idx in requested_global_indices if idx < 0 or idx >= total]
        if invalid:
            raise ValueError(f"Global indices out of range 0..{total - 1}: {invalid}")
        indices = list(requested_global_indices)
        return indices[:max_frames] if max_frames is not None else indices

    indices = []
    requested = set(requested_frames) if requested_frames is not None else None
    for idx, metadata in dataset.real_dataset.data_dict.items():
        _, frame = extract_scene_frame(metadata, idx)
        if requested is not None and frame not in requested:
            continue
        indices.append(idx)
        if max_frames is not None and len(indices) >= max_frames:
            break
    return indices


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = build_params(args.dataset_path, args.train_val_scenes, args.test_scenes)
    params["cfar_folder"] = args.cfar_folder
    dataset = RaDelftWrapper(mode="test", params=params)
    indices = selected_indices(dataset, args.frames, args.global_indices, args.max_frames)
    x_grid, y_grid = build_coordinate_grid(params)
    model = load_model(args)

    csv_path = output_dir / "visual_compare2d_metrics.csv"
    records = []

    print("=" * 70)
    print("启动 2D 对比可视化")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Model: {args.model}")
    print(f"Test scenes: {args.test_scenes}")
    print(f"Frames selected: {len(indices)}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    with torch.no_grad():
        for dataset_index in tqdm(indices, desc="Visualizing"):
            sample = dataset[dataset_index]
            metadata = sample["metadata"]
            scene, frame = extract_scene_frame(metadata, dataset_index)

            radar_cube = sample["radar_cube"].unsqueeze(0).to(args.device)
            gt_np = sample["occupancy_target"].squeeze().cpu().numpy() > 0.5
            energy = radar_energy_from_tensor(radar_cube)

            if gt_np.shape != energy.shape:
                raise ValueError(f"Shape mismatch at Scene {scene} Frame {frame}: gt={gt_np.shape}, energy={energy.shape}")

            cfar = cfar_mask(energy, args.cfar_threshold)
            proposed = proposed_mask(model, radar_cube, args.alpha)

            gt_points = points_from_mask(gt_np, x_grid, y_grid)
            cfar_points = points_from_mask(cfar, x_grid, y_grid)
            proposed_points = points_from_mask(proposed, x_grid, y_grid)
            ignacio_path, ignacio_points = load_ignacio_points(metadata, args.ignacio_folder)

            cfar_cd = compute_chamfer_distance_2d(gt_points, cfar_points)
            proposed_cd = compute_chamfer_distance_2d(gt_points, proposed_points)
            ignacio_cd = compute_chamfer_distance_2d(gt_points, ignacio_points)

            save_path = output_dir / f"scene_{scene}_frame_{frame}.{args.output_format}"
            frame_data = {
                "scene": scene,
                "frame": frame,
                "metadata": metadata,
                "x_grid": x_grid,
                "y_grid": y_grid,
                "energy": energy,
                "gt_points": gt_points,
                "cfar_points": cfar_points,
                "proposed_points": proposed_points,
                "ignacio_points": ignacio_points,
                "cfar_cd": cfar_cd,
                "proposed_cd": proposed_cd,
                "ignacio_cd": ignacio_cd,
            }
            save_visualization(frame_data, save_path, args.dpi, args.output_format)

            records.append(
                {
                    "Scene": scene,
                    "Frame": frame,
                    "Checkpoint": args.checkpoint_path,
                    "Model": args.model,
                    "Alpha": args.alpha,
                    "CFAR_Method": "CACFAR_train5_guard5",
                    "CFAR_Threshold": args.cfar_threshold,
                    "CFAR_Chamfer": cfar_cd,
                    "Proposed_Chamfer": proposed_cd,
                    "Ignacionetwork_Chamfer": ignacio_cd,
                    "PowerPath": metadata["power_path"],
                    "CameraPath": metadata["cam_path"],
                    "GTPath": metadata["gt_path"],
                    "CFARPath": metadata["cfar_path"],
                    "IgnacionetworkPath": str(ignacio_path),
                    "OutputPath": str(save_path),
                }
            )

    fieldnames = [
        "Scene",
        "Frame",
        "Checkpoint",
        "Model",
        "Alpha",
        "CFAR_Method",
        "CFAR_Threshold",
        "CFAR_Chamfer",
        "Proposed_Chamfer",
        "Ignacionetwork_Chamfer",
        "PowerPath",
        "CameraPath",
        "GTPath",
        "CFARPath",
        "IgnacionetworkPath",
        "OutputPath",
    ]
    # with open(csv_path, "w", newline="") as f:
    #     writer = csv.DictWriter(f, fieldnames=fieldnames)
    #     writer.writeheader()
    #     writer.writerows(records)

    print(f"完成：保存 {len(records)} 张图，CSV: {'None'}")


if __name__ == "__main__":
    main()
