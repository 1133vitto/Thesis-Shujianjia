import sys
import os
import argparse
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import cKDTree

current_dir = Path(__file__).resolve().parent
radelft_dir = current_dir / "radelft"
sys.path.insert(0, str(radelft_dir))
sys.path.insert(0, str(current_dir))

from radelft.loaders.rad_cube_loader import RADCUBE_DATASET
from radelft.utils.compute_metrics import compute_pd_pfa
from radelft.data_preparation import data_preparation


def compute_chamfer_distance(point_cloud1, point_cloud2):
    if len(point_cloud1) == 0 or len(point_cloud2) == 0:
        return np.nan
    tree1 = cKDTree(point_cloud1)
    tree2 = cKDTree(point_cloud2)
    dist1, _ = tree1.query(point_cloud2)
    dist2, _ = tree2.query(point_cloud1)
    return np.mean(dist1) + np.mean(dist2)


def points_from_mask(mask, x_grid, y_grid):
    return np.column_stack((x_grid[mask], y_grid[mask]))


def extract_scene_frame(dp, fallback_index):
    power_path = str(dp.get('power_path', ''))
    scene_m = re.search(r'Scene(\d+)', power_path)
    frame_m = re.search(r'Pow_Frame_(\d+)', power_path)
    scene = int(scene_m.group(1)) if scene_m else f'unk_{fallback_index}'
    frame = int(frame_m.group(1)) if frame_m else fallback_index
    return scene, frame


def selected_items(data_dict, requested_frames, requested_global_indices, max_frames):
    if requested_frames is not None and requested_global_indices is not None:
        raise ValueError("Use either --frames for per-scene Pow_Frame numbers or --global_indices for dataset indices, not both.")

    items = list(data_dict.items())
    if requested_global_indices is not None:
        data_indices = {idx for idx, _ in items}
        invalid = [idx for idx in requested_global_indices if idx not in data_indices]
        if invalid:
            raise ValueError(f"Global indices not found in dataset: {invalid}")
        selected = [(idx, data_dict[idx]) for idx in requested_global_indices]
        return selected[:max_frames] if max_frames is not None else selected

    requested = set(requested_frames) if requested_frames is not None else None
    selected = []
    for idx, dp in items:
        _, frame = extract_scene_frame(dp, idx)
        if requested is not None and frame not in requested:
            continue
        selected.append((idx, dp))
        if max_frames is not None and len(selected) >= max_frames:
            break
    return selected


def main():
    parser = argparse.ArgumentParser(description='Visualize pre-saved npy prediction point clouds')
    parser.add_argument('--cfar_folder', type=str, default='radar_ososos2D',
                        help='CFAR folder name in dataset (e.g. radar_ososos2D)')
    parser.add_argument('--npy_folder', type=str, default='network2d_original',
                        help='NPY prediction folder name (e.g. network2d), '
                             'replaces cfar_folder in path to locate npy files')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--max_frames', type=int, default=10,
                        help='Maximum frames to visualize')
    parser.add_argument('--frames', type=int, nargs='+', default=None,
                        help='Pow_Frame numbers to visualize within the selected test scenes')
    parser.add_argument('--global_indices', type=int, nargs='+', default=None,
                        help='Dataset indices to visualize')
    parser.add_argument('--output_format', type=str, default='pdf', choices=['png', 'pdf'])
    parser.add_argument('--dpi', type=int, default=150)
    parser.add_argument('--dataset_path', type=str, default='/scratch/shujianjia/dataset/')
    parser.add_argument('--test_scenes', type=int, nargs='+', default=[2, 6],
                        help='Test scene numbers')
    args = parser.parse_args()

    vis_dir = os.path.join(args.output_dir, 'ignacio2d')
    os.makedirs(vis_dir, exist_ok=True)

    # ---- Dataset params ----
    params = data_preparation.get_default_params()
    params["dataset_path"] = args.dataset_path
    params["train_val_scenes"] = [1, 3, 4, 5, 7]
    params["test_scenes"] = args.test_scenes
    params["bev"] = True
    params["cfar_folder"] = args.cfar_folder

    gt_dataset = RADCUBE_DATASET(mode='test', params=params)

    # ---- Coordinate axes for TP/FP/FN point extraction ----
    # Must use the SAME axes as lidarpc_to_lidarcube mask creation.
    # lidarpc_to_lidarcube voxelizes on params axes then flips axis=1,
    # so mask column j corresponds to params['azimuth_axis'][::-1][j].
    range_axis_mask = params['range_axis']
    azimuth_axis_mask = params['azimuth_axis'][::-1]  # match np.flip(axis=1)
    THETA_mask, R_mask = np.meshgrid(azimuth_axis_mask, range_axis_mask)
    X_mask = R_mask * np.sin(THETA_mask)
    Y_mask = R_mask * np.cos(THETA_mask)

    metrics_records = []
    selected = selected_items(gt_dataset.data_dict, args.frames, args.global_indices, args.max_frames)
    subplot_title_fontsize = 22

    print("=" * 70)
    print(f"NPY Evaluation — {len(gt_dataset.data_dict)} dataset frames")
    print(f"  CFAR folder:  {args.cfar_folder}")
    print(f"  NPY folder:   {args.npy_folder}")
    print(f"  Frames selected: {len(selected)}")
    print(f"  Results:      {vis_dir}")
    print("=" * 70)

    for dataset_index, dp in tqdm(selected, desc="Processing"):
        cfar_path = dp.get('cfar_path')
        if cfar_path is None:
            continue

        scene, frame = extract_scene_frame(dp, dataset_index)
        npy_path = cfar_path.replace(args.cfar_folder, args.npy_folder)
        if not os.path.exists(npy_path):
            continue

        # --- Load GT (matching compute_metrics.py: np.load, no negation, no transform) ---
        gt_pc = np.load(dp['gt_path'])
        gt_cube = data_preparation.lidarpc_to_lidarcube(gt_pc, params)
        gt_mask = np.squeeze(gt_cube)

        # --- Load prediction (matching compute_metrics.py lines 36-39) ---
        pred_pc = np.load(npy_path)
        if pred_pc.shape[1] == 4:
            pred_pc = pred_pc[:, :-1]  # remove speed channel
        pred_pc[:, 1] = -pred_pc[:, 1]

        pred_cube = data_preparation.lidarpc_to_lidarcube(pred_pc, params)
        pred_mask = np.squeeze(pred_cube)

        # --- Metrics ---
        pd_val, pfa_val = compute_pd_pfa(gt_mask, pred_mask)
        gt_points = points_from_mask(gt_mask > 0.5, X_mask, Y_mask)
        pred_points = points_from_mask(pred_mask > 0.5, X_mask, Y_mask)
        cd_val = compute_chamfer_distance(gt_points, pred_points)

        metrics_records.append({
            'Scene': scene, 'Frame': frame,
            'Pd': pd_val, 'Pfa': pfa_val, 'Chamfer_Dist': cd_val
        })

        # --- TP / FP / FN from masks ---
        tp_mask = (pred_mask > 0.5) & (gt_mask > 0.5)
        fp_mask = (pred_mask > 0.5) & (gt_mask <= 0.5)
        fn_mask = (pred_mask <= 0.5) & (gt_mask > 0.5)
        tp_x, tp_y = X_mask[tp_mask], Y_mask[tp_mask]
        fp_x, fp_y = X_mask[fp_mask], Y_mask[fp_mask]
        fn_x, fn_y = X_mask[fn_mask], Y_mask[fn_mask]

        # --- Save path ---
        save_path = os.path.join(vis_dir, f"scene_{scene}_frame_{frame}.{args.output_format}")
        if os.path.exists(save_path):
            continue

        # ==========================================
        # Visualization (axis convention: pc[:,1]→x, pc[:,0]→y)
        # ==========================================
        layout = [
            ["camera", "camera", "camera", "pred_pc"],
            ["gt_pc", "tp_points", "fp_points", "fn_points"],
        ]
        fig, axd = plt.subplot_mosaic(layout, figsize=(20, 10), layout='constrained')
        fig.suptitle(f"DL baseline Evaluation - Scene {scene} | Frame {frame}  "
                     f"(Pd={pd_val:.4f}  Pfa={pfa_val:.6f}  CD={cd_val:.4f})", fontsize=20)

        # Camera
        cam_path = dp.get('cam_path', None)
        if cam_path and os.path.exists(cam_path):
            img = plt.imread(cam_path)
            if img.ndim >= 3:
                img = img[500:-150, :, :]
                img = np.fliplr(img)
            axd["camera"].imshow(img, aspect='auto')
        axd["camera"].axis('off')
        axd["camera"].set_title("Camera Reference", fontsize=subplot_title_fontsize)

        # Pred point cloud
        axd["pred_pc"].scatter(pred_pc[:, 1], pred_pc[:, 0], s=3, c='blue', marker='o', edgecolors='none', rasterized=True)
        axd["pred_pc"].set_title(f"Prediction (n={len(pred_pc)})", fontsize=subplot_title_fontsize)

        # GT point cloud
        axd["gt_pc"].scatter(gt_pc[:, 1], gt_pc[:, 0], s=3, c='green', marker='o', edgecolors='none', rasterized=True)
        axd["gt_pc"].set_title(f"Ground Truth (n={len(gt_pc)})", fontsize=subplot_title_fontsize)

        # TP / FP / FN — scatter(x, y) must match pred/gt convention:
        #   x_axis = cross-range = X_mask,  y_axis = down-range = Y_mask
        axd["tp_points"].scatter(tp_x, tp_y, s=5, c='limegreen', marker='o', edgecolors='none', rasterized=True)
        axd["tp_points"].set_title(f"True Positives (n={len(tp_x)})", fontsize=subplot_title_fontsize)

        axd["fp_points"].scatter(fp_x, fp_y, s=5, c='red', marker='o', edgecolors='none', rasterized=True)
        axd["fp_points"].set_title(f"False Positives (n={len(fp_x)})", fontsize=subplot_title_fontsize)

        axd["fn_points"].scatter(fn_x, fn_y, s=5, c='dodgerblue', marker='o', edgecolors='none', rasterized=True)
        axd["fn_points"].set_title(f"False Negatives (n={len(fn_x)})", fontsize=subplot_title_fontsize)

        # Unified axis settings
        y_max = np.max(Y_mask) if np.max(Y_mask) > 0 else 50
        for key in ["pred_pc", "gt_pc", "tp_points", "fp_points", "fn_points"]:
            axd[key].set_aspect('equal')
            axd[key].set_xlim(-30, 30)
            axd[key].set_ylim(0, y_max)
            axd[key].set_xlabel('x (m)')
            axd[key].set_ylabel('y (m)')
            axd[key].grid(True, linestyle=':', alpha=0.6)

        plt.savefig(save_path, dpi=args.dpi, bbox_inches='tight', format=args.output_format)
        plt.close(fig)

        print(f"  Scene {scene} Frame {frame} -> Pd={pd_val:.4f} Pfa={pfa_val:.6f} CD={cd_val:.4f}")

    # ---- Summary ----
    if metrics_records:
        print(f"\nProcessed {len(metrics_records)} frames")
        valid_cd = [m['Chamfer_Dist'] for m in metrics_records if not np.isnan(m['Chamfer_Dist'])]
        if valid_cd:
            print(f"  Avg Pd: {np.mean([m['Pd'] for m in metrics_records]):.4f}")
            print(f"  Avg Pfa: {np.mean([m['Pfa'] for m in metrics_records]):.6f}")
            print(f"  Avg Chamfer Dist: {np.mean(valid_cd):.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
