"""
testvisual2d.py - HDD 模型推理与可视化脚本
对齐 test2d.py 的模型和推理逻辑，专注于可视化输出
"""
import sys
import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from scipy.spatial import cKDTree

current_dir = Path(__file__).resolve().parent
radelft_dir = current_dir / "radelft"
sys.path.insert(0, str(radelft_dir))
sys.path.insert(0, str(current_dir))

from model import RadarResUNet
from radelft.loaders.rad_cube_loader import RADCUBE_DATASET
from radelft.utils.compute_metrics import compute_pd_pfa
from radelft.data_preparation import data_preparation
from torch.utils.data import DataLoader


def compute_chamfer_distance_2d(gt_pc, pred_pc):
    """物理级倒角距离 (单位: 米)"""
    if len(gt_pc) == 0 or len(pred_pc) == 0:
        return np.nan
    tree_gt = cKDTree(gt_pc)
    tree_pred = cKDTree(pred_pc)
    dist_pred_to_gt, _ = tree_gt.query(pred_pc)
    dist_gt_to_pred, _ = tree_pred.query(gt_pc)
    return np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred)


def main():
    parser = argparse.ArgumentParser(description='HDD 模型推理与可视化')
    parser.add_argument('--checkpoint_path', type=str,
                        default='./checkpoints/run_20260317_013246/best_epoch_11_loss_0.0118.pth')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='L2 norm detection threshold (default: 1.0)')
    parser.add_argument('--max_frames', type=int, default=10,
                        help='Maximum frames to visualize (for quick testing)')
    args = parser.parse_args()

    vis_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # ==========================================
    # 坐标轴构建 (对齐 test2d.py)
    # ==========================================
    range_cell_size = 0.1004
    range_axis_full = np.arange(range_cell_size, 51.4242 + 1e-5, range_cell_size)
    range_axis = range_axis_full[10:-2]

    angle_fft_size = 256
    wx_vec_full = np.linspace(-np.pi, np.pi, angle_fft_size)
    wx_vec_full = wx_vec_full[::-1]
    wx_vec = wx_vec_full[8:248]
    sin_theta = np.clip(wx_vec / (2 * np.pi * 0.4972), -1.0, 1.0)
    azimuth_axis = np.arcsin(sin_theta)

    THETA, R = np.meshgrid(azimuth_axis, range_axis)
    X = R * np.sin(THETA)
    Y = R * np.cos(THETA)

    print("=" * 70)
    print("启动 HDD 推理与可视化")
    print(f" 加载模型: {args.checkpoint_path}")
    print(f" 结果保存至: {args.output_dir}")
    print(f" 检测阈值: {args.threshold}")
    print("=" * 70)

    # ==========================================
    # 数据集加载 (对齐 test2d.py)
    # ==========================================
    params = data_preparation.get_default_params()
    params["dataset_path"] = '/scratch/shujianjia/dataset/'
    params["train_val_scenes"] = [1, 3, 4, 5, 7]
    params["test_scenes"] = [2, 6]
    params["bev"] = True

    test_dataset = RADCUBE_DATASET(mode='test', params=params)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=args.workers if args.device == 'cuda' else 0)

    # ==========================================
    # 模型初始化 (对齐 test2d.py)
    # ==========================================
    model = RadarResUNet(n_doppler=128, out_dim=32).to(args.device)

    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    metrics_records = []
    step = 0

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Testing & Visualizing")):
            radar_cube, lidar_cube, meta = batch_data
            radar_cube = radar_cube.float().to(args.device)
            lidar_cube = lidar_cube.float().to(args.device)

            if lidar_cube.dim() == 3:
                occupancy_target = lidar_cube.unsqueeze(1)
            else:
                occupancy_target = lidar_cube

            # 模型前向传播
            outputs = model(radar_cube)
            outputs = outputs[:, :, :-12, 8:-8]  # crop to match GT: (B, 32, 500, 240)

            # 2D GT
            occupancy_target_2d, _ = torch.max(occupancy_target, dim=1)
            gt_np = occupancy_target_2d.squeeze().cpu().numpy()

            # HDD 检测: L2 范数作为得分图
            score_map = torch.norm(outputs, p=2, dim=1)  # (B, 500, 240)
            binary_pred = (score_map > args.threshold).float()

            score_map_np = score_map.squeeze().cpu().numpy()
            pred_np = binary_pred.squeeze().cpu().numpy().astype(np.float32)

            # 雷达能量图 (max over Doppler channels, 用于可视化对比)
            radar_energy_full = radar_cube.max(dim=1)[0]
            if radar_energy_full.shape[-2:] == (512, 256):
                radar_energy_np = radar_energy_full[:, :-12, 8:-8].squeeze().cpu().numpy()
            else:
                radar_energy_np = radar_energy_full.squeeze().cpu().numpy()

            # ==========================================
            # 元数据提取
            # ==========================================
            scene_id = meta.get('scene', [f'unk_{batch_idx}'])[0]
            if isinstance(scene_id, torch.Tensor):
                scene_id = scene_id.item()
            frame_id = meta.get('frame', [batch_idx])[0]
            if isinstance(frame_id, torch.Tensor):
                frame_id = frame_id.item()

            title_info = f"Scene: {scene_id} | Frame: {frame_id}"
            save_path = os.path.join(vis_dir, f"scene_{scene_id}_frame_{frame_id}_t{args.threshold}.png")
            if os.path.exists(save_path):
                continue

            # ==========================================
            # 指标计算
            # ==========================================
            pred_pc_x = X[pred_np > 0.5]
            pred_pc_y = Y[pred_np > 0.5]
            gt_pc_x = X[gt_np > 0.5]
            gt_pc_y = Y[gt_np > 0.5]

            pd_val, pfa_val = compute_pd_pfa(gt_np, pred_np)

            gt_pc_array = np.column_stack((gt_pc_x, gt_pc_y))
            pred_pc_array = np.column_stack((pred_pc_x, pred_pc_y))
            cd_val = compute_chamfer_distance_2d(gt_pc_array, pred_pc_array)

            print(f"{title_info} -> Pd: {pd_val:.4f}, Pfa: {pfa_val:.6f}, Chamfer: {cd_val:.4f}")

            metrics_records.append({
                'Threshold': args.threshold,
                'Scene': scene_id,
                'Frame': frame_id,
                'Pd': pd_val,
                'Pfa': pfa_val,
                'Chamfer_Dist': cd_val
            })

            # ==========================================
            # 可视化 (2行4列 mosaic 布局)
            # ==========================================
            layout = [
                ["camera", "camera", "radar_energy"],
                ["score_map", "pred_pc", "gt_pc"]
            ]

            fig, axd = plt.subplot_mosaic(layout, figsize=(20, 10), layout='constrained')
            fig.suptitle(f"HDD Detection — {title_info}  |  Threshold: {args.threshold}", fontsize=18)

            # --- Camera ---
            cam_path = meta.get('cam_path', [None])[0]
            if cam_path is not None and os.path.exists(cam_path):
                img = plt.imread(cam_path)
                img = img[500:-150, :, :]
                img = np.fliplr(img)
                axd["camera"].imshow(img, aspect='auto')
                axd["camera"].set_title("Camera Reference")
            else:
                axd["camera"].text(0.5, 0.5, "No Camera Image", ha='center', va='center',
                                   transform=axd["camera"].transAxes, fontsize=14)
                axd["camera"].set_title("Camera Reference (unavailable)")
            axd["camera"].axis('off')

            # --- Radar Energy (dB) ---
            radar_energy_db = 10 * np.log10(radar_energy_np + 1e-9) + 39.54
            im_re = axd["radar_energy"].pcolormesh(X, Y, radar_energy_db, cmap='jet', shading='gouraud')
            axd["radar_energy"].set_title("Radar Energy (dB)")
            plt.colorbar(im_re, ax=axd["radar_energy"], fraction=0.046, pad=0.04)

            # --- Score Map (L2 Norm) ---
            im_score = axd["score_map"].pcolormesh(X, Y, score_map_np, cmap='plasma', shading='gouraud')
            axd["score_map"].set_title(f"L2 Norm Score (thresh={args.threshold})")
            plt.colorbar(im_score, ax=axd["score_map"], fraction=0.046, pad=0.04)

            # --- Pred Point Cloud ---
            axd["pred_pc"].scatter(pred_pc_x, pred_pc_y, s=3, c='red', marker='o')
            axd["pred_pc"].set_title("Prediction (Point Cloud)")

            # --- GT Point Cloud ---
            axd["gt_pc"].scatter(gt_pc_x, gt_pc_y, s=3, c='green', marker='o')
            axd["gt_pc"].set_title("Ground Truth (Point Cloud)")

            # 统一 BEV 坐标轴
            for key in ["radar_energy", "score_map", "pred_pc", "gt_pc"]:
                axd[key].set_aspect('equal')
                axd[key].set_xlim(-30, 30)
                axd[key].set_ylim(0, np.max(Y) if np.max(Y) > 0 else 50)
                axd[key].set_xlabel('x - Lateral (m)')
                axd[key].set_ylabel('y - Forward (m)')
                axd[key].grid(True, linestyle=':', alpha=0.6)

            # 隐藏空白子图
            for key in ["."]:
                if key in axd:
                    axd[key].set_visible(False)

            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            step += 1
            if step >= args.max_frames:
                break

    # ==========================================
    # 汇总并保存指标
    # ==========================================
    df_metrics = pd.DataFrame(metrics_records)
    csv_path = os.path.join(args.output_dir, "visual_metrics.csv")
    df_metrics.to_csv(csv_path, index=False)

    valid_cd = df_metrics['Chamfer_Dist'].dropna()
    avg_cd = valid_cd.mean() if not valid_cd.empty else float('nan')
    avg_pd = df_metrics['Pd'].mean()
    avg_pfa = df_metrics['Pfa'].mean()

    print(f"\n  平均 Pd: {avg_pd:.4f} | 平均 Pfa: {avg_pfa:.6f} | 平均 Chamfer: {avg_cd:.4f}")
    print(f"  可视化保存至: {vis_dir}")
    print(f"  指标 CSV: {csv_path}")


if __name__ == "__main__":
    main()
