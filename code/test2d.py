"""
test2d.py - 模型推理脚本，对齐 train2d.py 的模型和推理逻辑
"""
import sys
import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
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
    parser = argparse.ArgumentParser(description='模型推理脚本')
    parser.add_argument('--checkpoint_path', type=str,
                        default='./checkpoints/run_20260317_013246/best_epoch_11_loss_0.0118.pth')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("启动测试推理")
    print(f" 加载模型: {args.checkpoint_path}")
    print(f" 结果保存至: {args.output_dir}")
    print("=" * 70)

    # Range Axis
    range_cell_size = 0.1004
    range_axis_full = np.arange(range_cell_size, 51.4242 + 1e-5, range_cell_size)
    range_axis = range_axis_full[10:-2]

    # Azimuth Axis
    angle_fft_size = 256
    wx_vec_full = np.linspace(-np.pi, np.pi, angle_fft_size)
    wx_vec_full = wx_vec_full[::-1]
    wx_vec = wx_vec_full[8:248]
    sin_theta = np.clip(wx_vec / (2 * np.pi * 0.4972), -1.0, 1.0)
    azimuth_axis = np.arcsin(sin_theta)

    THETA, R = np.meshgrid(azimuth_axis, range_axis)
    X = R * np.sin(THETA)
    Y = R * np.cos(THETA)

    # 数据集加载，对齐 train2d.py
    params = data_preparation.get_default_params()
    params["dataset_path"] = '/scratch/shujianjia/dataset/'
    params["train_val_scenes"] = [1, 3, 4, 5, 7]
    params["test_scenes"] = [2, 6]
    params["bev"]=True

    test_dataset = RADCUBE_DATASET(mode='test', params=params)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                            num_workers=args.workers if args.device == 'cuda' else 0)

    # 模型初始化，对齐 train2d.py
    model = RadarResUNet(n_doppler=128, out_dim=32).to(args.device)

    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 阈值扫描
    test_thresholds = [0.8, 0.9,1.0,1.1,1.2]
    metrics_records = []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Testing")):
            radar_cube, lidar_cube, meta = batch_data
            radar_cube = radar_cube.float().to(args.device)
            lidar_cube = lidar_cube.float().to(args.device)

            if lidar_cube.dim() == 3:
                occupancy_target = lidar_cube.unsqueeze(1)
            else:
                occupancy_target = lidar_cube

            # 模型前向传播，与 train2d.py 一致
            outputs = model(radar_cube)
            outputs = outputs[:, :, :-12, 8:-8]

            # 2D GT
            occupancy_target_2d, _ = torch.max(occupancy_target, dim=1)
            gt_np = occupancy_target_2d.squeeze().cpu().numpy()

            # 计算 L2 范数作为预测得分
            final_pred_2d = torch.norm(outputs, p=2, dim=1)

            scene_id = meta.get('scene', [f'unk_{batch_idx}'])[0]
            if isinstance(scene_id, torch.Tensor):
                scene_id = scene_id.item()
            frame_id = meta.get('frame', [batch_idx])[0]
            if isinstance(frame_id, torch.Tensor):
                frame_id = frame_id.item()

            title_info = f"Scene: {scene_id} | Frame: {frame_id}"

            for current_threshold in test_thresholds:
                binary_pred = (final_pred_2d > current_threshold).float()
                pred_np = binary_pred.squeeze().cpu().numpy().astype(np.float32)

                pd_val, pfa_val = compute_pd_pfa(gt_np, pred_np)

                pred_pc_x = X[pred_np > 0.5]
                pred_pc_y = Y[pred_np > 0.5]
                gt_pc_x = X[gt_np > 0.5]
                gt_pc_y = Y[gt_np > 0.5]

                gt_pc_array = np.column_stack((gt_pc_x, gt_pc_y))
                pred_pc_array = np.column_stack((pred_pc_x, pred_pc_y))
                cd_val = compute_chamfer_distance_2d(gt_pc_array, pred_pc_array)

                print(f"{title_info} -> threshold: {current_threshold}, Pd: {pd_val:.4f}, Pfa: {pfa_val:.6f}, Chamfer: {cd_val:.4f}",flush=True)
                
                metrics_records.append({
                    'Threshold': current_threshold,
                    'Scene': scene_id,
                    'Frame': frame_id,
                    'Pd': pd_val,
                    'Pfa': pfa_val,
                    'Chamfer_Dist': cd_val
                })

    # 保存结果
    df_metrics = pd.DataFrame(metrics_records)
    csv_path = os.path.join(args.output_dir, "504_输出32维度results.csv")
    df_metrics.to_csv(csv_path, index=False)

    avg_pd = df_metrics['Pd'].mean()
    avg_pfa = df_metrics['Pfa'].mean()
    valid_cd = df_metrics['Chamfer_Dist'].dropna()
    avg_cd = valid_cd.mean() if not valid_cd.empty else float('nan')

    print(f"\n 平均 Pd: {avg_pd:.4f} | 平均 Pfa: {avg_pfa:.6f} | 平均 Chamfer: {avg_cd:.4f}")
    print(f" 结果已保存至: {csv_path}")


if __name__ == "__main__":
    main()