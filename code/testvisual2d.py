import sys
import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import cdist
import pandas as pd

# 路径设置，与你的训练脚本保持一致
current_dir = Path(__file__).resolve().parent
radelft_dir = current_dir / "radelft"
sys.path.insert(0, str(radelft_dir)) 
sys.path.insert(0, str(current_dir))

from model import MaxPower2DModel
from radelft.loaders.rad_cube_loader import RADCUBE_DATASET
from radelft.utils.compute_metrics import compute_pd_pfa
from radelft.data_preparation import data_preparation

# ==========================================
# 辅助函数：计算 Chamfer Distance (2D)
# ==========================================
def compute_chamfer_distance_2d(gt_mask, pred_mask):
    """
    计算二值图上的倒角距离。
    将 gt 和 pred 中 > 0 的像素作为点集，计算双向最小距离的平均值。
    """
    gt_points = np.argwhere(gt_mask > 0)
    pred_points = np.argwhere(pred_mask > 0)
    
    # 边界情况处理
    if len(gt_points) == 0 and len(pred_points) == 0:
        return 0.0
    if len(gt_points) == 0 or len(pred_points) == 0:
        return np.nan # 某一方完全没有预测出目标，标记为 NaN 并在外层过滤
        
    # 计算所有点对的距离矩阵
    dist_matrix = cdist(gt_points, pred_points)
    
    # 双向最小距离求平均
    dist_gt_to_pred = np.mean(np.min(dist_matrix, axis=1))
    dist_pred_to_gt = np.mean(np.min(dist_matrix, axis=0))
    
    return (dist_gt_to_pred + dist_pred_to_gt) / 2.0


class RaDelftTestWrapper(torch.utils.data.Dataset):
    """
    复用你的 Wrapper，专用于测试集
    """
    def __init__(self, mode='val', params=None):
        self.real_dataset = RADCUBE_DATASET(mode=mode, params=params)

    def __len__(self):
        return len(self.real_dataset)

    def __getitem__(self, idx):
        input_cube, gt_cube, item_params = self.real_dataset[idx]
        power_cube = input_cube[0]
        power_cube = np.transpose(power_cube, (1, 0, 2))
        occupancy_target = np.squeeze(gt_cube) 
        
        return {
            'radar_cube': torch.from_numpy(power_cube).float(),
            'occupancy_target': torch.from_numpy(occupancy_target).float(),
            'metadata': item_params  # 包含 scene / frame 信息
        }

# ==========================================
# 推理与可视化主函数
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='模型推理与可视化脚本')
    # 替换为你实际的最佳模型路径
    parser.add_argument('--checkpoint_path', type=str, 
                        default='./checkpoints/run_20260317_013246/best_epoch_11_loss_0.0118.pth')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    # 1. 创建输出目录
    vis_dir = os.path.join(args.output_dir, 'visualizationsnew3')
    os.makedirs(vis_dir, exist_ok=True)
    
    print("="*70)
    print("启动测试推理与可视化")
    print(f" 加载模型: {args.checkpoint_path}")
    print(f" 结果保存至: {args.output_dir}")
    print("="*70)

    # 2. 准备数据集 (使用 batch_size=1 以便逐帧画图)
    params = data_preparation.get_default_params()
    params["dataset_path"] = '/scratch/shujianjia/dataset/'
    params["train_val_scenes"] = [1, 3, 4, 5, 7]
    params["test_scenes"] = [2, 6]  # 仅测试集
    
    test_dataset = RaDelftTestWrapper(mode='test', params=params) # 或者 mode='test' 看你的 dataloader 定义
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 3. 初始化模型并加载权重
    model = MaxPower2DModel(in_channels=2).to(args.device)
    
    # 解析字典并加载权重
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()

    # 4. 指标统计列表
    metrics_records = []
    step=0
    # 5. 推理循环
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Testing & Plotting")):
            radar_cube = batch_data['radar_cube'].to(args.device)
            occupancy_target = batch_data['occupancy_target']
            
            # 压缩 Z 轴 -> 2D GT
            occupancy_target_2d, _ = torch.max(occupancy_target, dim=1)
            occupancy_target_2d = occupancy_target_2d.to(args.device)
            
            # 模型前向传播
            outputs = model(radar_cube)
            
            # 维度截取 (根据你验证集的代码逻辑)
            occupancy_logits = outputs['occupancy_logits'].unsqueeze(0) # (B, 1, R, A)
            occupancy_logits = occupancy_logits[:, :-12, 8:-8]
            radar_energy = outputs['ra_energy'][:, :-12, 8:-8] if outputs['ra_energy'].dim() == 3 else outputs['ra_energy'][:, 0, :-12, 8:-8]
            
            occupancy_logits = occupancy_logits.unsqueeze(1) # (B, 1, R, A)
            radar_energy_4d = radar_energy.unsqueeze(1)      # (B, 1, R, A)
            
            # 计算 pred 和 bgenergy
            pred = 1.0 - occupancy_logits
            bgenergy = pred * radar_energy_4d
            
            # 计算局部背景噪声
            kernel_size = 5
            pad = kernel_size // 2
            local_bg_noise_sum = F.avg_pool2d(bgenergy, kernel_size=kernel_size, stride=1, padding=pad)
            
            # 最终的二值化预测
            alpha = 2.0
            final_pred_2d = radar_energy_4d > (alpha * local_bg_noise_sum)
            
            # ==============================
            # 数据转换为 Numpy (去除 B 和 C 维度)
            # ==============================
            gt_np = occupancy_target_2d.squeeze().cpu().numpy()
            radar_energy_np = radar_energy_4d.squeeze().cpu().numpy()
            pred_np = pred.squeeze().cpu().numpy()
            bg_noise_np = local_bg_noise_sum.squeeze().cpu().numpy()
            final_pred_np = final_pred_2d.squeeze().cpu().numpy().astype(np.float32)

            # 获取元数据信息用于命名
            meta = batch_data['metadata']
            scene_id = meta.get('scene', [f'unk_{batch_idx}'])[0]
            if isinstance(scene_id, torch.Tensor): scene_id = scene_id.item()
            frame_id = meta.get('frame', [batch_idx])[0]
            if isinstance(frame_id, torch.Tensor): frame_id = frame_id.item()

            title_info = f"Scene: {scene_id} | Frame: {frame_id}"
            save_path = os.path.join(vis_dir, f"scene_{scene_id}_frame_{frame_id}.png")
            if os.path.exists(save_path):
                continue# jump existing visualizations to save time
            # ==============================
            # 指标计算
            # ==============================
            pd_val, pfa_val = compute_pd_pfa(gt_np, final_pred_np)
            cd_val = compute_chamfer_distance_2d(gt_np, final_pred_np)
            print(f"{title_info} -> Pd: {pd_val:.4f}, Pfa: {pfa_val:.6f}, Chamfer Dist: {cd_val:.4f}")
            metrics_records.append({
                'Scene': scene_id,
                'Frame': frame_id,
                'Pd': pd_val,
                'Pfa': pfa_val,
                'Chamfer_Dist': cd_val
            })

            # ==============================
            # 画图 (1行5列)
            # ==============================
            fig, axes = plt.subplots(1, 5, figsize=(25, 5))
            fig.suptitle(f"Test Set Evaluation - {title_info}", fontsize=16)

            # 为了满足你“0-1颜色深浅图”的需求，使用 vmin=0, vmax=1 限制色域
            # radar_energy 和 bg_noise 可能会超出 1，如果超出的话，你可以去掉 vmax=1 或者做归一化
            
            # 1. Radar Energy
            radar_energy_db = 10 * np.log10(radar_energy_np + 1e-9)+39.54
            im0 = axes[0].imshow(radar_energy_db, cmap='viridis', aspect='auto',vmin=-10, vmax=40)
            axes[0].set_title("Radar Energy")
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            # 2. Pred (1.0 - occupancy_logits)
            im1 = axes[1].imshow(pred_np, cmap='plasma', vmin=0, vmax=1, aspect='auto')
            axes[1].set_title("Pred (Probability)")
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            # 3. Local BG Noise Sum
            im2 = axes[2].imshow(bg_noise_np, cmap='viridis', aspect='auto')
            axes[2].set_title("Local BG Noise Sum")
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

            # 4. Final Pred 2D (二值图)
            im3 = axes[3].imshow(final_pred_np, cmap='gray', vmin=0, vmax=1, aspect='auto')
            axes[3].set_title("Final Pred 2D")
            
            # 5. GT (Ground Truth)
            im4 = axes[4].imshow(gt_np, cmap='gray', vmin=0, vmax=1, aspect='auto')
            axes[4].set_title("Ground Truth")

            plt.tight_layout()
            
            # 保存图片
            save_path = os.path.join(vis_dir, f"scene_{scene_id}_frame_{frame_id}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig) # 防止内存泄漏
            step+=1
            if step>10:
                break# 先测试前10帧的可视化，节省时间:

    # ==============================
    # 汇总并保存所有指标
    # ==============================
    # df_metrics = pd.DataFrame(metrics_records)
    
    # # 过滤掉 Chamfer Distance 计算中的 NaN (比如 GT 或 Pred 全黑的情况)
    # valid_cd = df_metrics['Chamfer_Dist'].dropna()
    # avg_cd = valid_cd.mean() if not valid_cd.empty else float('nan')
    
    # avg_pd = df_metrics['Pd'].mean()
    # avg_pfa = df_metrics['Pfa'].mean()
    
    # 保存 CSV
    # csv_path = os.path.join(args.output_dir, "metrics_report_1.csv")
    # df_metrics.to_csv(csv_path, index=False)
    
    # 保存 TXT Summary
    # txt_path = os.path.join(args.output_dir, "summary.txt")
    # with open(txt_path, "w") as f:
    #     f.write("=== 2D Radar Fusion Model Test Summary ===\n")
    #     f.write(f"Model: {args.checkpoint_path}\n")
    #     f.write(f"Total Frames Tested: {len(df_metrics)}\n\n")
    #     f.write(f"Average Pd: {avg_pd:.4f}\n")
    #     f.write(f"Average Pfa: {avg_pfa:.6f}\n")
    #     f.write(f"Average Chamfer Distance: {avg_cd:.4f}\n")
        
    print("\n✅ 推理和可视化全部完成！")
    # print(f"👉 可视化图片文件夹: {vis_dir}")
    # print(f"👉 详细指标数据: {csv_path}")
    # print(f"👉 平均性能总结: {txt_path}")
    # print(f"   平均 Pd: {avg_pd:.4f} | 平均 Pfa: {avg_pfa:.6f} | 平均倒角距离: {avg_cd:.4f}")

if __name__ == "__main__":
    main()