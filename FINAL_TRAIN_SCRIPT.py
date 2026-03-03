
"""
========================================================================
✨ 最终训练脚本 - 二合一版本 ✨
========================================================================

这个脚本提供两种模式：
1. 模拟数据模式（默认）：直接运行，不需要 RaDelft 数据集
2. RaDelft 数据模式：设置 use_radelft=True，需要 RaDelft 数据集

我们的核心创新保留：
- model.py 中的网络架构（U-Net + 分位数输出）
- losses.py 中的 Loss Functions（Quantile/Pinball Loss 对应 CFAR）
========================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
from pathlib import Path

# 添加路径
project_root = Path(__file__).parent
import sys
sys.path.insert(0, str(project_root / "src"))

# 我们的核心创新（保留！）
from model import RadarLidarFusionModel
from losses import RadarLidarLoss, FocalLoss, QuantileLoss
from evaluation import Evaluator, PointCloudGenerator, ResultSaver
from radelft.data_preparation import data_preparation

# ==========================================
# 模式 1：模拟数据集（默认，不需要 RaDelft）
# ==========================================
class MockRadarLidarDataset(Dataset):
    """模拟数据集，不需要 RaDelft"""
    def __init__(self, num_samples=100, range_bins=64, doppler_bins=32, angle_bins=16):
        self.num_samples = num_samples
        self.range_bins = range_bins
        self.doppler_bins = doppler_bins
        self.angle_bins = angle_bins
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 模拟雷达 Cube
        radar_cube = np.abs(np.random.randn(
            self.range_bins, self.doppler_bins, self.angle_bins
        ))
        
        # 模拟 occupancy 真值
        occupancy = np.zeros((self.range_bins, self.angle_bins))
        for _ in range(np.random.randint(3, 10)):
            r = np.random.randint(0, self.range_bins)
            a = np.random.randint(0, self.angle_bins)
            occupancy[r, a] = 1.0
        
        # 模拟噪声真值
        noise_target = np.abs(np.random.randn(self.range_bins, self.angle_bins))
        
        # 模拟点云
        num_points = np.random.randint(50, 200)
        lidar_points = np.random.randn(num_points, 3) * 50
        
        return {
            'radar_cube': torch.from_numpy(radar_cube).float(),
            'occupancy_target': torch.from_numpy(occupancy).float(),
            'noise_target': torch.from_numpy(noise_target).float(),
            'lidar_points': torch.from_numpy(lidar_points).float()
        }


# ==========================================
# 模式 2：RaDelft 数据集（可选）
# ==========================================
def get_radelft_dataset(mode='train', params=None):
    """
    使用 RaDelft 的数据集
    你需要配置 dataset_path 和 params
    """
    try:
        from radelft.loaders.rad_cube_loader import RADCUBE_DATASET
        return RADCUBE_DATASET(mode=mode, params=params)
    except ImportError:
        print("⚠️  RaDelft 未正确配置，使用模拟数据")
        return MockRadarLidarDataset()


# ==========================================
# 训练主函数
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='最终训练脚本 - 二合一版本')
    parser.add_argument('--use_radelft', action='store_true', help='使用 RaDelft 数据集（需要配置）',default=True)
    parser.add_argument('--dataset_path', type=str, default='F:/radelft/Scene1_RadarCubes/', help='RaDelft 数据集路径')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--range_bins', type=int, default=500)
    parser.add_argument('--doppler_bins', type=int, default=128)
    parser.add_argument('--angle_bins', type=int, default=256)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--result_dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print("="*70)
    print("✨ 雷达-LiDAR 融合检测 - 最终训练脚本 ✨")
    print("="*70)
    print(f"使用设备: {args.device}")
    print(f"模式: {'RaDelft 数据' if args.use_radelft else '模拟数据'}")
    print()
    
    # 创建目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    
    # 数据集
    print("创建数据集...")
    if args.use_radelft:
        params = data_preparation.get_default_params()

        # Initialise parameters
        params["dataset_path"] = 'F:/radelft/Scene1_RadarCubes/'
        params["train_val_scenes"] = [1]
        params["test_scenes"] = [1]
        params["train_test_split_percent"] = 0.8
        params["cfar_folder"] = None
        params["quantile"] = False

        # This must be kept to false. If the network without elevation is needed, use network_noElevation.py instead
        params["bev"] = False
        # radelft_params = {
        #     'dataset_path': args.dataset_path,
        #     'train_val_scenes': [1],  # 示例，你需要修改
        #     'test_scenes': [1],
        #     'bev': False,
        #     'cfar_folder': None
        # }
        train_dataset = get_radelft_dataset('train', params=params)
        val_dataset = get_radelft_dataset('val', params=params)
    else:
        train_dataset = MockRadarLidarDataset(
            num_samples=args.num_samples,
            range_bins=args.range_bins,
            doppler_bins=args.doppler_bins,
            angle_bins=args.angle_bins
        )
        val_dataset = MockRadarLidarDataset(
            num_samples=max(20, args.num_samples // 5),
            range_bins=args.range_bins,
            doppler_bins=args.doppler_bins,
            angle_bins=args.angle_bins
        )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    
    # 模型（我们的核心创新！保留！）
    print("\n创建模型（我们的核心创新架构）...")
    model = RadarLidarFusionModel(
        range_bins=args.range_bins,
        doppler_bins=args.doppler_bins,
        angle_bins=args.angle_bins,
        quantiles=[0.1, 0.5, 0.9]
    ).to(args.device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")
    
    # Loss（我们的核心创新！保留！）
    criterion = RadarLidarLoss(
        weight_focal=1.0,
        weight_quantile=1.0,
        weight_chamfer=0.0,
        weight_consistency=0.0
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 评估器
    evaluator = Evaluator(save_dir=args.result_dir)
    
    # 训练循环
    print("\n开始训练...")
    print("-"*70)
    
    best_f1 = 0.0
    
    for epoch in range(args.num_epochs):
        # 训练
        model.train()
        total_loss = 0.0
        
        for batch_idx, batch_data in enumerate(train_loader):
            radar_cube = batch_data['radar_cube'].to(args.device)
            occupancy_target = batch_data['occupancy_target'].to(args.device)
            noise_target = batch_data['noise_target'].to(args.device)
            
            optimizer.zero_grad()
            model_output = model(radar_cube)
            
            loss_dict = criterion(
                occupancy_pred=model_output['occupancy'],
                occupancy_target=occupancy_target,
                quantile_pred=model_output['quantiles'],
                noise_target=noise_target
            )
            
            loss_dict['total'].backward()
            optimizer.step()
            
            total_loss += loss_dict['total'].item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch [{epoch+1}/{args.num_epochs}], "
                      f"Batch [{batch_idx}/{len(train_loader)}], "
                      f"Loss: {loss_dict['total'].item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}] 完成. 平均 Loss: {avg_loss:.4f}")
        
        # 定期评估和保存
        if (epoch + 1) % 5 == 0 or epoch == args.num_epochs - 1:
            print(f"\n开始评估 (Epoch {epoch+1})...")
            val_metrics = evaluator.evaluate(
                model, val_loader, args.device, 
                save_results=True, num_vis=2
            )
            
            print(f"  评估结果:")
            for key, value in val_metrics.items():
                print(f"    {key}: {value:.4f}")
            
            # 保存模型
            save_path = os.path.join(args.save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  模型已保存到 {save_path}")
            
            if 'AP' in val_metrics and val_metrics['AP'] > best_f1:
                best_f1 = val_metrics['AP']
                best_path = os.path.join(args.save_dir, "model_best.pth")
                torch.save(model.state_dict(), best_path)
                print(f"  最佳模型已更新 (AP: {best_f1:.4f})")
        
        print("-"*70)
    
    print("\n" + "="*70)
    print("训练完成!")
    print("="*70)


if __name__ == "__main__":
    main()

