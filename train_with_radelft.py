
"""
雷达-LiDAR 融合检测项目 - 使用 RaDelft 兼容层的训练脚本
保留我们的核心创新：网络模型和 Loss Function
参考：https://github.com/RaDelft/RaDelft-Dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import sys
import argparse
from pathlib import Path

# 添加 src 目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from model import RadarLidarFusionModel
from losses import RadarLidarLoss
from evaluation import Evaluator, PointCloudGenerator, ResultSaver
from radelft_compatibility import (
    RaDelftDataLoader, 
    RaDelftMetrics, 
    RaDelftVisualizer
)


class RaDelftDataset(Dataset):
    """
    使用 RaDelft 兼容层的 Dataset
    """
    
    def __init__(self, 
                 num_samples: int = 100,
                 range_bins: int = 64,
                 doppler_bins: int = 32,
                 angle_bins: int = 16):
        self.num_samples = num_samples
        self.range_bins = range_bins
        self.doppler_bins = doppler_bins
        self.angle_bins = angle_bins
        
        # RaDelft 兼容的数据加载器
        self.radelft_loader = RaDelftDataLoader(
            range_bins=range_bins,
            doppler_bins=doppler_bins,
            angle_bins=angle_bins
        )
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx: int):
        # 加载雷达数据（RaDelft 兼容接口）
        radar_cube = self.radelft_loader.load_radar_cube(idx)
        
        # 加载 LiDAR 点云（RaDelft 兼容接口）
        lidar_points = self.radelft_loader.load_lidar_pointcloud(idx)
        
        # 将 LiDAR 投影到 range-angle 作为 occupancy target
        from data_preprocessing import LidarPointCloudProcessor
        occupancy_target = LidarPointCloudProcessor.project_to_range_angle(
            lidar_points,
            range_bins=self.range_bins,
            angle_bins=self.angle_bins
        )
        
        # 模拟噪声目标（实际项目中用 RaDelft 的标注）
        noise_target = np.abs(np.random.randn(self.range_bins, self.angle_bins))
        
        return {
            'radar_cube': torch.from_numpy(radar_cube).float(),
            'lidar_points': torch.from_numpy(lidar_points).float(),
            'occupancy_target': torch.from_numpy(occupancy_target).float(),
            'noise_target': torch.from_numpy(noise_target).float()
        }


def parse_args():
    parser = argparse.ArgumentParser(description='雷达-LiDAR 融合检测 - RaDelft 兼容版')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--range_bins', type=int, default=64)
    parser.add_argument('--doppler_bins', type=int, default=32)
    parser.add_argument('--angle_bins', type=int, default=16)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--result_dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch_idx):
    model.train()
    total_loss_dict = {}
    
    for batch_idx, batch_data in enumerate(dataloader):
        radar_cube = batch_data['radar_cube'].to(device)
        occupancy_target = batch_data['occupancy_target'].to(device)
        noise_target = batch_data['noise_target'].to(device)
        
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
        
        for key, value in loss_dict.items():
            if key not in total_loss_dict:
                total_loss_dict[key] = 0.0
            total_loss_dict[key] += value.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch_idx+1}], Batch [{batch_idx}/{len(dataloader)}], "
                  f"Total Loss: {loss_dict['total'].item():.4f}")
    
    num_batches = len(dataloader)
    for key in total_loss_dict:
        total_loss_dict[key] /= num_batches
        
    return total_loss_dict


def main():
    args = parse_args()
    
    print("="*80)
    print("雷达-LiDAR 融合检测 - RaDelft 兼容版")
    print("="*80)
    print(f"使用设备: {args.device}")
    print(f"参考仓库: https://github.com/RaDelft/RaDelft-Dataset")
    print()
    
    # 创建目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    
    # 数据集（使用 RaDelft 兼容层）
    print("创建数据集 (RaDelft 兼容)...")
    train_dataset = RaDelftDataset(
        num_samples=args.num_samples,
        range_bins=args.range_bins,
        doppler_bins=args.doppler_bins,
        angle_bins=args.angle_bins
    )
    val_dataset = RaDelftDataset(
        num_samples=max(20, args.num_samples // 5),
        range_bins=args.range_bins,
        doppler_bins=args.doppler_bins,
        angle_bins=args.angle_bins
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 模型（我们自己的，保留创新！）
    print("创建模型 (保留我们的创新架构)...")
    model = RadarLidarFusionModel(
        range_bins=args.range_bins,
        doppler_bins=args.doppler_bins,
        angle_bins=args.angle_bins,
        quantiles=[0.1, 0.5, 0.9]
    ).to(args.device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")
    print()
    
    # 损失函数（我们自己的，保留创新！）
    criterion = RadarLidarLoss(
        weight_focal=1.0,
        weight_quantile=1.0,
        weight_chamfer=0.1,
        weight_consistency=0.0
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 评估器
    evaluator = Evaluator(save_dir=args.result_dir)
    
    # 训练循环
    print("开始训练...")
    print("-"*80)
    
    best_f1 = 0.0
    
    for epoch in range(args.num_epochs):
        # 训练
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, args.device, epoch
        )
        
        print(f"Epoch [{epoch+1}/{args.num_epochs}] - 训练完成")
        print(f"  损失: Total={train_losses['total']:.4f}, "
              f"Focal={train_losses['focal']:.4f}, "
              f"Quantile={train_losses['quantile']:.4f}")
        
        # 定期评估
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
        
        print("-"*80)
    
    print()
    print("="*80)
    print("训练完成!")
    print("核心创新保留: 网络模型 + Quantile/Pinball Loss (CFAR)")
    print("="*80)


if __name__ == "__main__":
    main()

