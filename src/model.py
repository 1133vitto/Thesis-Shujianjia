
"""
雷达-LiDAR 融合检测项目 - 完整模型与训练架构
包含: 网络定义, 数据处理, 训练循环, 评估指标
Author: Your Name
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt

from radar_lidar_fusion_losses import (
    FocalLoss, QuantileLoss, ChamferDistance, 
    ConsistencyLoss, RadarLidarLoss
)


# ==================== 网络定义部分 ====================

class DopplerEncoder2DCNN(nn.Module):
    """
    2D CNN 编码器: 将 Doppler 维度编码为 Channels
    输入形状: (batch_size, range_bins, doppler_bins, angle_bins)
    输出形状: (batch_size, range_bins, angle_bins, out_channels)
    """
    def __init__(self, in_doppler_bins: int = 32, out_channels: int = 64):
        super().__init__()
        self.in_doppler = in_doppler_bins
        self.out_channels = out_channels
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 5), padding=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        with torch.no_grad():
            dummy = torch.randn(1, 1, 1, in_doppler_bins)
            out = self.conv_layers(dummy)
            self.compressed_doppler = out.shape[3]
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=(1, self.compressed_doppler))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, range_bins, doppler_bins, angle_bins = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size * angle_bins, 1, range_bins, doppler_bins)
        x = self.conv_layers(x)
        x = self.final_conv(x)
        x = x.squeeze(-1)
        x = x.reshape(batch_size, angle_bins, self.out_channels, range_bins)
        x = x.permute(0, 3, 1, 2)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, 2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)


class UNet2D(nn.Module):
    """
    2D U-Net 网络 - 多任务输出: occupancy + quantiles
    """
    def __init__(self, n_channels: int = 64, n_quantiles: int = 3, bilinear: bool = False):
        super(UNet2D, self).__init__()
        self.n_channels = n_channels
        self.n_quantiles = n_quantiles
        self.bilinear = bilinear
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # 两个输出头
        self.outc_occupancy = OutConv(64, 1)  # occupancy 预测
        self.outc_quantiles = OutConv(64, n_quantiles)  # 分位数预测

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.permute(0, 3, 1, 2)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 两个输出
        logits_occupancy = self.outc_occupancy(x)
        logits_quantiles = self.outc_quantiles(x)
        
        occupancy = torch.sigmoid(logits_occupancy.squeeze(1))  # (B, R, A)
        quantiles = logits_quantiles.permute(0, 2, 3, 1)  # (B, R, A, Q)
        
        return occupancy, quantiles


class RadarLidarFusionModel(nn.Module):
    """
    雷达-LiDAR 融合检测完整模型
    """
    def __init__(self, 
                 range_bins: int = 64, 
                 doppler_bins: int = 32, 
                 angle_bins: int = 16,
                 cnn_out_channels: int = 64,
                 quantiles: list = None):
        super().__init__()
        
        self.range_bins = range_bins
        self.doppler_bins = doppler_bins
        self.angle_bins = angle_bins
        
        if quantiles is None:
            self.quantiles = [0.1, 0.5, 0.9]
        else:
            self.quantiles = quantiles
        
        self.doppler_encoder = DopplerEncoder2DCNN(
            in_doppler_bins=doppler_bins,
            out_channels=cnn_out_channels
        )
        
        self.unet = UNet2D(
            n_channels=cnn_out_channels,
            n_quantiles=len(self.quantiles)
        )
        
    def forward(self, radar_cube: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.doppler_encoder(radar_cube)
        occupancy, quantiles = self.unet(encoded)
        
        # 计算 range-angle 能量图 (对 Doppler 积分)
        ra_energy = torch.mean(radar_cube ** 2, dim=2)
        
        # 计算背景估计 (使用中位数分位数)
        median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else 1
        background_est = quantiles[..., median_idx]
        
        # 检测分数
        detection_score = ra_energy / (background_est + 1e-8)
        
        return {
            'occupancy': occupancy,
            'quantiles': quantiles,
            'background_est': background_est,
            'detection_score': detection_score,
            'ra_energy': ra_energy
        }


# ==================== 数据集部分 ====================

class RadarLidarDataset(Dataset):
    """
    雷达-LiDAR 数据集
    包含雷达 Cube 和 LiDAR 点云监督
    """
    def __init__(self, num_samples: int = 100,
                 range_bins: int = 64, doppler_bins: int = 32, angle_bins: int = 16):
        self.num_samples = num_samples
        self.range_bins = range_bins
        self.doppler_bins = doppler_bins
        self.angle_bins = angle_bins
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 模拟雷达数据 Cube
        radar_cube = np.abs(np.random.randn(
            self.range_bins, self.doppler_bins, self.angle_bins
        ))
        
        # 模拟 LiDAR 点云 (真值)
        num_lidar_points = np.random.randint(50, 200)
        lidar_points = np.random.randn(num_lidar_points, 3) * 20  # 模拟 3D 点
        
        # 生成 occupancy 真值 (从 LiDAR 投影)
        occupancy = np.zeros((self.range_bins, self.angle_bins))
        
        # 简单投影: 随机设置一些点为 1
        for _ in range(np.random.randint(3, 10)):
            r = np.random.randint(0, self.range_bins)
            a = np.random.randint(0, self.angle_bins)
            occupancy[r, a] = 1.0
        
        # 模拟噪声真值
        noise_target = np.abs(np.random.randn(self.range_bins, self.angle_bins))
        
        return {
            'radar_cube': torch.from_numpy(radar_cube).float(),
            'lidar_points': torch.from_numpy(lidar_points).float(),
            'occupancy_target': torch.from_numpy(occupancy).float(),
            'noise_target': torch.from_numpy(noise_target).float()
        }


# ==================== 评估指标部分 ====================

class DetectionMetrics:
    """
    检测评估指标: Pd, Pfa, F1, Chamfer Distance 等
    """
    @staticmethod
    def compute_pd_pfa(occupancy_pred: torch.Tensor, 
                       occupancy_target: torch.Tensor,
                       threshold: float = 0.5) -> Tuple[float, float]:
        """
        计算检测概率 Pd 和虚警概率 Pfa
        """
        pred_binary = (occupancy_pred > threshold).float()
        
        tp = torch.sum((pred_binary == 1) & (occupancy_target == 1)).float()
        fn = torch.sum((pred_binary == 0) & (occupancy_target == 1)).float()
        fp = torch.sum((pred_binary == 1) & (occupancy_target == 0)).float()
        tn = torch.sum((pred_binary == 0) & (occupancy_target == 0)).float()
        
        pd = tp / (tp + fn + 1e-8)
        pfa = fp / (fp + tn + 1e-8)
        
        return pd.item(), pfa.item()
    
    @staticmethod
    def compute_f1(occupancy_pred: torch.Tensor,
                   occupancy_target: torch.Tensor,
                   threshold: float = 0.5) -> float:
        """
        计算 F1-score
        """
        pred_binary = (occupancy_pred > threshold).float()
        
        tp = torch.sum((pred_binary == 1) & (occupancy_target == 1)).float()
        fp = torch.sum((pred_binary == 1) & (occupancy_target == 0)).float()
        fn = torch.sum((pred_binary == 0) & (occupancy_target == 1)).float()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return f1.item()


# ==================== 可视化部分 ====================

def visualize_results(model_output: Dict[str, torch.Tensor], 
                      batch_data: Dict[str, torch.Tensor],
                      save_path: str = "visualization.png"):
    """
    可视化结果
    """
    batch_idx = 0
    
    occupancy_pred = model_output['occupancy'][batch_idx].cpu().numpy()
    quantiles_pred = model_output['quantiles'][batch_idx].cpu().numpy()
    detection_score = model_output['detection_score'][batch_idx].cpu().numpy()
    occupancy_target = batch_data['occupancy_target'][batch_idx].cpu().numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 第一行
    im0 = axes[0, 0].imshow(occupancy_target, cmap='hot')
    axes[0, 0].set_title('Occupancy Target (LiDAR)')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(occupancy_pred, cmap='hot')
    axes[0, 1].set_title('Occupancy Prediction')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].imshow(detection_score, cmap='hot')
    axes[0, 2].set_title('Detection Score')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # 第二行: 分位数
    for i, q in enumerate([0.1, 0.5, 0.9]):
        if i < quantiles_pred.shape[-1]:
            im = axes[1, i].imshow(quantiles_pred[..., i], cmap='hot')
            axes[1, i].set_title(f'Quantile {q} Prediction')
            plt.colorbar(im, ax=axes[1, i])
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Visualization saved to {save_path}")

