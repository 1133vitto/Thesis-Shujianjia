"""
models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from einops import rearrange
import segmentation_models_pytorch as smp
from typing import Dict



# class RadarEncoder(nn.Module):
#     """
#     """
#     def __init__(self, in_doppler_bins: int = 128, out_channels: int = 64):
#         super().__init__()
        
#         # 直接使用 2D 卷积处理多通道输入
#         self.conv_layers = nn.Sequential(
#             # 把 128 维的多普勒特征，混合压缩成 64 维特征
#             nn.Conv2d(in_channels=in_doppler_bins, out_channels=out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             # 再加一层进一步融合周围的 Range 和 Azimuth 信息
#             nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # 输入 x 形状: (Batch, Range, Doppler, Azimuth) 
#         # 比如真实的: (B, 512, 128, 256)
        
#         # 1. 调整维度顺序，把 Doppler 放到 Channel 的位置 (PyTorch 的 2D 格式是 B, C, H, W)
#         x = x.permute(0, 2, 1, 3) 
#         # 此时形状变成 (B, 128, 512, 256)
        
#         # 2. 直接进行 2D 卷积
#         x = self.conv_layers(x) 
        
#         # 最终输出完美符合 U-Net 要求的格式: (B, 64通道, 512, 256)
#         return x


class FastFusionModel(nn.Module):
    """
    完整融合模型：编码器 + smp 提供的 U-Net
    """
    def __init__(self, angle_bins: int = 256, doppler_channels: int = 128):
        super().__init__()
        self.angle_bins = angle_bins
        
        # 分位数列表，用于估计背景噪声的分布
        self.quantiles = [0.1, 0.5, 0.9] 
        num_quantiles = len(self.quantiles)
        
        # 1. doppler编码器qudiao
        
        # self.encoder = RadarEncoder(in_doppler_bins=128, out_channels=cnn_out_channels)
        
        # 2. 【核心优化】直接使用 smp 库的 U-Net
        
        self.unet = smp.Unet(
            encoder_name="resnet18",      # 使用轻量级的 resnet18 作为主干提取特征
            encoder_weights="imagenet",         
            in_channels=doppler_channels, # 输入通道数等于doppler
            # 输出通道数 = 1个占据栅格预测(Occupancy) + 3个分位数预测(Quantiles)
            classes=1 + num_quantiles     
        )

    def forward(self, radar_cube: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1. (B, R, D, A) -> (B, D, R, A)
        # encoded = self.encoder(radar_cube)
        radar_cube = radar_cube.permute(0, 2, 1, 3)
        # 2. 丢进 U-Net：输出 (B, 4, R, A)
        unet_out = self.unet(radar_cube)
        
        # 3. 把 U-Net 的输出一分为二
        # 第 0 个通道是 Occupancy（预测有没有障碍物）
        occupancy_logits = unet_out[:, 0, :, :]  # (B, R, A)
        # 经过 Sigmoid 变成 0~1 之间的概率
        occupancy = torch.sigmoid(occupancy_logits) 
        
        # 第 1 到 3 个通道是 Quantiles（预测背景噪声）
        quantile_logits = unet_out[:, 1:, :, :]  # (B, 3, R, A)
        # 【核心优化】使用 softplus 激活函数！
        # 确保预测出来的噪声能量永远是正数，不然下面做除法会算出负的离谱分数
        quantiles = F.softplus(quantile_logits)
        quantiles = rearrange(quantiles, 'b q r a -> b r a q') # 把分位数放到最后一维方便取用
        
        # 4. 计算雷达原始能量图（对多普勒维度求平方平均）
        ra_energy = torch.mean(radar_cube ** 2, dim=1)
        
        # 5. 提取 (0.9) 对应的背景噪声估计
        median_idx = self.quantiles.index(0.9)
        background_est = quantiles[..., median_idx]
        
        # 6. 计算检测分数：信号能量 / (背景噪声 + 极小值防止除以0)
        # 也就是经典的 CFAR（恒虚警率）物理逻辑
        detection_score = ra_energy / (background_est + 1e-8)
        
        return {
            'occupancy': occupancy,
            'quantiles': quantiles,
            'background_est': background_est,
            'detection_score': detection_score,
            'occupancy_logits':occupancy_logits,
            'ra_energy': ra_energy
        }


# ==================== 简单测试一下 ====================
if __name__ == "__main__":
    print("正在初始化极简版雷达融合模型...")
    model = FastFusionModel()
    
    
    
    print("正在进行前向传播推理...")
    outputs = model(dummy_input)
    
    print("推理成功！输出结果维度如下：")
    print(f"  障碍物概率 (Occupancy): {outputs['occupancy'].shape}")
    print(f"  背景噪声 (Quantiles)  : {outputs['quantiles'].shape}")
    print(f"  最终检测分数          : {outputs['detection_score'].shape}")