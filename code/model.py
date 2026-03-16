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





class FastFusionModel(nn.Module):
    """
    完整融合模型：编码器 + smp 提供的 U-Net
    """
    def __init__(self, angle_bins: int = 256, doppler_channels: int = 128):
        super().__init__()
        self.angle_bins = angle_bins
        
        
        
        self.unet = smp.Unet(
            encoder_name="resnet18",      # 使用轻量级的 resnet18 作为主干提取特征
            encoder_weights="imagenet",         
            in_channels=doppler_channels, # 输入通道数等于doppler
            
            classes=1   
        )

    def forward(self, radar_cube: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1. (B, R, D, A) -> (B, D, R, A)
        # encoded = self.encoder(radar_cube)
        radar_cube = radar_cube.permute(0, 2, 1, 3)
        # 2. 丢进 U-Net：输出 (B, 4, R, A)
        unet_out = self.unet(radar_cube)
        
        # 3. 把 U-Net 的输出一分为二
        # 第 0 个通道是 Occupancy（预测有没有障碍物）
        occupancy_logits = unet_out[:, 0, :, :]  # (B,1, R, A)
        # occupancy_logits = unet_out # (B, 1, R, A)
        # 经过 Sigmoid 变成 0~1 之间的概率
        occupancy = torch.sigmoid(occupancy_logits) 
        
       
        ra_energy = torch.max(radar_cube, dim=1).values
        
    
        
        return {
            'occupancy': occupancy,
            # 'quantiles': quantiles,
            # 'background_est': background_est,
            # 'detection_score': detection_score,
            'occupancy_logits':occupancy_logits,
            'ra_energy': ra_energy
        }


class MaxPower2DModel(nn.Module):
    """
    
    input:(B, Range, Doppler, Azimuth) - (B, 512, 128, 256)
    Extract Max Doppler Power and corresponding Index as 2D feature input
    """
    def __init__(self, in_channels: int = 2, encoder_name: str = "resnet18"):
        super().__init__()
        
       
        # in_channels=2: Channel 0 是 Max Power, Channel 1 是 Normalized Doppler Index
        self.unet = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=in_channels, 
            classes=1 
        )

    def forward(self, radar_cube: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        radar_cube: (B, R, D, A)
        """
        
        # max_power: (B, R, A), max_indices: (B, R, A)
        max_power, max_indices = torch.max(radar_cube, dim=2)
        max_power = max_power.unsqueeze(1) # (B, 1, R, A)
        
        # 
        max_indices_norm = (max_indices.float() / 127.0).unsqueeze(1) # (B, 1, R, A)
        
        # 3. (B, 2, R, A)
        x = torch.cat([max_power, max_indices_norm], dim=1)
        
        
        logits = self.unet(x) 
        logits = torch.sigmoid(logits)
        
        return {
            'occupancy_logits': logits.squeeze(), # (B, R, A)
            'ra_energy': max_power,                # (B, R, A)
            'max_indices': max_indices             # (B, R, A)
        }