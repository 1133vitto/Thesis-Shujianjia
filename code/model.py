"""
models
"""

import torch
import torchvision
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


class old2DModel(nn.Module):
    """
    
    input:(B, Range, Doppler, Azimuth) - (B, 512, 128, 256)
    Extract Max Doppler Power and corresponding Index as 2D feature input
    """
    def __init__(self, in_channels: int = 2, encoder_name: str = "resnet18"):
        super().__init__()
        
       
        #in_channels=2: Channel 0 是 Max Power, Channel 1 是 Normalized Doppler Index
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
            'ra_energy': max_power,                # (B, 1,R, A)
            'max_indices': max_indices             # (B,1, R, A)
        }






class MaxPower2DModel(nn.Module):
    """
    
    input:(B, Range, Doppler, Azimuth) - (B, 512, 128, 256)
    Extract Max Doppler Power and corresponding Index as 2D feature input
    """
    def __init__(self, in_channels: int = 2, encoder_name: str = "resnet18"):
        super().__init__()
        
       
        # in_channels=2: Channel 0 是 Max Power, Channel 1 是 Normalized Doppler Index
        # self.unet = smp.UnetPlusPlus(
        #     encoder_name=encoder_name,
        #     encoder_weights="imagenet",
        #     in_channels=in_channels, 
        #     classes=1 
        # )
        self.unet = CustomUNetPlusPlus(
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
            'ra_energy': max_power,                # (B, 1,R, A)
            'max_indices': max_indices             # (B,1, R, A)
        }





#customize model
class ConvBlock(nn.Module):
    """
    (Conv + BN + ReLU) * 2
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x

class DecoderNode(nn.Module):
    """
    U-Net++ decoder:  upsampling features from the lower layer and concatenating with skip connections from the encoder and previous decoder nodes
    """
    def __init__(self, up_in_channels, skip_channels_list, out_channels):
        super().__init__()
        # 
        total_in_channels = up_in_channels + sum(skip_channels_list)
        self.block = ConvBlock(total_in_channels, out_channels)

    def forward(self, up_x, skip_xs):
        up_x = F.interpolate(up_x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat(skip_xs + [up_x], dim=1)
        return self.block(x)

class CustomResNet18Encoder(nn.Module):
    """提取 ResNet18 的 5 个阶段特征，并修改第一层以支持 2 通道输入"""
    def __init__(self, in_channels=2, pretrained=True):
        super().__init__()
        base_model = torchvision.models.resnet18(pretrained=pretrained)
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                self.conv1.weight.copy_(base_model.conv1.weight[:, :in_channels, :, :])
                
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

    def forward(self, x):
        features = []
        # Stage 0: (B, 64, H/2, W/2)
        x0 = self.relu(self.bn1(self.conv1(x)))
        features.append(x0) 
        
        # Stage 1: (B, 64, H/4, W/4)
        x1 = self.layer1(self.maxpool(x0))
        features.append(x1)
        
        # Stage 2: (B, 128, H/8, W/8)
        x2 = self.layer2(x1)
        features.append(x2)
        
        # Stage 3: (B, 256, H/16, W/16)
        x3 = self.layer3(x2)
        features.append(x3)
        
        # Stage 4: (B, 512, H/32, W/32)
        x4 = self.layer4(x3)
        features.append(x4)
        
        return features # 返回 5 个层级的特征字典 [x0_0, x1_0, x2_0, x3_0, x4_0]

class CustomUNetPlusPlus(nn.Module):
    def __init__(self, in_channels=2, classes=1):
        super().__init__()
        self.encoder = CustomResNet18Encoder(in_channels=in_channels, pretrained=True)
        
        # ResNet18 各层输出通道数: [64, 64, 128, 256, 512]
        # 定义解码层输出通道数 (可以根据需要调整，这里仿照标准结构)
        ch = [32, 64, 128, 256] 
        
        # ---------------------------------------------------------
        # Naming rules: node_{i}_{j}, i represents depth (0 is the shallowest), j represents horizontal progress
        # ---------------------------------------------------------
        
        # L1: The first column of intermediate nodes
        self.node_0_1 = DecoderNode(up_in_channels=64, skip_channels_list=[64], out_channels=ch[0])
        self.node_1_1 = DecoderNode(up_in_channels=128, skip_channels_list=[64], out_channels=ch[1])
        self.node_2_1 = DecoderNode(up_in_channels=256, skip_channels_list=[128], out_channels=ch[2])
        self.node_3_1 = DecoderNode(up_in_channels=512, skip_channels_list=[256], out_channels=ch[3])

        # L2: The second column of intermediate nodes
        self.node_0_2 = DecoderNode(up_in_channels=ch[1], skip_channels_list=[64, ch[0]], out_channels=ch[0])
        self.node_1_2 = DecoderNode(up_in_channels=ch[2], skip_channels_list=[64, ch[1]], out_channels=ch[1])
        self.node_2_2 = DecoderNode(up_in_channels=ch[3], skip_channels_list=[128, ch[2]], out_channels=ch[2])

        # L3: The third column of intermediate nodes
        self.node_0_3 = DecoderNode(up_in_channels=ch[1], skip_channels_list=[64, ch[0], ch[0]], out_channels=ch[0])
        self.node_1_3 = DecoderNode(up_in_channels=ch[2], skip_channels_list=[64, ch[1], ch[1]], out_channels=ch[1])

        # L4: The fourth column of final nodes
        self.node_0_4 = DecoderNode(up_in_channels=ch[1], skip_channels_list=[64, ch[0], ch[0], ch[0]], out_channels=ch[0])

        # Final split header (restore H/2, W/2 to H, W)
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final_conv = nn.Conv2d(ch[0], classes, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        x0_0, x1_0, x2_0, x3_0, x4_0 = features
        
        
        # Column 1
        x0_1 = self.node_0_1(up_x=x1_0, skip_xs=[x0_0])
        x1_1 = self.node_1_1(up_x=x2_0, skip_xs=[x1_0])
        x2_1 = self.node_2_1(up_x=x3_0, skip_xs=[x2_0])
        x3_1 = self.node_3_1(up_x=x4_0, skip_xs=[x3_0])

        # Column 2
        x0_2 = self.node_0_2(up_x=x1_1, skip_xs=[x0_0, x0_1])
        x1_2 = self.node_1_2(up_x=x2_1, skip_xs=[x1_0, x1_1])
        x2_2 = self.node_2_2(up_x=x3_1, skip_xs=[x2_0, x2_1])

        # Column 3
        x0_3 = self.node_0_3(up_x=x1_2, skip_xs=[x0_0, x0_1, x0_2])
        x1_3 = self.node_1_3(up_x=x2_2, skip_xs=[x1_0, x1_1, x1_2])

        # Column 4
        x0_4 = self.node_0_4(up_x=x1_3, skip_xs=[x0_0, x0_1, x0_2, x0_3])

        # 恢复到原分辨率
        out = self.final_up(x0_4)
        out = self.final_conv(out)
        
        return out

    def freeze_backbone(self):
        """
        freeze
        """
        print("正在冻结 ResNet18 的layer1 to layer4")
        
        for layer in [self.encoder.layer1, self.encoder.layer2, self.encoder.layer3, self.encoder.layer4]:
            for param in layer.parameters():
                param.requires_grad = False
                
        

    def unfreeze_backbone(self):
        """
        unfreeze
        """
        # print("正在解冻 ResNet18 全网络...")
        for param in self.encoder.parameters():
            param.requires_grad = True