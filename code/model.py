

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SingleConv(nn.Module):
    """UNet3+ 中用于对齐通道数的单次卷积"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class RadarResUNet3Plus(nn.Module):
    def __init__(self, n_doppler=128, out_dim=16, backbone='resnet18', pretrained=True):
        super(RadarResUNet3Plus, self).__init__()
        
        # ==========================================
        # 1. Doppler 投影器 (Stem 模块) - 保持你的神来之笔
        # ==========================================
        self.doppler_proj = nn.Sequential(
            nn.Conv2d(n_doppler, 3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # ==========================================
        # 2. Encoder 模块 (ResNet) - 砍掉一层，只保留到 enc3 (1/16)
        # ==========================================
        base_model = getattr(models, backbone)(pretrained=pretrained)
        
        # ResNet18 特征维度: enc0(64), enc1(64), enc2(128), enc3(256)
        self.enc0 = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu) # 1/2
        self.maxpool = base_model.maxpool # 1/4
        self.enc1 = base_model.layer1 # 1/4
        self.enc2 = base_model.layer2 # 1/8
        self.enc3 = base_model.layer3 # 1/16 
        # [哲学体现: 砍掉 layer4，避免过度下采样丢失雷达小目标]

        # ==========================================
        # 3. UNet3+ Decoder 模块 (全尺度特征融合)
        # 核心设计：每个融合节点的每个输入分支都被统一压缩到 C=64，拼接后为 64*4=256
        # ==========================================
        cat_ch = 64  # 统一对齐通道数
        up_ch = cat_ch * 4 # 拼接后的总通道数

        # --- D3 (1/16) ---
        self.d3_conv = SingleConv(256, up_ch) # 基础语义层

        # --- D2 (1/8) 融合层 ---
        self.d2_e0 = nn.Sequential(nn.MaxPool2d(4), SingleConv(64, cat_ch))
        self.d2_e1 = nn.Sequential(nn.MaxPool2d(2), SingleConv(64, cat_ch))
        self.d2_e2 = SingleConv(128, cat_ch)
        self.d2_d3 = SingleConv(up_ch, cat_ch)
        self.d2_fusion = DoubleConv(up_ch, up_ch)

        # --- D1 (1/4) 融合层 ---
        self.d1_e0 = nn.Sequential(nn.MaxPool2d(2), SingleConv(64, cat_ch))
        self.d1_e1 = SingleConv(64, cat_ch)
        self.d1_d2 = SingleConv(up_ch, cat_ch)
        self.d1_d3 = SingleConv(up_ch, cat_ch)
        self.d1_fusion = DoubleConv(up_ch, up_ch)

        # --- D0 (1/2) 融合层 ---
        self.d0_e0 = SingleConv(64, cat_ch)
        self.d0_d1 = SingleConv(up_ch, cat_ch)
        self.d0_d2 = SingleConv(up_ch, cat_ch)
        self.d0_d3 = SingleConv(up_ch, cat_ch)
        self.d0_fusion = DoubleConv(up_ch, up_ch)

        # ==========================================
        # 4. 恢复到原分辨率并投影
        # ==========================================
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = DoubleConv(up_ch, 32)
        self.projector = nn.Conv2d(32, out_dim, kernel_size=1)

    def forward(self, x):
        if x.dim() == 5: # [B, 1, R, A, D]
            x = x.squeeze(1) # [B, R, A, D]
            
        # [B, 128, R, A] -> [B, 3, R, A]
        x_rgb = self.doppler_proj(x)
        
        # ---------- Encoder ----------
        e0 = self.enc0(x_rgb)            # [B, 64, R/2, A/2]
        e1 = self.enc1(self.maxpool(e0)) # [B, 64, R/4, A/4]
        e2 = self.enc2(e1)               # [B, 128, R/8, A/8]
        e3 = self.enc3(e2)               # [B, 256, R/16, A/16]

        # ---------- Decoder (UNet3+) ----------
        # D3 (1/16)
        d3 = self.d3_conv(e3)

        # D2 (1/8)
        d2_e0 = self.d2_e0(e0)
        d2_e1 = self.d2_e1(e1)
        d2_e2 = self.d2_e2(e2)
        d2_d3 = self.d2_d3(F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True))
        d2 = self.d2_fusion(torch.cat([d2_e0, d2_e1, d2_e2, d2_d3], dim=1))

        # D1 (1/4)
        d1_e0 = self.d1_e0(e0)
        d1_e1 = self.d1_e1(e1)
        d1_d2 = self.d1_d2(F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True))
        d1_d3 = self.d1_d3(F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=True))
        d1 = self.d1_fusion(torch.cat([d1_e0, d1_e1, d1_d2, d1_d3], dim=1))

        # D0 (1/2)
        d0_e0 = self.d0_e0(e0)
        d0_d1 = self.d0_d1(F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=True))
        d0_d2 = self.d0_d2(F.interpolate(d2, scale_factor=4, mode='bilinear', align_corners=True))
        d0_d3 = self.d0_d3(F.interpolate(d3, scale_factor=8, mode='bilinear', align_corners=True))
        d0 = self.d0_fusion(torch.cat([d0_e0, d0_d1, d0_d2, d0_d3], dim=1))

        # ---------- Final Projection ----------
        out = self.final_up(d0)          # 恢复到 [B, C, R, A]
        out = self.final_conv(out)       # 降维到 32
        z = self.projector(out)          # 投射到隐空间 [B, out_dim, R, A]
        
        return z

# 测试模型
if __name__ == "__main__":
    # 假设 RDA 是 256x256x32
    model = RadarResUNet3Plus(n_doppler=32, out_dim=16)
    dummy_input = torch.randn(2, 1, 256, 256, 32)
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出特征图形状: {output.shape}") # 应该是 [2, 16, 256, 256]