import torch
import torch.nn as nn
import torchvision.models as models

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

class RadarResUNet(nn.Module):
    def __init__(self, n_doppler=128, out_dim=16, backbone='resnet18', pretrained=True):
        super(RadarResUNet, self).__init__()
        
        # ==========================================
        # 1. 你的神来之笔：Doppler 投影器 (Stem 模块)
        # 作用：1x1 卷积，不改变 Range-Azimuth 空间大小
        # 仅将 D 个多普勒通道“聪明地”压缩为 3 个伪 RGB 通道
        # ==========================================
        self.doppler_proj = nn.Sequential(
            nn.Conv2d(n_doppler, 3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # ==========================================
        # 2. 原封不动地加载预训练 ResNet
        # ==========================================
        base_model = getattr(models, backbone)(pretrained=pretrained)
        
        # Encoder 各阶段 (直接调用，0魔改)
        # self.first_conv = nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.enc0 = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu) # 1/2
        self.maxpool = base_model.maxpool # 1/4
        self.enc1 = base_model.layer1 # 1/4
        self.enc2 = base_model.layer2 # 1/8
        self.enc3 = base_model.layer3 # 1/16
        self.enc4 = base_model.layer4 # 1/32

        # ==========================================
        # 3. Decoder 模块 (手写上采样，解耦特征融合)
        # ==========================================
        # ResNet18 通道数: [64, 64, 128, 256, 512]
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up4 = DoubleConv(512 + 256, 256)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up3 = DoubleConv(256 + 128, 128)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up2 = DoubleConv(128 + 64, 64)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up1 = DoubleConv(64 + 64, 64)
        
        self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up0 = DoubleConv(64, 32)

        # Final Projector: 映射到你构思的高维隐空间 (out_dim 维)
        self.projector = nn.Conv2d(32, out_dim, kernel_size=1)

    def forward(self, x):
        # 假设输入 x shape: [B, 1, R, A, D] 或者 [B, R, A, D]
        if x.dim() == 5: # [B, 1, R, A, D]
            x = x.squeeze(1) # 变成 [B, R, A, D]
            
        # 1. 多普勒降维：[B, 128, R, A] -> [B, 3, R, A]
        x_rgb = self.doppler_proj(x)
        
        # 2. Encoder: 提取空间特征
        x0 = self.enc0(x_rgb)    # [B, 64, R/2, A/2]
        x1 = self.enc1(self.maxpool(x0)) # [B, 64, R/4, A/4]
        x2 = self.enc2(x1)    # [B, 128, R/8, A/8]
        x3 = self.enc3(x2)    # [B, 256, R/16, A/16]
        x4 = self.enc4(x3)    # [B, 512, R/32, A/32]

        # 3. Decoder: 逐层融合
        d4 = self.up4(x4)
        d4 = torch.cat([d4, x3], dim=1)
        d4 = self.conv_up4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.conv_up3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, x1], dim=1)
        d2 = self.conv_up2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, x0], dim=1)
        d1 = self.conv_up1(d1)

        d0 = self.up0(d1)
        d0 = self.conv_up0(d0)

        # 4. 投射到统计隐空间
        z = self.projector(d0) # [B, out_dim, R, A]
        return z

# 测试模型
if __name__ == "__main__":
    # 假设 RDA 是 256x256x32
    model = RadarResUNet(n_doppler=32, out_dim=16)
    dummy_input = torch.randn(2, 1, 256, 256, 32)
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出特征图形状: {output.shape}") # 应该是 [2, 16, 256, 256]