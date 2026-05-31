import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Simple conv block: Conv2d + BN + ReLU. Used in stem and transitions."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """ResNet bottleneck with expansion=4. Used only in Stage 1."""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        mid_channels = out_channels // self.expansion

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.use_shortcut = (stride != 1) or (in_channels != out_channels)
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x) if self.use_shortcut else x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        return self.relu(out + identity)


class BasicBlock(nn.Module):
    """ResNet basic block (no bottleneck). Used in Stages 2-3."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.use_shortcut = (stride != 1) or (in_channels != out_channels)
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x) if self.use_shortcut else x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        return self.relu(out + identity)


class HighResolutionModule(nn.Module):
    """Multi-resolution cross-fusion."""
    def __init__(self, num_branches, channels_list):
        super().__init__()
        self.num_branches = num_branches
        self.channels_list = channels_list

        self.fuse_layers = nn.ModuleList()
        for i in range(num_branches):
            row = nn.ModuleList()
            for j in range(num_branches):
                in_ch = channels_list[j]
                out_ch = channels_list[i]

                if j == i:
                    row.append(nn.Identity())
                elif j > i:
                    scale = 2 ** (j - i)
                    layers = [
                        nn.Conv2d(in_ch, out_ch, 1, bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)
                    ]
                    row.append(nn.Sequential(*layers))
                else:  # j < i
                    layers = []
                    for k in range(i - j):
                        conv_in = in_ch if k == 0 else out_ch
                        layers.extend([
                            nn.Conv2d(conv_in, out_ch, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(out_ch),
                            nn.ReLU(inplace=True),
                        ])
                    row.append(nn.Sequential(*layers))
            self.fuse_layers.append(row)

    def forward(self, x):
        out = []
        for i in range(self.num_branches):
            fused = self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                fused = fused + self.fuse_layers[i][j](x[j])
            out.append(F.relu(fused))
        return out


class HRNetV1_W18(nn.Module):
    """
    HRNetV1-W18, Pruned for Radar Physics & High-Efficiency.
    
    去掉了毫无物理意义的 Stage 4 (144通道分支)，深度止于 Stage 3 (72通道)。
    极大降低算力负荷，避免过度平滑，像素级守护雷达目标的边界。

    Input:  radar_cube (B, 128, 512, 256)
    Output: dict with 'occupancy_prob': (B, 512, 256)
    """
    def __init__(self):
        super().__init__()

        # ---- Step 0: Doppler compression ----
        self.doppler_conv = nn.Conv2d(128, 3, kernel_size=1, stride=1)

        # ---- Step 1: Stem ----
        self.stem = nn.Sequential(
            ConvBlock(6, 64, 3, 2),  
            ConvBlock(64, 64, 3, 1),
        )

        # ---- Step 2: Stage 1 (4x Bottleneck) ----
        stage1_blocks = [Bottleneck(64, 256)]
        for _ in range(3):
            stage1_blocks.append(Bottleneck(256, 256))
        self.stage1 = nn.Sequential(*stage1_blocks)

        # ---- Step 3: Transition 1 ----
        self.transition1 = nn.ModuleList([
            ConvBlock(256, 18, 3, 1),    # Branch 0 (hr): 256x128
            ConvBlock(256, 36, 3, 2),    # Branch 1 (mr): 128x64
        ])

        # ---- Stage 2 branch blocks ----
        self.stage2_hr_blocks = nn.Sequential(*[BasicBlock(18, 18) for _ in range(4)])
        self.stage2_mr_blocks = nn.Sequential(*[BasicBlock(36, 36) for _ in range(4)])
        self.stage2_fusion = HighResolutionModule(2, [18, 36])

        # ---- Step 5: Transition 2 ----
        self.transition2 = ConvBlock(36, 72, 3, 2)  # Branch 2 (lr): 64x32

        # ---- Stage 3 branch blocks ----
        self.stage3_hr_blocks = nn.Sequential(*[BasicBlock(18, 18) for _ in range(4)])
        self.stage3_mr_blocks = nn.Sequential(*[BasicBlock(36, 36) for _ in range(4)])
        self.stage3_lr_blocks = nn.Sequential(*[BasicBlock(72, 72) for _ in range(4)])
        
        # Stage 3 融合模块 (深度终点站)
        self.stage3_fusions = nn.ModuleList([
            HighResolutionModule(3, [18, 36, 72]) for _ in range(4)
        ])

        # ✂️ 以前写在这里的 Transition 3 和 Stage 4 模块已被冷酷斩杀

        # ---- Step 9: V1 Final Head (物理锚点对齐) ----
        # 直接提取 Stage 3 结束后的高分辨率分支（18通道，256x128），拉伸回原图尺寸
        self.upsample_hr = nn.Upsample(size=(512, 256), mode='bilinear', align_corners=False)
        
        # 20通道融合：18个语义通道 + 2个原始物理通道（Max/Mean）
        self.final_conv = nn.Conv2d(20, 1, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        # 原始特征提取与归一化
        x_max = torch.max(x, dim=1, keepdim=True)[0]       
        x_mean = torch.mean(x, dim=1, keepdim=True)       
        x_argmax = torch.argmax(x, dim=1, keepdim=True).float() / 127.0  

        # 头部注入
        x_conv = self.doppler_conv(x)                     
        x = torch.cat([x_conv, x_max, x_mean, x_argmax], dim=1)  

        # 骨干网络前向传播
        x = self.stem(x)                                    
        x = self.stage1(x)                                  

        hr = self.transition1[0](x)                         
        mr = self.transition1[1](x)                         

        # Stage 2
        hr, mr = self.stage2_fusion([hr, mr])
        hr = self.stage2_hr_blocks(hr)
        mr = self.stage2_mr_blocks(mr)

        # Transition 2
        lr = self.transition2(mr)                           

        # Stage 3 (现在的核心深度阶段)
        for fusion in self.stage3_fusions:
            hr, mr, lr = fusion([hr, mr, lr])
            hr = self.stage3_hr_blocks(hr)
            mr = self.stage3_mr_blocks(mr)
            lr = self.stage3_lr_blocks(lr)

        # ✂️ 以前在这里的 Stage 4 循环迭代已被连根拔起

        # 尾部高分辨率对齐与最终输出
        hr_upsampled = self.upsample_hr(hr)                # (B, 18, 512, 256)
        out = torch.cat([hr_upsampled, x_max, x_mean], dim=1) # (B, 20, 512, 256)
        
        out = self.final_conv(out)                         
        out = out.squeeze(1)                                
        out = torch.sigmoid(out)                            

        return {'occupancy_prob': out}