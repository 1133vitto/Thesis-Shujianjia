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
    """ResNet basic block (no bottleneck). Used in Stages 2-4."""
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
    """
    Multi-resolution cross-fusion.
    """
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
                    # j is coarser → upsample to reach resolution i
                    # 彻底干掉以前的 for 循环拼接，一次缩放解决战斗，这就是“好品味”
                    scale = 2 ** (j - i)
                    layers = [
                        nn.Conv2d(in_ch, out_ch, 1, bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)
                    ]
                    row.append(nn.Sequential(*layers))

                else:  # j < i
                    # j is finer → downsample (i-j) times
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
    HRNetV1-W18, Refactored for Pragmatic Memory Usage.
    
    Stem downsamples 2x to save immense memory. 
    Final output maps back to original resolution to strictly preserve userspace interface.

    Input:  radar_cube (B, 512, 128, 256)  [Range, Doppler, Azimuth]
    Output: dict with 'occupancy_prob': (B, 512, 256), values in [0, 1]
    """

    def __init__(self):
        super().__init__()

        # ---- Step 0: Doppler compression ----
        self.doppler_conv = nn.Conv2d(128, 3, kernel_size=1, stride=1)

        # ---- Step 1: Stem (stride=2, drops to 256x128) ----
        self.stem = nn.Sequential(
            ConvBlock(3, 64, 3, 2),  # <-- 手术刀落下的地方：空间降维 2x
            ConvBlock(64, 64, 3, 1),
        )

        # ---- Step 2: Stage 1 (4x Bottleneck) ----
        stage1_blocks = [Bottleneck(64, 256)]
        for _ in range(3):
            stage1_blocks.append(Bottleneck(256, 256))
        self.stage1 = nn.Sequential(*stage1_blocks)

        # ---- Step 3: Transition 1 (256ch -> 2 branches) ----
        self.transition1 = nn.ModuleList([
            ConvBlock(256, 18, 3, 1),    # Branch 0 (hr): 256x128
            ConvBlock(256, 36, 3, 2),    # Branch 1 (mr): 128x64
        ])

        # ---- Stage 2 branch blocks ----
        self.stage2_hr_blocks = nn.Sequential(*[BasicBlock(18, 18) for _ in range(4)])
        self.stage2_mr_blocks = nn.Sequential(*[BasicBlock(36, 36) for _ in range(4)])

        # ---- Step 4: Stage 2 (1x fusion module) ----
        self.stage2_fusion = HighResolutionModule(2, [18, 36])

        # ---- Step 5: Transition 2 (add 3rd branch from branch 1) ----
        self.transition2 = ConvBlock(36, 72, 3, 2)   # lr: 64x32

        # ---- Stage 3 branch blocks ----
        self.stage3_hr_blocks = nn.Sequential(*[BasicBlock(18, 18) for _ in range(4)])
        self.stage3_mr_blocks = nn.Sequential(*[BasicBlock(36, 36) for _ in range(4)])
        self.stage3_lr_blocks = nn.Sequential(*[BasicBlock(72, 72) for _ in range(4)])

        # ---- Step 6: Stage 3 (4x fusion modules) ----
        self.stage3_fusions = nn.ModuleList([
            HighResolutionModule(3, [18, 36, 72]) for _ in range(4)
        ])

        # ---- Step 7: Transition 3 (add 4th branch from branch 2) ----
        self.transition3 = ConvBlock(72, 144, 3, 2)  # vlr: 32x16

        # ---- Stage 4 branch blocks ----
        self.stage4_hr_blocks = nn.Sequential(*[BasicBlock(18, 18) for _ in range(4)])
        self.stage4_mr_blocks = nn.Sequential(*[BasicBlock(36, 36) for _ in range(4)])
        self.stage4_lr_blocks = nn.Sequential(*[BasicBlock(72, 72) for _ in range(4)])
        self.stage4_vlr_blocks = nn.Sequential(*[BasicBlock(144, 144) for _ in range(4)])

        # ---- Step 8: Stage 4 (3x fusion modules) ----
        self.stage4_fusions = nn.ModuleList([
            HighResolutionModule(4, [18, 36, 72, 144]) for _ in range(3)
        ])

        # ---- Step 9: V1 Final Head ----
        # 履行向后兼容契约：提取 1ch，拉伸回 512x256
        self.final_conv = nn.Sequential(
            nn.Conv2d(18, 1, kernel_size=1, stride=1),
            nn.Upsample(size=(512, 256), mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        """
        Args:
            x: radar_cube (B, 512, 128, 256)
        Returns:
            {'occupancy_prob': (B, 512, 256)}
        """
        # Step 0: Doppler compression
        x = self.doppler_conv(x)                    # (B,3,512,256)

        # Step 1: Stem
        x = self.stem(x)                            # (B, 64, 256, 128)  <- 显存债务在这里被砍掉 75%

        # Step 2: Stage 1
        x = self.stage1(x)                          # (B, 256, 256, 128)

        # Step 3: Transition 1
        hr = self.transition1[0](x)                 # (B, 18, 256, 128)
        mr = self.transition1[1](x)                 # (B, 36, 128, 64)

        # Step 4: Stage 2
        hr, mr = self.stage2_fusion([hr, mr])
        hr = self.stage2_hr_blocks(hr)
        mr = self.stage2_mr_blocks(mr)

        # Step 5: Transition 2
        lr = self.transition2(mr)                   # (B, 72, 64, 32)

        # Step 6: Stage 3
        for fusion in self.stage3_fusions:
            hr, mr, lr = fusion([hr, mr, lr])
            hr = self.stage3_hr_blocks(hr)
            mr = self.stage3_mr_blocks(mr)
            lr = self.stage3_lr_blocks(lr)

        # Step 7: Transition 3
        vlr = self.transition3(lr)                  # (B, 144, 32, 16)

        # Step 8: Stage 4
        for fusion in self.stage4_fusions:
            hr, mr, lr, vlr = fusion([hr, mr, lr, vlr])
            hr = self.stage4_hr_blocks(hr)
            mr = self.stage4_mr_blocks(mr)
            lr = self.stage4_lr_blocks(lr)
            vlr = self.stage4_vlr_blocks(vlr)

        # Step 9: V1 Head
        out = self.final_conv(hr)                   # (B, 1, 512, 256) <- 插值撑回原状
        out = out.squeeze(1)                        # (B, 512, 256)
        out = torch.sigmoid(out)                    # (B, 512, 256)

        return {'occupancy_prob': out}