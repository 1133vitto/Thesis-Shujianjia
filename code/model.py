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

    Takes N feature maps at different resolutions, fuses them via
    element-wise SUM, and outputs N feature maps at the same resolutions.

    Parameters:
        num_branches: number of parallel branches
        channels_list: list of channel counts per branch [c0, c1, ..., c_{N-1}]
    """
    def __init__(self, num_branches, channels_list):
        super().__init__()
        self.num_branches = num_branches
        self.channels_list = channels_list

        # fuse_layers[i][j]: transform input branch j → output resolution i
        self.fuse_layers = nn.ModuleList()
        for i in range(num_branches):
            row = nn.ModuleList()
            for j in range(num_branches):
                in_ch = channels_list[j]
                out_ch = channels_list[i]

                if j == i:
                    row.append(nn.Identity())

                elif j > i:
                    # j is coarser → upsample (j-i) times to reach resolution i
                    layers = [
                        nn.Conv2d(in_ch, out_ch, 1, bias=False),
                        nn.BatchNorm2d(out_ch),
                    ]
                    for _ in range(j - i):
                        layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
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
        """
        Args:
            x: list of N tensors at decreasing resolutions
        Returns:
            list of N tensors at the same resolutions, fused
        """
        out = []
        for i in range(self.num_branches):
            fused = self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                fused = fused + self.fuse_layers[i][j](x[j])
            out.append(F.relu(fused))
        return out


class HRNetV1_W18(nn.Module):
    """
    HRNetV1-W18, full-resolution version.

    The highest-resolution branch stays at the input resolution (512x256)
    throughout the entire network. No stem downsampling.
    V1 head uses only the highest-resolution branch for the final output.

    Input:  radar_cube (B, 512, 128, 256)  [Range, Doppler, Azimuth]
    Output: dict with 'occupancy_prob': (B, 512, 256), values in [0, 1]
    """

    def __init__(self):
        super().__init__()

        # ---- Step 0: Doppler compression ----
        # (B, 512, 128, 256) -> permute -> (B, 128, 512, 256) -> Conv1x1 -> (B, 3, 512, 256)
        self.doppler_conv = nn.Conv2d(128, 3, kernel_size=1, stride=1)

        # ---- Step 1: Stem (stride=1, keeps 512x256) ----
        self.stem = nn.Sequential(
            ConvBlock(3, 64, 3, 1),
            ConvBlock(64, 64, 3, 1),
        )

        # ---- Step 2: Stage 1 (4x Bottleneck) ----
        stage1_blocks = [Bottleneck(64, 256)]
        for _ in range(3):
            stage1_blocks.append(Bottleneck(256, 256))
        self.stage1 = nn.Sequential(*stage1_blocks)

        # ---- Step 3: Transition 1 (256ch -> 2 branches) ----
        self.transition1 = nn.ModuleList([
            ConvBlock(256, 18, 3, 1),    # Branch 0 (hr): 512x256
            ConvBlock(256, 36, 3, 2),    # Branch 1 (mr): 256x128
        ])

        # ---- Stage 2 branch blocks ----
        self.stage2_hr_blocks = nn.Sequential(*[BasicBlock(18, 18) for _ in range(4)])
        self.stage2_mr_blocks = nn.Sequential(*[BasicBlock(36, 36) for _ in range(4)])

        # ---- Step 4: Stage 2 (1x fusion module) ----
        self.stage2_fusion = HighResolutionModule(2, [18, 36])

        # ---- Step 5: Transition 2 (add 3rd branch from branch 1) ----
        self.transition2 = ConvBlock(36, 72, 3, 2)   # lr: 128x64

        # ---- Stage 3 branch blocks ----
        self.stage3_hr_blocks = nn.Sequential(*[BasicBlock(18, 18) for _ in range(4)])
        self.stage3_mr_blocks = nn.Sequential(*[BasicBlock(36, 36) for _ in range(4)])
        self.stage3_lr_blocks = nn.Sequential(*[BasicBlock(72, 72) for _ in range(4)])

        # ---- Step 6: Stage 3 (4x fusion modules) ----
        self.stage3_fusions = nn.ModuleList([
            HighResolutionModule(3, [18, 36, 72]) for _ in range(4)
        ])

        # ---- Step 7: Transition 3 (add 4th branch from branch 2) ----
        self.transition3 = ConvBlock(72, 144, 3, 2)  # vlr: 64x32

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
        # Only the highest-resolution branch (18ch @ 512x256) -> 1ch
        self.final_conv = nn.Conv2d(18, 1, kernel_size=1, stride=1)

    def forward(self, x):
        """
        Args:
            x: radar_cube (B, 512, 128, 256)
        Returns:
            {'occupancy_prob': (B, 512, 256)}
        """
        # Step 0: Doppler compression
        # x is already (B, D, R, A) from RADCUBE_DATASET (bev=True)
        x = self.doppler_conv(x)                    # Conv2d(128,3,1): (B,128,512,256) -> (B,3,512,256)

        # Step 1: Stem
        x = self.stem(x)                            # (B, 64, 512, 256)

        # Step 2: Stage 1
        x = self.stage1(x)                          # (B, 256, 512, 256)

        # Step 3: Transition 1 -> 2 branches
        hr = self.transition1[0](x)                 # (B, 18, 512, 256)
        mr = self.transition1[1](x)                 # (B, 36, 256, 128)

        # Step 4: Stage 2
        hr, mr = self.stage2_fusion([hr, mr])
        hr = self.stage2_hr_blocks(hr)
        mr = self.stage2_mr_blocks(mr)

        # Step 5: Transition 2 -> 3 branches
        lr = self.transition2(mr)                   # (B, 72, 128, 64)

        # Step 6: Stage 3 (4 fusion layers)
        for fusion in self.stage3_fusions:
            hr, mr, lr = fusion([hr, mr, lr])
            hr = self.stage3_hr_blocks(hr)
            mr = self.stage3_mr_blocks(mr)
            lr = self.stage3_lr_blocks(lr)

        # Step 7: Transition 3 -> 4 branches
        vlr = self.transition3(lr)                  # (B, 144, 64, 32)

        # Step 8: Stage 4 (3 fusion layers)
        for fusion in self.stage4_fusions:
            hr, mr, lr, vlr = fusion([hr, mr, lr, vlr])
            hr = self.stage4_hr_blocks(hr)
            mr = self.stage4_mr_blocks(mr)
            lr = self.stage4_lr_blocks(lr)
            vlr = self.stage4_vlr_blocks(vlr)

        # Step 9: V1 Head — only highest-resolution branch
        out = self.final_conv(hr)                   # (B, 1, 512, 256)
        out = out.squeeze(1)                        # (B, 512, 256)
        out = torch.sigmoid(out)                    # (B, 512, 256)

        return {'occupancy_prob': out}
