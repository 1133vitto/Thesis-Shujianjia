# project.md — trial4: HRNetV1-W18 Full-Resolution Semantic Segmentation

## 核心思想

**全分辨率 HRNetV1-W18 语义分割**。网络是 HRNetV1-W18，最高分辨率分支全程保持 512×256 原分辨率（不做 stem 降采样），这是使用 HRNet 而非 U-Net 的根本原因——一条通路始终不丢失空间细节。输入 3D 雷达 cube，第一层用 1×1 卷积压缩 Doppler 维度到 3 通道伪 RGB，之后作为 2D 特征图处理。输出每像素的 occupancy probability（0~1）。推理时直接以 0.5 为阈值二值化：`occupancy_prob > 0.5`。

**设计哲学**: HRNet 的多分辨率并行分支在保持高分辨率通路的同时，通过交叉融合获得多尺度感受野。V1 输出头只取最高分辨率分支，避免低分辨率分支上采样带来的空间模糊。推理简单直接，不做 CFAR 后处理。

## 与另外两个 trial 的本质差异

| 维度 | trial4 | trial2-dist (PIN-CFAR) | trial3-HDD |
|------|--------|----------------------|------------|
| 方法论 | HRNet 全分辨率分割 + 0.5 阈值 | 物理衰减 + 可导 CFAR | 高维特征 + L2 阈值 |
| 网络输出 | occupancy_prob ∈ [0,1] | P_bg + Λ | z ∈ R³² |
| 判决方式 | prob > 0.5 | CFAR（在计算图中） | L2 threshold |
| 模型架构 | HRNetV1-W18（全分辨率） | 固定 Dual-Head U-Net | 固定 Doppler-ResUNet |
| 输入 | 3D radar cube (1×1 conv 压缩 Doppler) | 4（mean/max + sin/cos） | 128（全 Doppler 谱） |

## 数据流

### 输入
```python
# RaDelftWrapper 处理:
# input_cube: (2, 128, 512, 256) → power_cube: (128, 512, 256)
# transpose → (512, 128, 256) = (Range, Doppler, Azimuth)
# HRNetV1_W18 内部: permute → (128, 512, 256) → Conv2d(128, 3, 1) → (3, 512, 256)
radar_cube: (B, 512, 128, 256)
```

### GT 处理
```python
occupancy_target = np.squeeze(gt_cube)  # (500, 240)
```

### 输出
```python
# HRNetV1_W18 前向:
occupancy_prob:  (B, 512, 256)  # 裁剪后 [:-12, 8:-8] → (B, 500, 240)，sigmoid 激活
```

### 推理（直接阈值）
```python
detection = occupancy_prob > 0.5  # (B, 500, 240) bool tensor
```

## 模型架构：HRNetV1-W18 (Full-Resolution)

### 分辨率链
```
Branch 0 (hr):  512×256  ← 全程保持全分辨率，最终直接输出
Branch 1 (mr):  256×128
Branch 2 (lr):  128×64
Branch 3 (vlr): 64×32
```

### 网络结构
| 阶段 | 组件 | 输出 |
|------|------|------|
| Doppler 压缩 | permute + Conv2d(128→3, k=1) | (B, 3, 512, 256) |
| Stem | ConvBlock(3→64) ×2, stride=1 | (B, 64, 512, 256) |
| Stage 1 | Bottleneck(64→256) ×4 | (B, 256, 512, 256) |
| Transition 1 | 拆分为 2 分支 | hr(18@512×256), mr(36@256×128) |
| Stage 2 | HighResolutionModule(2, [18,36]) ×1 + BasicBlock ×4 | 同上 |
| Transition 2 | 从 mr 降采样 | + lr(72@128×64) |
| Stage 3 | HighResolutionModule(3, [18,36,72]) ×4 + BasicBlock ×4 | 同上 |
| Transition 3 | 从 lr 降采样 | + vlr(144@64×32) |
| Stage 4 | HighResolutionModule(4, [18,36,72,144]) ×3 + BasicBlock ×4 | 同上 |
| **V1 Head** | **只取 Branch 0** → Conv2d(18→1, k=1) → sigmoid | (B, 1, 512, 256) |

### V1 vs V2
- **V1 (本 trial 使用)**: 只取最高分辨率分支输出，Conv2d(18→1,1)，无需上采样，空间精度最高
- **V2**: 四分支全上采样拼接(270ch)后投影，多尺度特征但引入插值模糊

## 损失函数：PixelWiseNPLoss

```
第一阶段 (epoch < num_epochs//2): Loss = 1 - P_d
  其中 P_d = (2 × Σ(preds × targets)) / (Σ(preds) + Σ(targets) + ε)
  这是可微的检测率近似，直接优化检测性能

第二阶段 (epoch >= num_epochs//2): Loss = (1 - P_d) + λ × |虚警像素数 - α_pixels|
  加入虚警控制正则项，约束背景区域的误检像素数
```

## 训练流程

### 训练
```bash
cd /scratch/shujianjia/project/thesis/trial4/code
python train2d.py --batch_size 8 --num_epochs 50 --lr 1e-4 --name experiment_name
```

### 测试
```bash
cd /scratch/shujianjia/project/thesis/trial4/code
python test2d.py --checkpoint_path <path_to_checkpoint.pth>
```

### 可视化
```bash
cd /scratch/shujianjia/project/thesis/trial4/code
python testvisual2d.py --checkpoint_path <path_to_checkpoint.pth> --max_frames 10
```

### 关键超参数
- `lr=1e-4`, `AdamW`, `CosineAnnealingLR`
- `batch_size=8`
- `num_epochs=50`（第 25 epoch 起加入正则项）
- `lambda_reg=0.1`, `alpha_pixels=0.0`
- `seed=42`: 固定随机种子确保可复现
- Soft targets: Gaussian blur (kernel=[1,5], sigma=[0.1, 2.0]) 沿 azimuth 方向

## 关键文件

| 文件 | 用途 |
|------|------|
| `code/model.py` | HRNetV1_W18, ConvBlock, Bottleneck, BasicBlock, HighResolutionModule |
| `code/losses.py` | PixelWiseNPLoss（监督损失 + 虚警正则项） |
| `code/train2d.py` | 训练主脚本（含 RaDelftWrapper, 半程正则化开关） |
| `code/test2d.py` | 测试脚本（含 Chamfer Distance 评估, CA-CFAR, OS-CFAR） |
| `code/testvisual2d.py` | 2D 检测可视化 |
| `code/radelft/` | 共享的 RaDelft 数据加载库 |

## 注意事项

1. **HRNet 全程保持 512×256 高分辨率**，显存消耗较大。如果 OOM，优先减小 batch_size
2. **PixelWiseNPLoss 接受 [0,1] 概率值**（非 logits），模型输出已过 sigmoid
3. **正则项在半程后开启**：前一半 epoch 只优化检测率，后一半加入虚警控制
4. **推理直接阈值 0.5**：不做 CFAR 后处理，简洁直接
