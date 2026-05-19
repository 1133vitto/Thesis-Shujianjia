# project.md — Thesis-Shujianjia: Semantic Segmentation + CFAR Post-Processing

## 核心思想

**语义分割 + CFAR 后处理** —— 最接近传统深度学习方法的方案。网络是一个标准的语义分割模型，输入 2D 雷达特征图（Max Power + Normalized Doppler Index），输出每像素的 occupancy probability（0~1 之间）。推理时，用 `1 - occupancy_prob` 作为背景概率，对雷达能量加权后做局部平均来估计噪声基底，最后用经典 CFAR 判决：`radar_energy > α × local_bg_noise`。

**设计哲学**：将检测问题视为分割问题，让网络学习目标/背景的二分类。CFAR 在后处理阶段引入，**不在训练计算图中**，训练和推理是解耦的。

## 与另外两个 trial 的本质差异

| 维度 | Thesis-Shujianjia | trial2-dist (PIN-CFAR) | trial3-HDD |
|------|-------------------|----------------------|------------|
| 方法论 | 语义分割 + 后处理 CFAR | 物理衰减 + 可导 CFAR | 高维特征 + L2 阈值 |
| 网络输出 | occupancy_prob ∈ [0,1] | P_bg + Λ | z ∈ R³² |
| CFAR 是否可导 | 否（后处理） | 是（在计算图中） | 无 CFAR |
| 模型架构 | 可插拔（3 种 U-Net 变体） | 固定 Dual-Head U-Net | 固定 Doppler-ResUNet |
| 输入通道 | 2（MaxPower + DopplerIdx） | 4（mean/max + sin/cos） | 128（全 Doppler 谱） |

## 数据流

### 输入
```python
# RaDelftWrapper 处理:
# input_cube: (2, 128, 512, 256) → power_cube: (128, 512, 256)
# transpose → (512, 128, 256) = (Range, Doppler, Azimuth)
# MaxPower2DModel: max over Doppler → (B, 2, 512, 256)
# Channel 0: max_power
# Channel 1: max_doppler_index / 127.0 (归一化)
radar_input: (B, 2, 512, 256)
```

### GT 处理
```python
occupancy_target = np.squeeze(gt_cube)  # (34, 500, 240) → 直接 squeeze
# 训练时在 train2d.py 中: occupancy_target, _ = torch.max(occupancy_target, dim=1)
```

### 输出
```python
occupancy_prob:  (B, 500, 240)  # 裁剪后 [:-12, 8:-8]，sigmoid 激活
ra_energy:       (B, 1, 500, 240)  # max_power
max_indices:     (B, 500, 240)  # Doppler index
```

### 推理后处理（非可导）
```python
pred_bg = 1.0 - occupancy_prob                    # 背景概率
bgenergy = pred_bg * ra_energy                     # 加权能量
local_bg_noise = avg_pool2d(bgenergy, 5×5)        # 局部噪声估计
detection = ra_energy > (2.0 * local_bg_noise)     # CFAR 判决
```

## 模型动物园

通过 `MaxPower2DModel` 包装器，可切换四种 U-Net 变体。通过 `--model` 参数选择：

### 1. FastFusionModel
- 使用 `smp.Unet`（segmentation_models_pytorch）
- 输入: 128 通道（全 Doppler），permute RDA → DRA
- 最早的实验模型

### 2. CustomUNet
- 标准 U-Net，ResNet18 encoder + 4 层 Decoder + skip connections
- 最简洁的架构

### 3. CustomUNetPlusPlus (默认)
- U-Net++，稠密 skip connections（嵌套的 DecoderNode）
- 4 列中间节点，共 10 个 DecoderNode
- 更强的特征融合能力

### 4. CustomUNet3Plus
- UNet 3+，全尺度 skip connections
- 每个 Decoder 层聚合来自 **所有** Encoder 层和 **所有** 更浅 Decoder 层的特征
- 使用 MaxPool（下采样）和 Upsample（上采样）统一分辨率
- 参数最多，表达能力最强

## 损失函数：RadarFusionLoss

```
Total Loss = w_focal × FocalLoss + w_dice × DiceLoss + w_cfar × SoftCFARLoss
```

### Focal Loss
- 处理正负样本极度不平衡（目标像素 << 背景像素）
- `α=0.95`（正样本权重极高）, `γ=2.0`

### Dice Loss
- 直接优化预测与 GT 的重叠度
- 对目标尺度不敏感

### Soft CFAR Loss（可选，默认 weight=0）
- 可导 CFAR 的软近似
- `detection_score = radar_energy - α × local_bg_mean`
- `soft_detection = sigmoid(β × detection_score)`
- 用 BCE 监督 soft_detection 与 occupancy_target

**默认配置**: `weight_focal=1.0, weight_dice=0.1, weight_cfar=0.0`

## 训练流程

### 训练
```bash
cd /scratch/shujianjia/project/thesis/Thesis-Shujianjia/code

# 本地直接训练
python train2d.py --model CustomUNetPlusPlus --batch_size 8 --num_epochs 50 --lr 1e-4

# 或使用 FINAL_TRAIN_SCRIPT.py（几乎相同）

# HPC
sbatch /scratch/shujianjia/project/thesis/train1.sh
```

### 测试
```bash
cd /scratch/shujianjia/project/thesis/Thesis-Shujianjia/code
python test2d.py --checkpoint_path <path_to_checkpoint.pth> --model CustomUNet

# HPC
sbatch /scratch/shujianjia/project/thesis/test1.sh
```

### 关键超参数
- `model`: 模型选择 (`CustomUNet`, `CustomUNetPlusPlus`, `CustomUNet3Plus`)
- `in_channels=2`: Max Power + Normalized Doppler Index
- `lr=1e-4`, `AdamW`, `CosineAnnealingLR`
- `seed=42`: 固定随机种子确保可复现

## 关键文件

| 文件 | 用途 |
|------|------|
| `code/model.py` | `FastFusionModel`, `MaxPower2DModel`, `CustomUNet/PlusPlus/3Plus`, `CustomResNet18Encoder`, `ConvBlock`, `DecoderNode`, `Unet3ScaleConv` |
| `code/losses.py` | `RadarFusionLoss` (Focal + Dice + SoftCFAR), `StableFocalLoss`, `DiceLoss`, `SoftCFARLoss`, `MaskedQuantileLoss` |
| `code/train2d.py` | 训练主脚本（含 `RaDelftWrapper`） |
| `code/FINAL_TRAIN_SCRIPT.py` | 备用训练脚本（与 train2d.py 几乎相同，少 wandb） |
| `code/test2d.py` | 测试脚本（含 Chamfer Distance 评估） |
| `code/testvisual2d.py` | 2D 检测可视化 |
| `code/evaluation.py` | Evaluator 类 |
| `code/inference.py` | 推理脚本 |
| `code/radelft/` | 共享的 RaDelft 数据加载库 |

## 注意事项

1. **模型通过 `--model` 参数在 `MODEL_REGISTRY` 中查找**，添加新模型需同时注册
2. **GT 形状处理**: `RaDelftWrapper` 中 `np.squeeze(gt_cube)` 依赖 gt_cube 的 elevation 维度已经是 1，否则会出错。这是历史遗留的不健壮写法
3. **Soft CFAR Loss 默认不启用**（`weight_cfar=0.0`），因为它在训练早期容易导致数值不稳定
4. **后处理 CFAR 与训练解耦**：训练时优化 occupancy 分割，推理时独立做 CFAR 判决。这意味着更好的 occupancy 预测不一定带来更好的检测结果
