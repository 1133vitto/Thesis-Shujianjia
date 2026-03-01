
# 雷达-LiDAR 融合检测项目

> 深度学习与传统 CFAR 混合的毫米波雷达目标检测，使用 LiDAR 点云作为跨模态监督

---

## 📁 项目结构（已整理！）

```
radar_lidar_project/
├── README.md               # 本文档
├── train.py                # 主训练脚本 ✨
├── src/
│   ├── data_preprocessing.py  # 数据预处理模块 ✨
│   ├── model.py               # 模型架构、数据集
│   ├── losses.py              # Loss Functions
│   └── evaluation.py          # 评估、点云保存、可视化 ✨
├── checkpoints/            # 模型保存目录
├── results/                # 结果保存目录（点云、指标、可视化）
├── data/                   # 数据目录
└── configs/                # 配置文件目录
```

---

## 🚀 快速开始

### 1. 环境要求
```bash
pip install torch torchvision numpy matplotlib scikit-learn
# 可选（用于点云处理）：
pip install open3d
```

### 2. 运行训练
```bash
cd radar_lidar_project
python train.py --batch_size 4 --num_epochs 20
```

### 3. 查看结果
结果会保存在 `results/` 目录：
- `final_predicted_points.npy` - 最终预测点云
- `pd_pfa_curve.png` - Pd-Pfa 曲线
- `metrics.txt` - 性能指标
- `vis_batch_*.png` - 检测可视化图

---

## 🧩 模块说明

### `src/data_preprocessing.py` - 数据预处理
- `RadarCubeProcessor` - 雷达数据规范化、对数压缩、杂波去除
- `LidarPointCloudProcessor` - 点云投影、体素化、过滤
- `DataAugmentation` - 数据增强
- `FileLoader` - 文件加载与保存

### `src/model.py` - 模型架构
- 完整的 Radar-LiDAR 融合模型
- Doppler 编码器 + U-Net 多任务头
- 数据集类

### `src/losses.py` - Loss Functions
- Focal Loss - 处理不平衡
- Quantile/Pinball Loss - 噪声分位数预测（对应 CFAR）⭐
- Chamfer Loss - 点云对齐
- Consistency Loss - 2D-3D 一致性

### `src/evaluation.py` - 评估与点云保存
- `PointCloudGenerator` - 从检测结果生成点云 ✨
- `DetectionMetrics` - Pd/Pfa, F1, AP, Chamfer Distance ✨
- `ResultSaver` - 保存点云、指标、可视化 ✨
- `Evaluator` - 综合评估器 ✨

---

## 📊 输出文件

训练结束后，你会得到：

| 文件 | 说明 |
|------|------|
| `checkpoints/model_best.pth` | 最佳模型权重 |
| `results/final_predicted_points.npy` | 预测点云 |
| `results/final_predicted_points.ply` | PLY 格式点云（可选） |
| `results/pd_pfa_curve.png` | Pd-Pfa 曲线图 |
| `results/metrics.txt` | 性能指标数值 |
| `results/vis_batch_*.png` | 检测结果可视化 |

---

## 🔬 关于 radelft 仓库

> ⚠️ **注意**：由于环境网络/权限限制，暂时未能直接克隆 radelft 仓库。
>
> 但我已经帮你完成了完整的项目结构，包含：
> - 数据预处理
> - 网络架构
> - Loss Functions
> - 训练脚本
> - 评估指标
> - 点云生成与保存
>
> 如果你能提供 radelft 仓库的完整链接，或者告诉我需要复用其中的哪些具体模块，我可以进一步帮你整合！

---

## 💡 核心创新点

1. **Quantile/Pinball Loss 对应 CFAR 理论** - 你的想法！
2. **LiDAR 跨模态监督**
3. **完整的点云生成与保存** - 新增！
4. **综合评估指标（Pd/Pfa + Chamfer）** - 新增！

---

## 📝 下一步建议

1. 先跑通当前训练脚本（模拟数据）
2. 根据你的真实数据格式，修改数据加载部分
3. 如果有 radelft 代码，告诉我需要复用哪些部分

---

## 🔗 相关文件（旧版保留参考）

- `../radar_lidar_fusion_losses.py`
- `../radar_lidar_fusion_model.py`
- `../train_radar_lidar_fusion.py`
- `../INNOVATION_IDEAS.md` - 创新想法必读！

