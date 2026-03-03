
# 如何使用 RaDelft 的代码

## ✅ 现状总结

我们已经：
1. 找到了 RaDelft 仓库：https://github.com/RaDelft/RaDelft-Dataset
2. 获取到了 RaDelft 的 `rad_cube_loader.py` 代码片段
3. 把我们的项目结构准备好了

---

## 🚀 下一步：你手动克隆 RaDelft 仓库

因为当前环境无法直接访问 GitHub，**请你在本地电脑上操作**：

### 步骤 1：克隆 RaDelft 仓库
```bash
git clone https://github.com/RaDelft/RaDelft-Dataset.git
```

### 步骤 2：复制 RaDelft 的代码到我们的项目
把 RaDelft 的 `machine_learning_python/` 目录下的内容，
复制到我们项目的 `radar_lidar_project/src/radelft/` 目录：

```
我们的项目/
├── src/
│   ├── radelft/              # 把 RaDelft 的内容放在这里！
│   │   ├── data_preparation/
│   │   ├── loaders/
│   │   ├── networks/
│   │   ├── utils/
│   │   └── visualizers/
│   ├── data_preprocessing.py  # 我们自己的（保留）
│   ├── model.py             # 我们自己的（保留！核心创新！）
│   ├── losses.py            # 我们自己的（保留！核心创新！）
│   ├── evaluation.py         # 我们自己的（保留）
│   └── radelft_rad_cube_loader.py  # RaDelft 代码片段（参考）
└── ...
```

---

## 🎯 整合策略

### 我们保留的核心创新（不要改！）
- `src/model.py` - 我们的网络架构（U-Net + 分位数输出）
- `src/losses.py` - 我们的 Loss Functions（Quantile/Pinball Loss 对应 CFAR）

### 我们直接复用 RaDelft 的
- `src/radelft/loaders/` - 数据加载
- `src/radelft/utils/` - 指标计算
- `src/radelft/visualizers/` - 可视化
- `src/radelft/data_preparation/` - 数据预处理

---

## 📝 修改我们的训练脚本

把 RaDelft 的代码放好后，修改 `train_with_radelft.py` 来：

```python
# 替换我们自己的 data loader
from radelft.loaders.rad_cube_loader import RADCUBE_DATASET

# 替换我们自己的 metrics
from radelft.utils.compute_metrics import ...

# 但保留我们自己的 model 和 losses！
from model import RadarLidarFusionModel
from losses import RadarLidarLoss
```

---

## 💡 为什么这样设计？

- **RaDelft 的代码**：数据加载、指标计算、可视化 — 这些是基础设施，直接复用！
- **我们的代码**：网络架构和 Loss Function — 这是我们的创新点，要保留！

这样结合：**RaDelft 的基础设施 + 我们的核心创新** = 完美！

