# Thesis-Shujianjia 实验日志

本文件长期记录 `Thesis-Shujianjia` 方向的关键实验、代码状态、结论和下一步。每次出现新的训练、测试、sweep、可视化或结果日志时，都应追加到本文件；如果结果改变项目级理解，再同步更新同目录下的 `project.md`。

## 2026-06-21 - Thesis-Shujianjia test2d 2-layer UNet Alpha Sweep 更新

**运行设置**

- 更新脚本：`/scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/test2d.py`
- 目标 checkpoint：来自 `train2d.py` 的最新 2-layer UNet checkpoint
- 默认模型参数：`CustomUNet2Layer`
- Alpha sweep：`1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0`
- 输出 CSV：帧级 `test2d_metrics.csv` 和 alpha 级 `summary_test2d_metrics.csv`

**实现**

- 在 `test2d.py` 中注册 `CustomUNet2Layer`，并把它设为默认推理模型。
- 用 `data_preparation.get_default_params()` 的值替换硬编码的 range/azimuth 坐标构造。
- 在 Chamfer distance 计算前，增加坐标轴和裁剪后预测图的形状检查。
- 保留语义分割 + CFAR 后处理流程，并按请求 sweep alpha。

**验证**

- `python -m py_compile /scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/test2d.py` 通过。
- `python test2d.py --help` 在默认 shell 中失败，因为该 Python 缺少 `numpy`。
- 在 `condaenv` 中运行同一个 help 检查耗时过长后被中断；没有运行数据集或 checkpoint 推理。

**下一步**

- 在项目 conda 环境中，或通过 HPC job，使用训练好的 2-layer UNet checkpoint 运行 `test2d.py`，然后检查帧级 CSV 和 alpha summary CSV。

## 2026-06-21 - Thesis-Shujianjia 批量推理 Job 设置

**运行设置**

- 更新批处理脚本：`/scratch/shujianjia/project/thesis/test1.sh`
- 更新推理脚本：`/scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/test2d.py`
- 计划批量 checkpoint：
  - `2layer_20260617_epoch31`：`checkpoints/run_20260617_141331/best_epoch_31_loss_0.0119.pth`
  - `2layer_20260618_epoch19`：`checkpoints/run_20260618_223327/best_epoch_19_loss_0.0106.pth`
  - `2layer_20260619_epoch28`：`checkpoints/run_20260619_231422/best_epoch_28_loss_0.0962.pth`
- 所有计划 checkpoint 的模型类：`CustomUNet2Layer`
- Alpha sweep 保持为 `1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0`

**实现**

- `test1.sh` 在一个 4 小时 Slurm job 内顺序运行多个 model/checkpoint 组合。
- 每次运行写入 `results/batch_<jobid>_<timestamp>/<run_label>/`。
- 每个运行目录包含 `run_config.txt`、帧级 `test2d_metrics.csv` 和 alpha 级 `summary_test2d_metrics.csv`。
- 批处理脚本把所有 per-run summary 合并到 batch 根目录下的 `all_models_summary.csv`。
- `test2d.py` 在每个 CSV 行中记录 `Run`、`Model` 和 `Checkpoint`，并默认抑制逐帧 metric 日志。

**验证**

- `python -m py_compile /scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/test2d.py` 通过。
- `bash -n /scratch/shujianjia/project/thesis/test1.sh` 通过。
- 本次对话没有运行完整推理；应通过 Slurm 提交。

**下一步**

- 提交 `sbatch /scratch/shujianjia/project/thesis/test1.sh`，然后比较三个 checkpoint 的 `all_models_summary.csv`。

## 2026-06-21 - 批量推理输出命名清理

**运行设置**

- 更新脚本：`/scratch/shujianjia/project/thesis/test1.sh`
- 受影响的计划推理输出：
  - `results/2layer-20260617-141331/`
  - `results/2layer-20260618-223327/`
  - `results/2layer-20260619-231422/`

**实现**

- 移除 `results/batch_<jobid>_<timestamp>/...` 输出根目录。
- 每个 checkpoint 现在直接写入 `results/` 下由 checkpoint 语义命名的目录。
- 移除合并生成 `all_models_summary.csv` 的步骤。
- 每次运行只在自己的输出目录里保留 per-run CSV 文件。

**验证**

- `bash -n /scratch/shujianjia/project/thesis/test1.sh` 通过。
- 本次对话没有运行完整推理。

**下一步**

- 提交 `sbatch /scratch/shujianjia/project/thesis/test1.sh`，并分别比较每个运行目录中的 `summary_test2d_metrics.csv`。

## 2026-06-21 - 训练脚本快照与推理 Profile 列

**运行设置**

- 更新训练脚本：`/scratch/shujianjia/project/thesis/train1.sh`
- 更新训练代码：`/scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/train2d.py`
- 更新推理代码：`/scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/test2d.py`

**实现**

- `train2d.py` 接受 `--launch_script`，并把该 shell 脚本复制到 checkpoint run 目录中，命名为 `launch_script.sh`。
- `train2d.py` 还会在同一个 checkpoint run 目录写入 `train_args.txt`，包含命令行和解析后的训练参数。
- `train1.sh` 现在通过 `--launch_script` 传入 `/scratch/shujianjia/project/thesis/train1.sh`。
- `test2d.py` 默认对每个加载的模型 profile 一次，并把 profile 列追加到已有 metrics CSV 行中。
- 新增 profile 列：`Profile_Params`、`Profile_MACs`、`Profile_FLOPs` 和 `Profile_PeakMemoryMB`。

**验证**

- `python -m py_compile /scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/train2d.py` 通过。
- `python -m py_compile /scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/test2d.py` 通过。
- `bash -n /scratch/shujianjia/project/thesis/train1.sh` 通过。
- 本次对话没有运行训练或推理。

**下一步**

- 运行下一次训练 job，确认生成的 checkpoint 目录包含 `launch_script.sh` 和 `train_args.txt`；然后运行推理并检查 `test2d_metrics.csv` 中新增的 profile 列。

## 2026-06-22 - test1 可视化加新推理

**运行设置**

- 更新批处理脚本：`/scratch/shujianjia/project/thesis/test1.sh`
- 更新可视化脚本：`/scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/testvisual2d.py`
- 计划可视化任务：
  - `2layer-20260607-230629`，checkpoint `run_20260607_230629/best_epoch_46_loss_0.0155.pth`，alpha `2.0`，输出 `results/2layer-20260607-230629/`
  - `2layer-20260617_141331`，checkpoint `run_20260617_141331/best_epoch_31_loss_0.0119.pth`，alpha `3.5`，输出 `results/2layer-20260617_141331/`
- 计划推理任务：
  - `2layer-20260621-180733`，checkpoint `run_20260621_180733/best_epoch_48_loss_0.0789.pth`，输出 `results/2layer-20260621-180733/`

**实现**

- `testvisual2d.py` 现在接受 `--model` 和 `--alpha`。
- `testvisual2d.py` 现在直接把图片写入传入的 `--output_dir`，不再创建硬编码嵌套 visualization 文件夹。
- `test1.sh` 现在先运行两个可视化任务，再运行一个新的完整推理任务。

**验证**

- `python -m py_compile /scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/testvisual2d.py` 通过。
- `bash -n /scratch/shujianjia/project/thesis/test1.sh` 通过。
- 本次对话没有运行完整可视化或推理。

**下一步**

- 提交 `sbatch /scratch/shujianjia/project/thesis/test1.sh`，并检查两个已有结果文件夹中的 PNG，以及 `results/2layer-20260621-180733/` 中的 CSV。

## 2026-06-29 - 2D 三方法对比可视化脚本

**运行设置**

- 新增脚本：`/scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/visual_compare2d.py`
- 烟测 checkpoint：`/scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/checkpoints/run_20260621_180733/best_epoch_48_loss_0.0789.pth`
- 烟测模型：`CustomUNet2Layer`
- 烟测输出目录：`/scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/results/visual_compare2d_smoke`

**实现**

- 新脚本按固定 layout 保存相机、雷达能量、CA-CFAR g5t5、Proposed method、Ignacionetwork 和 LiDAR GT。
- Proposed method 使用 checkpoint 网络输出估计局部背景噪声后再检测，默认 `alpha=2.0`。
- CA-CFAR 使用 `CACFAR_train5_guard5`，默认 threshold factor 为 `1.5`。
- Ignacionetwork 默认读取 Scene 对应的 `rosDS/network2d`，按当前帧 `cfar_path` 的 timestamp 对齐。
- 输出帧图和 `visual_compare2d_metrics.csv`，CSV 记录 scene、frame、三种方法 Chamfer、输入路径和输出路径。

**验证**

- `python -m py_compile /scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/visual_compare2d.py` 通过。
- `python visual_compare2d.py --help` 在 `condaenv` 中通过。
- CPU 烟测 1 帧通过，生成 `scene_6_frame_1.png` 和 `visual_compare2d_metrics.csv`。
- 烟测帧指标：CFAR Chamfer `3.4891`，Proposed Chamfer `1.6521`，Ignacionetwork Chamfer `15.2030`。

**下一步**

- 用目标 checkpoint 对 Scene6 批量生成 PNG；筛选有价值帧后用 `--frames ... --output_format pdf` 反查并重画 PDF。

## 2026-07-03 - visual_compare2d 坐标翻转与字号修正

**运行设置**

- 更新脚本：`/scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/visual_compare2d.py`
- 参考逻辑：`/scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/radelft/visualizers/visualizer2D_matplotlib.py`
- 烟测 checkpoint：`/scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/checkpoints/run_20260621_180733/best_epoch_48_loss_0.0789.pth`
- 烟测输出目录：`/scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/results/visual_compare2d_flip_smoke`

**实现**

- 绘图时将 radar energy、CA-CFAR、Proposed method、Ignacionetwork 和 LiDAR GT 的横轴统一取反，以匹配当前相机显示方向。
- Ignacionetwork 点云不再直接使用前两列作为 `(x, y)`；改为参考旧 visualizer，先做 `azimuth_offset` transform，再用 `(pc[:, 1], pc[:, 0])` 作为 BEV 横纵坐标。
- 移除 Ignacionetwork 额外的 `y` 取反，避免点云向右偏斜。
- 放大总标题、小图标题、坐标轴标签、刻度和 colorbar 字号，并增大 scatter 点大小。

**验证**

- `python -m py_compile /scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/visual_compare2d.py` 通过。
- CPU 烟测 1 帧通过，生成 `scene_6_frame_1.png`。
- 目视检查烟测图：数据面板已水平翻转，Ignacionetwork 方向与 LiDAR/其他方法更一致，标题和坐标字号可读性提升。

**下一步**

- 用目标 checkpoint 批量重画候选帧；如需论文图，可用 `--frames ... --output_format pdf` 输出矢量版。

## 2026-07-03 - visual_compare2d 顶部版式优化

**运行设置**

- 更新脚本：`/scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/visual_compare2d.py`
- 烟测 checkpoint：`/scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/checkpoints/run_20260621_180733/best_epoch_48_loss_0.0789.pth`
- 烟测输出目录：`/scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/results/visual_compare2d_layout_smoke`

**实现**

- 相机图改为 `aspect="auto"`，去掉左上区域内图片左右两侧的大块空白，让照片尽量铺满三列空间。
- 所有点云/雷达坐标轴设置居中 anchor。
- Radar energy 的 colorbar 缩窄并减小 pad，让右上角图不再被 colorbar 和 constrained layout 明显挤到右边。

**验证**

- `python -m py_compile /scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/visual_compare2d.py` 通过。
- CPU 烟测 1 帧通过，生成 `scene_6_frame_1.png`。
- 目视检查烟测图：相机区域已铺满左上空间，radar energy 面板更居中紧凑。

**下一步**

- 对目标 checkpoint 批量重画；若后续仍觉得右上角 radar energy 太独立，可进一步改成独立宽度比例或把 colorbar 放到面板内侧。

## 2026-07-03 - visual_compare2d 相机比例与右上面板对齐修正

**运行设置**

- 更新脚本：`/scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/visual_compare2d.py`
- 烟测 checkpoint：`/scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/checkpoints/run_20260621_180733/best_epoch_48_loss_0.0789.pth`
- 烟测输出目录：`/scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/results/visual_compare2d_aspect_smoke`

**实现**

- 将顶部行高度比例从 `0.78` 增加到 `0.95`，让相机图上下更高，减少横向拉伸感。
- 将 radar energy 的 colorbar 改为面板内嵌 inset，不再由外置 colorbar 改变右上主坐标轴尺寸。
- 保持 radar energy 和右下 LiDAR GT 面板同一列宽度与边界对齐。

**验证**

- `python -m py_compile /scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/visual_compare2d.py` 通过。
- CPU 烟测 1 帧通过，生成 `scene_6_frame_1.png`。
- 目视检查烟测图：相机比例更协调，右上 radar energy 与右下 GT 面板对齐。

**下一步**

- 批量重画候选帧；如果希望照片完全无形变，下一步应改为固定比例裁剪而不是 `aspect="auto"` 拉伸。

## 2026-07-04 - Jupyter 交互式调图 notebook

**运行设置**

- 新增 notebook：`/scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/plot.ipynb`
- 复用脚本：`/scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/visual_compare2d.py`
- 默认 checkpoint：`/scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/checkpoints/run_20260621_180733/best_epoch_48_loss_0.0789.pth`
- 默认测试场景：Scene 6

**实现**

- notebook 在 Jupyter 中加载并 reload `visual_compare2d.py`，避免复制后逻辑漂移。
- 提供 `show_frame(...)` 单帧渲染函数，可直接调 `frame`、`top_height`、`camera_aspect`、`figsize`、`colorbar_inside`、`dpi` 和保存格式。
- 提供可选 `ipywidgets` frame slider；环境没有 `ipywidgets` 时可手动调用 `show_frame(...)`。
- 支持在 notebook 中保存调好的 PNG/PDF 到 `results/visual_compare2d_notebook`。

**验证**

- `python -m json.tool /scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/plot.ipynb` 通过。
- 本次没有在 Jupyter 内实际执行 notebook。

**下一步**

- 在交互式计算节点启动 Jupyter，打开 `plot.ipynb`，先运行配置和加载单元，再用 `show_frame(scene=6, frame=...)` 调整候选帧版式。

## 2026-07-04 - plot.ipynb 帧号查找容错

**运行设置**

- 更新 notebook：`/scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/plot.ipynb`
- 问题：`show_frame(scene=6, frame=1)` 在当前 notebook session 中触发 `KeyError: (6, 1)`。

**实现**

- 增加 `available_frames(scene=None)`，可查看每个 scene 的可用 frame。
- 增加 `resolve_dataset_index(...)`，支持 `frame=None` 默认取该 scene 第一帧，也支持直接传 `dataset_index`。
- 当 frame 不存在时，错误信息会打印 scene 的可用范围、总数和最近帧号，不再只有裸 `KeyError`。
- notebook 加载数据后会打印每个 scene 的 frame 范围，方便确认当前配置实际加载了哪些帧。

**验证**

- `python -m json.tool /scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/plot.ipynb` 通过。
- 本次没有在 Jupyter 内执行 notebook。

**下一步**

- 在 Jupyter 中重新运行加载 cell 和函数定义 cell；先用 `show_frame(scene=6, frame=None)` 验证第一帧，再按打印出的 frame 范围指定具体帧号。

## 2026-07-04 - plot.ipynb 显式显示 Matplotlib 图

**运行设置**

- 更新 notebook：`/scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/plot.ipynb`
- 问题：在 Jupyter cell 中调用 `fig, axd, frame_data = show_frame(...)` 或 `fig.show()` 后没有图像输出。

**实现**

- 在 notebook 中导入 `IPython.display.display`。
- 在导入并 reload `visual_compare2d.py` 后切回 inline backend。
- 给 `show_frame(...)` 增加 `display_inline=True` 参数，默认在函数内部 `display(fig)`。
- 示例调用显式传入 `display_inline=True`，避免赋值语句吞掉图像显示。

**验证**

- `python -m json.tool /scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/plot.ipynb` 通过。
- 本次没有在 Jupyter 内执行 notebook。

**下一步**

- 在当前 Jupyter session 中重新运行 import cell 和函数定义 cell，然后调用 `show_frame(scene=6, frame=None, display_inline=True)`。

## 2026-07-04 - visual_compare2d 全局帧索引与 Ignacio 去旋转

**运行设置**

- 更新脚本：`/scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/visual_compare2d.py`
- 烟测 checkpoint：`/scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/checkpoints/run_20260617_141331/best_epoch_31_loss_0.0119.pth`
- 烟测输出目录：`/scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/results/visual_compare2d_global_smoke`

**实现**

- Ignacio 点云读取后不再额外应用 `azimuth_offset=7` 度旋转，仅保留 `(pc[:, 1], pc[:, 0])` 的 BEV 坐标映射。
- 新增 `--global_indices` 参数，表示拼接后的 test dataset 0-based 全局样本序号。
- 保留 `--frames` 作为每个 scene 内部 `Pow_Frame_xxx` 选择；如果同时传 `--frames` 和 `--global_indices`，脚本会报错避免语义混淆。
- 对 `--global_indices` 增加范围检查，合法范围是 `0..len(dataset)-1`。

**验证**

- `python -m py_compile /scratch/shujianjia/project/thesis/Thesis-Shujianjia/code/visual_compare2d.py` 通过。
- `python visual_compare2d.py --help` 显示新增 `--global_indices`。
- CPU 烟测 `--test_scenes 2 6 --global_indices 1796` 只保存 1 张图。
- 0-based 全局索引 `1796` 映射为 `scene_6_frame_97.png`，符合 Scene2 1700 帧 + Scene6 后续帧的拼接顺序。

**下一步**

- 使用 `--global_indices 1796 1926 2189 3332 121 82 743` 重画候选帧；观察 Ignacio 去掉 7 度旋转后的 BEV 对齐是否更好。
