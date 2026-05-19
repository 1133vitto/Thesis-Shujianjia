"""
distvisual.py
=============
使用训练好的 UNet3+ 模型，对测试集前 N 帧进行推理，计算纯净背景功率
(bgenergy = (1 - occupancy_prob) * ra_energy)，并绘制目标/非目标区域的
功率分布直方图。

与 trial2-dist 的 distvisual.py 不同，这里的分布是基于模型输出的纯净背景，
而非原始雷达功率。
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

current_dir = Path(__file__).resolve().parent
radelft_dir = current_dir / "radelft"
sys.path.insert(0, str(radelft_dir))
sys.path.insert(0, str(current_dir))

from radelft.loaders.rad_cube_loader import RADCUBE_DATASET
from radelft.data_preparation.data_preparation import get_default_params
from model import CustomUNet3Plus, MaxPower2DModel


class RaDelftWrapper(Dataset):
    """与 train2d.py 保持一致的 RaDelft 数据集封装"""

    def __init__(self, mode='train', params=None):
        self.real_dataset = RADCUBE_DATASET(mode=mode, params=params)

    def __len__(self):
        return len(self.real_dataset)

    def __getitem__(self, idx):
        input_cube, gt_cube, item_params = self.real_dataset[idx]

        power_cube = input_cube  # (128, 512, 256)
        power_cube = np.transpose(power_cube, (1, 0, 2))  # (512, 128, 256)

        occupancy_target = np.squeeze(gt_cube)  # (500, 240)

        return {
            'radar_cube': torch.from_numpy(power_cube).float(),
            'occupancy_target': torch.from_numpy(occupancy_target).float(),
            'metadata': item_params,
        }


def plot_distribution_histograms(target_values, non_target_values, frame_idx, save_path,
                                  xlabel='Power'):
    """绘制目标和非目标区域的功率分布直方图（左右分列），y 轴对数尺度。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    all_values = np.concatenate([target_values, non_target_values])
    vmin = np.percentile(all_values, 0.01) if len(all_values) > 0 else 0
    vmax = np.percentile(all_values, 99.99) if len(all_values) > 0 else 1
    bins = np.linspace(vmin, vmax, 100)

    axes[0].hist(target_values, bins=bins, color='red', alpha=0.7, edgecolor='darkred')
    axes[0].set_title(f'Target (n={len(target_values)})')
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel('Count')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(non_target_values, bins=bins, color='blue', alpha=0.7, edgecolor='darkblue')
    axes[1].set_title(f'Non-target (n={len(non_target_values)})')
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel('Count')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f'Frame {frame_idx}: Clean Background Power Distribution', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_combined_histogram(target_values, non_target_values, frame_idx, save_path,
                             xlabel='Power'):
    """将目标和非目标绘制在同一张图上。"""
    fig, ax = plt.subplots(figsize=(10, 6))

    all_values = np.concatenate([target_values, non_target_values])
    if len(all_values) == 0:
        print(f"  Warning: No data for frame {frame_idx}, skipping combined plot")
        plt.close(fig)
        return

    vmin = np.percentile(all_values, 0.01)
    vmax = np.percentile(all_values, 99.99)
    bins = np.linspace(vmin, vmax, 100)

    ax.hist(target_values, bins=bins, color='red', alpha=0.6,
            label=f'Target (n={len(target_values)})', edgecolor='darkred')
    ax.hist(non_target_values, bins=bins, color='blue', alpha=0.6,
            label=f'Non-target (n={len(non_target_values)})', edgecolor='darkblue')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Count (log scale)', fontsize=12)
    ax.set_title(f'Frame {frame_idx}: Clean Background Power Distribution', fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_theoretical_distributions(save_path):
    """绘制理想指数分布（背景噪声）和莱斯分布（目标回波）的理论曲线。"""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.linspace(0, 5, 1000)

    lambda_exp = 1.0
    pdf_exp = lambda_exp * np.exp(-lambda_exp * x)

    sigma_rice = 0.5
    nu_rice = 1.5
    pdf_rice = (x / sigma_rice**2) * np.exp(-(x**2 + nu_rice**2) / (2 * sigma_rice**2)) * \
               np.i0(x * nu_rice / sigma_rice**2)
    pdf_rice = pdf_rice / pdf_rice.max() * pdf_exp.max()

    ax.plot(x, pdf_exp, color='blue', linewidth=2.5, label='Exponential (Background Noise)')
    ax.plot(x, pdf_rice, color='red', linewidth=2.5, label='Rician (Target)')

    ax.set_xlabel('Power', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('Theoretical CFAR Distributions', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, None)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='使用 UNet3+ 模型计算纯净背景功率并绘制分布图')
    parser.add_argument('--checkpoint_path', type=str,
                        default='checkpoints/run_20260502_024001/best_epoch_10_loss_0.0654.pth',
                        help='模型权重路径')
    parser.add_argument('--output_dir', type=str, default='results/dist',
                        help='输出目录')
    parser.add_argument('--num_frames', type=int, default=10,
                        help='处理的帧数')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备 (cuda/cpu)')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA 不可用，回退到 CPU")
        args.device = 'cpu'

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 加载数据集 ----
    params = get_default_params()
    params["dataset_path"] = '/scratch/shujianjia/dataset/'
    params["train_val_scenes"] = [1, 3, 4, 5, 7]
    params["test_scenes"] = [2, 6]
    params["bev"] = True

    test_dataset = RaDelftWrapper(mode='test', params=params)
    print(f"测试集大小: {len(test_dataset)}")

    # ---- 加载模型 ----
    model = MaxPower2DModel(model=CustomUNet3Plus, in_channels=2).to(args.device)
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"模型已加载: {args.checkpoint_path}")

    # ---- 绘制理论分布图（仅一次） ----
    plot_theoretical_distributions(os.path.join(args.output_dir, "theoretical_distributions.png"))

    # ---- 逐帧推理并绘图 ----
    print("=" * 60)
    print("纯净背景功率分布分析 - 测试集前 {} 帧".format(args.num_frames))
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)

    with torch.no_grad():
        for frame_idx in tqdm(range(min(args.num_frames, len(test_dataset))),
                               desc="Processing frames"):
            batch_data = test_dataset[frame_idx]

            radar_cube = batch_data['radar_cube'].unsqueeze(0).to(args.device)
            occupancy_target = batch_data['occupancy_target'].numpy()

            outputs = model(radar_cube)

            occupancy_prob = outputs['occupancy_prob']  # (1, 512, 256)
            ra_energy = outputs['ra_energy']             # (1, 1, 512, 256)

            # 裁剪: Range [:-12], Azimuth [8:-8]
            occupancy_prob = occupancy_prob[ :-12, 8:-8]       # (1, 500, 240)
            ra_energy = ra_energy[:, :, :-12, 8:-8]              # (1, 1, 500, 240)

            # 纯净背景: bgenergy = (1 - occupancy_prob) * ra_energy
            pred_bg = 1.0 - occupancy_prob                       # (1, 500, 240)
            bgenergy = pred_bg * ra_energy.squeeze(1)            # (1, 500, 240)

            bgenergy_np = bgenergy.squeeze(0).cpu().numpy()      # (500, 240)
            ra_energy_np = ra_energy.squeeze().cpu().numpy()  # (500, 240)

            # 按 GT 划分目标 / 非目标
            target_mask = occupancy_target > 0.5
            non_target_mask = ~target_mask

            target_values = ra_energy_np[target_mask].flatten()
            non_target_values = bgenergy_np[non_target_mask].flatten()

            print(f"\nFrame {frame_idx}:")
            print(f"  Target pixels: {len(target_values)}, "
                  f"mean={target_values.mean():.6f}, std={target_values.std():.6f}")
            print(f"  Non-target pixels: {len(non_target_values)}, "
                  f"mean={non_target_values.mean():.6f}, std={non_target_values.std():.6f}")

            # 分别绘制直方图
            plot_distribution_histograms(
                bgenergy_np.flatten(), ra_energy_np.flatten(), frame_idx,
                os.path.join(args.output_dir, f"frame_{frame_idx:04d}_separate背景vs背景.png"),
                xlabel='Clean Background Power')

            # 合并绘制
            plot_combined_histogram(
                bgenergy_np.flatten(), ra_energy_np.flatten(), frame_idx,
                os.path.join(args.output_dir, f"frame_{frame_idx:04d}_combined背景vs背景.png"),
                xlabel='Clean Background Power')

    print("\n" + "=" * 60)
    print("分析完成!")
    print(f"结果保存在: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
