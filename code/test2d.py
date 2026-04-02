import sys
import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import cdist
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree

# 路径设置，与你的训练脚本保持一致
current_dir = Path(__file__).resolve().parent
radelft_dir = current_dir / "radelft"
sys.path.insert(0, str(radelft_dir)) 
sys.path.insert(0, str(current_dir))

from model import MaxPower2DModel
from radelft.loaders.rad_cube_loader import RADCUBE_DATASET
from radelft.utils.compute_metrics import compute_pd_pfa
from radelft.data_preparation import data_preparation
from scipy.ndimage import distance_transform_edt
# ==========================================
# 辅助函数：计算 Chamfer Distance (2D)
# ==========================================
def compute_chamfer_distance_2d(gt_pc, pred_pc):
    """
    物理级倒角距离 (单位: 米)
    输入: gt_pc, pred_pc (N, 2) 形状的 numpy 数组，包含绝对物理坐标 [X, Y]
    速度: 使用 cKDTree，C语言底层实现，极速查询。
    """
    # 防御性编程：如果没有任何预测或真实目标，返回 NaN
    if len(gt_pc) == 0 or len(pred_pc) == 0:
        return np.nan

    # 1. 构建 KD 树 (空间索引建立)
    tree_gt = cKDTree(gt_pc)
    tree_pred = cKDTree(pred_pc)

    # 2. 查询最近邻距离 (米)
    # query 返回两个数组：距离数组，和对应的索引数组(这里用 _ 忽略)
    dist_pred_to_gt, _ = tree_gt.query(pred_pc)
    dist_gt_to_pred, _ = tree_pred.query(gt_pc)

    # 3. 提取均值并求倒角距离
    mean_dist_pred_to_gt = np.mean(dist_pred_to_gt)
    mean_dist_gt_to_pred = np.mean(dist_gt_to_pred)

    return (mean_dist_pred_to_gt + mean_dist_gt_to_pred)
# def compute_chamfer_distance_2d(gt_mask, pred_mask):
#     """
#     计算二值图上的倒角距离。
#     将 gt 和 pred 中 > 0 的像素作为点集，计算双向最小距离的平均值。
#     """
#     gt_points = np.argwhere(gt_mask > 0)
#     pred_points = np.argwhere(pred_mask > 0)
    
#     # 边界情况处理
#     if len(gt_points) == 0 and len(pred_points) == 0:
#         return 0.0
#     if len(gt_points) == 0 or len(pred_points) == 0:
#         return np.nan # 某一方完全没有预测出目标，标记为 NaN 并在外层过滤
        
#     # 计算所有点对的距离矩阵
#     dist_matrix = cdist(gt_points, pred_points)
    
#     # 双向最小距离求平均
#     dist_gt_to_pred = np.mean(np.min(dist_matrix, axis=1))
#     dist_pred_to_gt = np.mean(np.min(dist_matrix, axis=0))
    
#     return (dist_gt_to_pred + dist_pred_to_gt) / 2.0


class RaDelftTestWrapper(torch.utils.data.Dataset):
    """
    复用你的 Wrapper，专用于测试集
    """
    def __init__(self, mode='val', params=None):
        self.real_dataset = RADCUBE_DATASET(mode=mode, params=params)

    def __len__(self):
        return len(self.real_dataset)

    def __getitem__(self, idx):
        input_cube, gt_cube, item_params = self.real_dataset[idx]
        power_cube = input_cube[0]
        power_cube = np.transpose(power_cube, (1, 0, 2))
        occupancy_target = np.squeeze(gt_cube) 
        
        return {
            'radar_cube': torch.from_numpy(power_cube).float(),
            'occupancy_target': torch.from_numpy(occupancy_target).float(),
            'metadata': item_params  # 包含 scene / frame 信息
        }

# ==========================================
# 推理与可视化主函数
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='模型推理与可视化脚本')
    # 替换为你实际的最佳模型路径
    parser.add_argument('--checkpoint_path', type=str, 
                        default='./checkpoints/run_20260317_013246/best_epoch_11_loss_0.0118.pth')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    # 1. 创建输出目录
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    print("="*70)
    print("启动测试推理与可视化")
    print(f" 加载模型: {args.checkpoint_path}")
    print(f" 结果保存至: {args.output_dir}")
    print("="*70)

    # Range Axis
    range_cell_size = 0.1004
    # MATLAB: rangeCellSize:rangeCellSize:51.4242
    range_axis_full = np.arange(range_cell_size, 51.4242 + 1e-5, range_cell_size)
    # MATLAB 索引 11:end-2 对应 Python 索引 10:-2
    range_axis = range_axis_full[10:-2] 

    # Azimuth Axis
    angle_fft_size = 256
    # MATLAB: -pi:2*pi/(angleFFTSize-1):pi 
    wx_vec_full = np.linspace(-np.pi, np.pi, angle_fft_size)
    wx_vec_full = wx_vec_full[::-1] # flip
    # MATLAB 索引 9:248 对应 Python 索引 8:247
    wx_vec = wx_vec_full[8:248]
    # 防御性编程：避免因浮点精度导致超出 [-1, 1] 使得 arcsin 报错出现 NaN
    sin_theta = np.clip(wx_vec / (2 * np.pi * 0.4972), -1.0, 1.0)
    azimuth_axis = np.arcsin(sin_theta)



    # 2. 准备数据集 (使用 batch_size=1 以便逐帧画图)
    params = data_preparation.get_default_params()
    params["dataset_path"] = '/scratch/shujianjia/dataset/'
    params["train_val_scenes"] = [1, 3, 4, 5, 7]
    params["test_scenes"] = [2, 6]  # 仅测试集
    
    test_dataset = RaDelftTestWrapper(mode='test', params=params) # 或者 mode='test' 看你的 dataloader 定义
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    # 3. 初始化模型并加载权重
    model = MaxPower2DModel(in_channels=2).to(args.device)
    
    # 解析字典并加载权重
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()

    #CFAR 参数设置
    cfar_win_size = 7      # 
    cfar_guard_size = 3     # 

    cfar_kernel = torch.ones((1, 1, cfar_win_size, cfar_win_size), dtype=torch.float32, device=args.device)

    center = cfar_win_size // 2
    g_half = cfar_guard_size // 2
    cfar_kernel[:, :, center-g_half : center+g_half+1, center-g_half : center+g_half+1] = 0

    num_train_cells = cfar_kernel.sum().item()
    cfar_alpha = 2.0
    pad_cfar = cfar_win_size // 2


    #oscfar 参数设置
    os_win_size = 5      # 
    os_pad = os_win_size // 2

    # 1. 创建一个全 True 的 2D 掩码
    mask_2d = torch.ones((os_win_size, os_win_size), dtype=torch.bool, device=args.device)

    # 2. 把中心的保护单元和 CUT 挖空 (设为 False)
    center = os_win_size // 2
    mask_2d[center : center+1, center : center+1] = False

    # 3. 展平为 1D 掩码 (长度为 25)
    train_mask = mask_2d.flatten()
    num_train_cells = train_mask.sum().item()

    # 4. 确定 OS-CFAR 的排序索引 k (通常取 0.75 * N)
    # 意思是：从小到大排序，取第 75% 位置的值作为纯净背景代表
    k_index = int(0.75 * num_train_cells)

    print(f"✅ OS-CFAR 初始化: 窗口={os_win_size}x{os_win_size}, 训练单元数={num_train_cells}, k取值={k_index}")

    THETA, R = np.meshgrid(azimuth_axis, range_axis)
    X = R * np.sin(THETA)
    Y = R * np.cos(THETA)










    # 4. 
    metrics_records = []
    test_alphas = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    # 5. 
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Testing & Plotting")):
            radar_cube = batch_data['radar_cube'].to(args.device)
            occupancy_target = batch_data['occupancy_target']
            
            # 2D GT
            occupancy_target_2d, _ = torch.max(occupancy_target, dim=1)
            occupancy_target_2d = occupancy_target_2d.to(args.device)
            
            # 模型前向传播
            outputs = model(radar_cube)
            
            # 维度截取 (根据你验证集的代码逻辑)
            occupancy_logits = outputs['occupancy_logits'].unsqueeze(0) # (B, 1, R, A)
            occupancy_logits = occupancy_logits[:, :-12, 8:-8]
            radar_energy = outputs['ra_energy'][:, :-12, 8:-8] if outputs['ra_energy'].dim() == 3 else outputs['ra_energy'][:, 0, :-12, 8:-8]
            
            occupancy_logits = occupancy_logits.unsqueeze(1) # (B, 1, R, A)
            radar_energy_4d = radar_energy.unsqueeze(1)      # (B, 1, R, A)
            
            # 计算 pred 和 bgenergy
            pred = 1.0 - occupancy_logits
            bgenergy = pred * radar_energy_4d
            
            # 计算局部背景噪声
            kernel_size = 5
            pad = kernel_size // 2
            bgenergy = F.pad(bgenergy, (pad, pad, pad, pad), mode='replicate')
            # unfolded = F.unfold(bgenergy, kernel_size=os_win_size, padding=os_pad)
            # valid_cells = unfolded[:, train_mask, :]
            # local_bg_noise_sum, _ = torch.kthvalue(valid_cells, k_index, dim=1)
            # local_bg_noise_sum = local_bg_noise_sum.view(1, 1, 500, 240)
            # local_bg_noise_sum = F.conv2d(bgenergy, cfar_kernel, stride=1, padding=pad_cfar)
            # local_bg_weight_sum = local_bg_noise_sum / (num_train_cells) 
            local_bg_noise_sum = F.avg_pool2d(bgenergy, kernel_size=kernel_size, stride=1, padding=0)
            # local_bg_weight_sum = F.avg_pool2d(pred, kernel_size=kernel_size, stride=1, padding=pad)

            # local_bg_noise_mean = local_bg_noise_sum / (local_bg_weight_sum + 1e-5)
            
            
           
            # 转换为 Numpy 
           
            gt_np = occupancy_target_2d.squeeze().cpu().numpy()
            radar_energy_np = radar_energy_4d.squeeze().cpu().numpy()
            pred_np = pred.squeeze().cpu().numpy()
            bg_noise_np = local_bg_noise_sum.squeeze().cpu().numpy()
            # final_pred_np = final_pred_2d.squeeze().cpu().numpy().astype(np.float32)

            # 命名
            meta = batch_data['metadata']
            scene_id = meta.get('scene', [f'unk_{batch_idx}'])[0]
            if isinstance(scene_id, torch.Tensor): scene_id = scene_id.item()
            frame_id = meta.get('frame', [batch_idx])[0]
            if isinstance(frame_id, torch.Tensor): frame_id = frame_id.item()

            title_info = f"Scene: {scene_id} | Frame: {frame_id}"
            
            # ==============================
            # 指标计算
            # ==============================
            for current_alpha in test_alphas:
                break
                final_pred_2d = radar_energy_np > (current_alpha * bg_noise_np)
                final_pred_np = final_pred_2d.squeeze().astype(np.float32)
                pd_val, pfa_val = compute_pd_pfa(gt_np, final_pred_np)

                pred_pc_x = X[final_pred_np>0.5]
                pred_pc_y = Y[final_pred_np>0.5]

                gt_pc_x = X[gt_np>0.5]
                gt_pc_y = Y[gt_np>0.5]
                gt_pc_array = np.column_stack((gt_pc_x, gt_pc_y))
                pred_pc_array = np.column_stack((pred_pc_x, pred_pc_y))
                cd_val = compute_chamfer_distance_2d(gt_pc_array, pred_pc_array)
                print(f"{title_info} -> alpha: {current_alpha}, Pd: {pd_val:.4f}, Pfa: {pfa_val:.6f}, Chamfer Dist: {cd_val:.4f}",flush=True)

                tqdm.write(f"{title_info} -> alpha: {current_alpha}, Pd: {pd_val:.4f}, Pfa: {pfa_val:.6f}, Chamfer Dist: {cd_val:.4f}")
                metrics_records.append({
                    'Alpha': current_alpha,
                    'Scene': scene_id,
                    'Frame': frame_id,
                    'Pd': pd_val,
                    'Pfa': pfa_val,
                    'Chamfer_Dist': cd_val
                })



            ###cfar ososos
            gt_pc_x = X[gt_np>0.5]
            gt_pc_y = Y[gt_np>0.5]
            gt_pc_array = np.column_stack((gt_pc_x, gt_pc_y))
            radar_energy_pad= F.pad(radar_energy_4d, (os_pad, os_pad, os_pad, os_pad), mode='replicate')
            unfolded = F.unfold(radar_energy_pad, kernel_size=os_win_size, padding=0)
            valid_cells = unfolded[:, train_mask, :]
            cfar_noise_mean, _ = torch.kthvalue(valid_cells, k_index, dim=1)
            cfar_noise_mean = cfar_noise_mean.view(1, 1, 500, 240)
            cfar_pred = (radar_energy_4d > (cfar_alpha * cfar_noise_mean))

            # radar_energy_pad= F.pad(radar_energy_4d, (pad_cfar, pad_cfar, pad_cfar, pad_cfar), mode='replicate')    
            # cfar_noise_sum = F.conv2d(radar_energy_pad, cfar_kernel, padding=0)
            # cfar_noise_mean = cfar_noise_sum / num_train_cells
            ###cfar ososos
            
            
            cfar_pred_np = cfar_pred.squeeze().cpu().numpy().astype(np.float32)
            cfar_pd, cfar_pfa = compute_pd_pfa(gt_np, cfar_pred_np)

            cfar_pred_pc_x = X[cfar_pred_np>0.5]
            cfar_pred_pc_y = Y[cfar_pred_np>0.5]
            cfar_pred_pc_array = np.column_stack((cfar_pred_pc_x, cfar_pred_pc_y))  

            cfar_cd = compute_chamfer_distance_2d(gt_pc_array, cfar_pred_pc_array)
            print(f"{title_info} -> CFAR (alpha={cfar_alpha}): Pd: {cfar_pd:.4f}, Pfa: {cfar_pfa:.6f}, Chamfer Dist: {cfar_cd:.4f}",flush=True)
            metrics_records.append({
                'Alpha': f'CFAR_{cfar_alpha}',
                'Scene': scene_id,
                'Frame': frame_id,
                'Pd': cfar_pd,
                'Pfa': cfar_pfa,
                'Chamfer_Dist': cfar_cd 
            })

            

    # ==============================
    # 汇总并保存所有指标
    # ==============================
    df_metrics = pd.DataFrame(metrics_records)
    
    # 
    valid_cd = df_metrics['Chamfer_Dist'].dropna()
    avg_cd = valid_cd.mean() if not valid_cd.empty else float('nan')
    
    avg_pd = df_metrics['Pd'].mean()
    avg_pfa = df_metrics['Pfa'].mean()
    
    # 保存 CSV
    csv_path = os.path.join(args.output_dir, "metrics_report331a.csv")
    df_metrics.to_csv(csv_path, index=False)
    
    # 保存 TXT Summary
    # txt_path = os.path.join(args.output_dir, "summary323.txt")
    # with open(txt_path, "w") as f:
    #     f.write("=== 2D Radar Fusion Model Test Summary ===\n")
    #     f.write(f"Model: {args.checkpoint_path}\n")
    #     f.write(f"Total Frames Tested: {len(df_metrics)}\n\n")
    #     f.write(f"Average Pd: {avg_pd:.4f}\n")
    #     f.write(f"Average Pfa: {avg_pfa:.6f}\n")
    #     f.write(f"Average Chamfer Distance: {avg_cd:.4f}\n")
        
    print("\n✅ 推理和可视化全部完成！")
    # print(f"👉 可视化图片文件夹: {vis_dir}")
    # print(f"👉 详细指标数据: {csv_path}")
    # print(f"👉 平均性能总结: {txt_path}")
    # print(f"   平均 Pd: {avg_pd:.4f} | 平均 Pfa: {avg_pfa:.6f} | 平均倒角距离: {avg_cd:.4f}")

if __name__ == "__main__":
    main()