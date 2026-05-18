"""
2026/march
========================================================================
"""
import sys
import os
import datetime
from pathlib import Path
import torch.nn.functional as F
current_dir = Path(__file__).resolve().parent
radelft_dir = current_dir / "radelft"
sys.path.insert(0, str(radelft_dir)) 
sys.path.insert(0, str(current_dir))
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm  
from radelft.utils.compute_metrics import compute_metrics_time, compute_pd_pfa
import torchvision.transforms.functional as TF
import wandb

from radelft.data_preparation import data_preparation
from model import RadarResUNet3Plus
from losses import focal_sam_loss
from radelft.loaders.rad_cube_loader import RADCUBE_DATASET
from scipy.spatial.distance import cdist
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree


torch.set_float32_matmul_precision('medium')

def get_lidar_elevation_mask(device):
    range_cell_size = 0.1004
    max_range = 51.4242
    range_axis = np.arange(range_cell_size, max_range + range_cell_size, range_cell_size)
    range_axis = range_axis[10:-3]
    
    angle_fft_size = 256 
    wx_vec = np.linspace(-np.pi, np.pi, angle_fft_size) 
    wx_vec = wx_vec[8:248] 
    azimuth_axis = np.arcsin(wx_vec / (2 * np.pi * 0.4972))
    
    ele_fft_size = 128 
    wz_vec = np.linspace(-np.pi, np.pi, ele_fft_size) 
    wz_vec = wz_vec[47:81] 
    elevation_axis = np.arcsin(wz_vec / (2 * np.pi * 0.4972))

    E = elevation_axis          # (34,)
    R = range_axis             # (500,)
    A = azimuth_axis           # (240,)

    E_grid, R_grid, A_grid = np.meshgrid(E, R, A, indexing='ij')

    Z = R_grid * np.sin(E_grid)
    z_min = -1.0   
    z_max = 2.5    
    mask = (Z >= z_min) & (Z <= z_max)
    
    # 转为 PyTorch Tensor，并加上 Batch 和 Channel 维度备用
    # 假设 Lidar Cube shape 是 [B, E, R, A]
    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).to(device)
    return mask_tensor

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

# ==========================================
# 训练主函数
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='训练脚本')
    parser.add_argument('--use_radelft', default=True ,action='store_true', help='使用真实的 RaDelft 数据集')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--range_bins', type=int, default=512)
    parser.add_argument('--doppler_bins', type=int, default=128)
    parser.add_argument('--angle_bins', type=int, default=256)
    parser.add_argument('--name', type=str, default='first_try')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--workers', type=int, default=8)

    args = parser.parse_args()
    
    wandb.init(project="model v1.0", name=args.name)
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

    THETA, R = np.meshgrid(azimuth_axis, range_axis)
    X = R * np.sin(THETA)
    Y = R * np.cos(THETA)

    print("="*70)
    print("启动2d训练")
    print(f" 运算设备: {args.device}")
    print(f" 数据模式: {'RaDelft 真实数据' if args.use_radelft else '模拟数据'}")
    print("="*70)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{current_time}"
    
    save_dir = os.path.join(args.save_dir, run_name) 
    os.makedirs(save_dir, exist_ok=True)
    print(f"本次训练的所有权重将保存在: {save_dir}")
    
        
    params = data_preparation.get_default_params()
    params["dataset_path"] = '/scratch/shujianjia/dataset/'
    params["train_val_scenes"] = [1,3,4,5,7]
    params["test_scenes"] = [2,6]
    params["bev"]=True
    train_dataset = RADCUBE_DATASET(mode='train', params=params)
    val_dataset = RADCUBE_DATASET(mode='val', params=params)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers if args.device=='cuda' else 0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.workers if args.device=='cuda' else 0)
    

    model = RadarResUNet3Plus(
        n_doppler=128,
        out_dim=32
    ).to(args.device)
    
    # model.unet.freeze_backbone()

    # 3.  Loss
    # criterion = focal_sam_loss()
    
    # 4. optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)


    
    best_val_loss = float('inf')
    
    # break to test validation loop
    step=0

    kernel_size=[1, 5]
    sigma=[0.1, 2.0]
    dummy_point = torch.zeros(1, 1, 31, kernel_size[1])
    dummy_point[0, 0, 15, kernel_size[1]//2] = 1.0
    # # 获取孤立单点模糊后的最大值 (比如 0.4)
    W_c = TF.gaussian_blur(dummy_point, kernel_size=kernel_size, sigma=sigma).max()
    # ele_mask = get_lidar_elevation_mask(args.device)
    # 5. training loop
    for epoch in range(args.num_epochs):
        model.train()
        total_train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.num_epochs}] Train")
        
        for batch_data in pbar:
            radar_cube, lidar_cube, data_dict = batch_data
            radar_cube=radar_cube
            
            # [修复] 将数据放入设备
            radar_cube = radar_cube.float().to(args.device)
            lidar_cube = lidar_cube.float().to(args.device)
            if lidar_cube.dim() == 3:
                occupancy_target = lidar_cube.unsqueeze(1)
            else:
                occupancy_target = lidar_cube
            # [补齐] 1. 高度截断 (利用提前生成的 mask)
            # 假设 lidar_cube 形状是 [B, E, R, A]
            # lidar_filtered = lidar_cube * ele_mask

            # [补齐] 2. 投影到 2D BEV 平面 (Range-Azimuth)
            # 对 Elevation (高度维，假设是第1维) 取最大值投影
            # occupancy_target, _ = torch.max(lidar_cube, dim=1, keepdim=True) # 形状: [B, 1, R, A]
            
            optimizer.zero_grad()
            
            # [补齐] 3. 网络前向传播
            print(f"🧐 探针：当前 Batch 的 radar_cube 形状是 {radar_cube.shape}")
            outputs = model(radar_cube) # 形状: [B, 16, R, A]
            
            # 4. 生成 Soft Target (解决点云膨胀/位移)
            soft_targets = TF.gaussian_blur(occupancy_target, kernel_size=kernel_size, sigma=sigma)
            scaled = soft_targets / W_c
            soft_targets = torch.clamp(scaled, min=0, max=1.0)
            final_targets = torch.max(occupancy_target, soft_targets) # [B, 1, R, A]
            
            # [修复/补齐] 5. 计算损失
            # 此时 final_targets 是含有软响应的掩码，形状 [B, 1, R, A]
            # 我们直接把模型提取的高维特征 outputs 和这个软标签喂给 focal_sam_loss
            outputs = outputs[:, :, :-12, 8:-8]
            loss_dict = focal_sam_loss(z=outputs, mask_gt=final_targets, margin=3.0)
            
            loss = loss_dict
            loss.backward()
            
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # [补齐] 更新进度条，展示各项 Loss 的收敛情况
            pbar.set_postfix({
                'Tot': f"{loss.item():.3f}",
            })
            # step+=1
            # if step>3:
            #     break  # Only run a few batches to test the validation loop. delete this line during formal training.
        avg_train_loss = total_train_loss / len(train_loader)
        
        # ==========================================
        # 6. Validation Loop
        # ==========================================
        model.eval()

        # ----------------------------------------------------
        # 采用最简单粗暴的固定阈值法 (先跑通为主！)
        fixed_threshold = 1.0
        # ----------------------------------------------------

        total_val_loss = 0.0
        pd_list, pfa_list,cd_list = [], [],[]
        # count = 0

        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc="Validation", leave=False):
                # ... [前面的数据加载和前向传播不变] ...
                radar_cube, lidar_cube, _ = batch_data
                radar_cube = radar_cube.float().to(args.device)
                lidar_cube = lidar_cube.float().to(args.device)
                if lidar_cube.dim() == 3:
                    occupancy_target = lidar_cube.unsqueeze(1)
                else:
                    occupancy_target = lidar_cube
                
                outputs = model(radar_cube)
                outputs = outputs[:, :, :-12, 8:-8]
                loss_dict = focal_sam_loss(z=outputs, mask_gt=occupancy_target, margin=3.0)
                total_val_loss += loss_dict.item()


                
    
                
                # 1. 算出每个像素的特征模长 (得分图)
                final_pred_2d = torch.norm(outputs, p=2, dim=1) # 形状: [B, R, A]
                occupancy_target_2d = occupancy_target.squeeze(1) 
                
                # 2. 【修改点】：直接用固定阈值截断！
                binary_pred_2d = (final_pred_2d > fixed_threshold).float()

                # 3. 转换为 NumPy 喂给评估函数
                gt_2d_numpy = occupancy_target_2d.cpu().numpy()
                pred_2d_numpy = binary_pred_2d.cpu().numpy()

                # 计算 Pd 和 Pfa
                pd, pfa = compute_pd_pfa(gt_2d_numpy, pred_2d_numpy)

                # pred_pc_x = X[pred_2d_numpy>0.5]
                # pred_pc_y = Y[pred_2d_numpy>0.5]
                # gt_pc_x = X[gt_2d_numpy>0.5]
                # gt_pc_y = Y[gt_2d_numpy>0.5]
                # gt_pc_array = np.column_stack((gt_pc_x, gt_pc_y))
                # pred_pc_array = np.column_stack((pred_pc_x, pred_pc_y))
                # cd = compute_chamfer_distance_2d(gt_pc_array, pred_pc_array)
                
                pd_list.append(pd)
                pfa_list.append(pfa)

                # cd_list.append(cd)
                # step=step+1
                # if step>5:
                #     break



        avg_val_loss = total_val_loss / len(val_loader)
        mean_pd = np.mean(pd_list)
        mean_pfa = np.mean(pfa_list)
        # mean_cd=np.mean(cd_list)

        print(f"\n[Validation Result] -> Average Pd: {mean_pd:.4f} | Average Pfa: {mean_pfa:.4f}")
        print(f" Epoch [{epoch+1}] Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        wandb.log({
            "epoch": epoch + 1,  # 统一的 X 轴
            "Loss/Train": avg_train_loss,
            "Loss/Validation": avg_val_loss,
            "Metrics/Pd": mean_pd,
            "Metrics/Pfa": mean_pfa,
            "Learning_Rate": optimizer.param_groups[0]['lr'] # 顺手记录一下学习率的变化！
        })


        # save the parameters of the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            file_name = f"best_epoch_{epoch+1}_loss_{avg_val_loss:.4f}.pth"
            save_path = os.path.join(save_dir, file_name)
            count=0
            
            # Save the complete state dictionary, including model weights and optimizer state
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, save_path)
            print(f" 新的最佳模型已保存 -> {save_path}")
        else:
            count+=1
            print(f" 本轮验证损失未提升，当前连续未提升次数: {count}")
            if count>=5:
                print(" 验证损失连续5轮未提升，提前停止训练！")
                break
            
        # scheduler.step(avg_val_loss) # if ReduceLROnPlateau 
        scheduler.step() # if CosineAnnealingLR
        

    print("\n 训练圆满结束！")

if __name__ == "__main__":
    main()