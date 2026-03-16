import torch
import numpy as np
import os
import re 
import sys
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm 

# 动态路径
current_dir = Path(__file__).resolve().parent
radelft_dir = current_dir / "radelft"
sys.path.insert(0, str(radelft_dir))
sys.path.insert(0, str(current_dir))

import argparse
from radelft.utils.compute_metrics import compute_pd_pfa
from radelft.data_preparation import data_preparation
from torch.utils.data import DataLoader

from model import FastFusionModel
from FINAL_TRAIN_SCRIPT import RaDelftWrapper 

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = './checkpoints/run_20260310_002558/best_epoch_16_loss_0.0058.pth' # 你的神级权重
    
    print("="*50)
    print(" 开始初始化推理")
    
    #  load model
    model = FastFusionModel(angle_bins=256, doppler_channels=128)
    state_dict = torch.load(model_path, map_location=device, weights_only=True) 
    model.load_state_dict(state_dict['model_state_dict']) 
    model.to(device)
    model.eval() 
    print("模型已就绪！")

    # test set params
    params = data_preparation.get_default_params()
    params["dataset_path"] = '/scratch/shujianjia/dataset/'
    params["train_val_scenes"] = [1, 3, 4, 5, 7]
    params["test_scenes"] = [2, 6]
    
    # test dataset and loader
    test_dataset = RaDelftWrapper(mode='test', params=params)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=8,shuffle=False)
    
    print(f" 测试集已加载，共 {len(test_dataset)} 帧数据。")
    print("="*50)

    # 用来统计整个测试集的平均 2D 成绩
    all_pd_2d, all_pfa_2d = [], []

    # ==========================================================
    # 3. 推理开始
    # ==========================================================
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Inference Progress")):
            
            # --- 【数据载入与形状注释】 ---
            # radar_cube 形状: (B, Doppler, Range, Azimuth) -> (1, 128, 512, 256)
            radar_cube = batch_data['radar_cube'].to(device)
            # elevation_cube 形状: (B, Doppler, Range, Azimuth) -> (1, 128, 512, 256)
            elevation_cube = batch_data['elevation_cube'].to(device)
            # occupancy_target_2d 形状: (B, Range, Azimuth) -> (1, 512, 256)
            occ_target = batch_data['occupancy_target']
            occ_target_2d, _ = torch.max(occ_target, dim=1)
            # occ_target_2d=occ_target_2d.squeeze()
            occ_target_2d=occ_target_2d.to(device)
            occ_target_3d=occ_target.to(device)
            # metadata 包含这一帧对应的原始路径和时间戳字典
            metadata = batch_data['metadata'] 

            # --- 【模型前向传播】 ---
            outputs = model(radar_cube)

            # --- 【切除 Padding，还原真实尺寸】 ---
            # 真实尺寸：Range=500, Azimuth=240
            # 剥离后的预测图形状全部变为: (B, 500, 240) -> (1, 500, 240)
            occupancy = outputs['occupancy'][:, :-12, 8:-8]
            radar_energy = outputs['ra_energy'][:, :-12, 8:-8]
            # occ_target_2d_real = occ_target_2d[ :, :-12, 8:-8]

            # 4D 张量剥离 Padding (保留第 1 维 Doppler 不变)
            # 真实雷达能量图: (B, 128, 500, 240)
            radar_cube_real = radar_cube[:, :, :-12, 8:-8]
            # 真实高度映射图: (B, 128, 500, 240)
            elevation_cube_real = elevation_cube[:, :, :-12, 8:-8]

            # --- 【2D 背景评估与预测生成】 ---
            pred = 1.0 - occupancy # (1, 500, 240) 背景权重
            bgenergy = pred * radar_energy # (1, 500, 240) 加权后的背景能量
            
            kernel_size = 5
            pad = kernel_size // 2
            
            # 【修复警告】：avg_pool2d 必须吃 4D 张量 (B, C, H, W)，所以要 unsqueeze(1) 借一个通道，算完再 squeeze(1) 还回去
            local_bg_noise_sum = F.avg_pool2d(bgenergy, kernel_size=kernel_size, stride=1, padding=pad)
            
            # 生成 2D 的最终布尔预测图 final_pred_2d: (1, 500, 240)
            alpha = 1.0
            final_pred_2d = radar_energy > (alpha * local_bg_noise_sum)
            

            # --- 【计算并打印 2D 成绩】 ---
            gt_2d_np = occ_target_2d.squeeze().cpu().numpy()
            pred_2d_np = final_pred_2d.squeeze().cpu().numpy()
            pd_2d, pfa_2d = compute_pd_pfa(gt_2d_np, pred_2d_np)
            all_pd_2d.append(pd_2d)
            all_pfa_2d.append(pfa_2d)
            # 如果你不想每次都刷屏，可以注释掉下面这行 print
            # print(f"  Frame {batch_idx} -> 2D Pd: {pd_2d:.4f} | Pfa: {pfa_2d:.4f}")

            # --- 【核心升维：2.5D to 3D Lifting】 ---
            # 沿着 Doppler 维度 (dim=1) 找最大值。
            # max_doppler_idx 形状: (B, 500, 240)
            max_doppler_idx = torch.argmax(radar_cube_real, dim=1) 
            
            # 用最大索引去提取高度，提取后的 elevation_val 形状: (B, 500, 240)
            elevation_val = torch.gather(elevation_cube_real, dim=1, index=max_doppler_idx.unsqueeze(1)).squeeze(1)
            
            # 乘回 34 层并取整，得到目标所在的具体层号: (B, 500, 240) 整数
            elevation_indices = torch.clamp((elevation_val * 34).long(), 0, 33)
            
            # 创建 3D 预测的空壳 final_pred_3d: (B, 34, 500, 240)
            final_pred_3d = torch.zeros((1, 34, 500, 240), device=device)
            
            # 把 2D 的预测点填入 3D 空壳的对应高度层中
            final_pred_3d.scatter_(
                dim=1, 
                index=elevation_indices.unsqueeze(1), 
                src=final_pred_2d.unsqueeze(1).float() 
            )
            gt_3d_numpy = occ_target_3d.squeeze().cpu().detach().numpy()
            pred_3d_numpy = final_pred_3d.squeeze().cpu().detach().numpy()

            pd, pfa = compute_pd_pfa(gt_3d_numpy, pred_3d_numpy)
            
            print(f"  Frame {batch_idx} -> 2D Pd: {pd_2d:.4f} | Pfa: {pfa_2d:.4f}")
            print(f"3d Pd: {pd:.4f} | 3d Pfa: {pfa:.4f} ")

            # --- 【3D 预测图转点云与保存】 ---
            # 因为 batch_size 是 1，我们直接拿第 0 个样本的 3D 张量出来：(34, 500, 240)
            pred_3d_np = final_pred_3d[0].cpu().detach()
            
            # 把 3D 张量转成 N*3 的雷达点云坐标 (N 是点云数量，3 是 X, Y, Z)
            pred_pc = data_preparation.cube_to_pointcloud(pred_3d_np, None, None, mode='lidar')
            
            # 只有当画面里有检测到目标时，才做坐标系变换并保存
            if pred_pc.shape[0] > 0:
                pred_pc[:, 1] = -pred_pc[:, 1] # 坐标系翻转
                pred_pc = data_preparation.transform_point_cloud(
                    pred_pc, 
                    [0, 0, params['azimuth_offset']],
                    [-params['x_offset'] / 100, -params['y_offset'] / 100, 0]
                )
            else:
                # 这一帧啥也没检测到，存个空数组
                pred_pc = np.array([])

            # --- 【完美的路径替换与保存逻辑】 ---
            # 从 metadata 字典里拿出当前这一帧原来的 cfar 路径
            # metadata 里的 value 也是一个 list (长度为1)，所以加 [0]
            cfar_path = metadata["cfar_path"][0] 
            
            # 把路径里的 radar_ososos (或其他名字) 替换成我们自己的 network 文件夹名
            save_path = re.sub(r"radar_.+/", r"jsjnn/", cfar_path)
            
            # 【关键】：有可能 network 文件夹还不存在，必须帮它建好，不然保存必报错！
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 正式存为 .npy 文件
            np.save(save_path, pred_pc)

    print("="*50)
    print("🎉 整个测试集推理完成！所有点云文件已生成！")
    print(f"📊 整个测试集的平均 2D 指标 -> Pd: {np.mean(all_pd_2d):.4f} | Pfa: {np.mean(all_pfa_2d):.4f}")

if __name__ == "__main__":
    main()