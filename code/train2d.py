"""
2026/march
========================================================================
"""
import sys
import os
import datetime
import shutil
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


from model import FastFusionModel, MaxPower2DModel, CustomUNet, CustomUNetPlusPlus, CustomUNet3Plus, CustomUNet2Layer, CustomUNet2Level, CustomUNet4Level
from losses import RadarFusionLoss, AsymmetricTemperatureBottleneck
from lidar_filtering import build_filtered_lidar_gt
from radelft.loaders.rad_cube_loader import RADCUBE_DATASET

try:
    from evaluation import Evaluator
    HAS_EXTERNAL_EVALUATOR = True
except ImportError:
    HAS_EXTERNAL_EVALUATOR = False
    print("未找到外部 Evaluator，将使用内置的")

torch.set_float32_matmul_precision('medium')

class RaDelftWrapper(Dataset):
    """
    need to be modify later
    """
    def __init__(self, mode='train', params=None):
        # the same Dataset used in radelft
        self.real_dataset = RADCUBE_DATASET(mode=mode, params=params)

    def __len__(self):
        return len(self.real_dataset)

    def __getitem__(self, idx):
        
        input_cube, gt_cube, item_params = self.real_dataset[idx]
        
        # 1. seperate Elevation /Power
        #  input_cube  (2, 128, 512, 256) -> (Channel, Doppler, Range, Azimuth)
        # index 0 refers to Power
        power_cube = input_cube #  (128, 512, 256)
        # elevation_cube = input_cube[1] # (128, 512, 256) 
        
        # 2. adjust the order
        # 
        power_cube = np.transpose(power_cube, (1, 0, 2)) #  (512, 128, 256)
        # elevation_cube = np.transpose(elevation_cube, (1, 0, 2))
        
        # range_cell_size = 0.1004
        # max_range = 51.4242
        # range_axis = np.arange(range_cell_size, max_range + range_cell_size, range_cell_size)
        # range_axis = range_axis[10:-3]
        # # Azimuth Axis 
        # angle_fft_size = 256 
        # wx_vec = np.linspace(-np.pi, np.pi, angle_fft_size) 
        # wx_vec = wx_vec[8:248] 
        # azimuth_axis = np.arcsin(wx_vec / (2 * np.pi * 0.4972))
        # # Elevation Axis 
        # ele_fft_size = 128 
        # wz_vec = np.linspace(-np.pi, np.pi, ele_fft_size) 
        # wz_vec = wz_vec[47:81] 
        # elevation_axis = np.arcsin(wz_vec / (2 * np.pi * 0.4972))

        # E = elevation_axis          # (34,)
        # R = range_axis             # (500,)
        # A = azimuth_axis           # (240,)

        # E_grid, R_grid, A_grid = np.meshgrid(E, R, A, indexing='ij')

        # # 坐标变换
        # Z = R_grid * np.sin(E_grid)
        # X = R_grid * np.cos(E_grid) * np.cos(A_grid)
        # Y = R_grid * np.cos(E_grid) * np.sin(A_grid)
        # z_min = -1.0   # 地面以下一点
        # z_max = 2.5    # SUV / truck 上限
        # mask = (Z >= z_min) & (Z <= z_max)
        # gt_filtered = gt_cube * mask



        # occupancy_target = np.max(gt_filtered, axis=0)  # (500, 240)
        # occupancy_target = (occupancy_target > 0).astype(np.float32)  # 二值化
        # 
        occupancy_target = np.squeeze(gt_cube) #(500, 240)

        
        return {
            'radar_cube': torch.from_numpy(power_cube).float(),
            # 'elevation_cube': torch.from_numpy(elevation_cube).float(),
            'occupancy_target': torch.from_numpy(occupancy_target).float(),
            'metadata': item_params  
        }



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
    parser.add_argument('--seed', type=int, default=42, help='随机种子，确保结果可复现')
    parser.add_argument('--model', type=str, default='CustomUNetPlusPlus', help='选择模型类型')
    parser.add_argument('--loss', type=str, default='RadarFusionLoss', help='选择损失函数')
    parser.add_argument('--train_val_scenes', type=int, nargs='+', default=[1, 3, 4, 5, 7],
                        help='用于训练和验证的 RaDelft scene 列表')
    parser.add_argument('--test_scenes', type=int, nargs='+', default=[2, 6],
                        help='保存在 params 中的测试 scene 列表')
    parser.add_argument('--use_soft_target', action='store_true', default=False,
                        help='使用高斯模糊 soft target')
    parser.add_argument('--focal_loss_type', type=str, default='quality',
                        choices=['quality', 'stable'],
                        help='focal loss 类型: quality (连续) 或 stable (离散)')
    parser.add_argument('--use_atb', action='store_true', default=False,
                        help='训练 loss 前启用 AsymmetricTemperatureBottleneck')
    parser.add_argument('--filter_lidar_gt', action='store_true',
                        help='使用 z>2m 且 normalized SPAR<0.2 过滤后的 LiDAR GT')
    parser.add_argument('--launch_script', type=str, default=None,
                        help='提交本次训练的 shell 脚本路径，会复制到 checkpoint run 目录')
    args = parser.parse_args()
    
    wandb.init(project="model v1.0", name=args.name)

    print("="*70)
    print("启动2d训练")
    print(f" 运算设备: {args.device}")
    print(f" 数据模式: {'RaDelft 真实数据' if args.use_radelft else '模拟数据'}")
    print(f" Train/Val scenes: {args.train_val_scenes}")
    print(f" Test scenes: {args.test_scenes}")
    print(f" LiDAR GT过滤: {'开启' if args.filter_lidar_gt else '关闭'}")
    print("="*70)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{current_time}"
    
    save_dir = os.path.join(args.save_dir, run_name) 
    os.makedirs(save_dir, exist_ok=True)
    print(f"本次训练的所有权重将保存在: {save_dir}")
    with open(os.path.join(save_dir, "train_args.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n\n")
        for key, value in sorted(vars(args).items()):
            f.write(f"{key}: {value}\n")
    if args.launch_script:
        shutil.copy2(args.launch_script, os.path.join(save_dir, "launch_script.sh"))
    # os.makedirs(args.save_dir, exist_ok=True)
    
    #load the data
    if args.use_radelft:
        # RaDelft
        from radelft.loaders.rad_cube_loader import RADCUBE_DATASET
        from radelft.data_preparation import data_preparation
        params = data_preparation.get_default_params()
        params["dataset_path"] = '/scratch/shujianjia/dataset/'
        params["train_val_scenes"] = args.train_val_scenes
        params["test_scenes"] = args.test_scenes
        params["bev"] = True
        train_dataset = RaDelftWrapper(mode='train', params=params)
        val_dataset = RaDelftWrapper(mode='val', params=params)
    # else:
    #     train_dataset = SafeMockDataset(num_samples=200, range_bins=args.range_bins, doppler_bins=args.doppler_bins, angle_bins=args.angle_bins)
    #     val_dataset = SafeMockDataset(num_samples=40, range_bins=args.range_bins, doppler_bins=args.doppler_bins, angle_bins=args.angle_bins)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers if args.device=='cuda' else 0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.workers if args.device=='cuda' else 0)
    
    # 2. model
    # model = FastFusionModel(
    #     angle_bins=args.angle_bins,
    #     doppler_channels=128
    # ).to(args.device)
    MODEL_REGISTRY = {
    'CustomUNet': CustomUNet,
    'CustomUNetPlusPlus': CustomUNetPlusPlus,
    'CustomUNet3Plus': CustomUNet3Plus,
    'CustomUNet2Layer': CustomUNet2Layer,
    'CustomUNet2Level': CustomUNet2Level,
    'CustomUNet4Level': CustomUNet4Level
    }
    model_class =MODEL_REGISTRY[args.model]

    model = MaxPower2DModel(
        model=model_class,
        in_channels=2
    ).to(args.device)
    print(f"模型使用：{args.model}")
    
    # model.unet.freeze_backbone()

    # 3.  Loss
    criterion = RadarFusionLoss(weight_focal=1.0,
                                focal_loss_type=args.focal_loss_type)
    logit_modulator = AsymmetricTemperatureBottleneck() if args.use_atb else None
    
    # 4. optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4) # AdamW 比 Adam 更利于泛化
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)


    
    best_val_loss = float('inf')
    
    # break to test validation loop
    step=0

    if args.use_soft_target:
        kernel_size=[1, 5]
        sigma=[0.1, 2.0]
        dummy_point = torch.zeros(1, 1, 31, kernel_size[1])
        dummy_point[0, 0, 15, kernel_size[1]//2] = 1.0
        W_c = TF.gaussian_blur(dummy_point, kernel_size=kernel_size, sigma=sigma).max()
    # 5. training loop
    for epoch in range(args.num_epochs):
        model.train()
        total_train_loss = 0.0
        train_filter_stats = {"raw_points": 0, "dropped_points": 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.num_epochs}] Train")
        
        for batch_data in pbar:
            radar_cube = batch_data['radar_cube'].to(args.device)
            occupancy_target = batch_data['occupancy_target']
            if len(occupancy_target.shape) == 4:
                # height find max, 34 layers -> 1 layer
                # collapse the height dimension by taking the maximum value across it, resulting in a 2D occupancy map
                occupancy_target, _ = torch.max(occupancy_target, dim=1)
            occupancy_target=occupancy_target.to(args.device)
            if args.filter_lidar_gt:
                occupancy_target, filter_stats = build_filtered_lidar_gt(
                    radar_cube=radar_cube,
                    metadata=batch_data['metadata'],
                    params=params,
                    device=args.device,
                )
                train_filter_stats["raw_points"] += filter_stats["raw_points"]
                train_filter_stats["dropped_points"] += filter_stats["dropped_points"]
            optimizer.zero_grad()
            
            # forward
            outputs = model(radar_cube)
            occupancy_prob = outputs['occupancy_prob'][:, :-12, 8:-8]
            occupancy_logits = outputs['occupancy_logits'][:, :-12, 8:-8]
            radar_energy = outputs['ra_energy'][:, :, :-12, 8:-8]#( B, 1, R, A  )
            # quantile_preds = outputs['quantiles'][..., :-12, 8:-8]
            # radar_energy = outputs['ra_energy'][..., :-12, 8:-8]

            if args.use_soft_target:
                soft_targets = TF.gaussian_blur(occupancy_target.unsqueeze(1),
                                                kernel_size=[1, 5], sigma=[0.1, 2.0])
                scaled = soft_targets / W_c
                soft_targets = torch.clamp(scaled, min=0, max=1.0)
                final_targets = torch.max(occupancy_target.unsqueeze(1), soft_targets)
                occupancy_target = final_targets.squeeze(1)


            loss_logits = occupancy_logits
            if logit_modulator is not None:
                loss_logits = logit_modulator(loss_logits, occupancy_target)

            loss_dict = criterion(
                occupancy_logits=loss_logits,
                # quantile_preds=quantile_preds,
                occupancy_target=occupancy_target,
                radar_energy=radar_energy
            )
            
            loss = loss_dict['total_loss']
            loss.backward()
            
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            total_train_loss += loss.item()
            
            pbar.set_postfix({
                'Tot': f"{loss.item():.3f}",
                'Foc': f"{loss_dict['focal_loss'].item():.3f}"
                # 'Dice': f"{loss_dict['dice_loss'].item():.3f}",
                # 'CFAR': f"{loss_dict['cfar_loss'].item():.3f}",
                # 'Qnt': f"{loss_dict['quantile_loss'].item():.3f}"
            })
            # wandb.log({"epoch": epoch, "loss": loss})
            # step+=1
            # if step>2:
            #     break  # Only run a few batches to test the validation loop. delete this line during formal training.
        avg_train_loss = total_train_loss / len(train_loader)
        
        # 6. validation loop
        model.eval()
        total_val_loss = 0.0
        pd_list = []
        pfa_list = []
        val_filter_stats = {"raw_points": 0, "dropped_points": 0}
        count=0
        with torch.no_grad():
            for batch_data in val_loader:
                radar_cube = batch_data['radar_cube'].to(args.device) # (B, 512, 128, 256)
                occupancy_target = batch_data['occupancy_target']

                # occupancy_target_2d, _ = torch.max(occupancy_target, dim=1)
                occupancy_target_2d=occupancy_target.to(args.device)
                # occupancy_target_3d=occupancy_target.to(args.device)
                # occupancy_target = batch_data['occupancy_target'].to(args.device)
                if args.filter_lidar_gt:
                    occupancy_target_2d, filter_stats = build_filtered_lidar_gt(
                        radar_cube=radar_cube,
                        metadata=batch_data['metadata'],
                        params=params,
                        device=args.device,
                    )
                    val_filter_stats["raw_points"] += filter_stats["raw_points"]
                    val_filter_stats["dropped_points"] += filter_stats["dropped_points"]
                outputs = model(radar_cube)
                
                occupancy_logits = outputs['occupancy_logits'][:,  :-12, 8:-8]
                occupancy_prob = outputs['occupancy_prob'][:, :-12, 8:-8]
                radar_energy = outputs['ra_energy'][:, :, :-12, 8:-8]
                
                # qback_est=outputs['background_est'][:, :-12, 8:-8]

                radar_cube_real = radar_cube[:, :-12, :, 8:-8]

                occupancy_target = occupancy_target_2d #(B, R, A)

                if args.use_soft_target:
                    soft_targets = TF.gaussian_blur(occupancy_target.unsqueeze(1),
                                                    kernel_size=[1, 5], sigma=[0.1, 2.0])
                    scaled = soft_targets / W_c
                    soft_targets = torch.clamp(scaled, min=0, max=1.0)
                    final_targets = torch.max(occupancy_target.unsqueeze(1), soft_targets)
                    occupancy_target = final_targets.squeeze(1)

                loss_logits = occupancy_logits
                if logit_modulator is not None:
                    loss_logits = logit_modulator(loss_logits, occupancy_target)

                loss_dict = criterion(
                occupancy_logits=loss_logits,
                # quantile_preds=quantile_preds,
                occupancy_target=occupancy_target,
                radar_energy=radar_energy
                )
                total_val_loss += loss_dict['total_loss'].item()

                pred=1.0-occupancy_prob.unsqueeze(1)
                bgenergy=pred*radar_energy
                # print(f"radar_energy.shape: {radar_energy.shape} | bgenergy.shape: {bgenergy.shape}, occupancy_logits.shape: {occupancy_logits.shape}")
                kernel_size = 5
                pad = kernel_size // 2
            
                # avg pooling to get local background noise sum
                local_bg_noise_sum = F.avg_pool2d(bgenergy, kernel_size=kernel_size, stride=1, padding=pad)
                # local_bg_weight_sum = F.avg_pool2d(pred, kernel_size=kernel_size, stride=1, padding=pad)

                # local_bg_noise_mean = local_bg_noise_sum / (local_bg_weight_sum + 1e-5)
                alpha=2.0
                final_pred_2d = radar_energy > (alpha * local_bg_noise_sum)
                


                
                final_pred_2d = final_pred_2d.squeeze(1)
                # B, R, A = final_pred_2d.shape
                max_doppler_idx=outputs['max_indices'][:, :-12, 8:-8] #(B, 500, 240)
                # max_doppler_idx = torch.argmax(radar_cube_real, dim=2)#(B, 500, 240)


                gt_2d_numpy = occupancy_target_2d.cpu().detach().numpy()
                pred_2d_numpy = final_pred_2d.cpu().detach().numpy()

                pd, pfa = compute_pd_pfa(gt_2d_numpy, pred_2d_numpy)
                # qpred=qpred.cpu().detach().numpy()
                # gt_numpy=occupancy_target.cpu().detach().numpy()
                # final_pred_2d=final_pred_2d.cpu().detach().numpy()
                # pd, pfa = compute_pd_pfa(gt_numpy, final_pred_2d)
                # qpd ,qpfa= compute_pd_pfa(gt_numpy, qpred)
                if count%10==0:
                    print(f" Pd: {pd:.4f} |  Pfa: {pfa:.4f} ")
                pd_list.append(pd)
                pfa_list.append(pfa)
                # qpd_list.append(qpd)
                # qpfa_list.append(qpfa)
                count=count+1
                # step+=1
                # if step>3:
                #     break  # Only run a few batches to test the validation loop. delete this line during formal training.




                





        avg_val_loss = total_val_loss / len(val_loader)
        mean_pd = np.mean(pd_list)
        mean_pfa = np.mean(pfa_list)
        # mean_qpd = np.mean(qpd_list)
        # mean_qpfa = np.mean(qpfa_list)
        print(f"\n[Validation Result] -> Average Pd: {mean_pd:.4f} | Average Pfa: {mean_pfa:.4f}")
        print(f" Epoch [{epoch+1}] Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        if args.filter_lidar_gt:
            print(
                f" Filtered LiDAR GT | "
                f"train dropped {train_filter_stats['dropped_points']}/{train_filter_stats['raw_points']} raw points | "
                f"val dropped {val_filter_stats['dropped_points']}/{val_filter_stats['raw_points']} raw points"
            )
        
        wandb.log({
            "epoch": epoch + 1,  # 统一的 X 轴
            "Loss/Train": avg_train_loss,
            "Loss/Validation": avg_val_loss,
            "Metrics/Pd": mean_pd,
            "Metrics/Pfa": mean_pfa,
            "Learning_Rate": optimizer.param_groups[0]['lr'], # 顺手记录一下学习率的变化！
            "FilteredGT/Train_Dropped_Points": train_filter_stats["dropped_points"],
            "FilteredGT/Train_Raw_Points": train_filter_stats["raw_points"],
            "FilteredGT/Validation_Dropped_Points": val_filter_stats["dropped_points"],
            "FilteredGT/Validation_Raw_Points": val_filter_stats["raw_points"],
        })


        # save the parameters of the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            file_name = f"best_epoch_{epoch+1}_loss_{avg_val_loss:.4f}.pth"
            save_path = os.path.join(save_dir, file_name)
            
            # Save the complete state dictionary, including model weights and optimizer state
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, save_path)
            print(f" 新的最佳模型已保存 -> {save_path}")
            # save_path = os.path.join(args.save_dir, "best_fusion_model.pth")
            # torch.save(model.state_dict(), save_path)
            # print(f" 新的最佳模型已保存 -> {save_path}")
        # scheduler.step(avg_val_loss) # if ReduceLROnPlateau 
        scheduler.step() # if CosineAnnealingLR
        

    print("\n 训练圆满结束！")

if __name__ == "__main__":
    main()
