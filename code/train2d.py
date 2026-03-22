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


from model import FastFusionModel, MaxPower2DModel
from losses import RadarFusionLoss
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
        power_cube = input_cube[0] #  (128, 512, 256)
        # elevation_cube = input_cube[1] # (128, 512, 256) 
        
        # 2. adjust the order
        # 
        power_cube = np.transpose(power_cube, (1, 0, 2)) #  (512, 128, 256)
        # elevation_cube = np.transpose(elevation_cube, (1, 0, 2))
        
        # 3. GT
        # 
        occupancy_target = np.squeeze(gt_cube) 

        
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
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--range_bins', type=int, default=512)
    parser.add_argument('--doppler_bins', type=int, default=128)
    parser.add_argument('--angle_bins', type=int, default=256)
    
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    wandb.init(project="hpc-network-test", name="first-try")

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
    # os.makedirs(args.save_dir, exist_ok=True)
    
    #load the data
    if args.use_radelft:
        # RaDelft
        from radelft.loaders.rad_cube_loader import RADCUBE_DATASET
        from radelft.data_preparation import data_preparation
        params = data_preparation.get_default_params()
        params["dataset_path"] = '/scratch/shujianjia/dataset/'
        params["train_val_scenes"] = [1,3,4,5,7]
        params["test_scenes"] = [2,6]
        train_dataset = RaDelftWrapper(mode='train', params=params)
        val_dataset = RaDelftWrapper(mode='val', params=params)
    # else:
    #     train_dataset = SafeMockDataset(num_samples=200, range_bins=args.range_bins, doppler_bins=args.doppler_bins, angle_bins=args.angle_bins)
    #     val_dataset = SafeMockDataset(num_samples=40, range_bins=args.range_bins, doppler_bins=args.doppler_bins, angle_bins=args.angle_bins)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16 if args.device=='cuda' else 0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 2. model
    # model = FastFusionModel(
    #     angle_bins=args.angle_bins,
    #     doppler_channels=128
    # ).to(args.device)

    model = MaxPower2DModel(
        in_channels=2
    ).to(args.device)
    
    # 3.  Loss
    criterion = RadarFusionLoss(weight_focal=1.0)
    
    # 4. optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4) # AdamW 比 Adam 更利于泛化
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)


    
    best_val_loss = float('inf')
    
    # break to test validation loop
    # step=0

    # 5. training loop
    for epoch in range(args.num_epochs):
        model.train()
        total_train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.num_epochs}] Train")
        
        for batch_data in pbar:
            radar_cube = batch_data['radar_cube'].to(args.device)
            occupancy_target = batch_data['occupancy_target']
            if len(occupancy_target.shape) == 4:
                # height find max, 34 layers -> 1 layer
                # collapse the height dimension by taking the maximum value across it, resulting in a 2D occupancy map
                occupancy_target, _ = torch.max(occupancy_target, dim=1)
            occupancy_target=occupancy_target.to(args.device)
            optimizer.zero_grad()
            
            # forward
            outputs = model(radar_cube)
            occupancy_logits = outputs['occupancy_logits'][:, :-12, 8:-8]
            radar_energy = outputs['ra_energy'][:, :-12, 8:-8]
            # quantile_preds = outputs['quantiles'][:, :-12, 8:-8, :]
            # occupancy_logits = outputs['occupancy_logits'][..., :-12, 8:-8]
            # quantile_preds = outputs['quantiles'][..., :-12, 8:-8]
            # radar_energy = outputs['ra_energy'][..., :-12, 8:-8]
            occupancy_target = occupancy_target.unsqueeze(1) #(B, 1, R, A)
            soft_targets = TF.gaussian_blur(occupancy_target, kernel_size=[1, 5], sigma=[0.1, 2.0])
            batch_max = soft_targets.view(soft_targets.size(0), -1).max(dim=1).values
            batch_max = batch_max.view(-1, 1, 1, 1)
            soft_targets_norm = soft_targets / (batch_max + 1e-8)
            soft_targets = soft_targets_norm.squeeze(1) #(B, R, A)





            loss_dict = criterion(
                occupancy_logits=occupancy_logits, 
                # quantile_preds=quantile_preds,
                occupancy_target=soft_targets,
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
                'Foc': f"{loss_dict['focal_loss'].item():.3f}",
                # 'Qnt': f"{loss_dict['quantile_loss'].item():.3f}"
            })
            wandb.log({"epoch": epoch, "loss": loss})
            # step+=1
            # if step>3:
            #     break  # Only run a few batches to test the validation loop. delete this line during formal training.
        avg_train_loss = total_train_loss / len(train_loader)
        
        # 6. validation loop
        model.eval()
        total_val_loss = 0.0
        pd_list = []
        pfa_list = []
        qpd_list = []
        qpfa_list = []
        count=0
        with torch.no_grad():
            for batch_data in val_loader:
                radar_cube = batch_data['radar_cube'].to(args.device) # (B, 512, 128, 256)
                occupancy_target = batch_data['occupancy_target']
                occupancy_target_2d, _ = torch.max(occupancy_target, dim=1)
                occupancy_target_2d=occupancy_target_2d.to(args.device)
                occupancy_target_3d=occupancy_target.to(args.device)
                # occupancy_target = batch_data['occupancy_target'].to(args.device)
                outputs = model(radar_cube)
                
                occupancy_logits = outputs['occupancy_logits'][:,  :-12, 8:-8]
                radar_energy = outputs['ra_energy'][:, :, :-12, 8:-8]
                # occupancy=outputs['occupancy'][:, :-12, 8:-8]
                # quantile_preds = outputs['quantiles'][:, :-12, 8:-8, :]
                # qback_est=outputs['background_est'][:, :-12, 8:-8]

                radar_cube_real = radar_cube[:, :-12, :, 8:-8]

                occupancy_target = occupancy_target_2d.unsqueeze(1) #(B, 1, R, A)
                soft_targets = TF.gaussian_blur(occupancy_target, kernel_size=[1, 5], sigma=[0.1, 2.0])
                batch_max = soft_targets.view(soft_targets.size(0), -1).max(dim=1).values
                batch_max = batch_max.view(-1, 1, 1, 1)
                soft_targets_norm = soft_targets / (batch_max + 1e-8)
                soft_targets = soft_targets_norm.squeeze(1) #(B, R, A)
                
                




                loss_dict = criterion(
                occupancy_logits=occupancy_logits, 
                # quantile_preds=quantile_preds,
                occupancy_target=soft_targets,
                radar_energy=radar_energy
                )
                total_val_loss += loss_dict['total_loss'].item()

                occupancy_logits=occupancy_logits.unsqueeze(1) #(B, 1, R, A)
                pred=1.0-occupancy_logits
                bgenergy=pred*radar_energy
                print(f"radar_energy.shape: {radar_energy.shape} | bgenergy.shape: {bgenergy.shape}, occupancy_logits.shape: {occupancy_logits.shape}")
                kernel_size = 5
                pad = kernel_size // 2
            
                # avg pooling to get local background noise sum
                local_bg_noise_sum = F.avg_pool2d(bgenergy, kernel_size=kernel_size, stride=1, padding=pad)
                # local_bg_weight_sum = F.avg_pool2d(pred, kernel_size=kernel_size, stride=1, padding=pad)

                # local_bg_noise_mean = local_bg_noise_sum / (local_bg_weight_sum + 1e-5)
                alpha=2.0
                final_pred_2d = radar_energy > (alpha * local_bg_noise_sum)
                # qpred=radar_energy > qback_est


                
                final_pred_2d = final_pred_2d.squeeze()
                print(f"final_pred_2d.shape: {final_pred_2d.shape}")
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




                





        avg_val_loss = total_val_loss / len(val_loader)
        mean_pd = np.mean(pd_list)
        mean_pfa = np.mean(pfa_list)
        # mean_qpd = np.mean(qpd_list)
        # mean_qpfa = np.mean(qpfa_list)
        print(f"\n[Validation Result] -> Average Pd: {mean_pd:.4f} | Average Pfa: {mean_pfa:.4f}")
        print(f" Epoch [{epoch+1}] Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
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
        scheduler.step(avg_val_loss)
        # 如果有外部 Evaluator，可以在每个 Epoch 末尾调用它算 AP / F1
        # if HAS_EXTERNAL_EVALUATOR and (epoch + 1) % 5 == 0:
        #     evaluator = Evaluator()
        #     evaluator.evaluate(model, val_loader, args.device)

    print("\n 训练圆满结束！")

if __name__ == "__main__":
    main()