"""
2026/march
========================================================================
"""
import sys
import os
import datetime
from pathlib import Path
current_dir = Path(__file__).resolve().parent
radelft_dir = current_dir / "radelft"
sys.path.insert(0, str(radelft_dir)) 
sys.path.insert(0, str(current_dir))
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm  
from radelft.utils.compute_metrics import compute_pd_pfa
import torchvision.transforms.functional as TF
import wandb


from model import HRNetV1_W18
from losses import PixelWiseNPLoss
from radelft.loaders.rad_cube_loader import RADCUBE_DATASET


torch.set_float32_matmul_precision('medium')

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
    args = parser.parse_args()
    
    wandb.init(project="idea4", name=args.name)

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
    
    # load data: RADCUBE_DATASET with bev=True already returns (D,R,A) input and 2D occupancy GT
    from radelft.data_preparation import data_preparation
    params = data_preparation.get_default_params()
    params["dataset_path"] = '/scratch/shujianjia/dataset/'
    params["train_val_scenes"] = [1,3,4,5,7]
    params["test_scenes"] = [2,6]
    params["bev"] = True
    train_dataset = RADCUBE_DATASET(mode='train', params=params)
    val_dataset = RADCUBE_DATASET(mode='val', params=params)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers if args.device=='cuda' else 0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers if args.device=='cuda' else 0)
    
    # 2. model
    model = HRNetV1_W18().to(args.device)
    print("模型使用：HRNetV1_W18 (full-resolution)")

    # 3.  Loss
    criterion = PixelWiseNPLoss(epsilon=1e-6)
    
    # 4. optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4) # AdamW 比 Adam 更利于泛化
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)


    
    best_val_loss = float('inf')
    
    # break to test validation loop
    step=0

    # 5. training loop
    for epoch in range(args.num_epochs):
        model.train()
        total_train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.num_epochs}] Train")
        
        for batch_data in pbar:
            radar_cube, occupancy_target, _ = batch_data
            radar_cube = radar_cube.to(args.device)
            occupancy_target = occupancy_target.to(args.device)
            optimizer.zero_grad()
            
            # forward
            occupancy_prob = model(radar_cube)['occupancy_prob']  # (B, 512, 256)
            occupancy_prob_cropped = occupancy_prob[:, :-12, 8:-8]  # (B, 500, 240)

            # soft targets: Gaussian blur along azimuth for spatial alignment
            occupancy_target = occupancy_target.unsqueeze(1)  # (B, 1, R, A)

            # mid-training regularization switch
            use_reg = (epoch >= args.num_epochs // 10)
            loss = criterion(
                occupancy_prob_cropped,
                occupancy_target,
                use_regularization=use_reg,
                lambda_reg=1,
                pfa_set=0.02,
            )

            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_train_loss += loss.item()

            pbar.set_postfix({'Loss': f"{loss.item():.3f}"})
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
        count=0
        with torch.no_grad():
            for batch_data in val_loader:
                radar_cube, occupancy_target, _ = batch_data
                radar_cube = radar_cube.to(args.device)

                # occupancy_target_2d, _ = torch.max(occupancy_target, dim=1)
                occupancy_target_2d=occupancy_target.to(args.device)
                # occupancy_target_3d=occupancy_target.to(args.device)
                # occupancy_target = batch_data['occupancy_target'].to(args.device)
                occupancy_prob = model(radar_cube)['occupancy_prob'][:, :-12, 8:-8]  # (B, 500, 240)

                gt_2d = occupancy_target_2d.unsqueeze(1)  # (B, 1, R, A)
                
                




                loss = criterion(occupancy_prob, gt_2d, pfa_set=0.02)  
                total_val_loss += loss.item()

                final_pred_2d = (occupancy_prob > 0.5).float()


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