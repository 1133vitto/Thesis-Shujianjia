"""
2026/march
========================================================================
"""
import sys
import os
from pathlib import Path

current_dir = Path(__file__).resolve().parent
radelft_dir = current_dir / "radelft"
sys.path.insert(0, str(radelft_dir)) # 插到最前面，拥有最高优先级
sys.path.insert(0, str(current_dir))
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm  # 进度条神器

# 导入我们刚刚重构的核心利器
from model import FastFusionModel
from losses import RadarFusionLoss
from radelft.loaders.rad_cube_loader import RADCUBE_DATASET
# 如果你有自己写的 evaluator，可以在这里 try-except 导入
try:
    from evaluation import Evaluator
    HAS_EXTERNAL_EVALUATOR = True
except ImportError:
    HAS_EXTERNAL_EVALUATOR = False
    print("未找到外部 Evaluator，将使用内置的")

torch.set_float32_matmul_precision('medium')

class RaDelftWrapper(Dataset):
    """
    数据集适配器：
    1. 负责把 RADCUBE_DATASET 吐出的元组，转换成我们训练循环需要的字典格式。
    2. 负责剔除 Elevation，只保留 Power。
    3. 负责修复维度的顺序，对齐模型输入。
    """
    def __init__(self, mode='train', params=None):
        # 内部实例化你真实的 Dataset
        self.real_dataset = RADCUBE_DATASET(mode=mode, params=params)

    def __len__(self):
        return len(self.real_dataset)

    def __getitem__(self, idx):
        # 从真实 dataset 中拿到数据
        input_cube, gt_cube, item_params = self.real_dataset[idx]
        
        # 1. 剥离 Elevation，提取 Power
        # 此时 input_cube 形状是 (2, 128, 512, 256) -> (Channel, Doppler, Range, Azimuth)
        # 索引 0 就是 Power
        power_cube = input_cube[0] # 形状变成 (128, 512, 256)
        
        # 2. 修复维度顺序
        # 我们的模型需要 (Range, Doppler, Azimuth)，所以把第0维和第1维换一下
        power_cube = np.transpose(power_cube, (1, 0, 2)) # 形状完美变成 (512, 128, 256)
        
        # 3. 处理真值 (确保是 2D 的 Range x Azimuth)
        # 有时候 gt_cube 可能会带一个通道维度，比如 (1, 512, 256)，我们用 squeeze 把多余的 1 挤掉
        occupancy_target = np.squeeze(gt_cube) 

        # 4. 组装成我们 Loss 和 Model 需要的字典格式
        return {
            'radar_cube': torch.from_numpy(power_cube).float(),
            'occupancy_target': torch.from_numpy(occupancy_target).float()
        }



# ==========================================
# 训练主函数
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='雷达-LiDAR 融合检测训练脚本')
    parser.add_argument('--use_radelft', default=True ,action='store_true', help='使用真实的 RaDelft 数据集')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    # 网络维度参数 (需要跟你的真实数据对齐)
    parser.add_argument('--range_bins', type=int, default=512)
    parser.add_argument('--doppler_bins', type=int, default=128)
    parser.add_argument('--angle_bins', type=int, default=256)
    
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 启动训练：雷达-LiDAR 融合感知网络 🚀")
    print(f"🖥️  运算设备: {args.device}")
    print(f"📂 数据模式: {'RaDelft 真实数据' if args.use_radelft else '本地安全模拟数据'}")
    print("="*70)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1. 准备数据
    if args.use_radelft:
        # 这里保留你真实的 RaDelft 加载逻辑
        # 请确保 RaDelft dataset 返回的字典里有 'radar_cube' 和 'occupancy_target' 这两个 key
        from radelft.loaders.rad_cube_loader import RADCUBE_DATASET
        from radelft.data_preparation import data_preparation
        params = data_preparation.get_default_params()
        params["dataset_path"] = 'F:/radelft/Scene1_RadarCubes/'
        params["train_val_scenes"] = [1]
        params["test_scenes"] = [1]
        train_dataset = RaDelftWrapper(mode='train', params=params)
        val_dataset = RaDelftWrapper(mode='val', params=params)
    # else:
    #     train_dataset = SafeMockDataset(num_samples=200, range_bins=args.range_bins, doppler_bins=args.doppler_bins, angle_bins=args.angle_bins)
    #     val_dataset = SafeMockDataset(num_samples=40, range_bins=args.range_bins, doppler_bins=args.doppler_bins, angle_bins=args.angle_bins)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4 if args.device=='cuda' else 0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 2. 初始化极简融合模型
    model = FastFusionModel(
        angle_bins=args.angle_bins,
        cnn_out_channels=64
    ).to(args.device)
    
    # 3. 初始化物理严谨的融合 Loss
    criterion = RadarFusionLoss(weight_focal=1.0, weight_quantile=0.5)
    
    # 4. 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4) # AdamW 比 Adam 更利于泛化
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# 然后在每个 epoch 结束，算完 avg_val_loss 后，告诉 scheduler 现在的状态：
    
    best_val_loss = float('inf')
    
    # 5. 核心训练循环
    for epoch in range(args.num_epochs):
        model.train()
        total_train_loss = 0.0
        
        # 使用 tqdm 包装 train_loader 形成进度条
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.num_epochs}] Train")
        
        for batch_data in pbar:
            radar_cube = batch_data['radar_cube'].to(args.device)
            occupancy_target = batch_data['occupancy_target']
            if len(occupancy_target.shape) == 4:
                # 沿第 1 维 (高度维度) 取最大值。
                # 物理意义：只要这 34 层里有 1，投影下来的 2D 结果就是 1。
                occupancy_target, _ = torch.max(occupancy_target, dim=1)
            occupancy_target=occupancy_target.to(args.device)
            optimizer.zero_grad()
            
            # 前向传播 (输出包含了 logits, 概率, 分位数估计, 以及雷达原始能量)
            outputs = model(radar_cube)
            occupancy_logits = outputs['occupancy_logits'][:, :-12, 8:-8]
            radar_energy = outputs['ra_energy'][:, :-12, 8:-8]
            quantile_preds = outputs['quantiles'][:, :-12, 8:-8, :]
            # occupancy_logits = outputs['occupancy_logits'][..., :-12, 8:-8]
            # quantile_preds = outputs['quantiles'][..., :-12, 8:-8]
            # radar_energy = outputs['ra_energy'][..., :-12, 8:-8]
            # 计算 Loss (严格按照我们设计的 API)
            loss_dict = criterion(
                occupancy_logits=occupancy_logits, 
                quantile_preds=quantile_preds,
                occupancy_target=occupancy_target,
                radar_energy=radar_energy
            )
            
            loss = loss_dict['total_loss']
            loss.backward()
            
            # 梯度裁剪：防止训练初期极端噪声导致梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # 在进度条上实时显示各项 Loss
            pbar.set_postfix({
                'Tot': f"{loss.item():.3f}",
                'Foc': f"{loss_dict['focal_loss'].item():.3f}",
                'Qnt': f"{loss_dict['quantile_loss'].item():.3f}"
            })
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # 6. 验证循环
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                radar_cube = batch_data['radar_cube'].to(args.device)
                occupancy_target = batch_data['occupancy_target']
                if len(occupancy_target.shape) == 4:
                # 沿第 1 维 (高度维度) 取最大值。
                    occupancy_target, _ = torch.max(occupancy_target, dim=1)
                occupancy_target=occupancy_target.to(args.device)
                # occupancy_target = batch_data['occupancy_target'].to(args.device)
                outputs = model(radar_cube)
                
                occupancy_logits = outputs['occupancy_logits'][:, :-12, 8:-8]
                radar_energy = outputs['ra_energy'][:, :-12, 8:-8]
                quantile_preds = outputs['quantiles'][:, :-12, 8:-8, :]

                loss_dict = criterion(
                occupancy_logits=occupancy_logits, 
                quantile_preds=quantile_preds,
                occupancy_target=occupancy_target,
                radar_energy=radar_energy
                )
                total_val_loss += loss_dict['total_loss'].item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"👉 Epoch [{epoch+1}] Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(args.save_dir, "best_fusion_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"🌟 新的最佳模型已保存 -> {save_path}")
        scheduler.step(avg_val_loss)
        # 如果有外部 Evaluator，可以在每个 Epoch 末尾调用它算 AP / F1
        if HAS_EXTERNAL_EVALUATOR and (epoch + 1) % 5 == 0:
            evaluator = Evaluator()
            evaluator.evaluate(model, val_loader, args.device)

    print("\n🎉 训练圆满结束！")

if __name__ == "__main__":
    main()