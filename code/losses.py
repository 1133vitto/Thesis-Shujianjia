"""
极简重构版：雷达-LiDAR 融合检测项目 - 核心 Loss
包含：数值稳定的 Focal Loss，符合雷达物理逻辑的 Masked Quantile Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class StableFocalLoss(nn.Module):
    """
    数值稳定的 Focal Loss
    专门用于处理“大片空地（背景），极少障碍物（目标）”的极度不平衡问题。
    """
    def __init__(self, alpha: float = 0.95, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        注意：这里的 inputs 必须是未经 Sigmoid 激活的原始 logits！
        Args:
            logits: (Batch, Range, Angle) - 网络预测的占据栅格（原始输出）
            targets: (Batch, Range, Angle) - LiDAR 真值 (0 或 1)
        """
        # 1. 使用自带的 with_logits 函数计算基础交叉熵，PyTorch 底层做了 Log-Sum-Exp 优化，绝对不会数值溢出！
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 2. 手动计算预测概率 p，用于计算 Focal 的难易权重
        p = torch.sigmoid(logits)
        
        # 3. 计算 p_t (如果是正样本就是 p，如果是负样本就是 1-p)
        p_t = targets * p + (1 - targets) * (1 - p)
        
        # 4. 计算 alpha 权重 (正样本权重为 alpha，负样本为 1-alpha)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        # 5. 组合最终的 Focal Loss：alpha * (1 - p_t)^gamma * BCE
        focal_loss = alpha_t * ((1 - p_t) ** self.gamma) * bce_loss
        
        return focal_loss.mean()


class MaskedQuantileLoss(nn.Module):
    """
    带背景掩码的分位数损失 (Masked Pinball Loss)
    核心物理逻辑：只在纯背景（空地）区域学习雷达的底噪分布，遇到有车（目标）的地方直接闭眼不学。
    """
    def __init__(self, quantiles: list):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, predictions: torch.Tensor, radar_energy: torch.Tensor, occupancy_target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (Batch, Range, Angle, Quantiles) - 网络预测的分位数底噪
            radar_energy: (Batch, Range, Angle) - 雷达最原始的真实能量图
            occupancy_target: (Batch, Range, Angle) - LiDAR 提供的真值，1代表目标，0代表背景
        """
        # 为了能和 predictions (4维) 做计算，把能量图增加一个维度变成 (Batch, Range, Angle, 1)
        radar_energy = radar_energy.unsqueeze(-1)
        
        # 1. 提取真理掩码：我们只关心 LiDAR 说是空地（0）的地方
        # 形状变成 (Batch, Range, Angle, 1)
        background_mask = (occupancy_target == 0).unsqueeze(-1).float()
        
        # 2. 计算误差：真实能量 - 预测底噪
        errors = radar_energy - predictions
        
        total_loss = 0.0
        # 遍历每一个要预测的分位数 (比如 0.1, 0.5, 0.9)
        for i, q in enumerate(self.quantiles):
            error_q = errors[..., i]
            
            # 3. 核心魔法：Pinball Loss 的不对称惩罚机制
            loss_q = torch.max((q - 1) * error_q, q * error_q)
            
            # 4. 物理约束：乘以 background_mask，把有目标的地方强行抹零！
            masked_loss_q = loss_q * background_mask.squeeze(-1)
            
            # 5. 求平均时，只除以背景网格的真实总数，防止被目标网格稀释
            valid_pixels = background_mask.sum() + 1e-8
            total_loss += masked_loss_q.sum() / valid_pixels
            
        return total_loss / len(self.quantiles)


class RadarFusionLoss(nn.Module):
    """
    总loss
    """
    def __init__(self, 
                 weight_focal: float = 1.0, 
                #  weight_quantile: float = 0.5, # 分位数回归通常数字较大，稍微降低点权重防止带偏主线
                #  quantiles: list = [0.1, 0.5, 0.9]
                ):
        super().__init__()
        self.weight_focal = weight_focal
        # self.weight_quantile = weight_quantile
        
        self.focal_loss = StableFocalLoss()
        # self.quantile_loss = MaskedQuantileLoss(quantiles=quantiles)

    def forward(self, 
                occupancy_logits: torch.Tensor,   #  U-Net 没过 sigmoid 的原始输出
                # quantile_preds: torch.Tensor,     # U-Net 预测的分位数
                occupancy_target: torch.Tensor,   # LiDAR gt
                radar_energy: torch.Tensor        # radar original
                ) -> dict:
        
        # 1. 算检测 Loss（找目标）
        loss_focal = self.focal_loss(occupancy_logits, occupancy_target)
        
        # 2. 算底噪 Loss（学背景）
        # loss_quantile = self.quantile_loss(quantile_preds, radar_energy, occupancy_target)
        
        # 3. 按权重相加
        total_loss = self.weight_focal * loss_focal #+ self.weight_quantile * loss_quantile
        
        # 返回一个字典，方便你在训练循环里打印和监控每个 Loss 的下降情况
        return {
            'total_loss': total_loss,
            'focal_loss': loss_focal,
            # 'quantile_loss': loss_quantile
        }