"""
loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class StableFocalLoss(nn.Module):
    """
    Focal loss 
    """
    def __init__(self, alpha: float = 0.95, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        
        Args:
            logits: (Batch, Range, Angle) - 
            targets: (Batch, Range, Angle) - gt occupancy, 1 for target, 0 for background
        """
        # 1. bce first
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 2. compute the probability p from logits using sigmoid
        p = torch.sigmoid(logits)
        
        # 3. 
        p_t = targets * p + (1 - targets) * (1 - p)
        
        # 4.
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        # 5.  Focal Loss：alpha * (1 - p_t)^gamma * BCE
        focal_loss = alpha_t * ((1 - p_t) ** self.gamma) * bce_loss
        
        return focal_loss.mean()


class MaskedQuantileLoss(nn.Module):
    """
    Masked Pinball Loss
    """
    def __init__(self, quantiles: list):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, predictions: torch.Tensor, radar_energy: torch.Tensor, occupancy_target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (Batch, Range, Angle, Quantiles) - 
            radar_energy: (Batch, Range, Angle) - 
            occupancy_target: (Batch, Range, Angle) - 
        """
        # 
        radar_energy = radar_energy.unsqueeze(-1)
        
        # 
        # 
        background_mask = (occupancy_target == 0).unsqueeze(-1).float()
        
        # 2. 
        errors = radar_energy - predictions
        
        total_loss = 0.0
        # 
        for i, q in enumerate(self.quantiles):
            error_q = errors[..., i]
            
            # 3. 
            loss_q = torch.max((q - 1) * error_q, q * error_q)
            
            # 4. 
            masked_loss_q = loss_q * background_mask.squeeze(-1)
            
            # 5. 
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
        
        
        loss_focal = self.focal_loss(occupancy_logits, occupancy_target)
        
        
        # loss_quantile = self.quantile_loss(quantile_preds, radar_energy, occupancy_target)
        
        
        total_loss = self.weight_focal * loss_focal #+ self.weight_quantile * loss_quantile
        
        
        return {
            'total_loss': total_loss,
            'focal_loss': loss_focal,
            # 'quantile_loss': loss_quantile
        }