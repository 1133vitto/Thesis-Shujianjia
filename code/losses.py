"""
loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.
    Directly optimizes the overlap between prediction and target.
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (Batch, Range, Angle) - model output probability
            target: (Batch, Range, Angle) - binary target
        """
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class SoftCFARLoss(nn.Module):
    """
    End-to-end differentiable approximation of the CFAR detection criterion.

    The hard CFAR detection is:
        detection = radar_energy > alpha * local_bg_mean
    where:
        pred_bg = 1 - occupancy_prob   (background probability)
        weighted_energy = pred_bg * radar_energy
        local_bg_mean = avg_pool(weighted_energy) / avg_pool(pred_bg)

    This loss replaces the hard comparison with a soft sigmoid approximation,
    allowing gradients to flow during training.

    The loss encourages:
    - detection_score >> 0  when target = 1 (target present → high score desired)
    - detection_score << 0  when target = 0 (background → low score desired)
    """
    def __init__(self, alpha: float = 2.0, beta: float = 10.0):
        super().__init__()
        self.alpha = alpha  # CFAR scale factor
        self.beta = beta   # sigmoid temperature

    def forward(self,
               occupancy_prob: torch.Tensor,
               radar_energy: torch.Tensor,
               occupancy_target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            occupancy_prob: (B, R, A) - model predicted probability
            radar_energy:   (B, 1, R, A) or (B, R, A) - raw radar energy
            occupancy_target: (B, R, A) - binary GT
        """
        # Ensure radar_energy has same shape as occupancy_prob
        if radar_energy.dim() == 4:
            radar_energy = radar_energy.squeeze(1)  # (B, R, A)

        # Background probability
        pred_bg = 1.0 - occupancy_prob  # P(background)

        # Weighted energy and local background mean
        weighted_energy = pred_bg * radar_energy

        kernel_size = 5
        pad = kernel_size // 2

        # avg_pool2d needs 4D: (B, 1, H, W)
        weighted_energy_4d = weighted_energy.unsqueeze(1)
        pred_bg_4d = pred_bg.unsqueeze(1)

        local_bg_sum = F.avg_pool2d(weighted_energy_4d, kernel_size=kernel_size,
                                     stride=1, padding=pad).squeeze(1)
        pred_bg_sum = F.avg_pool2d(pred_bg_4d, kernel_size=kernel_size,
                                   stride=1, padding=pad).squeeze(1)

        local_bg_mean = local_bg_sum / (pred_bg_sum + 1e-8)

        # Soft CFAR detection score (before hard threshold)
        detection_score = radar_energy - self.alpha * local_bg_mean

        # Soft detection: sigmoid approximation of [detection_score > 0]
        soft_detection = torch.sigmoid(self.beta * detection_score)

        # Loss: for target=1, want soft_detection close to 1
        #       for target=0, want soft_detection close to 0
        loss = F.binary_cross_entropy(soft_detection, occupancy_target)

        return loss


class StableFocalLoss(nn.Module):
    """
    Focal loss for binary classification.
    Input is probability p in [0, 1] (after sigmoid of raw logits).
    """
    def __init__(self, alpha: float = 0.95, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (Batch, Range, Angle) - model output probability (after sigmoid)
            targets: (Batch, Range, Angle) - gt occupancy, 1 for target, 0 for background
        """
        # Clamp pred for numerical stability
        pred = pred.clamp(min=1e-6, max=1 - 1e-6)

        # BCE loss
        bce_loss = F.binary_cross_entropy(pred, targets, reduction='none')

        # p_t: probability of positive class
        p_t = targets * pred + (1 - targets) * (1 - pred)

        # Alpha weighting
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Focal loss = alpha * focal_weight * BCE
        focal_loss = alpha_t * focal_weight * bce_loss

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
    Combined loss: Focal + Dice + Soft CFAR
    """
    def __init__(self,
                 weight_focal: float = 1.0,
                 weight_dice: float = 0.1,
                 weight_cfar: float = 0.1,
                 # weight_quantile: float = 0.5,
                 # quantiles: list = [0.1, 0.5, 0.9]
                 ):
        super().__init__()
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice
        self.weight_cfar = weight_cfar

        self.focal_loss = StableFocalLoss()
        self.dice_loss = DiceLoss()
        self.cfar_loss = SoftCFARLoss(alpha=2.0, beta=10.0)
        # self.quantile_loss = MaskedQuantileLoss(quantiles=quantiles)

    def forward(self,
                occupancy_prob: torch.Tensor,    # model output probability (after sigmoid)
                # quantile_preds: torch.Tensor,
                occupancy_target: torch.Tensor,   # LiDAR gt
                radar_energy: torch.Tensor        # radar original
                ) -> dict:


        loss_focal = self.focal_loss(occupancy_prob, occupancy_target)
        loss_dice = self.dice_loss(occupancy_prob, occupancy_target)
        loss_cfar = self.cfar_loss(occupancy_prob, radar_energy, occupancy_target)

        # loss_quantile = self.quantile_loss(quantile_preds, radar_energy, occupancy_target)


        total_loss = (self.weight_focal * loss_focal +
                      self.weight_dice * loss_dice +
                      self.weight_cfar * loss_cfar)


        return {
            'total_loss': total_loss,
            'focal_loss': loss_focal,
            'dice_loss': loss_dice,
            'cfar_loss': loss_cfar,
            # 'quantile_loss': loss_quantile
        }