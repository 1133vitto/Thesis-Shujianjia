
"""
雷达-LiDAR 融合检测项目 - Loss Functions
包含 Focal Loss, Quantile/Pinball Loss, Chamfer Loss, Consistency Loss 等
Author: Your Name
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in occupancy grid
    论文: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, 
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch_size, range_bins, angle_bins) - 预测的 occupancy 概率
            targets: (batch_size, range_bins, angle_bins) - 真值 occupancy (0 或 1)
        
        Returns:
            Focal Loss 值
        """
        # 确保输入在 [0, 1] 范围内
        inputs = torch.clamp(inputs, 1e-7, 1 - 1e-7)
        
        # 计算交叉熵
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # 计算 focal weight
        p_t = targets * inputs + (1 - targets) * (1 - inputs)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            loss = alpha_t * loss
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class VarifocalLoss(nn.Module):
    """
    Varifocal Loss - Focal Loss 的改进版，对正负样本使用不同权重
    论文: https://arxiv.org/abs/2008.13367
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, 
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.clamp(inputs, 1e-7, 1 - 1e-7)
        
        # 正样本权重
        pos_weight = targets * (self.alpha * ((1 - inputs) ** self.gamma))
        # 负样本权重
        neg_weight = (1 - targets) * (inputs ** self.gamma)
        
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        loss = (pos_weight + neg_weight) * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class QuantileLoss(nn.Module):
    """
    Quantile Loss (Pinball Loss) for noise quantile prediction
    用于预测噪声的分位数，对应 CFAR 理论
    """
    def __init__(self, quantiles: list = None):
        super().__init__()
        if quantiles is None:
            self.quantiles = [0.1, 0.5, 0.9]  # 默认预测 10%, 50%, 90% 分位数
        else:
            self.quantiles = quantiles
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (batch_size, range_bins, angle_bins, num_quantiles)
            targets: (batch_size, range_bins, angle_bins) - 真实噪声值
        
        Returns:
            Quantile Loss
        """
        total_loss = 0.0
        num_quantiles = len(self.quantiles)
        
        for i, tau in enumerate(self.quantiles):
            pred = predictions[..., i]
            error = targets - pred
            
            # Pinball loss
            loss = torch.max(tau * error, (tau - 1) * error)
            total_loss = total_loss + loss.mean()
            
        return total_loss / num_quantiles


class ChamferDistance(nn.Module):
    """
    Chamfer Distance for point cloud comparison
    用于直接比较雷达预测点云和 LiDAR 真值点云
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, pred_points: torch.Tensor, target_points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_points: (batch_size, num_pred_points, 3) - 预测点云
            target_points: (batch_size, num_target_points, 3) - 真值点云
        
        Returns:
            Chamfer Distance
        """
        batch_size = pred_points.shape[0]
        
        # 计算点对距离矩阵
        dist_matrix = torch.cdist(pred_points, target_points, p=2)
        
        # 最小距离: 预测点到最近真值点
        min_dist_pred_to_target, _ = torch.min(dist_matrix, dim=2)
        # 最小距离: 真值点到最近预测点
        min_dist_target_to_pred, _ = torch.min(dist_matrix, dim=1)
        
        # Chamfer Distance = 双向平均最小距离
        chamfer_dist = min_dist_pred_to_target.mean(dim=1) + min_dist_target_to_pred.mean(dim=1)
        
        if self.reduction == 'mean':
            return chamfer_dist.mean()
        elif self.reduction == 'sum':
            return chamfer_dist.sum()
        else:
            return chamfer_dist


class ConsistencyLoss(nn.Module):
    """
    Consistency Loss - 确保 2D occupancy 预测和 3D 点云预测一致
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, occupancy_2d: torch.Tensor, 
                points_3d: torch.Tensor,
                grid_range: float = 100.0,
                range_bins: int = 64,
                angle_bins: int = 16) -> torch.Tensor:
        """
        Args:
            occupancy_2d: (batch_size, range_bins, angle_bins)
            points_3d: (batch_size, num_points, 3)
        
        Returns:
            Consistency Loss
        """
        batch_size = occupancy_2d.shape[0]
        
        # 将 3D 点云投影到 2D range-angle 平面
        projected_occupancy = []
        
        for b in range(batch_size):
            pts = points_3d[b]  # (num_points, 3)
            
            # 转换为极坐标
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
            r = torch.sqrt(x**2 + y**2)
            theta = torch.atan2(y, x)
            
            # 离散化到网格
            r_bin = torch.clip((r / grid_range) * range_bins, 0, range_bins - 1).long()
            theta_bin = torch.clip(((theta + torch.pi) / (2 * torch.pi)) * angle_bins, 0, angle_bins - 1).long()
            
            # 创建 occupancy 网格
            occ = torch.zeros((range_bins, angle_bins), device=occupancy_2d.device)
            occ[r_bin, theta_bin] = 1.0
            projected_occupancy.append(occ)
        
        projected_occupancy = torch.stack(projected_occupancy, dim=0)
        
        # 计算一致性损失 (MSE)
        loss = F.mse_loss(occupancy_2d, projected_occupancy, reduction=self.reduction)
        
        return loss


class RadarLidarLoss(nn.Module):
    """
    雷达-LiDAR 融合检测总 Loss
    组合 Focal Loss, Quantile Loss, Chamfer Loss, Consistency Loss
    """
    def __init__(self, 
                 weight_focal: float = 1.0,
                 weight_quantile: float = 1.0,
                 weight_chamfer: float = 0.1,
                 weight_consistency: float = 0.1,
                 quantiles: list = None):
        super().__init__()
        
        self.weight_focal = weight_focal
        self.weight_quantile = weight_quantile
        self.weight_chamfer = weight_chamfer
        self.weight_consistency = weight_consistency
        
        self.focal_loss = FocalLoss()
        self.quantile_loss = QuantileLoss(quantiles=quantiles)
        self.chamfer_loss = ChamferDistance()
        self.consistency_loss = ConsistencyLoss()
        
    def forward(self, 
                occupancy_pred: torch.Tensor,
                occupancy_target: torch.Tensor,
                quantile_pred: torch.Tensor,
                noise_target: torch.Tensor,
                pred_points: torch.Tensor = None,
                target_points: torch.Tensor = None,
                consistency_weight: float = 0.0) -> dict:
        """
        Returns:
            loss_dict: 包含各项 loss 的字典
        """
        loss_dict = {}
        
        # Focal Loss
        loss_focal = self.focal_loss(occupancy_pred, occupancy_target)
        loss_dict['focal'] = loss_focal
        
        # Quantile Loss
        loss_quantile = self.quantile_loss(quantile_pred, noise_target)
        loss_dict['quantile'] = loss_quantile
        
        # Total loss so far
        total_loss = self.weight_focal * loss_focal + self.weight_quantile * loss_quantile
        
        # Chamfer Loss (optional)
        if pred_points is not None and target_points is not None:
            loss_chamfer = self.chamfer_loss(pred_points, target_points)
            loss_dict['chamfer'] = loss_chamfer
            total_loss = total_loss + self.weight_chamfer * loss_chamfer
        
        # Consistency Loss (optional)
        if consistency_weight > 0 and pred_points is not None:
            loss_consistency = self.consistency_loss(
                occupancy_pred, pred_points
            )
            loss_dict['consistency'] = loss_consistency
            total_loss = total_loss + self.weight_consistency * loss_consistency
        
        loss_dict['total'] = total_loss
        
        return loss_dict

