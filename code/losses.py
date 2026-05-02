import torch
import torch.nn.functional as F

def focal_sam_loss(z, mask_gt, margin=2.0, gamma=2.0, top_k_ratio=0.1):
    """
    z: 网络输出的高维特征 [B, d, R, A]
    mask_gt: 伪标签 [B, 1, R, A], 1为目标，0为背景
    """
    # 1. 计算所有像素的模长 (加 epsilon 防止梯度为 0 导致 NaN)
    norm_z = torch.norm(z, p=2, dim=1, keepdim=True) + 1e-6 
    
    # 2. 分离目标和背景
    target_mask = (mask_gt == 1)
    bg_mask = (mask_gt == 0)
    
    # ========================================
    # 3. 计算 Focal Target Loss (针对难易正样本)
    # ========================================
    target_norms = norm_z[target_mask]
    if target_norms.numel() > 0:
        # 基础 Margin Error
        target_error = F.relu(margin - target_norms)
        # 归一化的 Focal 权重
        focal_weight = (target_error / margin) ** gamma
        # Focal Loss 计算
        loss_tgt = torch.mean(focal_weight * (target_error ** 2))
    else:
        loss_tgt = torch.tensor(0.0, device=z.device)
        
    # ========================================
    # 4. 计算 Hard Background Loss (保护卡方分布的难负样本挖掘)
    # ========================================
    bg_norms = norm_z[bg_mask]
    if bg_norms.numel() > 0:
        # 计算所有背景点的 L2 Loss (即模长平方)
        bg_losses = bg_norms ** 2
        
        # OHEM: 挑出最难的 Top-K% 背景点
        k = max(int(bg_losses.numel() * top_k_ratio), 1) # 至少保留1个
        hard_bg_losses, _ = torch.topk(bg_losses, k)
        
        # 纯净的均值，不加权重！
        loss_bg = torch.mean(hard_bg_losses)
    else:
        loss_bg = torch.tensor(0.0, device=z.device)
        
    # 5. 总 Loss，可以通过 lambda 调节两者平衡 (初期建议 lambda=1.0 即可)
    total_loss = loss_bg + 1.0 * loss_tgt
    
    return total_loss