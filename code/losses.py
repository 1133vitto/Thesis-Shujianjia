import torch
import torch.nn as nn
import torch.nn.functional as F
class PixelWiseNPLoss(nn.Module):
    def __init__(self, epsilon=1e-6, pool_size=3):
        """
        终极版：带局部 1对1 匹配机制的纯像素级可微 NP 损失
        :param epsilon: 防止 0/0 崩溃的平滑常数
        :param pool_size: 容错感受野大小 (默认 3x3)
        """
        super(PixelWiseNPLoss, self).__init__()
        self.epsilon = epsilon
        self.pool_size = pool_size
        self.padding = pool_size // 2  # 保证池化后特征图大小不变
        self.alpha = 0.995  # Focal Loss 中正负样本权重
        self.gamma = 2.0   # Focal Loss 中调节难易样本的指数

    def forward(self, preds, targets, use_regularization=False, lambda_reg=0.1, pfa_set=0.02):
        """
        :param preds: 预测概率图, 经过 sigmoid, 形状必须是 [Batch, 1, H, W]
        :param targets: 单像素真实标签 0/1, 形状必须是 [Batch, 1, H, W]
        :param use_regularization: False为第一阶段，True为第二阶段
        :param lambda_reg: 正则项系数
        :param pfa_set: 期望的虚警率
        """
        # 确保输入是 4D 张量 (Batch, Channel, Height, Width)，因为 max_pool2d 需要这个形状
        if preds.dim() == 3:
            preds = preds.unsqueeze(1)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        # ==========================================
        # 1. 算奖励 (Intersection) —— 反向膨胀预测值，实现局部 1对1
        # ==========================================
        # 对预测值做 3x3 的最大池化，扩大预测的“射程”
        preds_max_pooled = F.max_pool2d(preds, kernel_size=self.pool_size, stride=1, padding=self.padding)
        
        # 点乘单像素真值，拿到唯一最高分作为奖励
        intersection = torch.sum(preds_max_pooled * targets, dim=(1, 2, 3))

        # ==========================================
        # 2. 算监督损失 L_s (防止胖子惩罚，分母必须用原始 preds)
        # ==========================================
        sum_preds = torch.sum(preds, dim=(1, 2, 3))    # 注意：这里千万不能用 preds_max_pooled！
        sum_targets = torch.sum(targets, dim=(1, 2, 3))
        
        # 软化检测率 P_d (分子和分母都加上 epsilon，彻底修复空背景图直接崩 1.0 的 Bug)
        p_d = (2.0 * intersection + self.epsilon) / (sum_preds + sum_targets + self.epsilon)
        p_d = torch.clamp(p_d, max=1.0)
        # Dice 监督损失 (Batch 均值)
        loss_s = 1.0 - p_d.mean()

        # ==========================================
        # 3. 阶段分配：第一阶段的冷启动加速
        # ==========================================
        if not use_regularization:
            # 第一阶段：加上 BCE 损失，逼迫网络快速建立初始自信，防止 Loss 卡在 0.4
            bce_loss = F.binary_cross_entropy(preds, targets, reduction='none')
            p_t = targets * preds + (1 - targets) * (1 - preds)
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            focal_weight = (1 - p_t) ** self.gamma
            focal_loss = (alpha_t * focal_weight * bce_loss).mean()
            return 0.5*loss_s + focal_loss

        # ==========================================
        # 4. 算惩罚 (False Alarms) —— 引入真值豁免区 (Ignore Zone)
        # ==========================================
        # 对真值做 3x3 最大池化，生成豁免掩码
        gt_ignore_zone = F.max_pool2d(targets, kernel_size=self.pool_size, stride=1, padding=self.padding)
        
        # 算虚警：原始预测值 * 豁免区之外的掩码 (落在 gt_ignore_zone 里的 0.9 会被乘以 0 抹除)
        false_alarms = torch.sum(preds * (1.0 - gt_ignore_zone), dim=(1, 2, 3))
        sum_background = torch.sum(1.0 - targets, dim=(1, 2, 3))  # 背景像素总数
        false_alarm_rate = false_alarms / (sum_background + self.epsilon)  # 虚警率，分母用背景像素总数保持一致
        # 计算绝对误差惩罚
        loss_r = torch.abs(false_alarm_rate - pfa_set).mean()

        # 最终总损失
        total_loss = loss_s + lambda_reg * loss_r
        
        return total_loss