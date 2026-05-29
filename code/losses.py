import torch
import torch.nn as nn

class PixelWiseNPLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        """
        纯像素级的可微 NP 损失函数 (无固定尺寸约束，无点迹凝聚)
        :param epsilon: 防止分母为 0 的极小常数
        """
        super(PixelWiseNPLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, preds, targets, use_regularization=False, lambda_reg=1, false_alarm_set=0.02):
        """
        前向传播计算 Loss
        :param preds: 网络的预测值，经过 sigmoid，形状如 [Batch, 1, 500, 240] 或 [Batch, 500, 240]
        :param targets: 真实标签 0/1，形状与 preds 一致
        :param use_regularization: 是否开启第二阶段的虚警控制正则项
        :param lambda_reg: 正则项的权重
        :param alpha_pixels: 当前期望的【虚警像素个数】约束阈值
        """
        # 将输入展平为 [Batch, N_pixels]
        # 对于 500x240 的图像，N_pixels = 120000
        preds_flat = preds.reshape(preds.size(0), -1)
        targets_flat = targets.reshape(targets.size(0), -1)

        # ==========================================
        # 1. 计算监督损失 L_s = 1 - P_d (基于像素的软化检测率)
        # ==========================================
        # 分子: 目标区域的预测概率求和
        intersection = torch.sum(preds_flat * targets_flat, dim=1) 
        
        # 分母: 预测总和 + 真值总和
        sum_preds = torch.sum(preds_flat, dim=1)
        sum_targets = torch.sum(targets_flat, dim=1)
        
        # 像素级可微检测概率 p_d
        p_d = (2.0 * intersection) / (sum_preds + sum_targets + self.epsilon)
        
        # 监督损失 (取 Batch 均值)
        loss_s = 1.0 - p_d.mean()

        # 第一阶段: 直接返回监督损失
        if not use_regularization:
            return loss_s

        # ==========================================
        # 2. 计算正则损失 L_r = | 预测虚警像素数 - 期望虚警像素数 |
        # ==========================================
        # 背景处 (targets==0) 预测出的概率求和，即为 "假阳性/虚警像素数"
        false_alarm_pixels = torch.sum(preds_flat * (1.0 - targets_flat), dim=1)
        false_alarm_mean= false_alarm_pixels.mean()
        
        # 计算绝对误差惩罚
        loss_r = torch.abs(false_alarm_mean - false_alarm_set).mean()

        # 最终总损失
        total_loss = loss_s + lambda_reg * loss_r
        
        return total_loss