
"""
雷达-LiDAR 融合检测项目 - 评估与点云保存模块
包含：性能指标计算、点云生成与保存、可视化
Author: Your Name
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc


class PointCloudGenerator:
    """
    从检测结果生成点云
    """
    
    @staticmethod
    def detection_score_to_point_cloud(detection_score: np.ndarray,
                                        occupancy: np.ndarray,
                                        range_bins: int = 64,
                                        angle_bins: int = 16,
                                        max_range: float = 100.0,
                                        score_threshold: float = 0.5,
                                        occupancy_threshold: float = 0.5) -> np.ndarray:
        """
        将 2D detection score 和 occupancy 转换回 3D 点云
        """
        points = []
        
        for r in range(range_bins):
            for a in range(angle_bins):
                score = detection_score[r, a]
                occ = occupancy[r, a]
                
                if score > score_threshold or occ > occupancy_threshold:
                    # 转换回极坐标
                    range_val = (r / range_bins) * max_range
                    angle_val = (a / angle_bins) * 2 * np.pi - np.pi
                    
                    # 转换为笛卡尔坐标
                    x = range_val * np.cos(angle_val)
                    y = range_val * np.sin(angle_val)
                    z = 0.0  # 简单假设 z=0，或者用另一个网络预测
                    
                    points.append([x, y, z, score, occ])
        
        if len(points) == 0:
            return np.zeros((0, 5))
        
        return np.array(points)


class DetectionMetrics:
    """
    检测评估指标
    """
    
    @staticmethod
    def compute_pd_pfa(occupancy_pred: np.ndarray,
                       occupancy_target: np.ndarray,
                       thresholds: Optional[List[float]] = None) -> Tuple[List[float], List[float]]:
        """
        计算 Pd (检测概率) 和 Pfa (虚警概率) 在不同阈值下
        """
        if thresholds is None:
            thresholds = np.linspace(0, 1, 50)
        
        pd_list = []
        pfa_list = []
        
        for thresh in thresholds:
            pred_binary = (occupancy_pred > thresh).astype(float)
            
            tp = np.sum((pred_binary == 1) & (occupancy_target == 1))
            fn = np.sum((pred_binary == 0) & (occupancy_target == 1))
            fp = np.sum((pred_binary == 1) & (occupancy_target == 0))
            tn = np.sum((pred_binary == 0) & (occupancy_target == 0))
            
            pd = tp / (tp + fn + 1e-8)
            pfa = fp / (fp + tn + 1e-8)
            
            pd_list.append(pd)
            pfa_list.append(pfa)
        
        return pd_list, pfa_list
    
    @staticmethod
    def compute_precision_recall(occupancy_pred: np.ndarray,
                                 occupancy_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        计算 Precision-Recall 曲线和 AP
        """
        precision, recall, _ = precision_recall_curve(
            occupancy_target.flatten(),
            occupancy_pred.flatten()
        )
        ap = auc(recall, precision)
        return precision, recall, ap
    
    @staticmethod
    def compute_chamfer_distance(pred_points: np.ndarray,
                                  target_points: np.ndarray) -> float:
        """
        计算 Chamfer Distance (点云级别)
        """
        if len(pred_points) == 0 or len(target_points) == 0:
            return 1000.0  # 大的惩罚
        
        # 计算距离矩阵
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(pred_points[:, :3], target_points[:, :3])
        
        # 双向最小距离
        min_dist_pred_to_target = np.min(dist_matrix, axis=1)
        min_dist_target_to_pred = np.min(dist_matrix, axis=0)
        
        chamfer = np.mean(min_dist_pred_to_target) + np.mean(min_dist_target_to_pred)
        return chamfer


class ResultSaver:
    """
    结果保存器
    """
    
    def __init__(self, save_dir: str = "./results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def save_point_cloud(self, points: np.ndarray, 
                         filename: str = "predicted_points.npy"):
        """
        保存预测点云
        """
        file_path = self.save_dir / filename
        np.save(file_path, points)
        print(f"Point cloud saved to {file_path}")
        
        # 同时保存为 PLY 格式（如果可能）
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            o3d.io.write_point_cloud(str(file_path).replace('.npy', '.ply'), pcd)
            print(f"PLY saved to {str(file_path).replace('.npy', '.ply')}")
        except ImportError:
            pass
        
    def save_metrics(self, metrics_dict: Dict, 
                     filename: str = "metrics.txt"):
        """
        保存指标
        """
        file_path = self.save_dir / filename
        
        with open(file_path, 'w') as f:
            for key, value in metrics_dict.items():
                f.write(f"{key}: {value}\n")
        
        print(f"Metrics saved to {file_path}")
        
    def save_detection_visualization(self, 
                                      model_output: Dict,
                                      batch_data: Dict,
                                      batch_idx: int = 0,
                                      filename: str = "detection_vis.png"):
        """
        保存检测可视化图
        """
        occupancy_pred = model_output['occupancy'][batch_idx].cpu().numpy()
        detection_score = model_output['detection_score'][batch_idx].cpu().numpy()
        occupancy_target = batch_data['occupancy_target'][batch_idx].cpu().numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        im0 = axes[0].imshow(occupancy_target, cmap='hot')
        axes[0].set_title('Occupancy Target (LiDAR)')
        plt.colorbar(im0, ax=axes[0])
        
        im1 = axes[1].imshow(occupancy_pred, cmap='hot')
        axes[1].set_title('Occupancy Prediction')
        plt.colorbar(im1, ax=axes[1])
        
        im2 = axes[2].imshow(detection_score, cmap='hot')
        axes[2].set_title('Detection Score')
        plt.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        save_path = self.save_dir / filename
        plt.savefig(save_path)
        plt.close()
        print(f"Visualization saved to {save_path}")
        
    def save_pd_pfa_curve(self, pd_list: List[float], 
                           pfa_list: List[float],
                           filename: str = "pd_pfa_curve.png"):
        """
        保存 Pd-Pfa 曲线
        """
        plt.figure(figsize=(8, 6))
        plt.plot(pfa_list, pd_list, 'b-', linewidth=2)
        plt.xlabel('False Alarm Probability (Pfa)')
        plt.ylabel('Detection Probability (Pd)')
        plt.title('Pd-Pfa Curve')
        plt.grid(True)
        plt.ylim([0, 1.05])
        plt.xlim([0, 1])
        
        save_path = self.save_dir / filename
        plt.savefig(save_path)
        plt.close()
        print(f"Pd-Pfa curve saved to {save_path}")


class Evaluator:
    """
    综合评估器
    """
    
    def __init__(self, save_dir: str = "./results"):
        self.saver = ResultSaver(save_dir)
        
    def evaluate(self, 
                 model, 
                 dataloader, 
                 device, 
                 save_results: bool = True,
                 num_vis: int = 3) -> Dict:
        """
        完整评估
        """
        model.eval()
        
        all_occupancy_pred = []
        all_occupancy_target = []
        all_pred_points = []
        all_target_points = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                radar_cube = batch_data['radar_cube'].to(device)
                
                model_output = model(radar_cube)
                
                # 收集
                occ_pred = model_output['occupancy'].cpu().numpy()
                occ_target = batch_data['occupancy_target'].numpy()
                
                all_occupancy_pred.append(occ_pred)
                all_occupancy_target.append(occ_target)
                
                # 生成点云
                for i in range(occ_pred.shape[0]):
                    pred_points = PointCloudGenerator.detection_score_to_point_cloud(
                        model_output['detection_score'][i].cpu().numpy(),
                        occ_pred[i]
                    )
                    all_pred_points.append(pred_points)
                    
                    # 如果有 LiDAR 点云
                    if 'lidar_points' in batch_data:
                        target_pc = batch_data['lidar_points'][i].numpy()
                        all_target_points.append(target_pc)
                
                # 可视化
                if save_results and batch_idx < num_vis:
                    self.saver.save_detection_visualization(
                        model_output, batch_data, 
                        filename=f"vis_batch_{batch_idx}.png"
                    )
        
        # 拼接所有
        all_occupancy_pred = np.concatenate(all_occupancy_pred, axis=0)
        all_occupancy_target = np.concatenate(all_occupancy_target, axis=0)
        
        # 计算指标
        metrics = {}
        
        # Pd-Pfa
        pd_list, pfa_list = DetectionMetrics.compute_pd_pfa(
            all_occupancy_pred, all_occupancy_target
        )
        metrics['pd_at_pfa_1e-3'] = np.interp(1e-3, pfa_list, pd_list)
        metrics['pd_at_pfa_1e-2'] = np.interp(1e-2, pfa_list, pd_list)
        
        if save_results:
            self.saver.save_pd_pfa_curve(pd_list, pfa_list)
        
        # Precision-Recall
        _, _, ap = DetectionMetrics.compute_precision_recall(
            all_occupancy_pred, all_occupancy_target
        )
        metrics['AP'] = ap
        
        # Chamfer Distance (如果有点云)
        if len(all_pred_points) > 0 and len(all_target_points) > 0:
            total_chamfer = 0.0
            count = 0
            for pred_pc, target_pc in zip(all_pred_points, all_target_points):
                if len(pred_pc) > 0 and len(target_pc) > 0:
                    chamfer = DetectionMetrics.compute_chamfer_distance(pred_pc, target_pc)
                    total_chamfer += chamfer
                    count += 1
            
            if count > 0:
                metrics['Chamfer_Distance'] = total_chamfer / count
        
        if save_results:
            self.saver.save_metrics(metrics)
            
            # 保存最后一个预测点云
            if len(all_pred_points) > 0:
                self.saver.save_point_cloud(
                    all_pred_points[-1], 
                    filename="final_predicted_points.npy"
                )
        
        return metrics

