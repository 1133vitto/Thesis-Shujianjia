
"""
雷达-LiDAR 融合检测项目 - 数据预处理模块
包含：数据加载、预处理、数据增强
Author: Your Name
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional, List
import os
from pathlib import Path


class RadarCubeProcessor:
    """
    雷达数据 Cube 处理器
    """
    
    @staticmethod
    def normalize_range_doppler(radar_cube: np.ndarray) -> np.ndarray:
        """
        规范化 range-doppler 维度
        """
        # 简单的全局归一化
        mean = np.mean(radar_cube)
        std = np.std(radar_cube)
        normalized = (radar_cube - mean) / (std + 1e-8)
        return normalized
    
    @staticmethod
    def log_compression(radar_cube: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """
        对数压缩，用于雷达信号
        """
        return np.log1p(np.abs(radar_cube) + epsilon)
    
    @staticmethod
    def remove_static_clutter(radar_cube: np.ndarray, 
                              clutter_map: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        去除静态杂波
        """
        if clutter_map is None:
            # 简单方法：用时间平均作为杂波估计
            clutter_map = np.mean(radar_cube, axis=1, keepdims=True)  # 对 Doppler 平均
        
        radar_cube_clean = radar_cube - clutter_map
        return radar_cube_clean, clutter_map


class LidarPointCloudProcessor:
    """
    LiDAR 点云处理器
    """
    
    @staticmethod
    def project_to_range_angle(points: np.ndarray,
                               range_bins: int = 64,
                               angle_bins: int = 16,
                               max_range: float = 100.0) -> np.ndarray:
        """
        将 3D LiDAR 点云投影到 2D range-angle 平面作为 occupancy grid
        """
        occupancy = np.zeros((range_bins, angle_bins), dtype=np.float32)
        
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)  # [-pi, pi]
        
        # 离散化
        r_bin = np.clip((r / max_range) * range_bins, 0, range_bins - 1).astype(int)
        theta_bin = np.clip(((theta + np.pi) / (2 * np.pi)) * angle_bins, 0, angle_bins - 1).astype(int)
        
        # 填充 occupancy
        for rb, ab in zip(r_bin, theta_bin):
            if 0 <= rb < range_bins and 0 <= ab < angle_bins:
                occupancy[rb, ab] = 1.0
                
        return occupancy
    
    @staticmethod
    def voxelize(points: np.ndarray,
                 voxel_size: float = 0.5,
                 max_voxels: int = 10000) -> Dict[str, np.ndarray]:
        """
        将点云体素化
        """
        # 简单的体素化实现
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        voxel_indices = ((points - min_coords) / voxel_size).astype(int)
        _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
        
        return {
            'voxel_points': points[unique_indices],
            'voxel_indices': voxel_indices[unique_indices]
        }
    
    @staticmethod
    def filter_points(points: np.ndarray,
                      min_range: float = 1.0,
                      max_range: float = 100.0,
                      min_z: float = -3.0,
                      max_z: float = 5.0) -> np.ndarray:
        """
        过滤点云（范围限制）
        """
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r = np.sqrt(x**2 + y**2)
        
        mask = (r >= min_range) & (r <= max_range) & (z >= min_z) & (z <= max_z)
        return points[mask]


class DataAugmentation:
    """
    数据增强
    """
    
    @staticmethod
    def add_noise_to_radar(radar_cube: np.ndarray, 
                           noise_level: float = 0.1) -> np.ndarray:
        """
        给雷达数据加噪声
        """
        noise = np.random.randn(*radar_cube.shape) * noise_level
        return radar_cube + noise
    
    @staticmethod
    def random_flip_points(points: np.ndarray, 
                          flip_x: bool = True,
                          flip_y: bool = False) -> np.ndarray:
        """
        随机翻转点云
        """
        points_aug = points.copy()
        if flip_x and np.random.random() > 0.5:
            points_aug[:, 0] = -points_aug[:, 0]
        if flip_y and np.random.random() > 0.5:
            points_aug[:, 1] = -points_aug[:, 1]
        return points_aug
    
    @staticmethod
    def random_rotate_points(points: np.ndarray,
                            max_angle: float = np.pi / 12) -> np.ndarray:
        """
        随机旋转点云（沿 Z 轴）
        """
        angle = (np.random.random() - 0.5) * 2 * max_angle
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_matrix = np.array([[cos_a, -sin_a, 0],
                                [sin_a, cos_a, 0],
                                [0, 0, 1]])
        return points @ rot_matrix.T


class FileLoader:
    """
    文件加载器
    """
    
    @staticmethod
    def load_npy_file(file_path: str) -> np.ndarray:
        """
        加载 npy 文件
        """
        return np.load(file_path, allow_pickle=True)
    
    @staticmethod
    def load_point_cloud(file_path: str) -> np.ndarray:
        """
        加载点云文件（支持 npy, ply 等）
        """
        ext = Path(file_path).suffix.lower()
        
        if ext == '.npy':
            return FileLoader.load_npy_file(file_path)
        elif ext == '.ply':
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(file_path)
                return np.asarray(pcd.points)
            except ImportError:
                print("Open3D not available, using fallback")
                return np.load(file_path.replace('.ply', '.npy'))
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    @staticmethod
    def save_point_cloud(points: np.ndarray, file_path: str):
        """
        保存点云文件
        """
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        ext = Path(file_path).suffix.lower()
        
        if ext == '.npy':
            np.save(file_path, points)
        elif ext == '.ply':
            try:
                import open3d as o3d
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                o3d.io.write_point_cloud(file_path, pcd)
            except ImportError:
                np.save(file_path.replace('.ply', '.npy'), points)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

