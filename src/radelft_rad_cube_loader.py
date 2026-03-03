
"""
RaDelft 的 rad_cube_loader.py (从 RaDelft/RaDelft-Dataset 获取
仓库获取
来源: https://github.com/RaDelft/RaDelft-Dataset/machine_learning_python/loaders/rad_cube_loader.py
"""

# 注意：这是 RaDelft 的代码片段
# 你需要把 RaDelft 仓库 clone 下来后，
# 将完整的文件放到这个位置

import sys
import torch

# append the absolute path of the parent directory
sys.path.append(sys.path[0] + "/..")

from torch.utils.data import Dataset, DataLoader
import os
import scipy.io
from data_preparation import data_preparation
import numpy as np
from loaders.TopicLoader import TopicIndex


class RADCUBE_DATASET(Dataset):
    """
    Data Loader for the RaDelf dataset.
    It initialises a dictionary with the paths to the files of the radar camera and lidar.
    This is the version for single frame as input, no temporal information.
    
    Attributes:
        mode: train, val or test
        params: a dictionary with the parameters defined in data_preparation.py
    """
    def __init__(self, mode='train', params=None):
        if mode != 'train' and mode != 'val' and mode != 'test':
            raise ValueError("mode should be either train, val or test")
        
        self.dataset_path = params['dataset_path']
        self.train_val_scenes = params['train_val_scenes']
        self.test_scenes = params['test_scenes']
        self.params = params
        
        # files are named as ELE_Frame_xxx and Pow_Frame_xxx. Lets get all files that matches these
        if mode == 'train' or mode == 'val':
            scene_set = self.train_val_scenes
        else:
            scene_set = self.test_scenes
            
        # make a dictionary, indices are keys, elevation, power, and gt_paths are values
        self.data_dict = {}
        global_array_index = 0
        
        # IMPORTANT: It is assumed that the folders structure is as given in the dataset.
        # If the folder structure is changed this will not work.
        for scene_number in scene_set:
            # Here it is assumed the folders structure is as given in the dataset.
            # If modified, this lines have to be changed, specially "Scene" "and RadarCubes"
            scene_dir = self.dataset_path + '/Scene' + str(scene_number)
            cubes_dir = scene_dir + '/RadarCubes'
            all_files = os.listdir(cubes_dir)
            power_files = [file for file in all_files if "Pow_Frame" in file]
            power_numbers = [int(file.split("_")[-1].split(".")[0]) for file in power_files]
            power_numbers.sort()
            indices = power_numbers.copy()
            indices = np.array(indices)
            
            # if train: Take 9 indices and skip one. 90% training in the train_val dataset
            if mode == 'train':
                reminder = len(indices) % 10
                if reminder != 0:
                    indices_aux = indices[:-reminder]
                    indices_aux = indices_aux.reshape(-1, 10)[:, :9].reshape(-1)
                    indices = np.concatenate([indices_aux, indices[-reminder:]])
                else:
                    indices = indices.reshape(-1, 10)[:, :9].reshape(-1)
            # if val: Skip 9 indices and take the 10th. 10% val in the train_val dataset
            elif mode == 'val':
                reminder = len(indices) % 10
                if reminder != 0:
                    indices = indices[:-reminder]
                indices = indices.reshape(-1, 10)[:, -1].reshape(-1)
            # if test we keep all the indices
            
            # get timestamp mapping
            timestamps_path = cubes_dir + '/timestamps.mat'
            frame_num_to_timestamp = scipy.io.loadmat(timestamps_path)
            frame_num_to_timestamp = frame_num_to_timestamp["unixDateTime"]
            
            rosDS_path = scene_dir + '/rosDS'
            lidar_path = rosDS_path + '/rslidar_points_clean'
            camera_dir = rosDS_path + '/ueye_left_image_rect_color'
            if params['cfar_folder'] is not None:
                cfar_dir = rosDS_path + '/' + params['cfar_folder']
                
            # get lidar timestamps
            lidar_timestamps_and_paths = data_preparation.get_timestamps_and_paths(lidar_path)
            camera_timestamps_and_paths = data_preparation.get_timestamps_and_paths(camera_dir)
            if params['cfar_folder'] is not None:
                cfar_timestamps_and_paths = data_preparation.get_timestamps_and_paths(cfar_dir)
                
            for index in indices:
                self.data_dict[global_array_index] = {}
                ## handle radar
                self.data_dict[global_array_index]["elevation_path"] = os.path.join(cubes_dir, "Ele_Frame_" + str(index) + ".mat")
                self.data_dict[global_array_index]["power_path"] = os.path.join(cubes_dir, "Pow_Frame_" + str(index) + ".mat")
                self.data_dict[global_array_index]["timestamp"] = (frame_num_to_timestamp[index - 1][0]) * 10 ** 9
                self.data_dict[global_array_index]["numpy_cube_path"] = os.path.join(cubes_dir, "radar_cube_" + str( index) + ".npy")
                
                ## handle LiDAR
                closest_lidar_time = data_preparation.closest_timestamp(self.data_dict[global_array_index]["timestamp"], lidar_timestamps_and_paths)
                self.data_dict[global_array_index]["gt_path"] = lidar_timestamps_and_paths[closest_lidar_time]
                self.data_dict[global_array_index]["gt_timestamp"] = closest_lidar_time
                
                ## handle camera
                closest_cam_time = data_preparation.closest_timestamp(self.data_dict[global_array_index]["timestamp"], camera_timestamps_and_paths)
                self.data_dict[global_array_index]["cam_path"] = camera_timestamps_and_paths[closest_cam_time]
                self.data_dict[global_array_index]["cam_timestamp"] = closest_cam_time
                
                ## handle CFAR
                if params['cfar_folder'] is not None:
                    closest_cfar_time = data_preparation.closest_timestamp(
                        self.data_dict[global_array_index]["timestamp"], cfar_timestamps_and_paths)
                    self.data_dict[global_array_index]["cfar_path"] = cfar_timestamps_and_paths[closest_cfar_time]
                    self.data_dict[global_array_index]["cfar_timestamp"] = closest_cfar_time
                
                global_array_index = global_array_index + 1
        
        # print division line
        print("-"*50)
        print(mode + " dataset loaded with " + str(len(self.data_dict)) + " samples")
        print("scenes used: " + str(scene_set))
        # print division line
        print("-"*50)
        
    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, idx):
        # load elevation and power
        if not self.params['bev']:
            elevation = scipy.io.loadmat(self.data_dict[idx]["elevation_path"])["elevationIndex"]
            elevation = elevation.astype(np.single)
            elevation = np.nan_to_num(elevation, nan=17.0)
            elevation = elevation / 34
            power = scipy.io.loadmat(self.data_dict[idx]["power_path"])["radarCube"]
            power = power.astype(np.single)
            # Hardcoded maximum value after data exploration
            power = power / 8998.5576
            # combine them into a single cube with 2 channels
            if not self.params['bev']:
                input_cube = np.stack((power, elevation))
            else:
                input_cube = power
        
        # load gt
        gt_cloud = data_preparation.read_pointcloud(self.data_dict[idx]["gt_path"], mode="rs_lidar_clean")
        item_params = self.data_dict[idx]  # this is a dictionary with all the paths and timestamps
        gt_cube = data_preparation.lidarpc_to_lidarcube(gt_cloud, self.params)
        
        if not self.params['bev']:
            zero_pad = np.zeros([2, 12, 128, 240], dtype='single')
            input_cube = np.concatenate([input_cube, zero_pad], axis=1)
            zero_pad = np.zeros([2, 512, 128, 8], dtype='single')
            input_cube = np.concatenate([zero_pad, input_cube, zero_pad], axis=3)
            input_cube = np.transpose(input_cube, (0, 2, 1, 3))  # (C, H, W)
        else:
            zero_pad = np.zeros([12, 128, 240], dtype='single')
            input_cube = np.concatenate([input_cube, zero_pad])
            zero_pad = np.zeros([512, 128, 8], dtype='single')
            input_cube = np.concatenate([zero_pad, input_cube, zero_pad], axis=2)
            input_cube = np.transpose(input_cube, (1, 0, 2))  # (C, H, W)
        
        return input_cube, gt_cube, item_params


# ==========================================
# 说明：
# ==========================================
# 这是 RaDelft 的 rad_cube_loader.py 片段
# 要完整使用 RaDelft 的代码，你需要：
#
# 1. 在你本地 clone RaDelft 仓库：
#    git clone https://github.com/RaDelft/RaDelft-Dataset.git
#
# 2. 把 machine_learning_python/ 下的内容复制到我们项目的 src/radelft/
#
# 3. 然后我们的项目就可以直接 import RaDelft 的 loaders, utils, visualizers 了！
#
# 我们保留：
# - model.py 中的网络架构（我们的创新）
# - losses.py 中的 Loss Functions（我们的创新）
# ==========================================

