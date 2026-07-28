import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from radelft.loaders.rad_cube_loader import RADCUBE_DATASET
from radelft.data_preparation import data_preparation
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import kurtosis, entropy  # 引入统计库高效计算特征
import scipy.ndimage as ndimage  # 用于高效实现 NMS

if __name__ == "__main__":
    params = data_preparation.get_default_params()

    params["dataset_path"] = "/scratch/shujianjia/dataset/"
    params["train_val_scenes"] = [1, 3, 4, 5, 7]
    params["test_scenes"] = [2, 6]
    params["bev"] = True
    params["cfar_folder"] = 'radar_ososos2D'

    test_dataset = RADCUBE_DATASET(mode='test', params=params)
    count = 1
    
    # 确保输出目录存在
    save_dir = '/scratch/shujianjia/visual_423/'
    os.makedirs(save_dir, exist_ok=True)

    for batchdata in test_dataset:
        # Load Data (原有的点云变量被我注释了，防止未定义报错，这里专注画特征图)
        radarcube, lidar, paths_dict = batchdata
        
        # 1. 数据预处理
        # 确保数据在 CPU 上且为 numpy 数组
        if torch.is_tensor(radarcube):
            radarcube = radarcube.cpu().numpy()
        P=radarcube
        # 提取功率信号 P (如果是复数矩阵，必须取模平方，如果是实数直接用)
        # if np.iscomplexobj(radarcube):
        #     P = np.abs(radarcube) ** 2
        # else:
        #     P = np.abs(radarcube)  # 假设已经是幅度或功率
            
        epsilon = 1e-10 # 防止除零和 log(0)
        
        # -----------------------------------------------------
        # 2. 沿着 Doppler 维度 (axis=0) 提取五大特征图
        # 最终得到的每个 map 维度均为 (512, 256) 即 (Range, Azimuth)
        # -----------------------------------------------------
        
        # (1) 绝对最大功率图 (Max Power Map)
        max_map = np.max(P, axis=0)
        max_map_db = 10 * np.log10(max_map + epsilon) # 转换为 dB 显示
        
        # (2) 平均功率图 (Mean Power Map)
        mean_map = np.mean(P, axis=0)
        mean_map_db = 10 * np.log10(mean_map + epsilon) # 转换为 dB 显示
        
        # (3) 峰值均值比 (PAPR Map)
        papr_map = max_map / (mean_map + epsilon)
        papr_map_db = 10 * np.log10(papr_map + epsilon) # PAPR 通常也用 dB 表示
        
        # (4) 峰度图 (Kurtosis Map)
        # fisher=False 代表高斯分布(纯噪声)的峰度约为3，有目标的点远大于3
        kurt_map = kurtosis(P, axis=0, fisher=False)
        kurt_map = np.nan_to_num(kurt_map, nan=0.0) # 防止全零导致 NaN
        
        # (5) 多普勒熵图 (Doppler Entropy Map)
        # scipy.stats.entropy 会自动沿 axis=0 计算概率分布并求信息熵
        # 噪声的熵大(频谱均匀)，有目标的熵小(能量集中在某个频点)
        ent_map = entropy(P, axis=0)
        ent_map = np.nan_to_num(ent_map, nan=np.log(128)) # 128 是 Doppler bin数量，最大熵为 ln(128)

        max_minus_mean = np.clip(max_map - mean_map, a_min=0, a_max=None)
        max_mul_papr = max_map * papr_map
        
        # (3) 方位角 NMS (非极大值抑制)
        local_max = ndimage.maximum_filter1d(max_map, size=3) # 窗口设为5，抑制左右各2个bin的旁瓣
        mask=(max_map == local_max) & (max_map > 0)
        nms_max=max_map * mask

        # -----------------------------------------------------
        # 3. 可视化绘图
        # -----------------------------------------------------
        cam = paths_dict['cam_path']
        img = plt.imread(cam)
        img = img[500:-150,:,:]

        # 重新设计布局：2行3列 (包含原图和5个特征图)
        layout = [
            ["camera", "max_power", "mean_power"],
            ["papr",    "kurtosis",  "entropy"],
            ["snr",    "maxpapr",  "nms_max"]
        ]
        
        fig, axd = plt.subplot_mosaic(layout, figsize=(16, 9), layout='constrained')
        
        # 绘制 Camera
        axd["camera"].imshow(img)
        axd["camera"].set_title("Camera Image")
        axd["camera"].axis('off')

        # 封装一个辅助画图函数，保证外观一致
        def plot_ra_map(ax_name, data, title, cmap='jet'):
            im = axd[ax_name].imshow(data, aspect='auto', origin='lower', cmap=cmap)
            axd[ax_name].set_title(title)
            axd[ax_name].set_xlabel('Azimuth Bins')
            axd[ax_name].set_ylabel('Range Bins')
            fig.colorbar(im, ax=axd[ax_name], fraction=0.046, pad=0.04)

        # 绘制五个特征图
        # Max 和 Mean 使用相同的 cmap，方便看出区别
        plot_ra_map("max_power", max_map, "Max Power (dB)")
        plot_ra_map("mean_power", mean_map, "Mean Power (dB)")
        
        # PAPR 越高越可能是目标
        plot_ra_map("papr", papr_map, "PAPR (Peak-to-Average, dB)", cmap='magma')
        
        # 峰度越大越尖锐(目标)，噪声通常在3左右
        plot_ra_map("kurtosis", kurt_map, "Kurtosis", cmap='inferno')
        
        # 熵越小越可能是目标(能量集中)。为了直观，这里使用不同色带或取反均可
        plot_ra_map("entropy", ent_map, "Doppler Entropy", cmap='viridis')

        plot_ra_map("snr", max_minus_mean, "Max - Mean (SNR)")
        plot_ra_map("maxpapr", max_mul_papr, "Max * PAPR", cmap='magma')
        plot_ra_map("nms_max", nms_max, "NMS Max Power")
        plt.show()
        # 保存图像
        # save_path = os.path.join(save_dir, f'{count}.png') # 建议开发阶段存为 png 速度快，论文阶段再存 eps
        # fig.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        print(f"Frame {count} processed and saved.")

        break  # 目前只处理一帧，测试完毕后可以去掉这个 break 来处理整个测试集