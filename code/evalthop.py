import sys
import os
import datetime
from pathlib import Path
import torch.nn.functional as F
current_dir = Path(__file__).resolve().parent
radelft_dir = current_dir / "radelft"
sys.path.insert(0, str(radelft_dir)) 
sys.path.insert(0, str(current_dir))
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm  
from radelft.utils.compute_metrics import compute_metrics_time, compute_pd_pfa
import torchvision.transforms.functional as TF
import wandb
from thop import profile, clever_format

from model import FastFusionModel, MaxPower2DModel, CustomUNet, CustomUNetPlusPlus, CustomUNet3Plus
from losses import RadarFusionLoss
from radelft.loaders.rad_cube_loader import RADCUBE_DATASET

def main():
    MODEL_REGISTRY = {
    'CustomUNet': CustomUNet,
    'CustomUNetPlusPlus': CustomUNetPlusPlus,
    'CustomUNet3Plus': CustomUNet3Plus,
    }
    dummy_input = torch.randn(1, 512, 128, 256)
    for key in MODEL_REGISTRY:
        print(f"{key}: {MODEL_REGISTRY[key]}")
        model_class =MODEL_REGISTRY[key]

        model = MaxPower2DModel(
            model=model_class,
            in_channels=2
        )
        flops, params = profile(model, inputs=(dummy_input, ))
        flops_formatted, params_formatted = clever_format([flops, params], "%.3f")
        print(f"{key} -> FLOPs: {flops}, Params: {params}")
        print(f"{key} -> FLOPs: {flops_formatted}, Params: {params_formatted}")

if __name__ == "__main__":
    main()
    