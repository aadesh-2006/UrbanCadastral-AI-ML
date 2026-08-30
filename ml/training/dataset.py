"""
PyTorch Dataset for SpaceNet 2 Las Vegas Building Footprint Segmentation.
Pairs raw 3-band PS-RGB GeoTIFF tiles with binary ground-truth PNG masks.
"""

import os
import rasterio
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class SpaceNetBuildingDataset(Dataset):
    def __init__(self, image_paths, mask_paths, target_size=(256, 256)):
        assert len(image_paths) == len(mask_paths), "Image and mask lists must have identical length"
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.target_size = target_size # (H, W)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # 1. Read GeoTIFF (3, H, W)
        with rasterio.open(img_path) as src:
            raw_img = src.read() # (3, H, W)

        # 2. Normalize 16-bit to float32 [0.0, 1.0] using 2-98 percentile stretch
        img = np.transpose(raw_img, (1, 2, 0)).astype(np.float32) # (H, W, 3)
        p2, p98 = np.percentile(img, (2, 98))
        if p98 > p2:
            img = np.clip((img - p2) / (p98 - p2), 0.0, 1.0)
        else:
            img = np.clip(img / (img.max() + 1e-5), 0.0, 1.0)

        # 3. Read binary mask (H, W)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found at {mask_path}")
        mask = (mask == 255).astype(np.float32) # [0.0, 1.0]

        # 4. Resize for efficient CPU processing
        if self.target_size is not None:
            H, W = self.target_size
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        # 5. Convert to PyTorch Tensors
        # Image: (H, W, 3) -> (3, H, W)
        img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        # Mask: (H, W) -> (1, H, W)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "img_path": img_path,
            "mask_path": mask_path
        }
