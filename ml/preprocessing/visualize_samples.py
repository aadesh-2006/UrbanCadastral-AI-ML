"""
Visual quality check: Generate ground-truth building footprint overlays on aerial imagery.
Produces at least 5 sample overlay images saved in dataset/samples/.
"""

import os
import random
import glob
import re
import json
import numpy as np
import rasterio
import cv2
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_IMAGES_DIR = os.path.join(BASE_DIR, "dataset", "raw", "images")
MASKS_DIR = os.path.join(BASE_DIR, "dataset", "masks")
SAMPLES_DIR = os.path.join(BASE_DIR, "dataset", "samples")
METADATA_DIR = os.path.join(BASE_DIR, "dataset", "metadata")

os.makedirs(SAMPLES_DIR, exist_ok=True)

def create_overlay_sample(img_path, mask_path, out_path, sample_num, tile_id, info=""):
    # Read 3-band GeoTIFF
    with rasterio.open(img_path) as src:
        raw_img = src.read() # (3, H, W)
    
    # Read binary mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask at {mask_path}")

    # Convert 16-bit to 8-bit RGB with robust percentile stretch
    rgb = np.transpose(raw_img, (1, 2, 0)).astype(np.float32)
    p2, p98 = np.percentile(rgb, (2, 98))
    if p98 > p2:
        rgb_norm = np.clip((rgb - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
    else:
        rgb_norm = np.clip(rgb / (rgb.max() + 1e-5) * 255.0, 0, 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    bgr = cv2.cvtColor(rgb_norm, cv2.COLOR_RGB2BGR)

    # Create semi-transparent overlay
    # Golden-yellow fill for building footprint: BGR (0, 215, 255)
    overlay = bgr.copy()
    color_fill = np.array([0, 220, 255], dtype=np.uint8)
    building_mask = (mask == 255)
    overlay[building_mask] = (overlay[building_mask] * 0.45 + color_fill * 0.55).astype(np.uint8)

    # Find and draw crisp building footprint contour boundaries
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Bright green boundary line (0, 255, 0)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    # Add header banner with metadata
    banner_height = 36
    h, w, _ = overlay.shape
    canvas = np.zeros((h + banner_height, w, 3), dtype=np.uint8)
    # Dark slate banner background
    canvas[:banner_height, :] = (30, 30, 30)
    canvas[banner_height:, :] = overlay

    # Text label
    header_text = f"Sample {sample_num:02d} | Tile {tile_id} | SpaceNet 2 Vegas | Footprints: {len(contours)} buildings"
    cv2.putText(canvas, header_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

    # Save visualization
    cv2.imwrite(out_path, canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return len(contours)

def main():
    print("=" * 60)
    print("Generating Sample Ground-Truth Visualization Overlays")
    print("=" * 60)

    index_file = os.path.join(METADATA_DIR, "dataset_index.json")
    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            records = json.load(f)
    else:
        print("Metadata index not found! Run rasterize_buildings.py first.")
        return

    # Filter tiles that have building footprints
    rich_tiles = [r for r in records if r.get("valid_polygons", 0) >= 10]
    if len(rich_tiles) < 5:
        rich_tiles = records

    # Fix random seed for reproducible sample selection
    random.seed(42)
    selected_samples = random.sample(rich_tiles, min(5, len(rich_tiles)))

    print(f"Selected {len(selected_samples)} representative tiles for visual inspection.")

    for idx, item in enumerate(selected_samples, 1):
        tile_id = item["tile_id"]
        img_path = os.path.join(RAW_IMAGES_DIR, item["image_filename"])
        mask_path = os.path.join(MASKS_DIR, item["mask_filename"])
        out_filename = f"sample_{idx:02d}_overlay.jpg"
        out_path = os.path.join(SAMPLES_DIR, out_filename)

        count = create_overlay_sample(img_path, mask_path, out_path, idx, tile_id)
        print(f"  [{idx:02d}/05] Saved {out_filename} (Tile {tile_id}, {count} buildings) -> {out_path}")

    print("\n" + "=" * 60)
    print("Visual Samples Generated Successfully!")
    print(f"Stored in: {SAMPLES_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
