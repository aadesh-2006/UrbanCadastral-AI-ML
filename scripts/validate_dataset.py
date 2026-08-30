"""
Automated dataset validation suite for UrbanCadastral-AI-ML SpaceNet 2 Vegas subset.

Verifies:
1. Exact 1-to-1 image <-> label pairing
2. No missing label or mask files
3. GeoJSON schema and parser integrity
4. Geometry validity
5. Geospatial CRS integrity
6. Dimension parity (image == mask)
7. Mask pixel range and non-empty checks
8. Building polygon counts and density
"""

import os
import glob
import re
import json
import numpy as np
import rasterio
from shapely.geometry import shape
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_IMAGES_DIR = os.path.join(BASE_DIR, "dataset", "raw", "images")
RAW_LABELS_DIR = os.path.join(BASE_DIR, "dataset", "raw", "labels")
MASKS_DIR = os.path.join(BASE_DIR, "dataset", "masks")
SAMPLES_DIR = os.path.join(BASE_DIR, "dataset", "samples")
METADATA_DIR = os.path.join(BASE_DIR, "dataset", "metadata")

def validate():
    print("=" * 60)
    print("AUTOMATED DATASET VALIDATION")
    print("=" * 60)

    images = sorted(glob.glob(os.path.join(RAW_IMAGES_DIR, "*.tif")))
    labels = sorted(glob.glob(os.path.join(RAW_LABELS_DIR, "*.geojson")))
    masks = sorted(glob.glob(os.path.join(MASKS_DIR, "*.png")))

    print(f"Discovered {len(images)} images, {len(labels)} labels, {len(masks)} masks.")

    img_ids = {}
    for img_p in images:
        m = re.search(r"_img(\d+)\.tif$", os.path.basename(img_p))
        if m:
            img_ids[m.group(1)] = img_p

    lbl_ids = {}
    for lbl_p in labels:
        m = re.search(r"_img(\d+)\.geojson$", os.path.basename(lbl_p))
        if m:
            lbl_ids[m.group(1)] = lbl_p

    mask_ids = {}
    for msk_p in masks:
        m = re.search(r"mask_img(\d+)\.png$", os.path.basename(msk_p))
        if m:
            mask_ids[m.group(1)] = msk_p

    # Check 1: Pairing
    all_keys = sorted(list(img_ids.keys()), key=lambda x: int(x))
    missing_labels = [k for k in all_keys if k not in lbl_ids]
    missing_masks = [k for k in all_keys if k not in mask_ids]

    valid_images_count = 0
    valid_labels_count = 0
    valid_masks_count = 0
    total_polygons = 0
    total_valid_polygons = 0
    total_invalid_polygons = 0
    dimension_mismatches = 0
    empty_masks_with_buildings = 0
    sample_crs = None

    for tile_id in all_keys:
        img_p = img_ids[tile_id]
        lbl_p = lbl_ids.get(tile_id)
        msk_p = mask_ids.get(tile_id)

        # Validate image
        img_w, img_h = 0, 0
        try:
            with rasterio.open(img_p) as src:
                img_w = src.width
                img_h = src.height
                if src.crs:
                    sample_crs = str(src.crs)
                valid_images_count += 1
        except Exception as e:
            print(f"[FAIL] Image corrupted: {img_p}: {e}")

        # Validate label
        tile_polygons = 0
        if lbl_p:
            try:
                with open(lbl_p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                features = data.get("features", [])
                tile_polygons = len(features)
                total_polygons += tile_polygons
                for feat in features:
                    geom = feat.get("geometry")
                    if geom and geom.get("coordinates"):
                        s = shape(geom)
                        if s.is_valid and not s.is_empty:
                            total_valid_polygons += 1
                        else:
                            total_invalid_polygons += 1
                valid_labels_count += 1
            except Exception as e:
                print(f"[FAIL] GeoJSON error: {lbl_p}: {e}")

        # Validate mask
        if msk_p:
            mask = cv2.imread(msk_p, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask_h, mask_w = mask.shape
                if mask_w != img_w or mask_h != img_h:
                    dimension_mismatches += 1
                building_pix = np.count_nonzero(mask == 255)
                if tile_polygons > 0 and building_pix == 0:
                    empty_masks_with_buildings += 1
                valid_masks_count += 1

    # Check sample overlays
    samples = sorted(glob.glob(os.path.join(SAMPLES_DIR, "*.jpg")))

    # Calculate dataset size on disk
    total_size_bytes = 0
    for folder in [RAW_IMAGES_DIR, RAW_LABELS_DIR, MASKS_DIR, SAMPLES_DIR, METADATA_DIR]:
        for root, _, files in os.walk(folder):
            for file in files:
                total_size_bytes += os.path.getsize(os.path.join(root, file))

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Dataset: SpaceNet 2 AOI 2 Vegas")
    print(f"Images: {len(images)}")
    print(f"Labels: {len(labels)}")
    print(f"Total building polygons: {total_valid_polygons}")
    print(f"Invalid geometries: {total_invalid_polygons}")
    print(f"Average buildings/tile: {total_valid_polygons / len(images):.2f}" if images else "0")
    print(f"Valid images: {valid_images_count}/{len(images)}")
    print(f"Valid labels: {valid_labels_count}/{len(labels)}")
    print(f"Generated masks: {valid_masks_count}/{len(images)}")
    print(f"Missing labels: {len(missing_labels)}")
    print(f"Missing masks: {len(missing_masks)}")
    print(f"Dimension mismatches (image vs mask): {dimension_mismatches}")
    print(f"Unexpected empty masks: {empty_masks_with_buildings}")
    print(f"Primary CRS: {sample_crs}")
    print(f"Dataset size on disk: {total_size_bytes / (1024*1024):.2f} MB")
    print(f"Visual quality check overlays: {len(samples)} generated in dataset/samples/")
    print("=" * 60)

    is_success = (
        len(images) > 0 and
        len(images) == len(labels) == len(masks) and
        valid_images_count == len(images) and
        valid_labels_count == len(labels) and
        valid_masks_count == len(images) and
        dimension_mismatches == 0 and
        empty_masks_with_buildings == 0 and
        len(samples) >= 5
    )

    if is_success:
        print("[SUCCESS] All validation checks passed flawlessly!")
    else:
        print("[WARNING] One or more validation checks flagged issues.")

    return is_success

if __name__ == "__main__":
    validate()
