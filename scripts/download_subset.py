"""
Download a conservative 50-tile subset of SpaceNet 2 (AOI 2 - Las Vegas)
Pairs high-resolution 3-band PS-RGB GeoTIFFs with corresponding building-footprint GeoJSONs.

Designed for resource safety on laptop hardware:
- Low-concurrency / single-worker sequential downloading
- Minimal memory footprint (< 50MB)
- Conservative disk footprint (~120MB)
- Strict pair matching by Image ID
"""

import os
import re
import sys
import json
import time
import urllib.request
import xml.etree.ElementTree as ET

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_IMAGES_DIR = os.path.join(BASE_DIR, "dataset", "raw", "images")
RAW_LABELS_DIR = os.path.join(BASE_DIR, "dataset", "raw", "labels")
METADATA_DIR = os.path.join(BASE_DIR, "dataset", "metadata")

os.makedirs(RAW_IMAGES_DIR, exist_ok=True)
os.makedirs(RAW_LABELS_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

S3_BASE_URL = "https://spacenet-dataset.s3.amazonaws.com"
IMG_PREFIX = "spacenet/SN2_buildings/train/AOI_2_Vegas/PS-RGB/"
LBL_PREFIX = "spacenet/SN2_buildings/train/AOI_2_Vegas/geojson_buildings/"

TARGET_TILE_COUNT = 50

def list_s3_keys(prefix, max_keys=1000):
    url = f"{S3_BASE_URL}/?prefix={prefix}&max-keys={max_keys}"
    req = urllib.request.Request(url, headers={"User-Agent": "UrbanCadastral-Downloader/1.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        root = ET.fromstring(response.read())
    return [
        elem.find("{http://s3.amazonaws.com/doc/2006-03-01/}Key").text
        for elem in root.findall("{http://s3.amazonaws.com/doc/2006-03-01/}Contents")
    ]

def download_file(url, local_path, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "UrbanCadastral-Downloader/1.0"})
            with urllib.request.urlopen(req, timeout=40) as resp, open(local_path, "wb") as out_file:
                while True:
                    chunk = resp.read(65536) # 64KB chunks
                    if not chunk:
                        break
                    out_file.write(chunk)
            return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"Error downloading {url}: {e}")
                return False

def main():
    print("=" * 60)
    print("SpaceNet 2 Vegas: Downloading 50-Tile Laptop-Safe Subset")
    print("=" * 60)

    print("Querying available tiles on AWS Open Data S3...")
    img_keys = list_s3_keys(IMG_PREFIX, max_keys=300)
    lbl_keys = list_s3_keys(LBL_PREFIX, max_keys=300)

    img_map = {}
    for k in img_keys:
        m = re.search(r"_img(\d+)\.tif$", k)
        if m:
            img_map[m.group(1)] = k

    lbl_map = {}
    for k in lbl_keys:
        m = re.search(r"_img(\d+)\.geojson$", k)
        if m:
            lbl_map[m.group(1)] = k

    common_ids = sorted(list(set(img_map.keys()) & set(lbl_map.keys())), key=lambda x: int(x))
    print(f"Discovered {len(common_ids)} valid matching candidate pairs in S3.")

    # Select tiles that contain buildings by checking GeoJSON size / features
    selected_pairs = []
    print(f"Screening candidate pairs for building density...")

    for img_id in common_ids:
        lbl_key = lbl_map[img_id]
        lbl_url = f"{S3_BASE_URL}/{lbl_key}"
        
        try:
            req = urllib.request.Request(lbl_url, headers={"User-Agent": "UrbanCadastral-Downloader/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            features = data.get("features", [])
            poly_count = len(features)
            
            # Select tiles with real building footprints (at least 5 buildings)
            if poly_count >= 5:
                selected_pairs.append({
                    "id": img_id,
                    "img_key": img_map[img_id],
                    "lbl_key": lbl_key,
                    "poly_count": poly_count
                })
                print(f"  [Tile img{img_id}]: {poly_count} building footprints (Selected {len(selected_pairs)}/{TARGET_TILE_COUNT})")
            
            if len(selected_pairs) >= TARGET_TILE_COUNT:
                break
                
            time.sleep(0.05) # Polite delay
        except Exception as e:
            continue

    print(f"\nFinal selection: {len(selected_pairs)} high-quality matching pairs.")

    metadata_list = []
    total_downloaded_bytes = 0

    print("\nStarting sequential download (low resource usage)...")
    for idx, pair in enumerate(selected_pairs, 1):
        img_id = pair["id"]
        img_filename = os.path.basename(pair["img_key"])
        lbl_filename = os.path.basename(pair["lbl_key"])

        local_img_path = os.path.join(RAW_IMAGES_DIR, img_filename)
        local_lbl_path = os.path.join(RAW_LABELS_DIR, lbl_filename)

        img_url = f"{S3_BASE_URL}/{pair['img_key']}"
        lbl_url = f"{S3_BASE_URL}/{pair['lbl_key']}"

        # Download GeoJSON
        if not os.path.exists(local_lbl_path) or os.path.getsize(local_lbl_path) == 0:
            download_file(lbl_url, local_lbl_path)

        # Download GeoTIFF
        if not os.path.exists(local_img_path) or os.path.getsize(local_img_path) == 0:
            download_file(img_url, local_img_path)

        img_size = os.path.getsize(local_img_path) if os.path.exists(local_img_path) else 0
        lbl_size = os.path.getsize(local_lbl_path) if os.path.exists(local_lbl_path) else 0
        total_downloaded_bytes += (img_size + lbl_size)

        metadata_list.append({
            "tile_id": img_id,
            "image_filename": img_filename,
            "label_filename": lbl_filename,
            "image_size_bytes": img_size,
            "label_size_bytes": lbl_size,
            "polygon_count": pair["poly_count"],
            "download_status": "success" if img_size > 0 and lbl_size > 0 else "failed"
        })

        print(f"  [{idx:02d}/{TARGET_TILE_COUNT}] Downloaded tile {img_id}: {img_filename} ({img_size/1024:.0f} KB) + GeoJSON ({lbl_size/1024:.1f} KB)")
        time.sleep(0.1)

    # Save metadata index
    index_file = os.path.join(METADATA_DIR, "dataset_index.json")
    with open(index_file, "w") as f:
        json.dump(metadata_list, f, indent=2)

    print("\n" + "=" * 60)
    print("Download Complete!")
    print(f"Total tiles downloaded: {len(metadata_list)}")
    print(f"Total download size: {total_downloaded_bytes / (1024*1024):.2f} MB")
    print(f"Metadata index saved to: {index_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
