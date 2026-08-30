"""
Rasterize SpaceNet 2 GeoJSON building footprints into binary ground-truth masks.

Convention:
0   = background
255 = building footprint

Uses each GeoTIFF's actual geospatial affine transform and CRS.
Ensures exact dimension matching (height x width).
Processes one tile at a time for minimal CPU and RAM usage.
"""

import os
import re
import glob
import json
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_IMAGES_DIR = os.path.join(BASE_DIR, "dataset", "raw", "images")
RAW_LABELS_DIR = os.path.join(BASE_DIR, "dataset", "raw", "labels")
MASKS_DIR = os.path.join(BASE_DIR, "dataset", "masks")
METADATA_DIR = os.path.join(BASE_DIR, "dataset", "metadata")

os.makedirs(MASKS_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

def rasterize_tile(img_path, lbl_path, mask_path):
    with rasterio.open(img_path) as src:
        width = src.width
        height = src.height
        transform = src.transform
        crs = str(src.crs)
        bounds = list(src.bounds)

    with open(lbl_path, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    shapes_to_draw = []
    invalid_geometries = 0
    raw_features = geojson_data.get("features", [])

    for feat in raw_features:
        geom = feat.get("geometry")
        if not geom or not geom.get("coordinates"):
            continue
        try:
            poly = shape(geom)
            if not poly.is_valid:
                poly = poly.buffer(0) # Attempt standard GIS self-intersection fix
            if poly.is_valid and not poly.is_empty:
                shapes_to_draw.append((poly.__geo_interface__, 255))
            else:
                invalid_geometries += 1
        except Exception:
            invalid_geometries += 1

    # Exact affine rasterization matching GeoTIFF coordinate grid
    if shapes_to_draw:
        mask = rasterize(
            shapes=shapes_to_draw,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            default_value=255,
            dtype=np.uint8
        )
    else:
        mask = np.zeros((height, width), dtype=np.uint8)

    # Save mask as lossless 8-bit PNG
    mask_img = Image.fromarray(mask, mode="L")
    mask_img.save(mask_path, format="PNG")

    building_pixels = int(np.count_nonzero(mask == 255))
    building_pct = (building_pixels / (width * height)) * 100.0

    return {
        "width": width,
        "height": height,
        "crs": crs,
        "bounds": bounds,
        "total_polygons": len(raw_features),
        "valid_polygons": len(shapes_to_draw),
        "invalid_geometries": invalid_geometries,
        "building_pixels": building_pixels,
        "building_pixel_pct": round(building_pct, 3),
        "mask_path": mask_path
    }

def main():
    print("=" * 60)
    print("Rasterizing Building Footprints from GeoJSON to Binary Masks")
    print("=" * 60)

    image_files = sorted(glob.glob(os.path.join(RAW_IMAGES_DIR, "*.tif")))
    if not image_files:
        print(f"No GeoTIFF images found in {RAW_IMAGES_DIR}!")
        return

    print(f"Discovered {len(image_files)} raw image files to process.")

    results = []
    total_buildings = 0
    total_valid_polygons = 0
    total_invalid = 0

    for idx, img_path in enumerate(image_files, 1):
        filename = os.path.basename(img_path)
        m = re.search(r"_img(\d+)\.tif$", filename)
        if not m:
            continue
        img_id = m.group(1)

        lbl_filename = f"SN2_buildings_train_AOI_2_Vegas_geojson_buildings_img{img_id}.geojson"
        lbl_path = os.path.join(RAW_LABELS_DIR, lbl_filename)

        if not os.path.exists(lbl_path):
            print(f"  [ERROR] Missing label file for tile {img_id}: {lbl_path}")
            continue

        mask_filename = f"mask_img{img_id}.png"
        mask_path = os.path.join(MASKS_DIR, mask_filename)

        stats = rasterize_tile(img_path, lbl_path, mask_path)
        stats["tile_id"] = img_id
        stats["image_filename"] = filename
        stats["label_filename"] = lbl_filename
        stats["mask_filename"] = mask_filename

        results.append(stats)
        total_buildings += stats["total_polygons"]
        total_valid_polygons += stats["valid_polygons"]
        total_invalid += stats["invalid_geometries"]

        print(f"  [{idx:02d}/{len(image_files)}] Tile {img_id}: {stats['valid_polygons']} buildings rasterized | "
              f"Mask: {stats['width']}x{stats['height']} | Building pixels: {stats['building_pixels']} ({stats['building_pixel_pct']}%)")

    # Update index metadata
    index_file = os.path.join(METADATA_DIR, "dataset_index.json")
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Generate dataset summary
    summary = {
        "dataset": "SpaceNet 2 Building Detection (AOI 2 - Las Vegas)",
        "total_images": len(results),
        "total_labels": len(results),
        "total_masks": len(results),
        "total_polygons": total_buildings,
        "valid_polygons": total_valid_polygons,
        "invalid_geometries": total_invalid,
        "average_buildings_per_tile": round(total_valid_polygons / len(results), 2) if results else 0,
        "crs": results[0]["crs"] if results else "Unknown",
        "resolution": "650x650 px (30cm GSD)",
        "mask_convention": "0 = background, 255 = building footprint"
    }

    summary_file = os.path.join(METADATA_DIR, "dataset_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("Rasterization Complete!")
    print(f"Total masks generated: {len(results)}/{len(image_files)}")
    print(f"Total building footprints: {total_valid_polygons}")
    print(f"Average buildings per tile: {summary['average_buildings_per_tile']}")
    print(f"Summary saved to: {summary_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
