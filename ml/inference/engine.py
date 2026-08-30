"""
AerialInferenceEngine: Reusable deep learning inference engine for aerial imagery.
Powered by LightUNet trained on high-resolution SpaceNet 2 satellite imagery.

Features:
- Multi-format ingestion: GeoTIFF (.tif, .tiff), JPG, JPEG, PNG, WEBP
- Strict CPU-first safety profile (pinned to 4 threads, single-tile/memory-safe)
- Automatic georeferencing detection & coordinate transformation
- Tiled inference for large images & direct resize mode for tile patches
- Mask post-processing and conservative polygonization
- GeoJSON export (geographic CRS84 for GeoTIFFs, pixel coords for plain images)
- Clean visual overlay generation
"""

import os
import sys
import json
import time
import math
import cv2
import numpy as np
import psutil
import torch
from shapely.geometry import Polygon, mapping

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from ml.models.light_unet import get_model
from ml.training.losses import compute_dice, compute_iou

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "ml", "outputs", "best_model.pth")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "ml", "outputs", "inference")

class AerialInferenceEngine:
    def __init__(self, model_path=None, num_threads=4):
        self.num_threads = num_threads
        torch.set_num_threads(num_threads)

        self.model_path = model_path or DEFAULT_MODEL_PATH
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        # Initialize LightUNet on CPU
        self.model = get_model(in_channels=3, num_classes=1, base_filters=16)
        state_dict = torch.load(self.model_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.proc = psutil.Process()

    def load_image(self, image_path):
        """
        Loads and validates an aerial image.
        Returns:
            rgb_float32: (H, W, 3) normalized float32 in [0.0, 1.0]
            is_georeferenced: bool
            affine_transform: rasterio Affine or None
            crs_str: str or None
            orig_h: int
            orig_w: int
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        ext = os.path.splitext(image_path)[1].lower()
        is_georeferenced = False
        affine_transform = None
        crs_str = None

        # Attempt GeoTIFF loading if rasterio is available
        if ext in [".tif", ".tiff"] and HAS_RASTERIO:
            try:
                with rasterio.open(image_path) as src:
                    raw_img = src.read() # (C, H, W)
                    orig_h = src.height
                    orig_w = src.width
                    if src.crs:
                        crs_str = str(src.crs)
                        affine_transform = src.transform
                        is_georeferenced = True

                    # Extract RGB channels (first 3 channels)
                    if raw_img.shape[0] >= 3:
                        raw_rgb = raw_img[:3]
                    else:
                        # Grayscale to 3 channels
                        raw_rgb = np.repeat(raw_img[:1], 3, axis=0)

                    img_hwc = np.transpose(raw_rgb, (1, 2, 0)).astype(np.float32)
            except Exception:
                img_hwc = None
        else:
            img_hwc = None

        # Fallback to OpenCV / standard image loading
        if img_hwc is None:
            bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if bgr is None:
                raise ValueError(f"Unsupported or corrupted image format: {image_path}")

            if len(bgr.shape) == 2:
                # Grayscale -> RGB
                rgb = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
            elif bgr.shape[2] == 4:
                # BGRA -> RGB
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)
            else:
                # BGR -> RGB
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            orig_h, orig_w = rgb.shape[:2]
            img_hwc = rgb.astype(np.float32)

        # Robust 2-98% percentile stretch (matches SpaceNet training preprocessing)
        p2, p98 = np.percentile(img_hwc, (2, 98))
        if p98 > p2:
            rgb_norm = np.clip((img_hwc - p2) / (p98 - p2), 0.0, 1.0)
        else:
            rgb_norm = np.clip(img_hwc / (img_hwc.max() + 1e-5), 0.0, 1.0)

        return rgb_norm, is_georeferenced, affine_transform, crs_str, orig_h, orig_w

    def _infer_patch(self, patch_hwc):
        """Runs LightUNet on a single (256, 256, 3) float32 patch."""
        tensor = torch.from_numpy(np.transpose(patch_hwc, (2, 0, 1))).unsqueeze(0).float()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits)
        return probs.squeeze().numpy() # (256, 256)

    def _predict_resize(self, rgb_norm, orig_h, orig_w):
        """Standard resize inference mode (matches Milestones 5, 6, 7)."""
        patch = cv2.resize(rgb_norm, (256, 256), interpolation=cv2.INTER_LINEAR)
        prob_256 = self._infer_patch(patch)
        # Scale probability map back to original dimensions
        prob_orig = cv2.resize(prob_256, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return prob_orig

    def _predict_tiled(self, rgb_norm, orig_h, orig_w, tile_size=256, stride=192):
        """
        Memory-safe sliding window inference for larger images.
        Sequentially tiles image, handles borders, blends overlaps.
        """
        pad_h = max(0, int(math.ceil((orig_h - tile_size) / stride)) * stride + tile_size - orig_h) if orig_h > tile_size else (tile_size - orig_h)
        pad_w = max(0, int(math.ceil((orig_w - tile_size) / stride)) * stride + tile_size - orig_w) if orig_w > tile_size else (tile_size - orig_w)

        padded = np.pad(rgb_norm, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        padded_h, padded_w = padded.shape[:2]

        prob_accum = np.zeros((padded_h, padded_w), dtype=np.float32)
        weight_accum = np.zeros((padded_h, padded_w), dtype=np.float32)

        # 2D Hann window for smooth cosine blending across tile seams
        hann_1d = np.hanning(tile_size)
        hann_2d = np.outer(hann_1d, hann_1d) + 1e-4

        y_steps = range(0, padded_h - tile_size + 1, stride)
        x_steps = range(0, padded_w - tile_size + 1, stride)

        for y in y_steps:
            for x in x_steps:
                patch = padded[y:y+tile_size, x:x+tile_size]
                patch_prob = self._infer_patch(patch)
                prob_accum[y:y+tile_size, x:x+tile_size] += patch_prob * hann_2d
                weight_accum[y:y+tile_size, x:x+tile_size] += hann_2d

        blended = prob_accum / np.maximum(weight_accum, 1e-5)
        # Crop out padding to match original dimensions
        return blended[:orig_h, :orig_w]

    def run(self, image_path, output_dir=None, mode="auto", gt_mask_path=None):
        """
        Executes end-to-end aerial building inference.
        Returns the structured output contract.
        """
        output_dir = output_dir or DEFAULT_OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        ram_start = self.proc.memory_info().rss / (1024 * 1024)

        # 1. Load image
        rgb_norm, is_georeferenced, affine_transform, crs_str, orig_h, orig_w = self.load_image(image_path)

        # 2. Select execution mode
        if mode == "auto":
            # For images up to 768x768 (like SpaceNet 650x650), use resize mode for exact parity
            mode_used = "resize" if max(orig_h, orig_w) <= 768 else "tiled"
        else:
            mode_used = mode

        # 3. Model Inference Timing
        t0 = time.perf_counter()
        if mode_used == "tiled":
            prob_map = self._predict_tiled(rgb_norm, orig_h, orig_w)
        else:
            prob_map = self._predict_resize(rgb_norm, orig_h, orig_w)
        t1 = time.perf_counter()
        inference_time_ms = (t1 - t0) * 1000.0

        # Threshold at 0.5
        binary_mask = (prob_map > 0.5).astype(np.uint8) * 255

        # 4. Mask Cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        # Connected component filtering: min area scaled to image dimensions
        # At 256x256 min area was 20; scale proportional to area (H*W / 256^2)
        scale_area = (orig_h * orig_w) / (256.0 * 256.0)
        min_comp_area = max(15, int(20 * scale_area))

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
        final_mask = np.zeros_like(cleaned_mask)
        for lbl_idx in range(1, num_labels):
            if stats[lbl_idx, cv2.CC_STAT_AREA] >= min_comp_area:
                final_mask[labels == lbl_idx] = 255

        # Save Mask PNG
        mask_out_path = os.path.join(output_dir, f"{base_name}_mask.png")
        cv2.imwrite(mask_out_path, final_mask)

        # 5. Polygonization
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_polygons = []
        geojson_features = []
        confidences = []

        for idx, cnt in enumerate(contours, 1):
            area_px = cv2.contourArea(cnt)
            if area_px < (min_comp_area * 0.8):
                continue

            # Conservative simplification
            approx = cv2.approxPolyDP(cnt, epsilon=1.2 * math.sqrt(scale_area), closed=True)
            if len(approx) < 3:
                continue

            # Compute genuine confidence: mean probability under contour
            cnt_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
            mean_conf = float(np.mean(prob_map[cnt_mask == 255]))
            confidences.append(mean_conf)

            # Coordinates handling
            coords = []
            for pt in approx:
                px = float(pt[0][0])
                py = float(pt[0][1])
                if is_georeferenced and affine_transform is not None:
                    # Affine mapping: (px, py) -> (lon, lat)
                    geo_x, geo_y = affine_transform * (px, py)
                    coords.append([round(geo_x, 7), round(geo_y, 7)])
                else:
                    # Pure image coordinates [x, y]
                    coords.append([round(px, 1), round(py, 1)])

            if coords[0] != coords[-1]:
                coords.append(coords[0])

            try:
                poly_geom = Polygon(coords)
                if not poly_geom.is_valid:
                    poly_geom = poly_geom.buffer(0)

                if poly_geom.is_valid and not poly_geom.is_empty:
                    valid_polygons.append({
                        "id": f"BLD-{idx:03d}",
                        "contour": cnt,
                        "confidence": round(mean_conf, 3),
                        "geom": poly_geom
                    })

                    feature = {
                        "type": "Feature",
                        "id": f"BLD-{idx:03d}",
                        "geometry": mapping(poly_geom),
                        "properties": {
                            "id": f"BLD-{idx:03d}",
                            "class": "building",
                            "source": "LightUNet",
                            "confidence": round(mean_conf, 3),
                            "pixel_area": round(area_px, 1),
                            "georeferenced": is_georeferenced
                        }
                    }
                    if is_georeferenced:
                        # 30cm GSD approximation: 0.3 * 0.3 = 0.09 m2/px
                        feature["properties"]["estimated_ground_area_sqm"] = round(area_px * 0.09, 1)

                    geojson_features.append(feature)
            except Exception:
                continue

        # 6. Export GeoJSON
        geojson_out_path = os.path.join(output_dir, f"{base_name}_buildings.geojson")
        geojson_data = {
            "type": "FeatureCollection",
            "name": f"Inference_{base_name}_Buildings",
            "properties": {
                "input_image": os.path.basename(image_path),
                "model": "LightUNet",
                "checkpoint": os.path.basename(self.model_path),
                "inference_time_ms": round(inference_time_ms, 2),
                "mode": mode_used,
                "georeferenced": is_georeferenced,
                "crs": crs_str if is_georeferenced else "None (pixel coordinates)",
                "building_count": len(valid_polygons),
                "mean_confidence": round(float(np.mean(confidences)), 3) if confidences else 0.0
            },
            "features": geojson_features
        }
        if is_georeferenced:
            geojson_data["crs"] = {
                "type": "name",
                "properties": {"name": crs_str or "urn:ogc:def:crs:OGC:1.3:CRS84"}
            }

        with open(geojson_out_path, "w", encoding="utf-8") as f:
            json.dump(geojson_data, f, indent=2)

        # 7. Render Visual Overlay
        bgr_full = cv2.cvtColor((rgb_norm * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        preview_out_path = os.path.join(output_dir, f"{base_name}_preview.jpg")
        cv2.imwrite(preview_out_path, bgr_full, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        overlay = bgr_full.copy()
        fill_color = np.array([0, 215, 255], dtype=np.uint8) # Amber fill

        for item in valid_polygons:
            cnt = item["contour"]
            mask_single = np.zeros((orig_h, orig_w), dtype=np.uint8)
            cv2.drawContours(mask_single, [cnt], -1, 255, -1)

            poly_bool = (mask_single == 255)
            overlay[poly_bool] = (overlay[poly_bool] * 0.55 + fill_color * 0.45).astype(np.uint8)
            cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), 2)

            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                conf_str = f"{int(item['confidence']*100)}%"
                cv2.putText(overlay, conf_str, (cX - 12, cY + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        banner_h = 36
        canvas = np.zeros((orig_h + banner_h, orig_w, 3), dtype=np.uint8)
        canvas[:banner_h, :] = (30, 30, 30)
        canvas[banner_h:, :] = overlay

        mean_c_str = f"{float(np.mean(confidences))*100:.1f}%" if confidences else "N/A"
        header_text = f"Image: {base_name} | LightUNet Inference -> {len(valid_polygons)} Buildings | Mean Conf: {mean_c_str}"
        cv2.putText(canvas, header_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

        overlay_out_path = os.path.join(output_dir, f"{base_name}_overlay.jpg")
        cv2.imwrite(overlay_out_path, canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # 8. Optional Ground-Truth Evaluation (if provided for benchmarking)
        eval_metrics = {}
        if gt_mask_path and os.path.exists(gt_mask_path):
            gt_mask_raw = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask_raw is not None:
                gt_resized = cv2.resize(gt_mask_raw, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                gt_t = torch.from_numpy((gt_resized == 255).astype(np.float32)).unsqueeze(0).unsqueeze(0)
                pr_t = torch.from_numpy((final_mask == 255).astype(np.float32)).unsqueeze(0).unsqueeze(0)
                eval_metrics["validation_dice"] = round(compute_dice(pr_t, gt_t), 4)
                eval_metrics["validation_iou"] = round(compute_iou(pr_t, gt_t), 4)

        # 9. Output Contract
        pixel_polys = []
        for item, feat in zip(valid_polygons, geojson_features):
            pts = item["contour"].reshape(-1, 2).tolist()
            pixel_polys.append({
                "id": item["id"],
                "confidence": item["confidence"],
                "points": pts,
                "properties": feat["properties"]
            })

        contract = {
            "input_image": image_path,
            "model": "LightUNet",
            "model_checkpoint": os.path.basename(self.model_path),
            "image_size": [orig_h, orig_w, 3],
            "building_count": len(valid_polygons),
            "mean_confidence": round(float(np.mean(confidences)), 3) if confidences else 0.0,
            "inference_time_ms": round(inference_time_ms, 2),
            "georeferenced": is_georeferenced,
            "crs": crs_str if is_georeferenced else "None (pixel coordinates)",
            "preview_path": preview_out_path,
            "mask_path": mask_out_path,
            "geojson_path": geojson_out_path,
            "overlay_path": overlay_out_path,
            "polygons_pixel": pixel_polys
        }
        if eval_metrics:
            contract["ground_truth_metrics"] = eval_metrics

        return contract
