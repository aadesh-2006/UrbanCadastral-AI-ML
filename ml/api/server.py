"""
FastAPI Server for UrbanCadastral-AI-ML Inference Engine.
Connects the React/TypeScript frontend to the real LightUNet inference engine.

Hardware Safety:
- Single-inference execution, CPU-only (pinned to 4 threads)
- Lightweight memory footprint, releases buffers after inference
"""

import os
import sys
import re
import time
import json
import shutil
import logging
import traceback
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger("uvicorn.error")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from ml.inference.engine import AerialInferenceEngine

app = FastAPI(
    title="UrbanCadastral AI-ML API",
    description="Real LightUNet Aerial Building Footprint Inference Engine",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "ml", "outputs")
INFERENCE_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "inference")
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "dataset", "uploads")
RAW_IMAGES_DIR = os.path.join(PROJECT_ROOT, "dataset", "raw", "images")
MASKS_DIR = os.path.join(PROJECT_ROOT, "dataset", "masks")

os.makedirs(INFERENCE_OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Mount static outputs
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

# Initialize Engine (singleton, lazy-loaded on startup)
engine = None

@app.on_event("startup")
def startup_event():
    global engine
    engine = AerialInferenceEngine(num_threads=4)

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "model": "LightUNet (1.94M params)",
        "checkpoint": "best_model.pth (Epoch 18)",
        "device": "cpu",
        "threads": 4,
        "supported_formats": [".tif", ".tiff", ".jpg", ".jpeg", ".png"]
    }

@app.get("/api/presets")
def get_presets():
    """Returns verified presets for instant demo testing."""
    return [
        {
            "id": "tile_127",
            "name": "SpaceNet Tile 127 (Residential)",
            "description": "Dense suburban street in Las Vegas. Best validation tile (88.0% Dice).",
            "format": "GeoTIFF",
            "dimensions": "650 x 650",
            "georeferenced": True,
            "crs": "EPSG:4326",
            "preview_url": "/outputs/inference/SN2_buildings_train_AOI_2_Vegas_PS-RGB_img127_preview.jpg"
        },
        {
            "id": "tile_1006",
            "name": "SpaceNet Tile 1006 (Townhomes & Utility)",
            "description": "Multi-family townhomes and industrial substation yard. Complex validation tile (66.5% Dice).",
            "format": "GeoTIFF",
            "dimensions": "650 x 650",
            "georeferenced": True,
            "crs": "EPSG:4326",
            "preview_url": "/outputs/inference/SN2_buildings_train_AOI_2_Vegas_PS-RGB_img1006_preview.jpg"
        },
        {
            "id": "plain_jpg",
            "name": "Standard Aerial Photograph (JPEG)",
            "description": "Standard unreferenced 8-bit JPEG image. Polygonized in local image pixel coordinates.",
            "format": "JPEG",
            "dimensions": "650 x 650",
            "georeferenced": False,
            "crs": "None (pixel coordinates)",
            "preview_url": "/outputs/inference/test_plain_aerial_preview.jpg"
        },
        {
            "id": "external_drone",
            "name": "External Drone Orthophoto (Domain Shift)",
            "description": "Rettern village drone orthophoto (Germany, CC-BY-SA). Unseen terracotta roofs, solar arrays & lush vegetation.",
            "format": "JPEG",
            "dimensions": "650 x 650",
            "georeferenced": False,
            "crs": "None (pixel coordinates)",
            "preview_url": "/outputs/inference/external_drone_rettern_crop650_preview.jpg"
        }
    ]

@app.post("/api/inference")
async def run_inference(
    file: Optional[UploadFile] = File(None),
    preset_id: Optional[str] = Query(None),
    mode: str = Query("auto", pattern="^(auto|resize|tiled)$")
):
    global engine
    if engine is None:
        engine = AerialInferenceEngine(num_threads=4)

    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(INFERENCE_OUTPUT_DIR, exist_ok=True)

    target_image_path = None
    gt_mask_path = None

    try:
        # Handle preset selection
        if preset_id:
            if preset_id == "tile_127":
                target_image_path = os.path.join(RAW_IMAGES_DIR, "SN2_buildings_train_AOI_2_Vegas_PS-RGB_img127.tif")
                gt_mask_path = os.path.join(MASKS_DIR, "mask_img127.png")
            elif preset_id == "tile_1006":
                target_image_path = os.path.join(RAW_IMAGES_DIR, "SN2_buildings_train_AOI_2_Vegas_PS-RGB_img1006.tif")
                gt_mask_path = os.path.join(MASKS_DIR, "mask_img1006.png")
            elif preset_id == "plain_jpg":
                target_image_path = os.path.join(INFERENCE_OUTPUT_DIR, "test_plain_aerial.jpg")
                if not os.path.exists(target_image_path):
                    tile127_p = os.path.join(RAW_IMAGES_DIR, "SN2_buildings_train_AOI_2_Vegas_PS-RGB_img127.tif")
                    rgb, _, _, _, _, _ = engine.load_image(tile127_p)
                    import cv2
                    cv2.imwrite(target_image_path, cv2.cvtColor((rgb * 255).astype("uint8"), cv2.COLOR_RGB2BGR))
            elif preset_id == "external_drone":
                target_image_path = os.path.join(UPLOADS_DIR, "external_drone_rettern_crop650.jpg")
            else:
                raise HTTPException(status_code=400, detail=f"Unknown preset ID: {preset_id}")

        elif file:
            filename = file.filename or "uploaded_image.jpg"
            ext = os.path.splitext(filename)[1].lower()
            if ext not in [".tif", ".tiff", ".jpg", ".jpeg", ".png", ".webp"]:
                raise HTTPException(status_code=400, detail=f"Unsupported format '{ext}'. Must be GeoTIFF, JPG, or PNG.")

            # Sanitize filename and clamp length to prevent Windows MAX_PATH (260 chars) overflow
            raw_base = os.path.splitext(os.path.basename(filename))[0]
            clean_base = re.sub(r"[^a-zA-Z0-9_\-]", "_", raw_base)
            clean_base = clean_base[:32].strip("_") or "uploaded"
            timestamp = int(time.time() * 1000)
            safe_filename = f"{clean_base}_{timestamp}{ext}"

            save_path = os.path.join(UPLOADS_DIR, safe_filename)
            with open(save_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            target_image_path = save_path
        else:
            raise HTTPException(status_code=400, detail="Must provide either an uploaded file or preset_id.")

        if not os.path.exists(target_image_path):
            raise HTTPException(status_code=404, detail="Target image file not found.")

        # Run real LightUNet inference
        result = engine.run(
            image_path=target_image_path,
            output_dir=INFERENCE_OUTPUT_DIR,
            mode=mode,
            gt_mask_path=gt_mask_path
        )

        base_name = os.path.splitext(os.path.basename(target_image_path))[0]

        # Read GeoJSON to include features directly in response
        with open(result["geojson_path"], "r", encoding="utf-8") as f:
            geojson_content = json.load(f)

        # Build web URLs
        result["preview_url"] = f"/outputs/inference/{base_name}_preview.jpg"
        result["mask_url"] = f"/outputs/inference/{base_name}_mask.png"
        result["overlay_url"] = f"/outputs/inference/{base_name}_overlay.jpg"
        result["geojson_url"] = f"/outputs/inference/{base_name}_buildings.geojson"
        result["geojson_data"] = geojson_content

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference execution failed: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
