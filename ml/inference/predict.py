"""
CLI for the UrbanCadastral-AI-ML Aerial Inference Engine.

Usage:
    py ml/inference/predict.py --image path/to/aerial_image.jpg
    py ml/inference/predict.py --image path/to/image.tif --mode tiled
"""

import os
import sys
import json
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from ml.inference.engine import AerialInferenceEngine

def main():
    parser = argparse.ArgumentParser(description="UrbanCadastral-AI-ML: Aerial Building Footprint Inference")
    parser.add_argument("--image", "-i", required=True, help="Path to input aerial image (.tif, .jpg, .png)")
    parser.add_argument("--output-dir", "-o", default=None, help="Directory to save predictions")
    parser.add_argument("--mode", "-m", default="auto", choices=["auto", "resize", "tiled"], help="Inference mode")
    parser.add_argument("--model", default=None, help="Path to model weights checkpoint (.pth)")
    parser.add_argument("--gt-mask", default=None, help="Optional ground-truth mask for validation benchmarking")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads to use (default: 4)")

    args = parser.parse_args()

    engine = AerialInferenceEngine(model_path=args.model, num_threads=args.threads)
    result = engine.run(
        image_path=args.image,
        output_dir=args.output_dir,
        mode=args.mode,
        gt_mask_path=args.gt_mask
    )

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
