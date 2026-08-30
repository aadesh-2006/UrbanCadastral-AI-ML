"""
Milestone 5: Controlled Train/Validation Experiment
Trains LightUNet on 27 tiles and evaluates on 7 unseen validation tiles.

Resource profile:
- Strictly CPU (Intel Core i7-12700H)
- Capped at 4 threads to stay well below the 50% CPU safety ceiling
- Batch size = 1, sequential tile loading
- Continuous CPU and RAM monitoring via psutil
"""

import os
import sys
import re
import glob
import json
import time
import random
import cv2
import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from ml.models.light_unet import get_model
from ml.training.dataset import SpaceNetBuildingDataset
from ml.training.losses import BCEDiceLoss, compute_iou, compute_dice

# Hardware constraint: Cap CPU threads to 4
CPU_THREADS = 4
torch.set_num_threads(CPU_THREADS)

RAW_IMAGES_DIR = os.path.join(PROJECT_ROOT, "dataset", "raw", "images")
MASKS_DIR = os.path.join(PROJECT_ROOT, "dataset", "masks")
METADATA_DIR = os.path.join(PROJECT_ROOT, "dataset", "metadata")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "ml", "outputs")

os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

RANDOM_SEED = 42
EPOCHS = 20
BATCH_SIZE = 1
LEARNING_RATE = 1e-3
IMAGE_SIZE = (256, 256)

def get_train_val_split():
    img_files = sorted(glob.glob(os.path.join(RAW_IMAGES_DIR, "*.tif")))
    all_pairs = []

    for img_p in img_files:
        fn = os.path.basename(img_p)
        m = re.search(r"_img(\d+)\.tif$", fn)
        if m:
            tile_id = m.group(1)
            mask_p = os.path.join(MASKS_DIR, f"mask_img{tile_id}.png")
            if os.path.exists(mask_p):
                all_pairs.append({"tile_id": tile_id, "image_path": img_p, "mask_path": mask_p})

    # Deterministic split: 27 train, 7 validation
    rng = random.Random(RANDOM_SEED)
    shuffled = list(all_pairs)
    rng.shuffle(shuffled)

    train_pairs = shuffled[:27]
    val_pairs = shuffled[27:]

    split_data = {
        "random_seed": RANDOM_SEED,
        "total_pairs": len(all_pairs),
        "train_count": len(train_pairs),
        "val_count": len(val_pairs),
        "train_tile_ids": [p["tile_id"] for p in train_pairs],
        "val_tile_ids": [p["tile_id"] for p in val_pairs]
    }

    split_path = os.path.join(METADATA_DIR, "train_val_split.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2)

    return train_pairs, val_pairs, split_data

def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    dices = []
    ious = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"]
            masks = batch["mask"]
            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            dices.append(compute_dice(preds, masks))
            ious.append(compute_iou(preds, masks))

    avg_loss = total_loss / len(val_loader)
    avg_dice = float(np.mean(dices))
    avg_iou = float(np.mean(ious))
    return avg_loss, avg_dice, avg_iou

def create_val_visualization(model, val_dataset, out_path, sample_idx, tile_id):
    model.eval()
    sample = val_dataset[sample_idx]
    with torch.no_grad():
        img_tensor = sample["image"].unsqueeze(0)
        gt_tensor = sample["mask"].unsqueeze(0)
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

    dice = compute_dice(preds, gt_tensor)
    iou = compute_iou(preds, gt_tensor)

    gt_np = (gt_tensor.squeeze().numpy() * 255).astype(np.uint8)
    pred_np = (preds.squeeze().numpy() * 255).astype(np.uint8)

    rgb_np = sample["image"].numpy().transpose((1, 2, 0))
    rgb_bgr = cv2.cvtColor((rgb_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Panel 2: Ground Truth overlay
    gt_vis = rgb_bgr.copy()
    gt_bool = (gt_np == 255)
    gt_vis[gt_bool] = (gt_vis[gt_bool] * 0.45 + np.array([255, 200, 0]) * 0.55).astype(np.uint8)
    contours_gt, _ = cv2.findContours(gt_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(gt_vis, contours_gt, -1, (0, 255, 0), 1)

    # Panel 3: Prediction overlay
    pred_vis = rgb_bgr.copy()
    pred_bool = (pred_np == 255)
    pred_vis[pred_bool] = (pred_vis[pred_bool] * 0.45 + np.array([0, 140, 255]) * 0.55).astype(np.uint8)
    contours_pred, _ = cv2.findContours(pred_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(pred_vis, contours_pred, -1, (0, 255, 255), 1)

    # Panel 4: Difference map (Green = Hit, Red = FP, Blue = Miss)
    tp = (gt_np == 255) & (pred_np == 255)
    fp = (gt_np == 0) & (pred_np == 255)
    fn = (gt_np == 255) & (pred_np == 0)
    diff_vis = np.zeros_like(rgb_bgr)
    diff_vis[tp] = [0, 255, 0]   # Hit (Green)
    diff_vis[fp] = [0, 0, 255]   # FP (Red)
    diff_vis[fn] = [255, 0, 0]   # Miss (Blue)

    panel_h, panel_w, _ = rgb_bgr.shape
    banner_h = 32
    total_w = panel_w * 4
    total_h = panel_h + banner_h

    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    canvas[:banner_h, :] = (25, 25, 25)

    canvas[banner_h:, :panel_w] = rgb_bgr
    canvas[banner_h:, panel_w:panel_w*2] = gt_vis
    canvas[banner_h:, panel_w*2:panel_w*3] = pred_vis
    canvas[banner_h:, panel_w*3:] = diff_vis

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, f"Unseen Val Tile {tile_id} (Aerial)", (10, 22), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Ground Truth Mask", (panel_w + 10, 22), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Prediction (Dice: {dice*100:.1f}%)", (panel_w * 2 + 10, 22), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Diff (IoU: {iou*100:.1f}%)", (panel_w * 3 + 10, 22), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return dice, iou

def main():
    print("=" * 65)
    print("CONTROLLED TRAIN/VALIDATION EXPERIMENT — LIGHTUNET")
    print("=" * 65)

    proc = psutil.Process()
    ram_init = proc.memory_info().rss / (1024 * 1024)
    print(f"Hardware: ASUS Vivobook S16 (Intel Core i7-12700H)")
    print(f"Device: CPU | Active Threads: {CPU_THREADS} (Target <= 50% CPU ceiling)")
    print(f"Initial Process RAM: {ram_init:.1f} MB")

    # 1. Dataset Partitioning
    train_pairs, val_pairs, split_info = get_train_val_split()
    print(f"\nDataset Partitioning (Seed = {RANDOM_SEED}):")
    print(f"  Training tiles:   {len(train_pairs)} tiles -> {split_info['train_tile_ids']}")
    print(f"  Validation tiles: {len(val_pairs)} tiles -> {split_info['val_tile_ids']}")

    # 2. DataLoaders
    train_ds = SpaceNetBuildingDataset(
        [p["image_path"] for p in train_pairs],
        [p["mask_path"] for p in train_pairs],
        target_size=IMAGE_SIZE
    )
    val_ds = SpaceNetBuildingDataset(
        [p["image_path"] for p in val_pairs],
        [p["mask_path"] for p in val_pairs],
        target_size=IMAGE_SIZE
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. Model & Optimizer
    model = get_model(in_channels=3, num_classes=1, base_filters=16)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: LightUNet | Parameters: {param_count:,} ({param_count/1e6:.2f}M)")

    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # 4. Training Loop with Tracking
    best_val_dice = -1.0
    best_epoch = 0
    history = []
    peak_cpu_readings = []
    peak_ram_mb = ram_init

    print(f"\nBeginning {EPOCHS} epochs of training...")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        train_loss_accum = 0.0

        for batch in train_loader:
            images = batch["image"]
            masks = batch["mask"]

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()

        # Measure system resource usage during epoch
        cpu_pct = psutil.cpu_percent(interval=None)
        cur_ram = proc.memory_info().rss / (1024 * 1024)
        peak_cpu_readings.append(cpu_pct)
        if cur_ram > peak_ram_mb:
            peak_ram_mb = cur_ram

        avg_train_loss = train_loss_accum / len(train_loader)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion)
        epoch_duration = time.time() - epoch_start

        # Record history
        record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_dice": round(val_dice, 4),
            "val_iou": round(val_iou, 4),
            "epoch_time_s": round(epoch_duration, 2),
            "cpu_percent": round(cpu_pct, 1),
            "ram_mb": round(cur_ram, 1)
        }
        history.append(record)

        # Check for best validation score
        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(OUTPUTS_DIR, "best_model.pth"))

        marker = " (*Best*)" if is_best else ""
        print(f"  Epoch [{epoch:02d}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice*100:.1f}% | Val IoU: {val_iou*100:.1f}% | "
              f"Time: {epoch_duration:.1f}s | CPU: {cpu_pct:.0f}%{marker}")

    total_training_time = time.time() - start_time
    torch.save(model.state_dict(), os.path.join(OUTPUTS_DIR, "final_model.pth"))

    print("\n" + "=" * 65)
    print("TRAINING RUN COMPLETE")
    print("=" * 65)
    print(f"Total training time: {total_training_time:.1f}s ({total_training_time/EPOCHS:.2f}s/epoch)")
    print(f"Best Validation Epoch: {best_epoch} (Val Dice: {best_val_dice*100:.1f}%)")
    print(f"Typical CPU Utilization: {np.mean(peak_cpu_readings):.1f}% (Peak: {np.max(peak_cpu_readings):.1f}%)")
    print(f"Peak Process RAM: {peak_ram_mb:.1f} MB")

    # 5. Save History & Final Metrics
    with open(os.path.join(OUTPUTS_DIR, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Load best model for validation artifact generation
    best_weights = os.path.join(OUTPUTS_DIR, "best_model.pth")
    model.load_state_dict(torch.load(best_weights, map_location="cpu"))
    final_val_loss, final_val_dice, final_val_iou = evaluate(model, val_loader, criterion)

    val_metrics = {
        "model": "LightUNet",
        "parameters": param_count,
        "train_tiles": len(train_pairs),
        "val_tiles": len(val_pairs),
        "epochs": EPOCHS,
        "best_epoch": best_epoch,
        "best_val_dice": round(final_val_dice, 4),
        "best_val_iou": round(final_val_iou, 4),
        "best_val_loss": round(final_val_loss, 4),
        "final_train_loss": history[-1]["train_loss"],
        "total_time_seconds": round(total_training_time, 2),
        "mean_cpu_percent": round(float(np.mean(peak_cpu_readings)), 1),
        "peak_cpu_percent": round(float(np.max(peak_cpu_readings)), 1),
        "peak_ram_mb": round(peak_ram_mb, 1)
    }

    with open(os.path.join(OUTPUTS_DIR, "validation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)

    # 6. Generate Validation Visualizations on Unseen Tiles
    print("\nGenerating 4-panel visual comparisons on unseen validation tiles...")
    val_vis_paths = []
    # Generate for 3 diverse unseen validation tiles
    for vis_idx in range(min(3, len(val_ds))):
        val_tile_id = val_pairs[vis_idx]["tile_id"]
        out_vis_file = f"val_sample_{vis_idx+1:02d}_tile{val_tile_id}.jpg"
        out_vis_path = os.path.join(OUTPUTS_DIR, out_vis_file)
        d, u = create_val_visualization(model, val_ds, out_vis_path, vis_idx, val_tile_id)
        val_vis_paths.append(out_vis_path)
        print(f"  [Val Sample {vis_idx+1}] Tile {val_tile_id}: Dice={d*100:.1f}%, IoU={u*100:.1f}% -> {out_vis_file}")

    print("\nArtifacts saved in ml/outputs/:")
    print(f"  - Best Checkpoint: ml/outputs/best_model.pth")
    print(f"  - Final Checkpoint: ml/outputs/final_model.pth")
    print(f"  - Training History: ml/outputs/training_history.json")
    print(f"  - Validation Metrics: ml/outputs/validation_metrics.json")
    for p in val_vis_paths:
        print(f"  - Visualization: {p}")
    print("=" * 65)

if __name__ == "__main__":
    main()
