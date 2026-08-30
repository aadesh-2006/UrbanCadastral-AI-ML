"""
Tiny Sanity-Training Experiment on SpaceNet 2 Vegas.
Trains LightUNet on 3 real image/mask pairs to verify:
1. Data pipeline integrity
2. Gradient flow & loss convergence
3. Model capacity to overfit a small sample
4. Visual fidelity of predicted building masks

Resource conservative:
- Runs strictly on CPU with 4 threads (< 25% CPU utilization on 14-core i7-12700H)
- Memory usage < 150MB
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from ml.models.light_unet import get_model
from ml.training.dataset import SpaceNetBuildingDataset
from ml.training.losses import BCEDiceLoss, compute_iou, compute_dice

# Set conservative thread count to ensure CPU <= 50%
torch.set_num_threads(4)

RAW_IMAGES_DIR = os.path.join(PROJECT_ROOT, "dataset", "raw", "images")
MASKS_DIR = os.path.join(PROJECT_ROOT, "dataset", "masks")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "ml", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Select 3 representative tiles
SAMPLE_IDS = ["10", "109", "116"]

def main():
    print("=" * 65)
    print("LIGHTUNET SANITY-TRAINING EXPERIMENT (CPU ONLY)")
    print("=" * 65)
    print(f"Device: CPU | Threads: {torch.get_num_threads()} (Conservative Laptop Profile)")
    
    # 1. Prepare sample pairs
    img_paths = [os.path.join(RAW_IMAGES_DIR, f"SN2_buildings_train_AOI_2_Vegas_PS-RGB_img{i}.tif") for i in SAMPLE_IDS]
    mask_paths = [os.path.join(MASKS_DIR, f"mask_img{i}.png") for i in SAMPLE_IDS]

    for p in img_paths + mask_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    print(f"Loaded {len(img_paths)} sanity pairs (Tiles: {', '.join(SAMPLE_IDS)})")

    # 2. Dataset & DataLoader
    train_dataset = SpaceNetBuildingDataset(img_paths, mask_paths, target_size=(256, 256))
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    # 3. Model
    model = get_model(in_channels=3, num_classes=1, base_filters=16)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: LightUNet | Parameters: {param_count:,} ({param_count/1e6:.2f}M)")

    # 4. Criterion & Optimizer
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 5. Training Loop
    epochs = 45
    loss_history = []
    print(f"\nStarting {epochs} epochs of sanity training...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            images = batch["image"] # (1, 3, 256, 256)
            masks = batch["mask"]   # (1, 1, 256, 256)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append({"epoch": epoch, "loss": round(avg_loss, 4)})

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:02d}/{epochs}] - Loss: {avg_loss:.4f}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f}s ({total_time/epochs:.2f}s per epoch on CPU)")
    print(f"Initial Loss: {loss_history[0]['loss']:.4f} -> Final Loss: {loss_history[-1]['loss']:.4f}")

    # 6. Save Model Checkpoint & Loss Log
    checkpoint_path = os.path.join(OUTPUTS_DIR, "sanity_model.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

    loss_log_path = os.path.join(OUTPUTS_DIR, "sanity_loss.json")
    with open(loss_log_path, "w") as f:
        json.dump(loss_history, f, indent=2)
    print(f"Saved loss history: {loss_log_path}")

    # 7. Evaluate on one of the trained samples and generate visual comparison
    model.eval()
    eval_dataset = SpaceNetBuildingDataset(img_paths, mask_paths, target_size=(256, 256))
    sample = eval_dataset[1] # Tile 109 (55 buildings, dense residential)

    with torch.no_grad():
        img_tensor = sample["image"].unsqueeze(0) # (1, 3, 256, 256)
        gt_tensor = sample["mask"].unsqueeze(0)   # (1, 1, 256, 256)
        pred_logits = model(img_tensor)
        pred_probs = torch.sigmoid(pred_logits)
        pred_binary = (pred_probs > 0.5).float()

    iou = compute_iou(pred_binary, gt_tensor)
    dice = compute_dice(pred_binary, gt_tensor)
    print(f"\nEvaluation on Tile 109:")
    print(f"  Dice Score: {dice:.4f} ({dice*100:.1f}%)")
    print(f"  IoU Score:  {iou:.4f} ({iou*100:.1f}%)")

    # 8. Save predicted & ground-truth binary masks
    gt_np = (gt_tensor.squeeze().numpy() * 255).astype(np.uint8)
    pred_np = (pred_binary.squeeze().numpy() * 255).astype(np.uint8)

    gt_path = os.path.join(OUTPUTS_DIR, "sanity_gt.png")
    pred_path = os.path.join(OUTPUTS_DIR, "sanity_pred.png")
    cv2.imwrite(gt_path, gt_np)
    cv2.imwrite(pred_path, pred_np)
    print(f"Saved ground-truth mask: {gt_path}")
    print(f"Saved predicted mask:    {pred_path}")

    # 9. Create 4-panel comparison visualization
    # Original aerial RGB
    rgb_np = sample["image"].numpy().transpose((1, 2, 0))
    rgb_bgr = cv2.cvtColor((rgb_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Ground truth panel (cyan overlay)
    gt_vis = rgb_bgr.copy()
    gt_bool = (gt_np == 255)
    gt_vis[gt_bool] = (gt_vis[gt_bool] * 0.45 + np.array([255, 200, 0]) * 0.55).astype(np.uint8)
    contours_gt, _ = cv2.findContours(gt_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(gt_vis, contours_gt, -1, (0, 255, 0), 1)

    # Predicted mask panel (magenta overlay)
    pred_vis = rgb_bgr.copy()
    pred_bool = (pred_np == 255)
    pred_vis[pred_bool] = (pred_vis[pred_bool] * 0.45 + np.array([0, 140, 255]) * 0.55).astype(np.uint8)
    contours_pred, _ = cv2.findContours(pred_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(pred_vis, contours_pred, -1, (0, 255, 255), 1)

    # Direct GT vs Prediction Difference panel (Green = True Positive, Red = False Positive, Blue = False Negative)
    tp = (gt_np == 255) & (pred_np == 255)
    fp = (gt_np == 0) & (pred_np == 255)
    fn = (gt_np == 255) & (pred_np == 0)
    diff_vis = np.zeros_like(rgb_bgr)
    diff_vis[tp] = [0, 255, 0]   # Green: Correctly predicted
    diff_vis[fp] = [0, 0, 255]   # Red: False positive
    diff_vis[fn] = [255, 0, 0]   # Blue: Missed

    # Combine 4 panels horizontally
    panel_h, panel_w, _ = rgb_bgr.shape
    banner_h = 32
    total_w = panel_w * 4
    total_h = panel_h + banner_h

    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    canvas[:banner_h, :] = (25, 25, 25)

    # Place panels
    canvas[banner_h:, :panel_w] = rgb_bgr
    canvas[banner_h:, panel_w:panel_w*2] = gt_vis
    canvas[banner_h:, panel_w*2:panel_w*3] = pred_vis
    canvas[banner_h:, panel_w*3:] = diff_vis

    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "[1] Original Aerial Image", (10, 22), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "[2] Ground Truth Mask", (panel_w + 10, 22), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"[3] Model Prediction (Dice: {dice*100:.1f}%)", (panel_w * 2 + 10, 22), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "[4] Green=Hit, Red=FP, Blue=Miss", (panel_w * 3 + 10, 22), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    comparison_path = os.path.join(OUTPUTS_DIR, "sanity_comparison.jpg")
    cv2.imwrite(comparison_path, canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"Saved 4-panel comparison visualization: {comparison_path}")

    print("\n" + "=" * 65)
    if dice >= 0.70:
        print("[SUCCESS] Model successfully overfit the sanity sample!")
        print("Pipeline is verified: data loader, forward pass, backward pass, and predictions all work.")
    else:
        print("[WARNING] Model loss converged, but Dice score is below expected sanity threshold.")
    print("=" * 65)

if __name__ == "__main__":
    main()
