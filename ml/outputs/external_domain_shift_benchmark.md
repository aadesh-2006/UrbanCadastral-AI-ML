# Milestone 10 — Real-World External Image Domain-Shift Benchmark Report

## 1. Test Overview

* **Objective:** Evaluate `LightUNet` (trained exclusively on SpaceNet 2 Las Vegas 30 cm satellite imagery) on a genuinely external, un-annotated high-resolution aerial drone orthophoto.
* **Test Image Source:** Wikimedia Commons — *Rettern Drone Orthophoto (Bavaria, Germany)*, CC-BY-SA 4.0 by Ermell.
* **Evaluation Type:** **Strictly Qualitative / Domain-Shift Assessment**. Ground-truth polygon labels do not exist for this image; therefore, **no synthetic accuracy, Dice, or IoU scores are fabricated**.
* **Model Checkpoint:** `ml/outputs/best_model.pth` (Unchanged, Epoch 18).

---

## 2. Ingestion & Quantitative Pipeline Metrics

| Metric | Measured Value | SpaceNet 2 Vegas Baseline |
| :--- | :--- | :--- |
| **Image Resolution** | **650 x 650 pixels** | 650 x 650 pixels |
| **Color Channels** | 3-band RGB (8-bit JPEG) | 3-band RGB (from 16-bit GeoTIFF) |
| **Georeferencing Status** | **Unreferenced (Pixel Coordinates)** | Georeferenced (`EPSG:4326` / WGS 84) |
| **Coordinate System** | Local image space (`[0, 650]`) | WGS 84 (`[-115.3°, 36.1°]`) |
| **Inference Latency** | **64.51 ms** (CPU, 4 threads) | ~60-80 ms |
| **Detected Building Regions** | **13 distinct polygons** | 16–27 regions/tile |
| **Mean Model Confidence** | **78.6%** | 91.6% (Tile 127) |
| **Geometry Validity** | **13/13 valid Shapely polygons (100%)** | 100% valid |
| **Non-Georeferenced Alert** | **Triggered & Verified** | None (Georeferenced) |

---

## 3. Qualitative Inspection & Domain-Shift Findings

Based on visual inspection of [external_domain_shift_diagnostic.jpg](./external_domain_shift_diagnostic.jpg) and [external_drone_rettern_crop650_overlay.jpg](./inference/external_drone_rettern_crop650_overlay.jpg):

### 1. What the Model Detected Accurately:
* **Terracotta Tiled Residential Houses:** Pitched residential roofs with orange/red terracotta tiles (center-left and center-right) were detected with high confidence (88% - 89%). The model recognized the distinct rectangular ridgelines and planar pitch facets despite the Bavarian architectural style.
* **Detached Outbuildings:** Freestanding rectangular barns with pitched gables were correctly bounded.

### 2. Primary Failure Modes & Domain Shifts:
* **Gravel/Asphalt Courtyard Confusion (False Positives):** In the center and bottom-right, unpaved gravel farm courtyards and light asphalt driveways were partially merged into adjacent building boundaries. In the SpaceNet Las Vegas training domain, flat commercial roofs feature light gray/tan gravel ballast, which visually resembles unpaved European farm driveways.
* **Photovoltaic Solar Panel Omission (Under-Segmentation):** A modern home at the bottom center features a large rooftop blue solar array. The model detected the surrounding roof perimeter but **completely carved out the solar panels as background**. SpaceNet 2 (2016 imagery) had virtually no rooftop solar annotations, causing dark blue reflective silicon panels to be treated as non-roof surfaces.
* **Adjacent Farm Compound Merging:** Closely packed rural European barns and sheds with shared eaves were polygonized into contiguous compound blocks rather than individual cadastral parcels.
* **Parked Vehicle Confusion:** In the upper-left gravel parking lot, tightly parked cars on pale gravel were grouped into a building polygon (79% confidence).
* **Tree Canopy Shadows:** Deep shadows cast by large deciduous trees caused localized notch artifacts along north-facing building edges.

---

## 4. SpaceNet vs. External Domain Comparison

| Attribute | SpaceNet 2 (Las Vegas) | External Drone (Rettern, Germany) | Model Impact |
| :--- | :--- | :--- | :--- |
| **Sensor & Platform** | WorldView-3 Satellite (30 cm GSD) | Drone Camera (Orthomosaic) | Drone imagery has sharper edge gradients and micro-textures. |
| **Environment** | Arid southwestern desert | Temperate European village | Green vegetation is well-filtered; gravel courtyards trigger false positives. |
| **Roofing Materials** | Asphalt shingles, gravel ballast, tile | Terracotta clay tiles, slate, solar panels | Terracotta tiles generalize well; solar panels are omitted. |
| **Structure Density** | Regular suburban subdivisions | Irregular historical European cluster | Adjoining rural outbuildings merge across narrow gaps. |

---

## 5. Output Verification Artifacts

* **4-Panel Diagnostic Grid:** [external_domain_shift_diagnostic.jpg](./external_domain_shift_diagnostic.jpg)
* **Visual Polygon Overlay:** [external_drone_rettern_crop650_overlay.jpg](./inference/external_drone_rettern_crop650_overlay.jpg)
* **Binary Mask PNG:** [external_drone_rettern_crop650_mask.png](./inference/external_drone_rettern_crop650_mask.png)
* **Pixel-Coordinate GeoJSON:** [external_drone_rettern_crop650_buildings.geojson](./inference/external_drone_rettern_crop650_buildings.geojson)
* **Machine-Readable JSON Benchmark:** [external_domain_shift_benchmark.json](./external_domain_shift_benchmark.json)
