# Milestone 7 — Multi-Image Generalization Benchmark Report

## 1. Executive Summary & Aggregate Metrics

| Metric | Benchmark Result across 7 Unseen Tiles | Milestone 5 Epoch 18 Validation |
| :--- | :--- | :--- |
| **Validation Tiles** | **7 unseen tiles** (`127`, `1006`, `14`, `146`, `113`, `10`, `101`) | Same 7 tiles |
| **Mean Dice Score** | **76.54%** (±7.07%) | 76.5% |
| **Mean IoU Score** | **62.53%** | 62.5% |
| **Best-Performing Tile** | **Tile 127 (87.99% Dice)** | Tile 127 (88.0%) |
| **Representative/Avg Tile** | **Tile 146 (76.51% Dice)** | Tile 10 (76.2%) |
| **Worst-Performing Tile** | **Tile 1006 (66.49% Dice)** | Tile 1006 (66.5%) |
| **Average Inference Latency** | **45.26 ms** per tile (CPU) | ~60 ms |
| **Total Benchmark Time** | **0.67 seconds** | — |
| **RAM Utilization** | **380.2 MB** | 467.1 MB |

---

## 2. Per-Tile Generalization Benchmark Table

| Tile ID | Dice | IoU | Inference Time | Predicted Regions | Main Observation |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **127** | **87.99%** | 78.56% | 68.68 ms | 16 | Dense suburban street with distinct single-family homes; excellent delineation and high Dice. |
| **1006** | **66.49%** | 49.81% | 44.16 ms | 21 | Townhomes and electrical substation utility yard; utility metallic equipment and bright gravel caused false-positive block. |
| **14** | **76.16%** | 61.50% | 46.89 ms | 16 | Curving residential tract; accurate roof boundary tracing, minor merging on adjoining eaves. |
| **146** | **76.51%** | 61.96% | 39.22 ms | 14 | Sparse residential with large gravel yards; lower building density, minor false positives on bright driveway borders. |
| **113** | **81.01%** | 68.08% | 34.54 ms | 27 | High-density residential neighborhood; captured majority of houses, some adjacent roofs merged. |
| **10** | **67.37%** | 50.79% | 36.52 ms | 20 | Mixed suburban and apartment complexes; solid footprint detection on large structures. |
| **101** | **80.23%** | 66.99% | 46.83 ms | 20 | Residential tract; good overall localization, slight under-segmentation on low-contrast roofs. |

---

## 3. Visual Quality & Representative Examples

### [BEST] Tile 127 — Dice: 87.99% | IoU: 78.56%
* **4-Panel Diagnostic:** [benchmark_best.jpg](./benchmark_best.jpg)
* **Polygon Overlay:** [benchmark_best_polygons.jpg](./benchmark_best_polygons.jpg)
* **GeoJSON:** [benchmark_best_polygons.geojson](./benchmark_best_polygons.geojson) (16 polygons)

### [AVERAGE] Tile 146 — Dice: 76.51% | IoU: 61.96%
* **4-Panel Diagnostic:** [benchmark_average.jpg](./benchmark_average.jpg)

### [WORST] Tile 1006 — Dice: 66.49% | IoU: 49.81%
* **4-Panel Diagnostic:** [benchmark_worst.jpg](./benchmark_worst.jpg)
* **Polygon Overlay:** [benchmark_worst_polygons.jpg](./benchmark_worst_polygons.jpg)
* **GeoJSON:** [benchmark_worst_polygons.geojson](./benchmark_worst_polygons.geojson) (19 polygons)

---

## 4. Failure Mode Analysis (Worst-Performing Tile 1006)

Detailed inspection of Tile 1006 reveals the following primary failure drivers:
1. **False Positive Industrial Utility Surfaces:** An electrical substation in the lower quadrant features metallic equipment and high-albedo gravel, which share reflectance characteristics with commercial gravel-topped roofs.
2. **Adjoining Roof Merging:** In dense multi-family rows, narrow roof separations (< 1 meter) are occasionally bridged by the 256x256 model resolution, producing a single merged polygon.
3. **Pavement & Driveway Contrast:** Minor false positive fringes occur along bright concrete driveways directly adjoining garage doors.

---

## 5. Polygonization Verification

* **Best Tile (127):** 16 polygons extracted, all 100% valid geometries.
* **Worst Tile (1006):** 19 polygons extracted, all 100% valid geometries.
* **Geospatial Integrity:** All coordinates strictly derived via affine transformation to standard WGS 84 (`EPSG:4326` / `CRS84`).
