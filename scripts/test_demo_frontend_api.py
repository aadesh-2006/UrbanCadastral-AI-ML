"""
End-to-end API test script for Milestone 9 Demo Frontend.
Tests:
1. GET /api/health
2. GET /api/presets
3. POST /api/inference with preset tile_127
4. POST /api/inference with preset tile_1006
5. POST /api/inference with preset plain_jpg
6. POST /api/inference with multipart file upload
Verifies contract integrity, coordinate handling, and consistency.
"""

import os
import sys
import json
from starlette.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ml.api.server import app

def main():
    print("=" * 68)
    print("TESTING DEMO FRONTEND FASTAPI BACKEND")
    print("=" * 68)

    client = TestClient(app)

    # 1. Health Check
    print("\n[1/6] Testing GET /api/health...")
    res_health = client.get("/api/health")
    assert res_health.status_code == 200, f"Health check failed: {res_health.status_code}"
    health_data = res_health.json()
    print(f"  Status: {health_data['status']} | Model: {health_data['model']} | Threads: {health_data['threads']}")

    # 2. Presets
    print("\n[2/6] Testing GET /api/presets...")
    res_presets = client.get("/api/presets")
    assert res_presets.status_code == 200, f"Presets failed: {res_presets.status_code}"
    presets_data = res_presets.json()
    print(f"  Returned {len(presets_data)} presets: {[p['id'] for p in presets_data]}")

    # 3. Preset Tile 127
    print("\n[3/6] Testing POST /api/inference?preset_id=tile_127...")
    res_127 = client.post("/api/inference?preset_id=tile_127")
    assert res_127.status_code == 200, f"Tile 127 failed: {res_127.status_code}, {res_127.text}"
    data_127 = res_127.json()
    print(f"  Buildings: {data_127['building_count']}")
    print(f"  Latency:   {data_127['inference_time_ms']} ms")
    print(f"  Conf:      {data_127['mean_confidence']*100:.1f}%")
    print(f"  Georef:    {data_127['georeferenced']} ({data_127['crs']})")
    print(f"  Polygons:  {len(data_127.get('polygons_pixel', []))} pixel polygons returned")
    assert data_127["building_count"] == 18, f"Expected 18 buildings, got {data_127['building_count']}"
    assert data_127["georeferenced"] is True

    # 4. Preset Tile 1006
    print("\n[4/6] Testing POST /api/inference?preset_id=tile_1006...")
    res_1006 = client.post("/api/inference?preset_id=tile_1006")
    assert res_1006.status_code == 200, f"Tile 1006 failed: {res_1006.status_code}"
    data_1006 = res_1006.json()
    print(f"  Buildings: {data_1006['building_count']}")
    print(f"  Latency:   {data_1006['inference_time_ms']} ms")
    print(f"  Conf:      {data_1006['mean_confidence']*100:.1f}%")
    print(f"  Georef:    {data_1006['georeferenced']} ({data_1006['crs']})")
    assert data_1006["building_count"] == 21, f"Expected 21 buildings, got {data_1006['building_count']}"

    # 5. Preset Plain JPG
    print("\n[5/6] Testing POST /api/inference?preset_id=plain_jpg...")
    res_plain = client.post("/api/inference?preset_id=plain_jpg")
    assert res_plain.status_code == 200, f"Plain JPG failed: {res_plain.status_code}"
    data_plain = res_plain.json()
    print(f"  Buildings: {data_plain['building_count']}")
    print(f"  Latency:   {data_plain['inference_time_ms']} ms")
    print(f"  Conf:      {data_plain['mean_confidence']*100:.1f}%")
    print(f"  Georef:    {data_plain['georeferenced']} ({data_plain['crs']})")
    assert data_plain["georeferenced"] is False, "Expected plain JPG to have georeferenced=False"
    assert "pixel coordinates" in data_plain["crs"]

    # 6. File Upload
    print("\n[6/6] Testing POST /api/inference with multipart file upload...")
    test_img_path = os.path.join(PROJECT_ROOT, "ml", "outputs", "inference", "test_plain_aerial.jpg")
    with open(test_img_path, "rb") as f:
        files = {"file": ("uploaded_aerial.jpg", f, "image/jpeg")}
        res_upload = client.post("/api/inference", files=files)
    assert res_upload.status_code == 200, f"Upload failed: {res_upload.status_code}, {res_upload.text}"
    data_upload = res_upload.json()
    print(f"  Uploaded Buildings: {data_upload['building_count']}")
    print(f"  Uploaded Latency:   {data_upload['inference_time_ms']} ms")
    assert data_upload["building_count"] == data_plain["building_count"]

    print("\n" + "=" * 68)
    print("[SUCCESS] All 6 API End-to-End Tests Passed Cleanly!")
    print("=" * 68)

if __name__ == "__main__":
    main()
