"""
Verification script for Milestone 9 Frontend State Fix.
Simulates and validates:
1. Preset tile_127 extraction -> georeferenced=True, CRS=EPSG:4326, 18 buildings
2. Preset tile_1006 extraction -> georeferenced=True, CRS=EPSG:4326, 21 buildings
3. Preset plain_jpg extraction -> georeferenced=False, CRS=None (pixel coordinates), 18 buildings, pixel coordinates in GeoJSON
"""

import urllib.request
import json

BASE_URL = "http://127.0.0.1:8000"

def test_preset(preset_id, expected_name, expected_georef, expected_crs_sub):
    print(f"\nTesting preset '{preset_id}'...")
    url = f"{BASE_URL}/api/inference?preset_id={preset_id}"
    req = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200, f"Failed with status {resp.status}"
        data = json.loads(resp.read().decode("utf-8"))

    print(f"  Building Count:  {data['building_count']}")
    print(f"  Mean Confidence: {data['mean_confidence']*100:.1f}%")
    print(f"  Georeferenced:   {data['georeferenced']}")
    print(f"  CRS:             {data['crs']}")

    assert data["georeferenced"] == expected_georef, f"Expected georeferenced={expected_georef}, got {data['georeferenced']}"
    assert expected_crs_sub in data["crs"], f"Expected '{expected_crs_sub}' in CRS, got {data['crs']}"

    # Verify GeoJSON coordinate domain
    coords = data["geojson_data"]["features"][0]["geometry"]["coordinates"][0]
    pt = coords[0]
    if expected_georef:
        # WGS84: lon in [-180, 180], lat in [-90, 90] (Vegas is approx -115.3, 36.15)
        assert -180 <= pt[0] <= 180 and -90 <= pt[1] <= 90, f"Expected geographic coords, got {pt}"
        print(f"  GeoJSON Coord sample: [lon: {pt[0]}, lat: {pt[1]}] (WGS 84 verified)")
    else:
        # Pixel coordinates: x in [0, 650], y in [0, 650]
        assert 0 <= pt[0] <= 650 and 0 <= pt[1] <= 650, f"Expected pixel coordinates, got {pt}"
        print(f"  GeoJSON Coord sample: [px_x: {pt[0]}, px_y: {pt[1]}] (Pixel coords verified)")

    return data

def main():
    print("=" * 68)
    print("VALIDATING INFERENCE ENDPOINTS FOR FRONTEND INTEGRITY")
    print("=" * 68)

    # 1. Tile 127
    d127 = test_preset("tile_127", "SpaceNet Tile 127 (Residential)", True, "EPSG:4326")
    assert d127["building_count"] == 18

    # 2. Tile 1006
    d1006 = test_preset("tile_1006", "SpaceNet Tile 1006 (Townhomes & Utility)", True, "EPSG:4326")
    assert d1006["building_count"] == 21

    # 3. Plain JPG
    d_jpg = test_preset("plain_jpg", "Standard Aerial Photograph (JPEG)", False, "pixel coordinates")
    assert d_jpg["building_count"] == 18

    print("\n" + "=" * 68)
    print("[SUCCESS] All 3 presets validated with 100% integrity!")
    print("=" * 68)

if __name__ == "__main__":
    main()
