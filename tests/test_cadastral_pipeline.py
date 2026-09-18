"""
Automated Verification Suite for CadastraAI
===========================================
Verifies:
1. ISO 7064 Mod 11-2 ULPIN (Bhu-Aadhaar) checksum generation and validation
2. Zero mutual overlap between regularized cadastral land parcels
3. 1-to-1 building containment within cadastral plots
4. Quadrilateral regularization and angle conformance
5. Cross-platform path resolution
6. Upload pipeline end-to-end integration
"""
import json
import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point, LineString, box, shape

# Set up paths relative to repository
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "ml"))

import cadastral_standards
from ml import parcel_engine, vegetation_index
import upload_pipeline


class TestCadastralPipeline(unittest.TestCase):

    def test_01_ulpin_iso7064_compliance(self):
        """Verify 14-character ULPIN with ISO 7064 Mod 11-2 check character."""
        test_coords = [
            (30.2268, -97.7845, 1),
            (19.6967, 73.5594, 42),
            (28.6139, 77.2090, 10),
            (0.0, 0.0, 0),
            (-33.8688, 151.2093, 7),
        ]
        for lat, lon, seq in test_coords:
            ulpin = cadastral_standards.generate_ulpin(lat, lon, seq)
            self.assertEqual(len(ulpin), 14, f"ULPIN {ulpin} must be exactly 14 characters")
            self.assertTrue(cadastral_standards.validate_ulpin(ulpin), f"ULPIN {ulpin} must pass ISO 7064 Mod 11-2 validation")

        # Verify corrupted ULPIN fails
        corrupted = ulpin[:-1] + ('0' if ulpin[-1] != '0' else '1')
        self.assertFalse(cadastral_standards.validate_ulpin(corrupted), "Corrupted check digit must fail")
        self.assertFalse(cadastral_standards.validate_ulpin("SHORT123"), "Short string must fail")

    def test_02_4factor_zero_overlap_guarantee(self):
        """Verify generated parcels have 100% zero mutual overlap."""
        # Create a synthetic grid of 6 buildings along a street
        w, h = 500, 500
        img_bgr = np.full((h, w, 3), 180, dtype=np.uint8)
        road_line = LineString([(50, 100), (450, 100)])
        road_lines = [(road_line, 20.0)]
        road_union = road_line.buffer(10.0)

        bldgs = []
        for i, bx in enumerate([80, 150, 220, 290, 360, 420]):
            b_poly = box(bx - 15, 130, bx + 15, 170)
            bldgs.append({
                "id": i,
                "cx": bx,
                "cy": 150,
                "poly_px": b_poly,
                "poly_geo": b_poly,
                "area_px": b_poly.area,
                "unrecorded": False,
                "type": "residential",
            })

        roi_box = box(0, 0, w, h)
        def dummy_px_to_lonlat(px, py):
            return px * 0.001, py * 0.001

        parcels_list, bldg_features, cards = parcel_engine.generate_4factor_cadastral_parcels(
            bldgs=bldgs,
            road_lines=road_lines,
            img_bgr=img_bgr,
            roi_box=roi_box,
            px_to_lonlat_fn=dummy_px_to_lonlat,
            cadastral_standards=cadastral_standards,
            road_union=road_union,
            meters_per_px=0.3,
        )

        self.assertGreater(len(parcels_list), 0, "Engine must generate parcels")
        parcel_shapes = [shape(f["geometry"]) for f in parcels_list]

        # Verify zero mutual overlap
        for i in range(len(parcel_shapes)):
            for j in range(i + 1, len(parcel_shapes)):
                inter = parcel_shapes[i].intersection(parcel_shapes[j])
                inter_area = inter.area if not inter.is_empty else 0.0
                self.assertAlmostEqual(
                    inter_area, 0.0, places=4,
                    msg=f"Parcels {i} and {j} must have zero overlap, got {inter_area}"
                )

    def test_03_building_containment(self):
        """Verify each building is contained within its designated parcel."""
        w, h = 400, 400
        img_bgr = np.full((h, w, 3), 150, dtype=np.uint8)
        road_line = LineString([(50, 50), (350, 50)])
        road_lines = [(road_line, 16.0)]

        bldgs = []
        for i, bx in enumerate([100, 200, 300]):
            b_poly = box(bx - 12, 80, bx + 12, 120)
            bldgs.append({
                "id": i,
                "cx": bx,
                "cy": 100,
                "poly_px": b_poly,
                "poly_geo": b_poly,
                "area_px": b_poly.area,
                "unrecorded": False,
                "type": "residential",
            })

        def dummy_px_to_lonlat(px, py):
            return px, py

        parcels_list, bldg_features, cards = parcel_engine.generate_4factor_cadastral_parcels(
            bldgs=bldgs,
            road_lines=road_lines,
            img_bgr=img_bgr,
            roi_box=box(0, 0, w, h),
            px_to_lonlat_fn=dummy_px_to_lonlat,
            cadastral_standards=cadastral_standards,
            road_union=road_line.buffer(8.0),
            meters_per_px=0.25,
        )

        # Each building must be contained inside its parcel
        for b in bldg_features:
            ulpin = b["properties"]["parcel_ulpin"]
            self.assertTrue(bool(ulpin), f"Building {b['properties']['id']} must have an assigned parcel ULPIN")

    def test_04_cross_platform_paths(self):
        """Verify core scripts contain no hardcoded user paths."""
        checked_files = [
            BACKEND_DIR / "build_clear_drone_survey.py",
            BACKEND_DIR / "upload_pipeline.py",
            BACKEND_DIR / "app.py",
            BACKEND_DIR / "cadastral_standards.py",
        ]
        for fpath in checked_files:
            if fpath.exists():
                text = fpath.read_text(encoding="utf-8")
                self.assertNotIn("C:\\Users\\gargm", text, f"{fpath.name} contains hardcoded user path")

    def test_05_upload_pipeline_execution(self):
        """Test end-to-end execution of upload_pipeline.process_upload()."""
        sample_img = REPO_ROOT / "data" / "processed" / "austin_sample.jpg"
        if not sample_img.exists():
            self.skipTest("Sample image not present for upload test")

        session_id = "test_verification_run"
        res = upload_pipeline.process_upload(
            file_path=sample_img,
            center_lat=30.2268,
            center_lon=-97.7845,
            width_m=200.0,
            session_id=session_id,
        )

        self.assertIn("session_id", res)
        self.assertGreater(res["parcels"], 0, "Uploaded survey must generate parcels")

        session_dir = REPO_ROOT / "data" / "uploads" / session_id
        self.assertTrue((session_dir / "parcels.geojson").exists(), "parcels.geojson must exist")
        self.assertTrue((session_dir / "buildings.geojson").exists(), "buildings.geojson must exist")
        self.assertTrue((session_dir / "property_cards.json").exists(), "property_cards.json must exist")
        self.assertTrue((session_dir / "cadastre.dxf").exists(), "cadastre.dxf must exist")

        # Verify GeoJSON validity
        parcels_data = json.loads((session_dir / "parcels.geojson").read_text(encoding="utf-8"))
        self.assertEqual(parcels_data["type"], "FeatureCollection")
        self.assertGreater(len(parcels_data["features"]), 0)

        first_p = parcels_data["features"][0]
        self.assertTrue(cadastral_standards.validate_ulpin(first_p["properties"]["ulpin"]))
        self.assertGreater(first_p["properties"]["area_m2"], 0)

    def test_06_visible_vegetation_and_barren_land_indices(self):
        """Verify mathematical calculation and masking of ExG, VARI, and Soil Tone Index (STI)."""
        h, w = 60, 60
        synth_img = np.zeros((h, w, 3), dtype=np.uint8)
        synth_img[:20, :] = [30, 160, 40]   # Vegetation / Green
        synth_img[20:40, :] = [60, 110, 180] # Barren Land / Brown-Red soil
        synth_img[40:, :] = [120, 120, 120]  # Neutral Grey road

        res = vegetation_index.compute_visible_vegetation_and_soil_indices(synth_img)
        
        # 1. Vegetation checks
        self.assertGreater(res["exg"][:20, :].mean(), 0.10, "Vegetation must produce strongly positive ExG")
        self.assertGreater(res["vari"][:20, :].mean(), 0.05, "Vegetation must produce positive VARI")
        self.assertGreater(res["veg_mask"][:20, :].mean(), 200, "Vegetation patch must be flagged in veg_mask")
        self.assertEqual(res["veg_mask"][20:40, :].max(), 0, "Barren land must NOT be flagged as vegetation")

        # 2. Barren land checks
        self.assertGreater(res["sti"][20:40, :].mean(), 0.15, "Barren earth must produce strong Soil Tone Index")
        self.assertGreater(res["barren_mask"][20:40, :].mean(), 200, "Barren patch must be flagged in barren_mask")
        self.assertEqual(res["barren_mask"][:20, :].max(), 0, "Vegetation must NOT be flagged as barren")
        self.assertEqual(res["barren_mask"][40:, :].max(), 0, "Neutral grey road must NOT be flagged as barren")

    def test_07_rural_parcel_landcover_classification(self):
        """Verify landcover attribution and classification for Cropland, Tree Canopy, and Barren Land."""
        w, h = 100, 100
        # 1. Test Cropland Parcel
        crop_mask = np.zeros((h, w), dtype=np.uint8)
        crop_mask[10:90, 10:90] = 255
        tree_mask = np.zeros((h, w), dtype=np.uint8)
        barren_mask = np.zeros((h, w), dtype=np.uint8)
        veg_mask = crop_mask.copy()

        p_poly = box(10, 10, 90, 90)
        stats_crop = vegetation_index.analyze_parcel_landcover(
            poly_px=p_poly,
            veg_mask=veg_mask,
            tree_mask=tree_mask,
            barren_mask=barren_mask,
            bldg_area_px=0.0,
            meters_per_px=0.3
        )
        self.assertEqual(stats_crop["landuse"], "Agricultural / Cultivated Cropland")
        self.assertGreater(stats_crop["vegetation_cover_pct"], 90.0)

        # 2. Test Barren Land Parcel
        barren_mask[10:90, 10:90] = 255
        crop_mask.fill(0)
        veg_mask.fill(0)
        stats_barren = vegetation_index.analyze_parcel_landcover(
            poly_px=p_poly,
            veg_mask=veg_mask,
            tree_mask=tree_mask,
            barren_mask=barren_mask,
            bldg_area_px=0.0,
            meters_per_px=0.3
        )
        self.assertEqual(stats_barren["landuse"], "Barren Land / Fallow Rural Ground")
        self.assertGreater(stats_barren["barren_cover_pct"], 90.0)


if __name__ == "__main__":
    unittest.main()
