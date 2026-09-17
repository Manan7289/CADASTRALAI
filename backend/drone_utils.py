"""Drone imagery and video ingestion utilities.

Handles:
- Automated EXIF extraction from drone photos (DJI, Autel, standard UAVs):
  GPS latitude/longitude, relative altitude, focal length, sensor dimensions.
- Ground Sample Distance (GSD, cm/px) and real ground footprint calculation.
- Automated georeferenced bounding box derivation (eliminates manual lat/lon/width typing).
- Video keyframe selection: samples frames and scores structural sharpness
  using Laplacian variance to select the optimal crisp survey frame (avoiding motion blur).
"""
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import ExifTags, Image

# Common sensor dimensions (width in mm) for drone models if not specified in EXIF
DEFAULT_SENSOR_WIDTH_MM = 13.2  # 1-inch sensor (DJI Mavic 2 Pro, Air 2S, Phantom 4 Pro)
DEFAULT_DRONE_ALTITUDE_M = 60.0  # 60m typical survey altitude


def _convert_to_degrees(value) -> Optional[float]:
    """Helper to convert GPS coordinates from EXIF format (deg, min, sec) to decimal degrees."""
    try:
        if isinstance(value, (float, int)):
            return float(value)
        if len(value) == 3:
            d = float(value[0])
            m = float(value[1])
            s = float(value[2])
            return d + (m / 60.0) + (s / 3600.0)
    except Exception:
        pass
    return None


def extract_drone_exif(file_path: Path) -> Dict:
    """Extract drone telemetry, GPS, and optical parameters from image EXIF."""
    info = {
        "has_gps": False,
        "lat": None,
        "lon": None,
        "altitude_m": None,
        "focal_length_mm": None,
        "make": None,
        "model": None,
        "gsd_cm_px": None,
        "width_m": None,
    }

    try:
        with Image.open(file_path) as img:
            exif_raw = img._getexif()
            if not exif_raw:
                return info

            exif = {}
            for tag, value in exif_raw.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                exif[tag_name] = value

            info["make"] = exif.get("Make")
            info["model"] = exif.get("Model")

            # Focal length
            fl = exif.get("FocalLength")
            if fl:
                try:
                    info["focal_length_mm"] = float(fl)
                except Exception:
                    pass

            # GPS Info
            gps_info = exif.get("GPSInfo")
            if gps_info:
                gps_tags = {}
                for t, v in gps_info.items():
                    sub_name = ExifTags.GPSTAGS.get(t, t)
                    gps_tags[sub_name] = v

                lat = _convert_to_degrees(gps_tags.get("GPSLatitude"))
                lat_ref = gps_tags.get("GPSLatitudeRef", "N")
                if lat is not None:
                    if lat_ref == "S":
                        lat = -lat
                    info["lat"] = round(lat, 7)

                lon = _convert_to_degrees(gps_tags.get("GPSLongitude"))
                lon_ref = gps_tags.get("GPSLongitudeRef", "E")
                if lon is not None:
                    if lon_ref == "W":
                        lon = -lon
                    info["lon"] = round(lon, 7)

                # Altitude
                alt = gps_tags.get("GPSAltitude")
                if alt:
                    try:
                        info["altitude_m"] = round(float(alt), 1)
                    except Exception:
                        pass

                if info["lat"] is not None and info["lon"] is not None:
                    info["has_gps"] = True

            # Calculate Ground Sample Distance (GSD) if focal length and altitude exist
            img_w, img_h = img.size
            alt_m = info["altitude_m"] or DEFAULT_DRONE_ALTITUDE_M
            fl_mm = info["focal_length_mm"] or 8.8  # Typical DJI 24mm equivalent

            # GSD = (altitude * sensor_width) / (focal_length * image_width)
            # In meters:
            gsd_m = (alt_m * (DEFAULT_SENSOR_WIDTH_MM / 1000.0)) / ((fl_mm / 1000.0) * img_w)
            info["gsd_cm_px"] = round(gsd_m * 100.0, 2)
            info["width_m"] = round(gsd_m * img_w, 1)

    except Exception as e:
        print(f"[drone_utils] EXIF extraction error: {e}")

    return info


def extract_sharpest_keyframe(video_path: Path, max_samples: int = 40) -> Tuple[Image.Image, float]:
    """Sample candidate frames from a drone survey video and select the frame
    with maximum structural sharpness (Laplacian variance) to avoid motion blur."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    if total_frames <= 1:
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise ValueError("Could not read frames from video")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), 0.0

    # Avoid the very first 5% and last 5% (takeoff / landing / yaw turns)
    start_f = int(total_frames * 0.05)
    end_f = int(total_frames * 0.95)
    step = max(1, (end_f - start_f) // max_samples)

    best_frame = None
    best_score = -1.0
    best_idx = 0

    for f_idx in range(start_f, end_f, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        # Evaluate sharpness via Laplacian variance on grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] > 600:
            s_factor = 600.0 / gray.shape[0]
            gray_small = cv2.resize(gray, (int(gray.shape[1] * s_factor), 600))
        else:
            gray_small = gray

        score = float(cv2.Laplacian(gray_small, cv2.CV_64F).var())

        if score > best_score:
            best_score = score
            best_frame = frame
            best_idx = f_idx

    cap.release()

    if best_frame is None:
        raise ValueError("Could not extract any valid survey frame from drone video.")

    print(f"[drone_utils] Best keyframe selected at frame #{best_idx}/{total_frames} with sharpness score {best_score:.1f}")
    rgb_img = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_img), best_score


def compute_drone_geobox(center_lat: float, center_lon: float, width_m: float, img_w: int, img_h: int) -> Dict:
    """Compute local geospatial bounding box in WGS84 coordinates from center and ground width."""
    height_m = width_m * (img_h / float(img_w))
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(center_lat))

    half_w_deg = (width_m / 2.0) / m_per_deg_lon
    half_h_deg = (height_m / 2.0) / m_per_deg_lat

    return {
        "width": img_w,
        "height": img_h,
        "width_m": round(width_m, 1),
        "height_m": round(height_m, 1),
        "meters_per_px": round(width_m / img_w, 4),
        "lon_nw": center_lon - half_w_deg,
        "lat_nw": center_lat + half_h_deg,
        "lon_se": center_lon + half_w_deg,
        "lat_se": center_lat - half_h_deg,
        "source": "UAV Drone Imagery (Georeferenced Ortho-Footprint)",
    }
