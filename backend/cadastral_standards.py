"""National Cadastral Standards, ULPIN (Bhu-Aadhaar) Generator,
Boundary Traverse Point Analysis, and Property Card Exporter.

Conforms to standards established under:
- Digital India Land Records Modernization Programme (DILRMP)
- SVAMITVA Scheme (Survey of Villages and Mapping with Improvised Technology in Village Areas)
- ISO 19152 Land Administration Domain Model (LADM)
"""
import hashlib
import json
import math
from typing import Dict, List, Tuple
from shapely.geometry import Polygon, shape, mapping


def compute_iso7064_mod11_2(data: str) -> str:
    """Compute the ISO 7064 Mod 11-2 check character for numeric or alphanumeric input string."""
    val = 0
    for char in data:
        if '0' <= char <= '9':
            d = int(char)
        elif 'A' <= char <= 'Z':
            d = ord(char) - ord('A') + 10
        elif 'a' <= char <= 'z':
            d = ord(char) - ord('a') + 10
        else:
            d = 0
        val = ((val + d) * 2) % 11
    check = (11 - val) % 11
    if check == 10:
        return 'X'
    return str(check)


def validate_ulpin(ulpin: str) -> bool:
    """Validate that a 14-character ULPIN has a valid ISO 7064 Mod 11-2 check character."""
    if not ulpin or len(ulpin) != 14:
        return False
    body = ulpin[:-1]
    expected = compute_iso7064_mod11_2(body)
    return ulpin[-1].upper() == expected


def generate_ulpin(lat: float, lon: float, parcel_seq: int = 0) -> str:
    """Generate a standard 14-character Unique Land Parcel Identification Number (ULPIN / Bhu-Aadhaar).
    Encodes spatial coordinates (degrees, minutes, seconds) with an ISO 7064 Mod 11-2 check character.
    Format: 13-character geographic plot identifier + 1-character ISO 7064 check character = 14 chars.
    """
    lat_abs = abs(lat)
    lon_abs = abs(lon)

    lat_deg = int(lat_abs) % 100
    lat_min = int((lat_abs - int(lat_abs)) * 60)
    lat_sec = int((((lat_abs - int(lat_abs)) * 60) - lat_min) * 60)

    lon_deg = int(lon_abs) % 100
    lon_min = int((lon_abs - int(lon_abs)) * 60)
    lon_sec = int((((lon_abs - int(lon_abs)) * 60) - lon_min) * 60)

    seq_digit = abs(parcel_seq) % 10

    # 13 characters: lat_deg(2) + lat_min(2) + lat_sec(2) + lon_deg(2) + lon_min(2) + lon_sec(2) + seq_digit(1)
    base = f"{lat_deg:02d}{lat_min:02d}{lat_sec:02d}{lon_deg:02d}{lon_min:02d}{lon_sec:02d}{seq_digit:01d}"
    check_char = compute_iso7064_mod11_2(base)
    return f"{base}{check_char}"


def _ensure_polygon(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda g: g.area)
    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if polys:
            p = max(polys, key=lambda g: g.area)
            if p.geom_type == "MultiPolygon":
                return max(p.geoms, key=lambda g: g.area)
            return p
    return None


def extract_traverse_points(polygon, proj=None) -> List[Dict]:
    """Extract ordered boundary traverse corner points (P1, P2, ... Pn)
    with GPS coordinates, local metric offsets, and segment lengths."""
    polygon = _ensure_polygon(polygon)
    if polygon is None or polygon.is_empty:
        return []

    coords = list(polygon.exterior.coords)
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]

    # Simplify collinear or sub-decimeter jitter points
    simplified_coords = []
    min_dist_deg = 0.00002  # ~2 meters
    for pt in coords:
        if not simplified_coords:
            simplified_coords.append(pt)
            continue
        prev = simplified_coords[-1]
        dist = math.hypot(pt[0] - prev[0], pt[1] - prev[1])
        if dist >= min_dist_deg:
            simplified_coords.append(pt)

    if len(simplified_coords) < 3:
        simplified_coords = coords

    traverse = []
    n = len(simplified_coords)
    deg_to_m_lat = 111320.0

    for i in range(n):
        curr = simplified_coords[i]
        nxt = simplified_coords[(i + 1) % n]

        mid_lat = (curr[1] + nxt[1]) / 2.0
        deg_to_m_lon = 111320.0 * math.cos(math.radians(mid_lat))

        dx_m = (nxt[0] - curr[0]) * deg_to_m_lon
        dy_m = (nxt[1] - curr[1]) * deg_to_m_lat
        segment_len_m = round(math.hypot(dx_m, dy_m), 2)

        point_info = {
            "point_id": f"P{i + 1}",
            "lon": round(curr[0], 7),
            "lat": round(curr[1], 7),
            "segment_to_next": f"P{i + 1} → P{((i + 1) % n) + 1}",
            "segment_length_m": segment_len_m,
        }
        traverse.append(point_info)

    return traverse


def compute_cadastral_metrics(parcel_geom: Polygon, building_geoms: List[Polygon]) -> Dict:
    """Compute complete multi-unit area, frontage, and building coverage ratios."""
    # Approximate metric area using centroid latitude
    centroid = parcel_geom.centroid
    lat_rad = math.radians(centroid.y)
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(lat_rad)
    scale_sq_m = m_per_deg_lat * m_per_deg_lon

    parcel_area_m2 = round(parcel_geom.area * scale_sq_m, 2)
    parcel_perimeter_m = round(parcel_geom.length * ((m_per_deg_lat + m_per_deg_lon) / 2.0), 2)

    # Unit conversions
    area_sq_ft = round(parcel_area_m2 * 10.7639, 1)
    area_guntha = round(parcel_area_m2 / 101.17, 3)  # Standard Guntha in Maharashtra/Western India
    area_acres = round(parcel_area_m2 / 4046.86, 4)

    # Built-up area calculation
    built_up_area_m2 = 0.0
    for b_geom in building_geoms:
        inter = parcel_geom.intersection(b_geom)
        if not inter.is_empty:
            built_up_area_m2 += inter.area * scale_sq_m

    built_up_area_m2 = round(built_up_area_m2, 2)
    open_space_m2 = max(0.0, round(parcel_area_m2 - built_up_area_m2, 2))
    gcr_pct = round((built_up_area_m2 / parcel_area_m2) * 100.0, 1) if parcel_area_m2 > 0 else 0.0

    return {
        "area_m2": parcel_area_m2,
        "area_sq_ft": area_sq_ft,
        "area_guntha": area_guntha,
        "area_acres": area_acres,
        "perimeter_m": parcel_perimeter_m,
        "built_up_area_m2": built_up_area_m2,
        "open_space_m2": open_space_m2,
        "ground_coverage_ratio_pct": gcr_pct,
    }


def render_parcel_svg_sketch(parcel_geom: Polygon, building_geoms: List[Polygon], traverse: List[Dict]) -> str:
    """Generate clean vector SVG cadastral sketch showing the parcel polygon,
    corner traverse points (P1..Pn), and interior building footprint."""
    parcel_geom = _ensure_polygon(parcel_geom)
    if parcel_geom is None or parcel_geom.is_empty:
        return '<svg viewBox="0 0 360 220" width="100%" height="220" style="background:#121A15;border-radius:8px;"></svg>'

    minx, miny, maxx, maxy = parcel_geom.bounds
    w = max(maxx - minx, 1e-6)
    h = max(maxy - miny, 1e-6)

    svg_w, svg_h = 360, 260
    pad = 35

    def to_svg(x, y):
        sx = pad + ((x - minx) / w) * (svg_w - 2 * pad)
        sy = svg_h - pad - ((y - miny) / h) * (svg_h - 2 * pad)
        return round(sx, 1), round(sy, 1)

    # Parcel exterior
    p_pts = " ".join(f"{to_svg(x, y)[0]},{to_svg(x, y)[1]}" for x, y in parcel_geom.exterior.coords)

    # Buildings inside
    b_polys_svg = []
    for b in building_geoms:
        inter = parcel_geom.intersection(b)
        if not inter.is_empty and inter.geom_type == "Polygon":
            b_coords = " ".join(f"{to_svg(x, y)[0]},{to_svg(x, y)[1]}" for x, y in inter.exterior.coords)
            b_polys_svg.append(f'<polygon points="{b_coords}" fill="#6CBE81" fill-opacity="0.45" stroke="#6CBE81" stroke-width="1.5"/>')

    # Traverse corner markers and labels
    markers_svg = []
    for pt in traverse:
        sx, sy = to_svg(pt["lon"], pt["lat"])
        markers_svg.append(f'<circle cx="{sx}" cy="{sy}" r="4" fill="#45C7B8" stroke="#0D1310" stroke-width="1.5"/>')
        markers_svg.append(f'<text x="{sx + 6}" y="{sy - 4}" font-family="monospace" font-size="10" font-weight="bold" fill="#E7EEE9">{pt["point_id"]}</text>')

    svg = f"""<svg viewBox="0 0 {svg_w} {svg_h}" width="100%" height="220" style="background:#121A15;border-radius:8px;border:1px solid #28352E;" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
      <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(255,255,255,0.04)" stroke-width="1"/>
    </pattern>
  </defs>
  <rect width="{svg_w}" height="{svg_h}" fill="url(#grid)" />
  <polygon points="{p_pts}" fill="#45C7B8" fill-opacity="0.16" stroke="#45C7B8" stroke-width="2.2" stroke-dasharray="6,3"/>
  {''.join(b_polys_svg)}
  {''.join(markers_svg)}
  <text x="14" y="24" font-family="monospace" font-size="10" fill="#9FB3A8">N ↑ CADASTRAAI PLOT SKETCH</text>
</svg>"""
    return svg


def generate_cadastral_property_card(parcel_props: Dict, parcel_geom: Polygon, building_geoms: List[Polygon], aoi_name: str = "Urban Ward") -> str:
    """Generate complete, printable SVAMITVA-compliant Cadastral Property Card (HTML)."""
    traverse = parcel_props.get("traverse_points") or extract_traverse_points(parcel_geom)
    metrics = compute_cadastral_metrics(parcel_geom, building_geoms)
    svg_sketch = render_parcel_svg_sketch(parcel_geom, building_geoms, traverse)

    ulpin = parcel_props.get("ulpin", generate_ulpin(parcel_props["gps_lat"], parcel_props["gps_lon"], parcel_props.get("id", 0)))
    pid = parcel_props.get("id", 0)
    landuse = parcel_props.get("landuse", "Residential")
    road_status = "Connected (Frontage Verified)" if parcel_props.get("road_connected") else f"Setback / ~{parcel_props.get('road_distance_m', 0)} m away"

    traverse_rows = "".join(f"""
    <tr>
      <td style="font-weight:600;color:#45C7B8;">{t['point_id']}</td>
      <td>{t['lat']:.7f}° N</td>
      <td>{t['lon']:.7f}° E</td>
      <td>{t['segment_to_next']}</td>
      <td style="font-weight:600;">{t['segment_length_m']} m</td>
    </tr>
    """ for t in traverse)

    alerts = parcel_props.get("alerts", [])
    if alerts:
        alerts_html = '<div style="background:#2C1B19;border-left:4px solid #E2685F;padding:10px 14px;border-radius:4px;margin-top:14px;">'
        alerts_html += '<strong style="color:#E2685F;">Compliance Advisory / Findings:</strong><ul style="margin:6px 0 0 0;padding-left:18px;font-size:0.82rem;color:#E7EEE9;">'
        for a in alerts:
            alerts_html += f"<li>{a.get('type', 'ALERT')}: {a.get('msg', '')} <em>({a.get('citation', '')})</em></li>"
        alerts_html += "</ul></div>"
    else:
        alerts_html = '<div style="background:#13261B;border-left:4px solid #6CBE81;padding:10px 14px;border-radius:4px;margin-top:14px;color:#6CBE81;font-size:0.85rem;">✔ No statutory setback or encroachment violations detected against automated reference buffers.</div>'

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Cadastral Property Card — {ulpin}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background:#0B100D; color:#E7EEE9; margin:0; padding:24px; }}
  .card-container {{ max-width:820px; margin:0 auto; background:#141C18; border:1px solid #28352E; border-radius:12px; padding:32px; box-shadow:0 12px 36px rgba(0,0,0,0.6); }}
  .header {{ display:flex; justify-content:space-between; align-items:flex-start; border-bottom:2px solid #45C7B8; padding-bottom:18px; margin-bottom:20px; }}
  .header h1 {{ margin:0 0 4px 0; font-size:1.6rem; letter-spacing:0.04em; text-transform:uppercase; color:#E7EEE9; }}
  .header p {{ margin:0; font-size:0.8rem; color:#9FB3A8; text-transform:uppercase; letter-spacing:0.06em; }}
  .ulpin-badge {{ background:rgba(69,199,184,0.12); border:1px solid #45C7B8; color:#45C7B8; padding:8px 14px; border-radius:8px; font-family:monospace; font-size:1.1rem; font-weight:bold; }}
  .grid2 {{ display:grid; grid-template-columns:1.2fr 1fr; gap:24px; margin-bottom:22px; }}
  .kv-table {{ width:100%; border-collapse:collapse; font-size:0.85rem; }}
  .kv-table td {{ padding:7px 8px; border-bottom:1px solid #212C26; }}
  .kv-table td.k {{ color:#9FB3A8; width:44%; font-weight:500; }}
  .kv-table td.v {{ color:#E7EEE9; font-weight:600; }}
  .traverse-table {{ width:100%; border-collapse:collapse; font-size:0.82rem; margin-top:10px; }}
  .traverse-table th {{ background:#1B2520; color:#9FB3A8; text-align:left; padding:8px 10px; font-size:0.75rem; text-transform:uppercase; }}
  .traverse-table td {{ padding:7px 10px; border-bottom:1px solid #212C26; }}
  .footer {{ margin-top:30px; padding-top:16px; border-top:1px solid #28352E; display:flex; justify-content:space-between; align-items:center; font-size:0.75rem; color:#66796F; }}
  @media print {{
    body {{ background:#fff !important; color:#000 !important; padding:0; }}
    .card-container {{ border:none; box-shadow:none; padding:10px; background:#fff !important; color:#000 !important; }}
    .header {{ border-bottom-color:#000; }}
    .header h1, .header p, .kv-table td, .traverse-table td {{ color:#000 !important; }}
    .kv-table td {{ border-bottom-color:#ddd !important; }}
    .traverse-table th {{ background:#eee !important; color:#000 !important; }}
    .ulpin-badge {{ border-color:#000; color:#000; background:#f0f0f0; }}
    .print-btn {{ display:none; }}
  }}
</style>
</head>
<body>
<div class="card-container">
  <div style="text-align:right;margin-bottom:12px;" class="print-btn">
    <button onclick="window.print()" style="background:#45C7B8;color:#04211D;border:none;padding:8px 16px;border-radius:6px;font-weight:bold;cursor:pointer;">🖨 Print / Save as PDF</button>
  </div>
  <div class="header">
    <div>
      <h1>Cadastral Property Record Card</h1>
      <p>SVAMITVA Standard Form · Automated Drone Cadastre · {aoi_name}</p>
    </div>
    <div class="ulpin-badge">
      ULPIN: {ulpin}
    </div>
  </div>

  <div class="grid2">
    <div>
      <table class="kv-table">
        <tr><td class="k">Parcel ID / Survey No.</td><td class="v">#{pid}</td></tr>
        <tr><td class="k">Land Use Category</td><td class="v">{landuse}</td></tr>
        <tr><td class="k">Total Parcel Area</td><td class="v">{metrics['area_m2']} m² ({metrics['area_sq_ft']} sq.ft)</td></tr>
        <tr><td class="k">Regional Measure</td><td class="v">{metrics['area_guntha']} Guntha / {metrics['area_acres']} Acres</td></tr>
        <tr><td class="k">Perimeter</td><td class="v">{metrics['perimeter_m']} m</td></tr>
        <tr><td class="k">Built-up Footprint Area</td><td class="v">{metrics['built_up_area_m2']} m²</td></tr>
        <tr><td class="k">Open Yard Area</td><td class="v">{metrics['open_space_m2']} m²</td></tr>
        <tr><td class="k">Ground Coverage Ratio</td><td class="v">{metrics['ground_coverage_ratio_pct']}%</td></tr>
        <tr><td class="k">Road Frontage Status</td><td class="v">{road_status}</td></tr>
        <tr><td class="k">Centroid Coordinates</td><td class="v">{parcel_props['gps_lat']:.6f}° N, {parcel_props['gps_lon']:.6f}° E</td></tr>
      </table>
    </div>
    <div>
      {svg_sketch}
    </div>
  </div>

  <h3 style="font-size:0.95rem;text-transform:uppercase;letter-spacing:0.04em;margin:18px 0 6px 0;color:#E7EEE9;">Boundary Traverse Coordinates (Demarcated Vertices)</h3>
  <table class="traverse-table">
    <thead>
      <tr>
        <th>Station</th>
        <th>Latitude (WGS84)</th>
        <th>Longitude (WGS84)</th>
        <th>Boundary Segment</th>
        <th>Dimension (Length)</th>
      </tr>
    </thead>
    <tbody>
      {traverse_rows}
    </tbody>
  </table>

  {alerts_html}

  <div class="footer">
    <div>Generated by CadastraAI Drone Cadastral Mapping System</div>
    <div>Digital Verification Signature Block: [VERIFIED AUTOMATED SURVEY]</div>
  </div>
</div>
</body>
</html>"""
    return html


def export_cadastral_dxf(parcels_fc: Dict, buildings_fc: Dict, out_path) -> None:
    """Generate standard ASCII DXF (Drawing Exchange Format) for AutoCAD / Land Surveyors."""
    lines = [
        "0", "SECTION",
        "2", "ENTITIES",
    ]

    # Export Parcels
    for f in parcels_fc.get("features", []):
        geom = shape(f["geometry"])
        pid = f["properties"].get("id", 0)
        ulpin = f["properties"].get("ulpin", f"PARCEL-{pid}")
        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        else:
            continue

        for p in polys:
            coords = list(p.exterior.coords)
            lines.extend([
                "0", "POLYLINE",
                "8", "CADASTRE_PARCELS",
                "62", "3",  # Green color
                "66", "1",
                "70", "1",  # Closed polyline
            ])
            for x, y in coords:
                lines.extend([
                    "0", "VERTEX",
                    "8", "CADASTRE_PARCELS",
                    "10", f"{x:.6f}",
                    "20", f"{y:.6f}",
                    "30", "0.0",
                ])
            lines.extend(["0", "SEQEND"])

            # Center text for ULPIN
            cx, cy = p.centroid.x, p.centroid.y
            lines.extend([
                "0", "TEXT",
                "8", "PARCEL_LABELS",
                "10", f"{cx:.6f}",
                "20", f"{cy:.6f}",
                "30", "0.0",
                "40", "0.0001",  # text height in degrees approx
                "1", str(ulpin),
            ])

    # Export Buildings
    for f in buildings_fc.get("features", []):
        geom = shape(f["geometry"])
        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        else:
            continue

        for p in polys:
            coords = list(p.exterior.coords)
            lines.extend([
                "0", "POLYLINE",
                "8", "BUILDING_FOOTPRINTS",
                "62", "1",  # Red color
                "66", "1",
                "70", "1",
            ])
            for x, y in coords:
                lines.extend([
                    "0", "VERTEX",
                    "8", "BUILDING_FOOTPRINTS",
                    "10", f"{x:.6f}",
                    "20", f"{y:.6f}",
                    "30", "0.0",
                ])
            lines.extend(["0", "SEQEND"])

    lines.extend(["0", "ENDSEC", "0", "EOF"])
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
