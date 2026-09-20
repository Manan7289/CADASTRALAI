"""One-page survey report (HTML, print-friendly): what was mapped, how well, and what needs checking.

A cadastral survey ends with a document for the file, not just GIS layers. The report pulls
everything from the survey folder: the map (image + parcels + buildings), counts, land use and
land cover, the road network by width, possible encroachments, topology status, review progress,
record / GNSS comparison when there is one, and the models used with their tested accuracy.
"""
import base64
import html
import io
import json
import math
import time

import numpy as np
from PIL import Image, ImageDraw

import reference
import survey

LANDUSE_COLOURS = {"Residential": "#F2C14E", "Residential - apartments": "#E8913A", "Commercial / mixed use": "#D1495B",
                   "Institutional / large complex": "#8F7CF6", "Vacant plot": "#C8955A", "Open space / green": "#8FD16A",
                   "Water body": "#3A86FF"}


def _map_png(d, meta, parcels, buildings, max_px=1400):
    img = Image.open(d / "ori.webp").convert("RGB")
    W, H = img.size
    sc = min(1.0, max_px / max(W, H))
    img = img.resize((int(W * sc), int(H * sc)))
    (s_, w_), (n_, e_) = meta["bounds"]
    my = lambda lat: math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    def px(lon, lat):
        return ((lon - w_) / (e_ - w_) * img.width, (my(n_) - my(lat)) / (my(n_) - my(s_)) * img.height)
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(over)
    def rings(g):
        if g["type"] == "Polygon":
            return [g["coordinates"][0]]
        if g["type"] == "MultiPolygon":
            return [p[0] for p in g["coordinates"]]
        return []
    for f in parcels["features"]:
        col = LANDUSE_COLOURS.get(f["properties"].get("land_use"), "#9FB3A8").lstrip("#")
        rgb = tuple(int(col[i:i + 2], 16) for i in (0, 2, 4))
        for r in rings(f["geometry"]):
            dr.polygon([px(*c) for c in r], fill=rgb + (70,), outline=(255, 255, 255, 230))
    for f in buildings["features"]:
        enc = f["properties"].get("encroachment_m2")
        for r in rings(f["geometry"]):
            dr.line([px(*c) for c in r] + [px(*r[0])], fill=(255, 45, 85, 255) if enc else (255, 107, 74, 220), width=3 if enc else 1)
    out = Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def build(sid):
    d = survey.survey_dir(sid)
    meta = json.loads((d / "meta.json").read_text())
    parcels = json.loads((d / "parcels.geojson").read_text())
    buildings = json.loads((d / "buildings.geojson").read_text())
    issues = json.loads((d / "issues.geojson").read_text())
    s = meta.get("stats", {})
    feats = [f["properties"] for f in parcels["features"]]
    e = html.escape
    counts = {k: sum(1 for p in feats if p.get("status") == k) for k in ("draft", "approved", "field_check", "rejected")}
    n = max(len(feats), 1)
    area = sum(p.get("area_m2") or 0 for p in feats)
    proc = meta.get("processing", {})
    lu = s.get("land_use_count") or {}
    lc = s.get("land_cover_fraction") or {}
    roads = s.get("road_length_by_class_m") or {}
    enc = sorted([f for f in buildings["features"] if f["properties"].get("encroachment_m2")],
                 key=lambda f: -f["properties"]["encroachment_m2"])
    cmp = None
    try:
        st = reference.status(sid)
        cmp = st.get("comparison")
    except Exception:
        pass

    def table(rows, head):
        return ("<table><tr>" + "".join(f"<th>{e(h)}</th>" for h in head) + "</tr>" +
                "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>")

    kpis = [("Parcels", len(feats)), ("Buildings", s.get("buildings", "—")),
            ("Roads & lanes", f"{round(s.get('corridor_length_m') or 0):,} m"), ("Area mapped", f"{proc.get('area_km2', round(area / 1e6, 3))} km²"),
            ("Topology problems", len(issues["features"])), ("Possible encroachments", s.get("encroachments", 0)),
            ("Reviewed", f"{round(100 * (n - counts['draft']) / n)}%"), ("Approved", counts["approved"])]
    body = [f"""
<header><div><h1>{e(meta['name'])}</h1><div class="sub">Cadastral survey report · CadastraAI · generated {time.strftime('%d %b %Y %H:%M')}</div></div>
<div class="meta">{e(meta['source'])}<br>CRS {e(meta['crs'])} · {round(meta['gsd_m'] * 100)} cm per pixel · survey {e(sid)}</div></header>
<section class="kpis">{''.join(f'<div><b>{e(str(v))}</b><span>{e(k)}</span></div>' for k, v in kpis)}</section>
<section class="two"><div><h2>Parcel map</h2><img src="data:image/jpeg;base64,{_map_png(d, meta, parcels, buildings)}">
<div class="note">Parcels coloured by suggested land use; buildings outlined in orange, possible encroachments in red.</div></div>
<div><h2>Land use (suggested)</h2>{table([[f'<i style="background:{LANDUSE_COLOURS.get(k, "#999")}"></i>{e(k)}', v, f"{round(100 * v / n)}%"] for k, v in lu.items() if v], ["Use", "Parcels", "Share"])}
<h2>Land cover</h2>{table([[e(k), f"{round(100 * v)}%"] for k, v in sorted(lc.items(), key=lambda kv: -kv[1]) if v >= 0.005], ["Class", "Share of area"])}
<h2>Road network</h2>{table([[e(k), f"{round(v):,} m"] for k, v in roads.items()], ["Class", "Length"]) if roads else '<div class="note">—</div>'}
<div class="note">Lane: under 3 m wide; street: 3–8 m; main road: over 8 m.</div></div></section>"""]
    body.append("<section><h2>Possible encroachments</h2>" + (
        table([[b["properties"].get("id"), f"{b['properties']['encroachment_m2']} m²", f"{b['properties'].get('encroachment_depth_m', '—')} m"] for b in enc[:12]],
              ["Building", "Area on the road / lane", "Sticks out by"]) +
        (f'<div class="note">Largest 12 of {len(enc)} shown; the full list is in the workbench (Review › Encroachment) and the export.</div>' if len(enc) > 12 else "") if enc else '<div class="note">None flagged.</div>') +
        '<div class="note">Buildings that extend at least 1 m onto a road or lane as detected in the imagery. Verify on site against the approved road width before any action.</div></section>')
    if cmp and (cmp.get("parcels") or cmp.get("gnss") or cmp.get("encroachment")):
        rows = []
        if cmp.get("parcels"):
            p = cmp["parcels"]
            rows += [["AI parcels matching the existing record", f"{p.get('agreement_pct')}%"], ["Mean IoU of matches", p.get("mean_iou_matched")],
                     ["Record parcels missing from the AI map", p.get("missing_from_ai")]]
        if cmp.get("gnss"):
            g = cmp["gnss"]
            rows += [["GNSS corner points scored", g.get("corner_points")], ["Corner error RMSE", f"{g.get('rmse_m')} m"], ["Corner error CE90", f"{g.get('ce90_m')} m"]]
        if cmp.get("encroachment"):
            en = cmp["encroachment"]
            rows += [["Buildings across a recorded boundary", en.get("crosses_record")], ["Buildings outside every recorded parcel", en.get("outside_record")]]
        body.append("<section><h2>Comparison with existing records and ground truth</h2>" + table(rows, ["Check", "Result"]) + "</section>")
    body.append("<section class='two'><div><h2>Topology validation</h2>" + (
        table([[e(i["properties"]["type"]), e(i["properties"]["message"])] for i in issues["features"][:15]], ["Problem", "Detail"])
        if issues["features"] else '<div class="ok">Topology clean: no overlaps, gaps, slivers, holes or invalid parcels.</div>') +
        "<h2>Review progress</h2>" + table([["To review", counts["draft"]], ["Approved", counts["approved"]], ["Field check", counts["field_check"]], ["Rejected", counts["rejected"]]], ["Status", "Parcels"]) +
        "</div><div><h2>Models and accuracy</h2>" +
        f"<p>{e((meta.get('model') or {}).get('name', '—'))}</p><p class='note'>{e((meta.get('model') or {}).get('summary', ''))}</p>" +
        f"<p class='note'><b>Known limits:</b> {e((meta.get('model') or {}).get('limits', ''))}</p>" +
        (f"<p class='note'>Processing: parcels, roads and topology built in {proc.get('build_seconds')} s" +
         (f"; AI models on Kaggle GPU in {round(proc['model_seconds'] / 60, 1)} min" if proc.get("model_seconds") else "") + ".</p>" if proc else "") +
        "<p class='note'>All parcels are preliminary until approved by a surveyor. Land use is a suggestion from buildings, storeys, road width and land cover.</p></div></section>")
    css = """:root{color-scheme:light}html{background:#fff}
body{font:13px/1.45 'IBM Plex Sans',system-ui,sans-serif;color:#1d2521;background:#fff;margin:28px auto;padding:0 28px;max-width:1150px}
header{display:flex;justify-content:space-between;gap:20px;border-bottom:3px solid #1f6f66;padding-bottom:10px;margin-bottom:14px}
h1{margin:0;font-size:24px}h2{font-size:14px;text-transform:uppercase;letter-spacing:.04em;color:#1f6f66;margin:16px 0 6px}
.sub,.meta,.note{color:#5b6b63;font-size:12px}.meta{text-align:right;max-width:460px}
.kpis{display:grid;grid-template-columns:repeat(8,1fr);gap:8px}.kpis div{border:1px solid #d6dfd9;border-radius:8px;padding:8px}
.kpis b{display:block;font-size:18px}.kpis span{font-size:11px;color:#5b6b63}
.two{display:grid;grid-template-columns:1.35fr 1fr;gap:22px}img{width:100%;border-radius:6px;border:1px solid #ccc}
table{border-collapse:collapse;width:100%;margin:4px 0}th,td{border-bottom:1px solid #e1e8e3;padding:4px 6px;text-align:left;font-size:12px}
th{background:#eef4f1}i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}
.ok{background:#e8f6ec;border:1px solid #9fd3ae;padding:8px;border-radius:6px}
@media print{body{margin:10mm}.kpis{grid-template-columns:repeat(4,1fr)}}"""
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{e(meta['name'])} — survey report</title>"
            f"<style>{css}</style></head><body>{''.join(body)}</body></html>")
