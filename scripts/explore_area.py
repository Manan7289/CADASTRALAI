"""Probe candidate bounding boxes against live Overpass API to find a real
area with a good mix of buildings, roads, railway, waterway and government
land -- so the demo AOI is chosen from actual data, not assumption."""
import requests

OVERPASS = "https://overpass-api.de/api/interpreter"

CANDIDATES = {
    "nashik_road":     (19.945, 73.800, 19.960, 73.815),
    "igatpuri":        (19.690, 73.555, 19.705, 73.570),
    "kalyan_east":     (19.235, 73.145, 19.248, 73.160),
    "lonavala":        (18.745, 73.405, 18.758, 73.420),
    "thane_kalwa":     (19.180, 72.975, 19.195, 72.995),
}

def query(bbox):
    s, w, n, e = bbox
    q = f"""
    [out:json][timeout:25];
    (
      way["building"]({s},{w},{n},{e});
      way["highway"]({s},{w},{n},{e});
      way["railway"="rail"]({s},{w},{n},{e});
      way["waterway"]({s},{w},{n},{e});
      way["natural"="water"]({s},{w},{n},{e});
      way["amenity"~"school|townhall|government_office"]({s},{w},{n},{e});
      node["amenity"~"school|townhall|government_office"]({s},{w},{n},{e});
    );
    out tags;
    """
    headers = {"User-Agent": "CadastraAI-SIH2026-Demo/1.0 (educational hackathon project)"}
    r = requests.post(OVERPASS, data={"data": q}, headers=headers, timeout=40)
    r.raise_for_status()
    els = r.json()["elements"]
    counts = {"building": 0, "highway": 0, "railway": 0, "waterway": 0, "water": 0, "govt": 0}
    for el in els:
        t = el.get("tags", {})
        if "building" in t: counts["building"] += 1
        if "highway" in t: counts["highway"] += 1
        if t.get("railway") == "rail": counts["railway"] += 1
        if "waterway" in t: counts["waterway"] += 1
        if t.get("natural") == "water": counts["water"] += 1
        if t.get("amenity") in ("school", "townhall", "government_office"): counts["govt"] += 1
    return counts

if __name__ == "__main__":
    for name, bbox in CANDIDATES.items():
        try:
            c = query(bbox)
            print(f"{name:15s} {bbox}  ->  {c}")
        except Exception as e:
            print(f"{name:15s} FAILED: {e}")
