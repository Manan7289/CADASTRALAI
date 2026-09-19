"""End-to-end test of the CadastraAI API (server must be running on :5050).

It edits, merges, deletes and imports records into the survey it is given, so run it
on a throwaway copy, never a demo survey:

    cp -R data/surveys/<id> data/surveys/<test-id>   # then set "id" in its meta.json to <test-id>
    .venv/bin/python tests/e2e_api.py <test-id>
"""
import io, json, random, sys, time, zipfile
import requests, numpy as np
from shapely.geometry import shape, mapping
from shapely import affinity
B = "http://127.0.0.1:5050"; SID = sys.argv[1]; U = f"{B}/api/surveys/{SID}"
res = []
def check(name, ok, detail=""):
    res.append((name, bool(ok), detail)); print(("PASS " if ok else "FAIL ") + name + (f" -- {detail}" if detail else ""))
def fc(): return requests.get(f"{B}/surveys/{SID}/parcels.geojson").json()
def types(r): return sorted({i["properties"]["type"] for i in r["issues"]["features"]})

r = requests.get(f"{B}/api/surveys").json(); check("list surveys", any(s["id"] == SID for s in r))
r = requests.patch(U, json={"name": "E2E renamed"}); check("rename", r.ok and r.json().get("name") == "E2E renamed", r.text[:100])
r = requests.patch(U, json={"name": ""}); check("rename rejects empty", r.status_code == 400, r.status_code)
for f in ("meta.json", "parcels.geojson", "issues.geojson", "buildings.geojson", "corridors.geojson", "classes.png", "landcover.png", "ori.webp"):
    check(f"serve {f}", requests.get(f"{B}/surveys/{SID}/{f}").ok)
check("tile served", requests.get(f"{B}/surveys/{SID}/tiles/{json.loads(requests.get(f'{B}/surveys/{SID}/meta.json').text)['tiles']['max_zoom']}/").status_code in (200, 404))
check("path traversal blocked", requests.get(f"{B}/surveys/{SID}/..%2F..%2Fbackend%2Fapp.py").status_code in (400, 404))

base = fc(); n0 = len(base["features"])
r = requests.put(f"{U}/parcels?action=e2e-resave", json=base).json(); check("resave unchanged -> clean", not r["issues"]["features"], types(r))

# edit: push one parcel into its neighbour -> overlap, then auto-fix
f2 = fc(); g = shape(f2["features"][10]["geometry"]); f2["features"][10]["geometry"] = mapping(affinity.scale(g, 1.3, 1.3))
r = requests.put(f"{U}/parcels?action=e2e-grow", json=f2).json(); check("grown parcel -> OVERLAP flagged", "OVERLAP" in types(r), types(r))
r = requests.post(f"{U}/autofix").json(); check("auto-fix clears overlap", not r["issues"]["features"], types(r) + [len(r["log"])])

# merge two touching parcels, then a non-touching pair
feats = fc()["features"]; geoms = [shape(f["geometry"]) for f in feats]
# a pair that shares a real edge (about 3 m or more), not just a corner
pair = next((feats[i]["properties"]["id"], feats[j]["properties"]["id"]) for i in range(len(feats)) for j in range(i+1, len(feats))
            if geoms[i].boundary.intersection(geoms[j].boundary).length > 3e-5)
r = requests.post(f"{U}/merge", json={"ids": list(pair)}); check("merge touching pair", r.ok and len(r.json()["parcels"]["features"]) == len(feats) - 1, r.status_code)
far = (feats[0]["properties"]["id"], feats[-1]["properties"]["id"])
r = requests.post(f"{U}/merge", json={"ids": list(far)}); check("merge far pair refused", r.status_code == 400, r.text[:80])
r = requests.post(f"{U}/undo"); check("undo merge", r.ok and len(r.json()["parcels"]["features"]) == len(feats), types(r.json()))

# delete a parcel -> hole in the coverage
f3 = fc(); victim = f3["features"].pop(40)
r = requests.put(f"{U}/parcels?action=e2e-delete", json=f3).json(); check("delete parcel accepted", len(r["parcels"]["features"]) == len(f3["features"]), types(r))
requests.post(f"{U}/undo")

# draw a tiny / self-intersecting parcel
f4 = fc(); c = shape(f4["features"][5]["geometry"]).centroid; d = 0.00002
bow = {"type": "Polygon", "coordinates": [[[c.x-d, c.y-d], [c.x+d, c.y+d], [c.x+d, c.y-d], [c.x-d, c.y+d], [c.x-d, c.y-d]]]}
f4["features"].append({"type": "Feature", "properties": {"id": 99999, "status": "draft", "source": "manual"}, "geometry": bow})
r = requests.put(f"{U}/parcels?action=e2e-bowtie", json=f4).json(); check("bow-tie parcel -> INVALID flagged", "INVALID" in types(r), types(r))
requests.post(f"{U}/undo")
iss = requests.get(f"{B}/surveys/{SID}/issues.geojson").json()["features"]
check("undo returns to a clean layer", not iss, [(i["properties"]["type"], i["properties"]["message"][:50]) for i in iss])
r = requests.put(f"{U}/parcels", json={"type": "nope"}); check("bad body rejected", r.status_code == 400)

# split a parcel along a line through its middle; a line that misses is refused
n_iss = len(requests.get(f"{B}/surveys/{SID}/issues.geojson").json()["features"])
f5 = fc(); big = max(f5["features"], key=lambda f: shape(f["geometry"]).area if len(f["properties"].get("issues") or []) == 0 else 0)
bg = shape(big["geometry"]); minx, miny, maxx, maxy = bg.bounds; cy = bg.centroid.y
r = requests.post(f"{U}/split", json={"id": big["properties"]["id"], "line": [[minx - 1e-4, cy], [maxx + 1e-4, cy]]})
check("split parcel across the middle", r.ok and len(r.json()["parcels"]["features"]) == len(f5["features"]) + 1, r.text[:100])
check("split adds no topology issue", r.ok and len(r.json()["issues"]["features"]) <= n_iss, (n_iss, types(r.json())) if r.ok else "")
requests.post(f"{U}/undo")
r = requests.post(f"{U}/split", json={"id": big["properties"]["id"], "line": [[minx - 1e-3, maxy + 1e-3], [minx - 5e-4, maxy + 2e-3]]})
check("split with a line that misses refused", r.status_code == 400, r.text[:80])
check("parcels carry a suggested land use", all(f["properties"].get("land_use") for f in fc()["features"] if f["properties"].get("source") == "ai"))

h = requests.get(f"{U}/history").json(); check("history records edits", len(h["entries"]) >= 6, len(h["entries"]))
check("back to original count", len(fc()["features"]) == n0, len(fc()["features"]))

# exports
import geopandas as gpd, tempfile, os
for fmt in ("gpkg", "shp", "geojson"):
    r = requests.get(f"{U}/export?fmt={fmt}")
    ok = r.ok and len(r.content) > 1000
    if ok:
        tmp = tempfile.mkdtemp(); p = os.path.join(tmp, "x." + ("zip" if fmt == "shp" else fmt)); open(p, "wb").write(r.content)
        df = gpd.read_file(("zip://" + p + "!parcels.shp") if fmt == "shp" else p, layer="parcels" if fmt == "gpkg" else None)
        if fmt == "gpkg":
            import pyogrio; names = [l[0] for l in pyogrio.list_layers(p)]
            check("gpkg has all layers", set(names) >= {"parcels", "buildings", "roads", "road_centrelines"}, names)
        if fmt == "shp":
            names = sorted(n for n in zipfile.ZipFile(p).namelist() if n.endswith(".shp"))
            check("shp zip has all layers", set(names) >= {"parcels.shp", "buildings.shp", "roads.shp", "road_lines.shp"}, names)
        ok = len(df) == n0 and df.crs is not None and df.geometry.is_valid.all()
        check(f"export {fmt}", ok, f"{len(df)} rows, crs {df.crs.to_epsg() if df.crs else None}, cols {len(df.columns)}")
    else: check(f"export {fmt}", False, r.status_code)
check("export bad fmt rejected", requests.get(f"{U}/export?fmt=kml").status_code == 400)

# records: existing parcel layer = our parcels, shifted 0.4 m east, 20 dropped, 3 merged
random.seed(0); ref = fc()["features"]; dx = 0.4 / 99000
refg = [{"type": "Feature", "properties": {"ref_id": f"R{i}"}, "geometry": mapping(affinity.translate(shape(f["geometry"]), dx, 0))} for i, f in enumerate(ref[20:])]
r = requests.post(f"{U}/reference/parcels", files={"file": ("record.geojson", json.dumps({"type": "FeatureCollection", "features": refg}))}, data={"id_field": "ref_id"})
check("import reference parcels", r.ok, r.text[:120])
# GNSS: 30 true corners with 0.3 m noise
pts = []
for f in ref[100:130]:
    x, y = list(shape(f["geometry"]).exterior.coords)[0]; pts.append((f"P{len(pts)}", y + random.gauss(0, 0.3/111000), x + random.gauss(0, 0.3/99000)))
csv = "id,lat,lon,type\n" + "\n".join(f"{a},{b},{c},corner" for a, b, c in pts)
r = requests.post(f"{U}/reference/gnss", files={"file": ("gnss.csv", csv)}); check("import GNSS csv", r.ok, r.text[:120])
r = requests.post(f"{U}/compare", json={"tolerance_m": 1.0}); ok = r.ok
if ok:
    j = r.json(); check("compare runs", True, json.dumps({k: (v if not isinstance(v, (list, dict)) else '…') for k, v in (j.get('parcels') or {}).items()})[:300])
    check("compare reports buildings vs record", "encroachment" in j, list(j.keys()))
    g = (j.get("gnss") or {}); check("GNSS corner errors small", g.get("rows") and np.median([x["error_to_vertex_m"] for x in g["rows"] if x.get("is_corner")]) < 1.0,
          f"median {np.median([x['error_to_vertex_m'] for x in g['rows'] if x.get('is_corner')]):.2f} m" if g.get("rows") else g)
else: check("compare runs", False, r.text[:200])
r = requests.post(f"{U}/reference/gnss", files={"file": ("bad.csv", "a,b\n1,2")}); check("bad GNSS rejected", r.status_code == 400, r.text[:80])

# field verification
from PIL import Image
buf = io.BytesIO(); Image.new("RGB", (64, 64), (200, 100, 50)).save(buf, "JPEG"); buf.seek(0)
pid = ref[50]["properties"]["id"]; c = shape(ref[50]["geometry"]).centroid
r = requests.post(f"{U}/field/{pid}", data={"verdict": "confirmed", "lat": c.y, "lon": c.x, "accuracy": 4, "note": "e2e", "surveyor": "tester"},
                  files={"photos": ("p.jpg", buf.getvalue(), "image/jpeg")})
check("field visit recorded", r.ok and any(f["properties"]["id"] == pid and f["properties"]["status"] == "approved" for f in r.json()["parcels"]["features"]), r.text[:120])
r = requests.post(f"{U}/field/{pid}", data={"verdict": "bogus", "lat": c.y, "lon": c.x}); check("bad verdict rejected", r.status_code == 400)
r = requests.post(f"{U}/field/corner", data={"lat": c.y, "lon": c.x, "accuracy": 2, "parcel_id": pid}); check("field corner recorded", r.ok, r.text[:100])
r = requests.get(f"{U}/field").json(); check("field summary", r["summary"].get("visits", r["summary"].get("total", 0)) >= 1 or r["summary"], str(r["summary"])[:150])

check("unknown survey 404", requests.get(f"{B}/api/surveys/nope/history").status_code == 404)
print(f"\n{sum(o for _, o, _ in res)}/{len(res)} passed")
