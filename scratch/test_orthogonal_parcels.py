import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon, box, mapping, shape
from shapely.ops import unary_union
import json

# 1. Load building contours
gt_path = 'data/datasets/inria_raw/data/train/gt/austin1.tif'
gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
crop_gt = gt[500:2500, 500:2500]

contours, _ = cv2.findContours(crop_gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
bldgs = []
for c in contours:
    a = cv2.contourArea(c)
    if a >= 70:
        eps = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) >= 4:
            p = Polygon(approx)
            if p.is_valid and p.area > 0:
                bldgs.append({"poly": p, "contour": c, "area": a})

print(f"Total valid buildings: {len(bldgs)}")

# 2. Highway corridor mask
# In austin_sample.jpg, the highway runs diagonally from (0, 480) to (2000, 1150)
# Let's create a clean highway corridor polygon
hw_pts = np.array([
    [0, 430], [500, 620], [1000, 800], [1500, 950], [2000, 1080],
    [2000, 1220], [1500, 1090], [1000, 940], [500, 770], [0, 590]
], dtype=np.int32)
highway_poly = Polygon(hw_pts)

# 3. Large Commercial Plot
# Find the largest building (the commercial mall/center)
commercial_bldg = max(bldgs, key=lambda b: b["area"])
# Commercial plot covers the building + parking lot
c_poly = commercial_bldg["poly"].buffer(18.0, join_style=2)
if c_poly.geom_type == 'MultiPolygon':
    c_poly = max(c_poly.geoms, key=lambda g: g.area)

print(f"Commercial building area: {commercial_bldg['area']:.0f} px, plot area: {c_poly.area:.0f} px")

# 4. Generate Clean, Rectangular / Orthogonal Cadastral Parcels for Residential Houses
residential_bldgs = [b for b in bldgs if b != commercial_bldg and not highway_poly.contains(b["poly"].centroid)]
print(f"Residential buildings to map: {len(residential_bldgs)}")

raw_parcels = []
# For each residential house, create an oriented rectangular cadastral plot
for b in residential_bldgs:
    rect = cv2.minAreaRect(b["contour"])
    (cx, cy), (w, h), angle = rect
    
    # Standard residential plot dimensions:
    # Width = house width + side setbacks (3-5m each side = 10-16px each side)
    # Depth = house depth + front yard (8-12m = 26-40px) + backyard (10-15m = 33-50px)
    plot_w = max(w + 24.0, 55.0)  # min 16.5m frontage
    plot_h = max(h + 60.0, 90.0)  # min 27m depth
    
    box_pts = cv2.boxPoints(((cx, cy), (plot_w, plot_h), angle))
    p_plot = Polygon(box_pts)
    if not p_plot.is_valid:
        p_plot = p_plot.buffer(0)
    
    # Clip by highway corridor
    if p_plot.intersects(highway_poly):
        p_plot = p_plot.difference(highway_poly)
        if p_plot.geom_type == 'MultiPolygon':
            p_plot = max(p_plot.geoms, key=lambda g: g.area) if p_plot.geoms else Polygon()
            
    # Clip by commercial plot
    if p_plot.intersects(c_poly):
        p_plot = p_plot.difference(c_poly)
        if p_plot.geom_type == 'MultiPolygon':
            p_plot = max(p_plot.geoms, key=lambda g: g.area) if p_plot.geoms else Polygon()

    if not p_plot.is_empty and p_plot.area >= 200:
        raw_parcels.append({"poly": p_plot, "bldg": b})

print(f"Generated {len(raw_parcels)} clean oriented rectangular cadastral plots!")

# Check visual sample: vertices count of residential plot vs Voronoi shard
v_counts = [len(p["poly"].exterior.coords) for p in raw_parcels[:10]]
print(f"Sample vertex counts per parcel (clean rectangles): {v_counts}")
