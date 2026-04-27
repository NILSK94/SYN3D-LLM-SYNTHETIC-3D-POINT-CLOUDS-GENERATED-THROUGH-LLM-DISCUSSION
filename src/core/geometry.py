import numpy as np
import math
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class GeometryProfile:
    profile_id: str
    description: str
    provide_box_surface: bool
    provide_minimal: bool
    provide_round: bool
    provide_specialized: bool

# ============================================================
# Geometry functions
# ============================================================

def box_surface(x_min, x_max, y_min, y_max, z_min, z_max, density=0.2):
    pts = []
    xs = np.arange(x_min, x_max + density, density)
    ys = np.arange(y_min, y_max + density, density)
    zs = np.arange(z_min, z_max + density, density)
    # Floor/Ceiling
    for x in xs:
        for y in ys:
            pts.append([x, y, z_min])
            pts.append([x, y, z_max])
    # Walls X
    for x in xs:
        for z in zs:
            pts.append([x, y_min, z])
            pts.append([x, y_max, z])
    # Walls Y
    for y in ys:
        for z in zs:
            pts.append([x_min, y, z])
            pts.append([x_max, y, z])
    return np.unique(np.array(pts), axis=0).tolist()

def plane_surface(x_min, x_max, y_min, y_max, z, density=0.2):
    pts = []
    xs = np.arange(x_min, x_max + density, density)
    ys = np.arange(y_min, y_max + density, density)
    for x in xs:
        for y in ys:
            pts.append([x, y, z])
    return np.unique(np.array(pts), axis=0).tolist()

def cylinder_surface(cx, cy, z_min, z_max, radius=0.15, density=0.05, n_theta=36):
    pts = []
    zs = np.arange(z_min, z_max + density, density)
    for z in zs:
        for i in range(n_theta):
            theta = 2.0 * math.pi * (i / float(n_theta))
            x = cx + radius * math.cos(theta)
            y = cy + radius * math.sin(theta)
            pts.append([x, y, z])
    return np.unique(np.array(pts), axis=0).tolist()

def rail_pair(x0, x1, y_center, gauge=1.435, z=0.05, density=0.05):
    pts = []
    ys = [y_center - gauge / 2.0, y_center + gauge / 2.0]
    xs = np.arange(x0, x1 + density, density)
    for y in ys:
        for x in xs:
            pts.append([x, y, z])
            pts.append([x, y, z + 0.03])
    return np.unique(np.array(pts), axis=0).tolist()

def sign_post(x, y, z0=0.0, z1=2.2, radius=0.05, density=0.05):
    pole = cylinder_surface(x, y, z0, z1, radius=radius, density=density, n_theta=24)
    board = box_surface(x - 0.25, x + 0.25, y - 0.02, y + 0.02, z1 - 0.6, z1 - 0.2, density=density)
    return pole + board

def traffic_light(x, y, z0=0.0, z1=3.0, density=0.05):
    pole = cylinder_surface(x, y, z0, z1, radius=0.06, density=density, n_theta=24)
    head = box_surface(x - 0.08, x + 0.08, y - 0.05, y + 0.05, z1 - 0.5, z1 - 0.2, density=density)
    return pole + head

def build_profile_functions(profile: GeometryProfile) -> Dict[str, Any]:
    fns: Dict[str, Any] = {}
    if profile.provide_box_surface:
        fns["box_surface"] = box_surface
    if profile.provide_minimal:
        fns["plane_surface"] = plane_surface
    if profile.provide_round:
        fns["cylinder_surface"] = cylinder_surface
    if profile.provide_specialized:
        fns["rail_pair"] = rail_pair
        fns["sign_post"] = sign_post
        fns["traffic_light"] = traffic_light
    return fns

# Profile definitions
GEOM_PROFILES = [
    GeometryProfile("G0_NoHelper", "No helper function is available. Sample points manually.", False, False, False, False),
    GeometryProfile("G1_Minimal", "Minimal plane_surface helper.", False, True, False, False),
    GeometryProfile("G2_BoxSurface", "Original box_surface helper.", True, False, False, False),
    GeometryProfile("G3_BoxPlusRound", "box_surface + cylinder_surface.", True, False, True, False),
    GeometryProfile("G4_Specialized", "Specialized helpers: rails, sign_post, traffic_light.", True, False, True, True),
]

def get_profile(pid: str) -> GeometryProfile:
    for p in GEOM_PROFILES:
        if p.profile_id == pid:
            return p
    raise KeyError(pid)
