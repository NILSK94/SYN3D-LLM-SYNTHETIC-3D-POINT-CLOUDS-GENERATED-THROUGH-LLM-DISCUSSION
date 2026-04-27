import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from ..config import ROOT_OUT


@dataclass
class SceneSpec:
    scene_id: str
    prompt: str
    blueprint: List[str]
    allowed_ifc: List[str]
    required_ifc: List[str]


# ============================================================
# IFC / Ontology registry
# ============================================================

ELEMENT_CLASSES: Dict[str, int] = {}
CLASS_NAMES_BY_ID: Dict[int, str] = {}
CLASS_COLORS: Dict[int, Tuple[float, float, float]] = {}


def get_class_id(name: str) -> int:
    if name not in ELEMENT_CLASSES:
        cid = len(ELEMENT_CLASSES)
        ELEMENT_CLASSES[name] = cid
        CLASS_NAMES_BY_ID[cid] = name
        r = (37 * cid % 255) / 255.0
        g = (89 * cid % 255) / 255.0
        b = (173 * cid % 255) / 255.0
        CLASS_COLORS[cid] = (r, g, b)
    return ELEMENT_CLASSES[name]


# ============================================================
# Scene templates (matching paper: high-rise buildings, industrial halls,
# streets with traffic signs, railway tracks, bridges, apartments)
# ============================================================

SCENES: List[SceneSpec] = [
    # --- Apartments (different shapes) ---
    SceneSpec(
        scene_id="Apartment_Rectangular",
        prompt="Apartment floor plan (rectangular shape): single storey with rooms and a corridor, including doors and windows.",
        blueprint=[
            "Rectangular footprint approx. 12 m × 9 m.",
            "Floor slab at z = 0.00 m; ceiling slab at z = 3.00 m.",
            "A corridor (~1.4 m wide) connects multiple enclosed rooms.",
            "Doors between corridor and each room; windows on all exterior walls.",
            "Use metric coordinates; keep all elements axis-aligned.",
        ],
        allowed_ifc=["IfcWall", "IfcSlab", "IfcColumn", "IfcDoor", "IfcWindow"],
        required_ifc=["IfcWall", "IfcSlab", "IfcDoor", "IfcWindow"],
    ),
    SceneSpec(
        scene_id="Apartment_L_Shape",
        prompt="Apartment floor plan (L-shape): single storey with rooms and a corridor, including doors and windows.",
        blueprint=[
            "L-shaped footprint: main wing 10 m × 6 m plus a side wing 6 m × 6 m.",
            "Floor slab z = 0.00 m; ceiling slab z = 3.00 m; slab must NOT fill the missing L-corner void.",
            "Corridor turns around the L-corner and connects rooms in both wings.",
            "Doors between corridor and rooms; windows on all exterior walls.",
        ],
        allowed_ifc=["IfcWall", "IfcSlab", "IfcColumn", "IfcDoor", "IfcWindow"],
        required_ifc=["IfcWall", "IfcSlab", "IfcDoor", "IfcWindow"],
    ),
    SceneSpec(
        scene_id="Apartment_Courtyard",
        prompt="Apartment floor plan (courtyard shape): single storey ring around a central courtyard, including doors and windows.",
        blueprint=[
            "Footprint approx. 18 m × 18 m with a central courtyard void approx. 6 m × 6 m.",
            "Floor slab z = 0.00 m; ceiling slab z = 3.00 m; slab must NOT cover the courtyard void.",
            "Ring corridor surrounding the courtyard; rooms arranged around the corridor.",
            "Windows on all exterior walls; optional railings around the inner courtyard edge.",
        ],
        allowed_ifc=["IfcWall", "IfcSlab", "IfcColumn", "IfcDoor", "IfcWindow", "IfcRailing"],
        required_ifc=["IfcWall", "IfcSlab", "IfcDoor", "IfcWindow"],
    ),
    # --- High-rise building ---
    SceneSpec(
        scene_id="Highrise",
        prompt="High-rise building: multiple storeys with a structural core, repeating floor plates, columns, and windows.",
        blueprint=[
            "Rectangular footprint approx. 18 m × 18 m.",
            "At least 6 storeys: floor slabs at z = 0, 3, 6, 9, 12, 15 m; roof slab at z = 18 m.",
            "Central core with structural walls; columns on the perimeter grid.",
            "Beams spanning between columns; windows on all four facades.",
            "A few doors on the ground-floor corridor level.",
        ],
        allowed_ifc=["IfcWall", "IfcSlab", "IfcColumn", "IfcDoor", "IfcWindow", "IfcBeam"],
        required_ifc=["IfcWall", "IfcSlab", "IfcColumn", "IfcWindow"],
    ),
    # --- Industrial hall ---
    SceneSpec(
        scene_id="IndustrialHall",
        prompt="Industrial hall: large single-span hall volume with a structural frame and loading gates.",
        blueprint=[
            "Rectangular footprint approx. 30 m × 18 m; clear height approx. 8 m.",
            "Floor slab at z = 0.00 m; roof slab at z = 8.00 m.",
            "Perimeter walls; columns on a regular grid; primary beams and secondary members forming the roof frame.",
            "Large gate doors on one long side for vehicle access.",
        ],
        allowed_ifc=["IfcWall", "IfcSlab", "IfcColumn", "IfcBeam", "IfcMember", "IfcDoor"],
        required_ifc=["IfcWall", "IfcSlab", "IfcColumn", "IfcBeam"],
    ),
    # --- Bridge ---
    SceneSpec(
        scene_id="Bridge",
        prompt="Bridge structure: deck, supporting columns, longitudinal beams, and edge railings.",
        blueprint=[
            "Bridge length approx. 45 m; deck width approx. 12 m.",
            "Ground reference plane at z = 0.00 m; bridge deck slab at z = 6.00 m.",
            "Columns at regular intervals supporting the deck; longitudinal beams under the deck.",
            "Continuous railings along both deck edges.",
        ],
        allowed_ifc=["IfcSlab", "IfcColumn", "IfcBeam", "IfcMember", "IfcRailing", "IfcWall"],
        required_ifc=["IfcSlab", "IfcColumn", "IfcBeam"],
    ),
    # --- Railway tracks ---
    SceneSpec(
        scene_id="RailwayTrack",
        prompt="Railway track segment: two parallel rails with wooden sleepers on a ballast bed, straight alignment.",
        blueprint=[
            "Track length approx. 50 m; standard gauge 1.435 m.",
            "Two rails (IfcRail) running parallel along the X axis; keep near ground (z ≈ 0.05–0.15 m).",
            "Sleepers (IfcBuildingElementProxy) spaced approx. 0.6 m apart, perpendicular to rail direction.",
            "Ballast bed (IfcBuildingElementProxy) as a trapezoidal slab beneath the sleepers.",
        ],
        allowed_ifc=["IfcRail", "IfcBuildingElementProxy", "IfcSlab"],
        required_ifc=["IfcRail", "IfcBuildingElementProxy"],
    ),
    # --- Streets with traffic signs ---
    SceneSpec(
        scene_id="Street_With_TrafficSigns",
        prompt="Urban street segment: asphalt road with a sidewalk, vertical traffic signs, and a traffic light.",
        blueprint=[
            "Road length approx. 60 m; total width approx. 10 m (road + two sidewalk strips).",
            "Road surface as IfcRoad; sidewalks as IfcSlab strips on both sides.",
            "Multiple vertical traffic sign posts with sign boards along the roadside.",
            "One traffic light with pole and signal head at one end of the segment.",
        ],
        allowed_ifc=["IfcRoad", "IfcSlab", "IfcBuildingElementProxy", "IfcSign", "IfcSignal", "IfcColumn"],
        required_ifc=["IfcRoad", "IfcBuildingElementProxy"],
    ),
    SceneSpec(
        scene_id="Street_Intersection",
        prompt="Urban road intersection: two roads crossing at right angles, with traffic signs and a traffic light.",
        blueprint=[
            "Two roads crossing at a right angle; intersection area approx. 20 m × 20 m.",
            "Road surfaces as IfcRoad; optional curb slabs at the four corner islands.",
            "Multiple traffic signs on posts at the corners; at least one traffic light pole with signal head.",
        ],
        allowed_ifc=["IfcRoad", "IfcSlab", "IfcBuildingElementProxy", "IfcSign", "IfcSignal", "IfcColumn"],
        required_ifc=["IfcRoad", "IfcBuildingElementProxy"],
    ),
]

# ============================================================
# Scene categories (for GUI grouping, matching paper descriptions)
# ============================================================

SCENE_CATEGORIES: Dict[str, List[str]] = {
    "Apartments":       ["Apartment_Rectangular", "Apartment_L_Shape", "Apartment_Courtyard"],
    "High-rise":        ["Highrise"],
    "Industrial Hall":  ["IndustrialHall"],
    "Bridge":           ["Bridge"],
    "Railway":          ["RailwayTrack"],
    "Street":           ["Street_With_TrafficSigns", "Street_Intersection"],
}

# ============================================================
# Prompt variants
# ============================================================

PROMPT_VARIANTS = [
    ("P0_Base",
     ""),
    ("P1_Specific",
     "Be explicit: use clear axis-aligned geometry. Avoid random jitter. Place openings precisely on walls."),
    ("P2_Robustness",
     "Robustness variant: include mild occlusion gaps and slight density variation across different elements."),
]

# ============================================================
# IFC ontology initialisation
# ============================================================

_ALL_IFC = sorted(set(t for s in SCENES for t in s.allowed_ifc))
for _t in _ALL_IFC:
    get_class_id(_t)

# ============================================================
# Custom scene persistence
# ============================================================

SCENES_CUSTOM_PATH = os.path.join(ROOT_OUT, "scenes_custom.json")


def scene_to_dict(s: SceneSpec) -> Dict[str, Any]:
    return {
        "scene_id": s.scene_id,
        "prompt": s.prompt,
        "blueprint": list(s.blueprint),
        "allowed_ifc": list(s.allowed_ifc),
        "required_ifc": list(s.required_ifc),
    }


def dict_to_scene(d: Dict[str, Any]) -> SceneSpec:
    return SceneSpec(
        scene_id=str(d.get("scene_id", "")).strip(),
        prompt=str(d.get("prompt", "")).strip(),
        blueprint=[str(x) for x in (d.get("blueprint") or [])],
        allowed_ifc=[str(x) for x in (d.get("allowed_ifc") or [])],
        required_ifc=[str(x) for x in (d.get("required_ifc") or [])],
    )


def save_custom_scenes(scenes: List[SceneSpec], path: str = SCENES_CUSTOM_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([scene_to_dict(s) for s in scenes], f, ensure_ascii=False, indent=2)


def load_custom_scenes(path: str = SCENES_CUSTOM_PATH) -> Optional[List[SceneSpec]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            arr = json.load(f)
        if not isinstance(arr, list):
            return None
        return [dict_to_scene(d) for d in arr if isinstance(d, dict) and d.get("scene_id")]
    except Exception:
        return None


def merge_scenes(default_scenes: List[SceneSpec], custom_scenes: Optional[List[SceneSpec]]) -> List[SceneSpec]:
    if not custom_scenes:
        return list(default_scenes)
    by_id = {s.scene_id: s for s in default_scenes}
    for cs in custom_scenes:
        by_id[cs.scene_id] = cs
    return sorted(by_id.values(), key=lambda s: s.scene_id)
