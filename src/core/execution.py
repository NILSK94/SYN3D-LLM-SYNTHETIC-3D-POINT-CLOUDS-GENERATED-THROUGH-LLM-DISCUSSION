import math
import numpy as np
from typing import List, Dict, Any, Tuple
from ..core.scenes import SceneSpec, get_class_id, CLASS_NAMES_BY_ID


def run_generated_code(code: str, profile_fns: Dict[str, Any]) -> List[Dict[str, Any]]:
    safe_builtins = {
        "range": range, "len": len, "min": min, "max": max, "abs": abs,
        "float": float, "int": int, "str": str, "bool": bool,
        "list": list, "dict": dict, "enumerate": enumerate, "zip": zip, "sum": sum,
        "math": math,
    }
    ns: Dict[str, Any] = {"__builtins__": safe_builtins}
    ns.update(profile_fns)
    local_ns: Dict[str, Any] = {}
    try:
        exec(code, ns, local_ns)
    except Exception as exc:
        raise RuntimeError(f"Exec failed: {exc}")

    fn = local_ns.get("build_scene")
    if fn is None:
        raise RuntimeError("build_scene() is not defined in the generated code.")
    pts = fn()
    if not isinstance(pts, list):
        raise RuntimeError("build_scene() must return a list.")
    return pts


def points_to_numpy(points: List[Dict[str, Any]], allowed_ifc: List[str]) -> Tuple[np.ndarray, List[str], List[str]]:
    allowed = set(allowed_ifc)
    arr = []
    elements = []
    labels = []
    for i, p in enumerate(points):
        for k in ("x", "y", "z", "element"):
            if k not in p:
                raise RuntimeError(f"Point {i} missing required key '{k}'.")
        x = float(p["x"])
        y = float(p["y"])
        z = float(p["z"])
        e = str(p["element"]).strip()
        if e not in allowed:
            raise RuntimeError(
                f"Disallowed IFC type '{e}' at point {i}. Allowed: {sorted(allowed)}"
            )
        cid = get_class_id(e)
        arr.append([x, y, z, cid])
        elements.append(e)
        labels.append(str(p.get("label", "")).strip())
    if not arr:
        return np.empty((0, 4), dtype=np.float64), [], []
    return np.array(arr, dtype=np.float64), elements, labels


def compute_stats(arr: np.ndarray) -> Dict[str, Any]:
    if arr.size == 0 or arr.shape[0] == 0:
        return {"num_points": 0, "class_counts": {}, "bbox": None}
    coords = arr[:, :3]
    cids = arr[:, 3].astype(int)
    mn = coords.min(axis=0)
    mx = coords.max(axis=0)
    class_counts = {}
    for cid in np.unique(cids):
        name = CLASS_NAMES_BY_ID.get(int(cid), f"CLASS_{cid}")
        class_counts[name] = int((cids == cid).sum())
    return {
        "num_points": int(arr.shape[0]),
        "class_counts": class_counts,
        "bbox": {
            "xmin": float(mn[0]), "xmax": float(mx[0]),
            "ymin": float(mn[1]), "ymax": float(mx[1]),
            "zmin": float(mn[2]), "zmax": float(mx[2]),
        },
    }


def constraint_check(
    scene: SceneSpec,
    arr: np.ndarray,
    elements: List[str],
    stats: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    issues = []
    if stats.get("num_points", 0) <= 0:
        issues.append("Point cloud is empty (no points).")
    present = set(elements)
    missing = [r for r in scene.required_ifc if r not in present]
    if missing:
        issues.append(f"Missing required IFC types: {missing}")
    bbox = stats.get("bbox")
    if bbox:
        dx = float(bbox["xmax"] - bbox["xmin"])
        dy = float(bbox["ymax"] - bbox["ymin"])
        dz = float(bbox["zmax"] - bbox["zmin"])
        if dx < 0.2 or dy < 0.2 or dz < 0.01:
            issues.append("Bounding box is degenerate (extents too small).")
        if dx > 500 or dy > 500 or dz > 200:
            issues.append("Bounding box is unrealistically large.")
    return len(issues) == 0, issues


def detect_string_indexing_bug(code: str) -> bool:
    """Detects common mistake: indexing helper output as a dict when it returns [x,y,z] lists."""
    bad_patterns = [
        '["x"]', "['x']",
        '["y"]', "['y']",
        '["z"]', "['z']",
        '["element"]', "['element']",
        '["label"]', "['label']",
    ]
    return any(p in code for p in bad_patterns)
