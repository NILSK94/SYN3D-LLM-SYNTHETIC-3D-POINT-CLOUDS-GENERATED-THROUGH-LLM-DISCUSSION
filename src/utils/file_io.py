import os
import csv
import json
import time
import threading
from typing import Dict, Any, List, Optional

import numpy as np
import open3d as o3d

from ..config import ROOT_OUT, DATA_DIR, LOG_DIR
from .visualization import build_open3d_pc

# ============================================================
# Thread safety
# ============================================================

_file_lock = threading.Lock()
_csv_lock = threading.Lock()

# ============================================================
# CSV log paths (per paper: discussions → .CSV and .JSON)
# ============================================================

RUNS_META_CSV = os.path.join(ROOT_OUT, "runs_metadata.csv")
DISCUSSION_CSV = os.path.join(ROOT_OUT, "discussion_log.csv")


def ensure_csv_headers() -> None:
    with _csv_lock:
        if not os.path.exists(RUNS_META_CSV):
            with open(RUNS_META_CSV, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "timestamp", "run_id", "geom_profile", "scene_id",
                    "prompt_variant", "seed", "n_dialog_iterations",
                    "constraint_pass", "num_points",
                    "total_prompt_tokens", "total_completion_tokens", "total_tokens",
                    "out_dir",
                ])
        if not os.path.exists(DISCUSSION_CSV):
            with open(DISCUSSION_CSV, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "timestamp", "run_id", "scene_id", "iteration",
                    "role", "prompt_tokens", "completion_tokens", "total_tokens",
                    "message_preview",
                ])


def append_run_meta(row: list) -> None:
    with _csv_lock:
        with open(RUNS_META_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
            f.flush()


def append_discussion_row(row: list) -> None:
    with _csv_lock:
        with open(DISCUSSION_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
            f.flush()


# ============================================================
# Per-run JSON discussion log
# ============================================================

class DiscussionLogger:
    """
    Logs the full Designer/Critic discussion for one run to a JSON file and
    appends per-turn rows to the shared discussion CSV.
    Matches paper: 'discussions are automatically logged to .CSV and .JSON files'.
    """

    def __init__(self, run_id: str, scene_id: str, log_dir: str = LOG_DIR):
        self.run_id = run_id
        self.scene_id = scene_id
        run_log_dir = os.path.join(log_dir, run_id)
        os.makedirs(run_log_dir, exist_ok=True)
        self.json_path = os.path.join(run_log_dir, f"discussion_{run_id}.json")
        self._entries: List[Dict[str, Any]] = []

    def log_turn(
        self,
        iteration: int,
        role: str,
        text: str,
        usage: Optional[Dict[str, int]] = None,
    ) -> None:
        usage = usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        entry = {
            "timestamp": time.time(),
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run_id": self.run_id,
            "scene_id": self.scene_id,
            "iteration": iteration,
            "role": role,
            "usage": usage,
            "text": text,
        }
        with _file_lock:
            self._entries.append(entry)

        preview = text[:200].replace("\n", " ")
        append_discussion_row([
            entry["iso_time"], self.run_id, self.scene_id, iteration,
            role,
            usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"],
            preview,
        ])

    def save_json(self) -> str:
        with _file_lock:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, ensure_ascii=False, indent=2)
        return self.json_path


# ============================================================
# Point cloud export
# Per paper: saved in .e57, .xyz, and .ply
# ============================================================

def export_pointcloud(arr: np.ndarray, elements: List[str], labels: List[str], out_dir: str, basename: str) -> Dict[str, str]:
    """
    Export point cloud in the three formats specified by the paper:
      - .xyz  (plain text XYZ coordinates)
      - .ply  (binary/ASCII via Open3D, includes per-class colour)
      - .e57  (industry standard; requires pye57)

    Returns a dict with format keys and their output paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    results: Dict[str, str] = {}

    # --- XYZ ---
    xyz_path = os.path.join(out_dir, basename + ".xyz")
    np.savetxt(xyz_path, arr[:, :3], fmt="%.6f")
    results["xyz"] = xyz_path

    # --- PLY ---
    ply_path = os.path.join(out_dir, basename + ".ply")
    pc = build_open3d_pc(arr)
    o3d.io.write_point_cloud(ply_path, pc, write_ascii=False)
    results["ply"] = ply_path

    # --- CSV (Full Data with Labels) ---
    csv_path = os.path.join(out_dir, basename + "_labeled.csv")
    colors = np.asarray(pc.colors)
    rgb8 = (colors * 255.0).astype(np.uint8)
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "z", "r", "g", "b", "class_id", "element", "label"])
            for i in range(arr.shape[0]):
                writer.writerow([
                    f"{arr[i, 0]:.6f}", f"{arr[i, 1]:.6f}", f"{arr[i, 2]:.6f}",
                    rgb8[i, 0], rgb8[i, 1], rgb8[i, 2],
                    int(arr[i, 3]), elements[i], labels[i]
                ])
        results["csv"] = csv_path
    except Exception as exc:
        results["csv"] = f"ERROR: {exc}"

    # --- E57 ---
    e57_path = os.path.join(out_dir, basename + ".e57")
    try:
        import pye57
        coords = arr[:, :3].astype(np.float64)
        colors = np.asarray(pc.colors)
        rgb8 = (colors * 255.0).astype(np.uint8)
        e57 = pye57.E57(e57_path, mode="w")
        e57.write_scan_raw({
            "cartesianX": coords[:, 0],
            "cartesianY": coords[:, 1],
            "cartesianZ": coords[:, 2],
            "colorRed":   rgb8[:, 0],
            "colorGreen": rgb8[:, 1],
            "colorBlue":  rgb8[:, 2],
        })
        e57.close()
        results["e57"] = e57_path
    except ImportError:
        results["e57"] = ""
    except Exception as exc:
        results["e57"] = f"ERROR: {exc}"

    return results


def export_iteration_pointcloud(arr: np.ndarray, elements: List[str], labels: List[str], step_dir: str, step_name: str) -> Dict[str, str]:
    """Export an intermediate iteration's point cloud (same formats)."""
    return export_pointcloud(arr, elements, labels, step_dir, step_name)
