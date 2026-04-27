import numpy as np
import open3d as o3d
import queue
import threading
import time
from typing import Optional, Dict
from ..core.scenes import CLASS_COLORS

# ============================================================
# Open3D Helpers
# ============================================================

def build_open3d_pc(arr: np.ndarray) -> o3d.geometry.PointCloud:
    coords = arr[:, :3]
    class_ids = arr[:, 3].astype(int)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(coords)
    colors = np.zeros_like(coords)
    for i, cid in enumerate(class_ids):
        colors[i, :] = CLASS_COLORS.get(int(cid), (0.5, 0.5, 0.5))
    pc.colors = o3d.utility.Vector3dVector(colors)
    return pc

def _interpolate_arrays(a: Optional[np.ndarray], b: np.ndarray, t: float) -> np.ndarray:
    if a is None:
        return b
    if a.size == 0 or b.size == 0:
        return b
    na = a.shape[0]
    nb = b.shape[0]
    n = min(na, nb)
    out = np.array(b, copy=True)
    out[:n, :3] = (1.0 - t) * a[:n, :3] + t * b[:n, :3]
    out[:n, 3] = b[:n, 3]
    return out


# ============================================================
# Live Preview Worker
# ============================================================

live_preview_queue = queue.Queue()
live_preview_stop = threading.Event()
live_preview_thread = None

def live_preview_push_open3d(arr: np.ndarray, title: str, subtitle: str, transition_s: float):
    try:
        live_preview_queue.put_nowait({
            "arr": np.array(arr, copy=True),
            "title": str(title),
            "subtitle": str(subtitle),
            "transition_s": float(transition_s),
            "ts": int(time.time()),
        })
    except Exception:
        pass

def live_preview_worker_open3d():
    vis = None
    geom = None
    last_arr: Optional[np.ndarray] = None
    has_geometry = False

    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="SYN3D Live Preview (Open3D)", width=1280, height=720, visible=True)

        try:
            ro = vis.get_render_option()
            ro.background_color = np.array([0.03, 0.03, 0.04])
            ro.point_size = 2.0
        except Exception:
            pass

        while not live_preview_stop.is_set():
            vis.poll_events()
            vis.update_renderer()

            try:
                item = live_preview_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            arr_new = item["arr"]
            transition_s = max(0.0, float(item.get("transition_s", 1.2)))

            if not has_geometry:
                geom = build_open3d_pc(arr_new)
                vis.add_geometry(geom)
                try:
                    vis.reset_view_point(True)
                except Exception:
                    pass
                has_geometry = True
                last_arr = np.array(arr_new, copy=True)
                continue

            frames = 1
            if transition_s > 0.01:
                frames = int(max(10, min(60, transition_s * 25.0)))

            start_t = time.time()
            for fi in range(frames):
                if live_preview_stop.is_set():
                    break
                t = 1.0 if frames <= 1 else (fi / float(frames - 1))
                arr_i = _interpolate_arrays(last_arr, arr_new, t)
                pc_i = build_open3d_pc(arr_i)
                geom.points = pc_i.points
                geom.colors = pc_i.colors
                vis.update_geometry(geom)
                vis.poll_events()
                vis.update_renderer()

                if transition_s > 0.01:
                    elapsed = time.time() - start_t
                    target = (fi + 1) * (transition_s / float(frames))
                    sleep_s = target - elapsed
                    if sleep_s > 0:
                        time.sleep(min(sleep_s, 0.02))

            last_arr = np.array(arr_new, copy=True)

        vis.destroy_window()
    except Exception:
        try:
            if vis is not None:
                vis.destroy_window()
        except Exception:
            pass

def start_open3d_live_preview():
    global live_preview_thread
    if live_preview_thread is not None and live_preview_thread.is_alive():
        return
    live_preview_stop.clear()
    while True:
        try:
            live_preview_queue.get_nowait()
        except queue.Empty:
            break
    live_preview_thread = threading.Thread(target=live_preview_worker_open3d, daemon=True)
    live_preview_thread.start()

def stop_open3d_live_preview():
    live_preview_stop.set()
