import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import queue
import threading
import os
import time
import json
from typing import Optional, List, Dict, Any

import open3d as o3d

from ..config import DEFAULT_MODEL, DEFAULT_API_KEY, set_openai_client, get_client, LOG_DIR, DATA_DIR
from ..core.scenes import (
    SCENES, PROMPT_VARIANTS, SceneSpec, ELEMENT_CLASSES, SCENE_CATEGORIES, get_class_id,
    load_custom_scenes, save_custom_scenes, merge_scenes, SCENES_CUSTOM_PATH,
    scene_to_dict
)
from ..core.geometry import GEOM_PROFILES, get_profile, build_profile_functions
from ..core.generator import (
    quick_api_ping,
    run_full_pipeline,
    build_designer_system_prompt,
    make_scene_text,
)
from ..core.execution import points_to_numpy, compute_stats, constraint_check
from ..utils.file_io import (
    ensure_csv_headers, append_run_meta, append_discussion_row,
    DiscussionLogger, export_pointcloud, export_iteration_pointcloud
)
from ..utils.visualization import (
    start_open3d_live_preview, stop_open3d_live_preview,
    live_preview_push_open3d
)
from .theme import apply_dark_mode

class Syn3dApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"SYN3D-LLM | {DEFAULT_MODEL}")
        self.geometry("1400x900")

        self.config_text_widget, self.colors = apply_dark_mode(self)
        
        # Data
        self.scenes = list(SCENES)
        self.custom_scenes = load_custom_scenes(SCENES_CUSTOM_PATH)
        if self.custom_scenes:
            self.scenes = merge_scenes(self.scenes, self.custom_scenes)
            
        # Variables
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.density_var = tk.StringVar(value="0.05")
        self.seeds_var = tk.StringVar(value="2")
        self.dialog_var = tk.StringVar(value="2")  # n_dialog_iterations (paper default: 2)
        self.codegen_tries_var = tk.StringVar(value="3")  # retries per round (paper default: 3)
        self.open3d_live_var = tk.IntVar(value=1)

        self.scene_var = tk.StringVar(value=self.scenes[0].scene_id if self.scenes else "")
        self.profile_var = tk.StringVar(value=GEOM_PROFILES[0].profile_id)
        self.pv_var = tk.StringVar(value=PROMPT_VARIANTS[0][0])

        self.all_scenes_var = tk.IntVar(value=1)
        self.all_profiles_var = tk.IntVar(value=1)
        self.all_pvs_var = tk.IntVar(value=1)

        self.api_key_var = tk.StringVar(value=DEFAULT_API_KEY)

        # Worker state
        self.stop_flag = False
        self.ui_queue: queue.Queue = queue.Queue()
        self.discussion_queue: queue.Queue = queue.Queue()

        self.setup_ui()
        self.after(50, self.process_ui_queue)

    def setup_ui(self):
        outer = ttk.Frame(self)
        outer.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        header = ttk.Frame(outer)
        header.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header, text="SYN3D-LLM", style="Title.TLabel").pack(side=tk.LEFT)

        notebook = ttk.Notebook(outer)
        notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_run = ttk.Frame(notebook)
        self.tab_scene = ttk.Frame(notebook)
        self.tab_custom = ttk.Frame(notebook)
        
        notebook.add(self.tab_run, text="Run")
        notebook.add(self.tab_scene, text="Scene Editor")
        notebook.add(self.tab_custom, text="Custom Prompt")
        
        self.tab_results = ttk.Frame(notebook)
        notebook.add(self.tab_results, text="Results Browser")

        self.setup_run_tab()
        self.setup_scene_tab()
        self.setup_custom_tab()
        self.setup_results_tab()

    def setup_run_tab(self):
        run_wrap = ttk.Frame(self.tab_run)
        run_wrap.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        left = ttk.Frame(run_wrap)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        right = ttk.Frame(run_wrap)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # API
        api_card = ttk.LabelFrame(left, text="API")
        api_card.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Entry(api_card, textvariable=self.api_key_var, show="•").pack(fill=tk.X, padx=6, pady=4)
        btn_frame = ttk.Frame(api_card)
        btn_frame.pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(btn_frame, text="Apply", command=self.on_apply_key).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Ping", command=self.on_ping).pack(side=tk.LEFT, padx=2)

        # Settings
        settings = ttk.LabelFrame(left, text="Settings")
        settings.pack(fill=tk.X, pady=(0, 10))
        
        def add_entry(lbl, var):
            f = ttk.Frame(settings)
            f.pack(fill=tk.X, padx=6, pady=2)
            ttk.Label(f, text=lbl, width=15).pack(side=tk.LEFT)
            ttk.Entry(f, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True)

        add_entry("Model", self.model_var)
        add_entry("Density (m)", self.density_var)
        add_entry("Seeds", self.seeds_var)
        add_entry("Iterations (N)", self.dialog_var)
        add_entry("Codegen Tries", self.codegen_tries_var)
        ttk.Checkbutton(settings, text="Open3D Live Preview", variable=self.open3d_live_var).pack(anchor="w", padx=6, pady=2)

        # Selection
        sel = ttk.LabelFrame(left, text="Selection")
        sel.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Combobox(sel, textvariable=self.scene_var, values=[s.scene_id for s in self.scenes], state="readonly").pack(fill=tk.X, padx=6, pady=4)
        ttk.Combobox(sel, textvariable=self.profile_var, values=[p.profile_id for p in GEOM_PROFILES], state="readonly").pack(fill=tk.X, padx=6, pady=4)
        # PV
        ttk.Combobox(sel, textvariable=self.pv_var, values=[pv[0] for pv in PROMPT_VARIANTS], state="readonly").pack(fill=tk.X, padx=6, pady=4)
        
        ttk.Checkbutton(sel, text="All Scenes", variable=self.all_scenes_var).pack(anchor="w", padx=6)
        ttk.Checkbutton(sel, text="All Profiles", variable=self.all_profiles_var).pack(anchor="w", padx=6)
        ttk.Checkbutton(sel, text="All Prompts", variable=self.all_pvs_var).pack(anchor="w", padx=6)

        # Actions
        actions = ttk.LabelFrame(left, text="Actions")
        actions.pack(fill=tk.X)
        self.btn_run_all = ttk.Button(actions, text="Run ALL", command=self.on_run_all)
        self.btn_run_all.pack(fill=tk.X, padx=6, pady=4)
        
        self.btn_run_one = ttk.Button(actions, text="Run ONE", command=self.on_run_one)
        self.btn_run_one.pack(fill=tk.X, padx=6, pady=4)

        self.btn_stop = ttk.Button(actions, text="STOP", command=self.on_stop, state="disabled")
        self.btn_stop.pack(fill=tk.X, padx=6, pady=4)

        # --- Right: run log + discussion log ---
        ttk.Label(right, text="Run Log").pack(anchor="w")
        self.log_box = tk.Text(right, width=80, height=10)
        self.log_box.pack(fill=tk.BOTH, expand=True)
        self.config_text_widget(self.log_box)

        ttk.Label(right, text="Designer \u2194 Critic Discussion").pack(anchor="w", pady=(6, 0))
        self.discussion_box = tk.Text(right, width=80, height=14)
        self.discussion_box.pack(fill=tk.BOTH, expand=True)
        self.config_text_widget(self.discussion_box)
        self.discussion_box.tag_config("designer", foreground="#7ec8e3")
        self.discussion_box.tag_config("critic",   foreground="#f4a261")
        self.discussion_box.tag_config("header",   foreground="#888888")

        self.after(100, self._process_discussion_queue)

    def setup_scene_tab(self):
        wrap = ttk.Frame(self.tab_scene)
        wrap.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # Load Existing
        top = ttk.LabelFrame(wrap, text="Load Existing Scene")
        top.pack(fill=tk.X, pady=(0, 10))
        
        self.scene_load_var = tk.StringVar()
        cb = ttk.Combobox(top, textvariable=self.scene_load_var, values=[s.scene_id for s in self.scenes], state="readonly", width=30)
        cb.pack(side=tk.LEFT, padx=6, pady=6)
        ttk.Button(top, text="Load into Editor", command=self.on_load_scene_to_editor).pack(side=tk.LEFT, padx=6)

        # Editor
        form = ttk.LabelFrame(wrap, text="Scene Definition")
        form.pack(fill=tk.BOTH, expand=True)

        self.ed_scene_id = tk.StringVar()
        self.ed_allowed = tk.StringVar()
        self.ed_required = tk.StringVar()
        
        def entry(row, lbl, var):
            ttk.Label(form, text=lbl).grid(row=row, column=0, sticky="w", padx=6, pady=4)
            ttk.Entry(form, textvariable=var, width=60).grid(row=row, column=1, sticky="we", padx=6, pady=4)
        
        entry(0, "Scene ID", self.ed_scene_id)
        
        ttk.Label(form, text="Prompt").grid(row=1, column=0, sticky="nw", padx=6, pady=4)
        self.ed_prompt = tk.Text(form, width=60, height=4)
        self.ed_prompt.grid(row=1, column=1, sticky="we", padx=6, pady=4)
        self.config_text_widget(self.ed_prompt)
        
        ttk.Label(form, text="Blueprint (lines)").grid(row=2, column=0, sticky="nw", padx=6, pady=4)
        self.ed_blueprint = tk.Text(form, width=60, height=6)
        self.ed_blueprint.grid(row=2, column=1, sticky="we", padx=6, pady=4)
        self.config_text_widget(self.ed_blueprint)
        
        entry(3, "Allowed IFC", self.ed_allowed)
        entry(4, "Required IFC", self.ed_required)
        
        form.columnconfigure(1, weight=1)

        # Buttons
        btns = ttk.Frame(form)
        btns.grid(row=5, column=1, sticky="w", pady=10)
        ttk.Button(btns, text="Apply to Session", command=self.on_apply_scene).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Save JSON", command=self.on_save_custom_scenes).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Reload JSON", command=self.on_reload_custom_scenes).pack(side=tk.LEFT, padx=4)

    def setup_custom_tab(self):
        wrap = ttk.Frame(self.tab_custom)
        wrap.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        
        form = ttk.LabelFrame(wrap, text="One-Off Custom Run")
        form.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.cus_scene_id = tk.StringVar(value="CustomScene")
        self.cus_allowed = tk.StringVar(value="IfcWall, IfcSlab, IfcDoor, IfcWindow")
        self.cus_required = tk.StringVar(value="IfcWall, IfcSlab")
        
        def entry(row, lbl, var):
            ttk.Label(form, text=lbl).grid(row=row, column=0, sticky="w", padx=6, pady=4)
            ttk.Entry(form, textvariable=var, width=60).grid(row=row, column=1, sticky="we", padx=6, pady=4)
            
        entry(0, "Scene ID", self.cus_scene_id)
        
        ttk.Label(form, text="Prompt").grid(row=1, column=0, sticky="nw", padx=6, pady=4)
        self.cus_prompt = tk.Text(form, width=60, height=4)
        self.cus_prompt.grid(row=1, column=1, sticky="we", padx=6, pady=4)
        self.config_text_widget(self.cus_prompt)
        self.cus_prompt.insert("1.0", "Describe your custom scene here.")
        
        ttk.Label(form, text="Blueprint").grid(row=2, column=0, sticky="nw", padx=6, pady=4)
        self.cus_blueprint = tk.Text(form, width=60, height=6)
        self.cus_blueprint.grid(row=2, column=1, sticky="we", padx=6, pady=4)
        self.config_text_widget(self.cus_blueprint)
        self.cus_blueprint.insert("1.0", "Height 3m.\nRectangular footprint.")
        
        entry(3, "Allowed IFC", self.cus_allowed)
        entry(4, "Required IFC", self.cus_required)
        
        self.cus_profile = tk.StringVar(value=GEOM_PROFILES[0].profile_id)
        ttk.Label(form, text="Geometry").grid(row=5, column=0, sticky="w", padx=6, pady=4)
        ttk.Combobox(form, textvariable=self.cus_profile, values=[p.profile_id for p in GEOM_PROFILES], state="readonly").grid(row=5, column=1, sticky="w", padx=6)
        
        form.columnconfigure(1, weight=1)
        
        ttk.Button(wrap, text="Run Custom", command=self.on_run_custom).pack(fill=tk.X, padx=6, pady=10)

    def setup_results_tab(self):
        wrap = ttk.Frame(self.tab_results)
        wrap.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        
        # Split: Left=Tree, Right=Info/Actions
        paned = ttk.PanedWindow(wrap, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        left = ttk.Frame(paned)
        right = ttk.Frame(paned, padding=(10, 0))
        paned.add(left, weight=1)
        paned.add(right, weight=1)
        
        # Tree
        ttk.Label(left, text="Runs / Data").pack(anchor="w")
        self.res_tree = ttk.Treeview(left, columns=("status",), displaycolumns=("status",))
        self.res_tree.heading("#0", text="Hierarchy")
        self.res_tree.heading("status", text="Info")
        self.res_tree.column("status", width=100)
        self.res_tree.pack(fill=tk.BOTH, expand=True)
        self.res_tree.bind("<<TreeviewSelect>>", self.on_res_select)
        
        ttk.Button(left, text="Refresh", command=self.refresh_results).pack(fill=tk.X, pady=4)
        
        # Right info
        lbl_frame = ttk.LabelFrame(right, text="Details")
        lbl_frame.pack(fill=tk.BOTH, expand=True)
        
        self.res_info = tk.Text(lbl_frame, width=40, height=10)
        self.res_info.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.config_text_widget(self.res_info)
        
        self.btn_viz = ttk.Button(right, text="Visualize Point Cloud", state="disabled", command=self.on_viz_result)
        self.btn_viz.pack(fill=tk.X, pady=4)
        
        self.btn_logs = ttk.Button(right, text="View Run Logs", state="disabled", command=self.on_view_logs)
        self.btn_logs.pack(fill=tk.X, pady=4)

        self.btn_del = ttk.Button(right, text="Delete Run", state="disabled", command=self.on_delete_run)
        self.btn_del.pack(fill=tk.X, pady=4)
        
        self.refresh_results()

    def refresh_results(self):
        # Clear
        for i in self.res_tree.get_children():
            self.res_tree.delete(i)
            
        # Walk DATA_DIR
        # Structure: DATA_DIR / RunID / Profile / Scene / PV / Seed / final.ply
        if not os.path.exists(DATA_DIR): return
        
        runs = sorted(os.listdir(DATA_DIR), reverse=True)
        for r in runs:
            r_path = os.path.join(DATA_DIR, r)
            if not os.path.isdir(r_path): continue
            
            rid = self.res_tree.insert("", "end", text=r, open=False)
            
            # Profiles
            for p in os.listdir(r_path):
                p_path = os.path.join(r_path, p)
                if not os.path.isdir(p_path): continue
                pid = self.res_tree.insert(rid, "end", text=p, open=False)
                
                # Scenes
                for s in os.listdir(p_path):
                    s_path = os.path.join(p_path, s)
                    if not os.path.isdir(s_path): continue
                    sid = self.res_tree.insert(pid, "end", text=s, open=False)
                    
                    # Prompt Variants (PV)
                    for pv in os.listdir(s_path):
                        pv_path = os.path.join(s_path, pv)
                        if not os.path.isdir(pv_path): continue
                        pvid = self.res_tree.insert(sid, "end", text=pv, open=False)
                        
                        # Seeds
                        for seed in os.listdir(pv_path):
                            seed_path = os.path.join(pv_path, seed)
                            if not os.path.isdir(seed_path): continue
                            
                            # Check status
                            final_ply = os.path.join(seed_path, "final.ply")
                            status = "DONE" if os.path.exists(final_ply) else "..."
                            self.res_tree.insert(pvid, "end", text=seed, values=(status,), tags=("leaf",))

    def on_res_select(self, event):
        sel = self.res_tree.selection()
        if not sel: return
        item = sel[0]
        tags = self.res_tree.item(item, "tags")
        
        # Traverse up to find path
        path_parts = []
        cur = item
        while cur:
            txt = self.res_tree.item(cur, "text")
            path_parts.insert(0, txt)
            cur = self.res_tree.parent(cur)
            
        full_path = os.path.join(DATA_DIR, *path_parts)
        self.selected_result_path = full_path
        
        info = f"Path: {full_path}\n"
        if os.path.exists(full_path):
            if "leaf" in tags:
                final = os.path.join(full_path, "final.ply")
                if os.path.exists(final):
                    info += f"Status: Complete\nSize: {os.path.getsize(final)/1024:.1f} KB"
                    self.btn_viz.config(state="normal")
                else:
                    info += "Status: Incomplete (no final.ply)"
                    self.btn_viz.config(state="disabled")
            else:
                self.btn_viz.config(state="disabled")
                
            # Check for logs (associated with Run ID)
            run_id = path_parts[0]
            log_f = os.path.join(LOG_DIR, run_id, f"discussion_{run_id}.json")
            if os.path.exists(log_f):
                self.btn_logs.config(state="normal")
                self.selected_log_path = log_f
                
                # Calculate tokens
                try:
                    total_tokens = 0
                    with open(log_f, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for entry in data:
                            u = entry.get("usage", {})
                            total_tokens += u.get("total_tokens", 0)
                    info += f"\nTotal Tokens: {total_tokens:,}"
                except Exception: pass
            else:
                self.btn_logs.config(state="disabled")
                
            # Enable delete if path exists
            self.btn_del.config(state="normal")
        else:
            self.btn_viz.config(state="disabled")
            self.btn_logs.config(state="disabled")
            self.btn_del.config(state="disabled")
            
        self.res_info.delete("1.0", tk.END)
        self.res_info.insert("1.0", info)

    def on_viz_result(self):
        if not hasattr(self, "selected_result_path"): return
        # Construct path to final ply
        ply_path = os.path.join(self.selected_result_path, "final.ply")
        if not os.path.exists(ply_path):
             messagebox.showerror("Error", f"Point cloud file not found:\n{ply_path}")
             return
             
        try:
            # Run in a separate thread to avoid freezing UI if O3D takes time
            def _viz():
                try:
                    pcd = o3d.io.read_point_cloud(ply_path)
                    if pcd.is_empty():
                         self.after(0, lambda: messagebox.showwarning("Warning", "Point cloud is empty."))
                         return
                    o3d.visualization.draw_geometries([pcd], window_name=f"Result: {os.path.basename(self.selected_result_path)}")
                except Exception as e:
                    err_msg = f"Open3D Error: {e}"
                    self.after(0, lambda: messagebox.showerror("Error", err_msg))
            
            threading.Thread(target=_viz, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch visualizer: {e}")

    def on_delete_run(self):
        if not hasattr(self, "selected_result_path"): return
        
        # Confirm
        if not messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete:\n{self.selected_result_path}?"):
            return
            
        try:
            import shutil
            shutil.rmtree(self.selected_result_path)
            self.refresh_results()
            self.res_info.delete("1.0", tk.END)
            self.res_info.insert("1.0", "Deleted.")
            self.btn_viz.config(state="disabled")
            self.btn_logs.config(state="disabled")
            self.btn_del.config(state="disabled")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete: {e}")

    def on_view_logs(self):
        if not hasattr(self, "selected_log_path"): return
        try:
            # Simple text viewer window
            top = tk.Toplevel(self)
            top.title(f"Logs: {os.path.basename(os.path.dirname(self.selected_log_path))}")
            top.geometry("800x600")
            txt = tk.Text(top)
            txt.pack(fill=tk.BOTH, expand=True)
            with open(self.selected_log_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                out = []
                for d in data:
                    out.append(f"[{d.get('iso_time','')}] {d.get('role','').upper()} (Iter {d.get('iteration')})")
                    out.append(d.get('text',''))
                    u = d.get('usage')
                    if u and u.get('total_tokens', 0) > 0:
                        out.append(f"[Tokens: {u['total_tokens']}]")
                    out.append("-" * 60)
                txt.insert("1.0", "\n".join(out))
            self.config_text_widget(txt)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # --- Scene Editor Handlers ---
    
    def on_load_scene_to_editor(self):
        sid = self.scene_load_var.get()
        if not sid: return
        try:
            s = next(s for s in self.scenes if s.scene_id == sid)
            self.ed_scene_id.set(s.scene_id)
            self.ed_allowed.set(", ".join(s.allowed_ifc))
            self.ed_required.set(", ".join(s.required_ifc))
            self.ed_prompt.delete("1.0", tk.END)
            self.ed_prompt.insert("1.0", s.prompt)
            self.ed_blueprint.delete("1.0", tk.END)
            self.ed_blueprint.insert("1.0", "\n".join(s.blueprint))
        except StopIteration:
            pass

    def on_apply_scene(self):
        try:
            s = SceneSpec(
                scene_id=self.ed_scene_id.get().strip(),
                prompt=self.ed_prompt.get("1.0", tk.END).strip(),
                blueprint=[l.strip() for l in self.ed_blueprint.get("1.0", tk.END).splitlines() if l.strip()],
                allowed_ifc=[x.strip() for x in self.ed_allowed.get().split(",") if x.strip()],
                required_ifc=[x.strip() for x in self.ed_required.get().split(",") if x.strip()]
            )
            if not s.scene_id: raise ValueError("ID required")
            
            # Update list
            existing = next((i for i, x in enumerate(self.scenes) if x.scene_id == s.scene_id), None)
            if existing is not None:
                self.scenes[existing] = s
            else:
                self.scenes.append(s)
            
            # Register types
            for t in s.allowed_ifc: get_class_id(t)
            
            self.log(f"Applied scene: {s.scene_id}", "INFO")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_save_custom_scenes(self):
        try:
            save_custom_scenes(self.scenes)
            messagebox.showinfo("Saved", f"Saved to {SCENES_CUSTOM_PATH}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_reload_custom_scenes(self):
        self.custom_scenes = load_custom_scenes(SCENES_CUSTOM_PATH)
        # Merge logic again
        # simplified: just re-merge
        if self.custom_scenes:
            self.scenes = merge_scenes(self.scenes, self.custom_scenes)
        self.log("Reloaded custom scenes", "INFO")

    # --- Custom Run Handler ---
    
    def on_run_custom(self):
        try:
            s = SceneSpec(
                scene_id=self.cus_scene_id.get().strip(),
                prompt=self.cus_prompt.get("1.0", tk.END).strip(),
                blueprint=[l.strip() for l in self.cus_blueprint.get("1.0", tk.END).splitlines() if l.strip()],
                allowed_ifc=[x.strip() for x in self.cus_allowed.get().split(",") if x.strip()],
                required_ifc=[x.strip() for x in self.cus_required.get().split(",") if x.strip()]
            )
            # Register types
            for t in s.allowed_ifc: get_class_id(t)
            
            profile = get_profile(self.cus_profile.get())
            
            # Override selections for the worker to pick up?
            # actually better to just launch a thread with specific args.
            
            self.stop_flag = False
            self.set_running(True)
            
            # Hack: define a worker wrapper for custom
            def custom_worker():
                self.log(f"Starting Custom Run: {s.scene_id}", "START")
                run_id = time.strftime("CustomRun_%Y%m%d_%H%M%S")
                ensure_csv_headers()
                try:
                    out_dir = os.path.join(DATA_DIR, run_id, profile.profile_id, s.scene_id, "Custom_Variant", "seed_0")
                    step_dir = os.path.join(out_dir, "_steps")
                    os.makedirs(step_dir, exist_ok=True)
                    
                    profile_fns = build_profile_functions(profile)
                    d_prompt = build_designer_system_prompt(s, profile, float(self.density_var.get()))
                    s_text = make_scene_text(s, "Custom")
                    
                    iters = int(self.dialog_var.get())
                    max_tries = int(self.codegen_tries_var.get())
                    disc_logger = DiscussionLogger(run_id, s.scene_id)
                    
                    def dialog_callback(iteration, role, text, _dl=disc_logger):
                        self.log_discussion(iteration, role, text)
                        _dl.log_turn(iteration, role, text)

                    def step_callback(iteration, code, arr, elems, labels, stats):
                        export_iteration_pointcloud(arr, elems, labels, step_dir, f"step_{iteration}")
                        if self.open3d_live_var.get():
                            start_open3d_live_preview()
                            live_preview_push_open3d(arr, s.scene_id, f"Round {iteration}", 1.2)

                    final_code, total_usage, trace, dialog_history = run_full_pipeline(
                        designer_system_prompt=d_prompt,
                        scene_text=s_text,
                        model_name=self.model_var.get(),
                        n_dialog_iterations=iters,
                        prev_code=None,
                        profile_fns=profile_fns,
                        scene=s,
                        max_codegen_tries=max_tries,
                        log_fn=lambda m: self.log(m, "DEBUG"),
                        dialog_callback=dialog_callback,
                        step_callback=step_callback,
                    )
                    
                    from ..core.execution import run_generated_code
                    pts = run_generated_code(final_code, profile_fns)
                    arr, elems, labels = points_to_numpy(pts, s.allowed_ifc)
                    stats = compute_stats(arr)
                    ok, issues = constraint_check(s, arr, elems, stats)
                    
                    if self.open3d_live_var.get():
                        start_open3d_live_preview()
                        live_preview_push_open3d(arr, s.scene_id, "Final", 1.2)
                        
                    if ok:
                        exported = export_pointcloud(arr, elems, labels, out_dir, "final")
                        disc_logger.save_json()
                        self.log(f"SUCCESS. Exported: {', '.join(k for k, v in exported.items() if v)}", "SUCCESS")
                        with open(os.path.join(out_dir, "final_code.py"), "w", encoding="utf-8") as f:
                            f.write(final_code)
                        append_run_meta([
                            time.strftime("%Y-%m-%dT%H:%M:%S"), run_id,
                            profile.profile_id, s.scene_id, "Custom", 0,
                            iters, True, stats["num_points"],
                            total_usage["prompt_tokens"],
                            total_usage["completion_tokens"],
                            total_usage["total_tokens"],
                            out_dir,
                        ])
                    else:
                        self.log(f"Constraint Fail: {issues}", "WARN")
                            
                except Exception as e:
                    self.log(f"Error: {e}", "ERROR")
                finally:
                    self.set_running(False)
                    self.refresh_results()
            
            t = threading.Thread(target=custom_worker, daemon=True)
            t.start()
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def log(self, msg: str, level: str = "INFO"):
        self.ui_queue.put(f"[{level}] {msg}")

    def log_discussion(self, iteration: int, role: str, text: str):
        self.discussion_queue.put((iteration, role, text))

    def process_ui_queue(self):
        try:
            while True:
                msg = self.ui_queue.get_nowait()
                self.log_box.insert(tk.END, msg + "\n")
                self.log_box.see(tk.END)
        except queue.Empty:
            pass
        self.after(50, self.process_ui_queue)

    def _process_discussion_queue(self):
        try:
            while True:
                iteration, role, text = self.discussion_queue.get_nowait()
                header = f"\n[Round {iteration} | {role.upper()}]\n"
                self.discussion_box.insert(tk.END, header, "header")
                self.discussion_box.insert(tk.END, text.strip() + "\n", role)
                self.discussion_box.see(tk.END)
        except queue.Empty:
            pass
        self.after(100, self._process_discussion_queue)

    def on_apply_key(self):
        k = self.api_key_var.get().strip()
        set_openai_client(k)
        messagebox.showinfo("API", "API Key Updated")

    def on_ping(self):
        ok, msg = quick_api_ping(self.model_var.get())
        if ok:
            self.log(f"Ping OK: {msg}", "INFO")
            messagebox.showinfo("Ping", f"OK: {msg}")
        else:
            self.log(f"Ping FAILED: {msg}", "ERROR")
            messagebox.showerror("Ping", f"Error: {msg}")

    def on_stop(self):
        self.stop_flag = True
        self.log("Stopping...", "WARN")

    def set_running(self, running: bool):
        state = "disabled" if running else "normal"
        self.btn_run_all.config(state=state)
        self.btn_run_one.config(state=state)
        self.btn_stop.config(state="normal" if running else "disabled")

    def get_selected_scene(self) -> SceneSpec:
        sid = self.scene_var.get()
        for s in self.scenes:
            if s.scene_id == sid:
                return s
        raise ValueError(f"Unknown scene: {sid}")

    def on_run_one(self):
        self.start_worker(single=True)

    def on_run_all(self):
        self.start_worker(single=False)

    def start_worker(self, single: bool):
        self.stop_flag = False
        self.set_running(True)
        t = threading.Thread(target=self.worker, args=(single,), daemon=True)
        t.start()

    def worker(self, single: bool):
        run_id = time.strftime("Run_%Y%m%d_%H%M%S")
        self.log(f"Started Run: {run_id}", "START")
        ensure_csv_headers()

        try:
            scenes = [self.get_selected_scene()] if single or not self.all_scenes_var.get() else self.scenes
            profiles = [get_profile(self.profile_var.get())] if single or not self.all_profiles_var.get() else GEOM_PROFILES

            selected_pv_name = self.pv_var.get()
            target_pvs = PROMPT_VARIANTS
            if single or not self.all_pvs_var.get():
                target_pvs = [pv for pv in PROMPT_VARIANTS if pv[0] == selected_pv_name]
                if not target_pvs:
                    target_pvs = [PROMPT_VARIANTS[0]]

            model = self.model_var.get()
            density = float(self.density_var.get())
            seeds = int(self.seeds_var.get()) if not single else 1
            n_iters = int(self.dialog_var.get())
            max_tries = int(self.codegen_tries_var.get())

            for scene in scenes:
                for profile in profiles:
                    for pv in target_pvs:
                        for seed in range(seeds):
                            if self.stop_flag:
                                raise RuntimeError("Stopped by user.")

                            run_label = f"{scene.scene_id} | {profile.profile_id} | {pv[0]} | Seed {seed}"
                            self.log(f"Running: {run_label}", "INFO")

                            out_dir = os.path.join(DATA_DIR, run_id, profile.profile_id, scene.scene_id, pv[0], f"seed_{seed}")
                            step_dir = os.path.join(out_dir, "_steps")
                            os.makedirs(step_dir, exist_ok=True)

                            scene_text = make_scene_text(scene, pv[1])
                            profile_fns = build_profile_functions(profile)
                            designer_sys = build_designer_system_prompt(scene, profile, density)

                            disc_logger = DiscussionLogger(run_id, scene.scene_id)

                            def dialog_callback(iteration, role, text, _dl=disc_logger):
                                self.log_discussion(iteration, role, text)
                                _dl.log_turn(iteration, role, text)

                            def step_callback(iteration, code, arr, elems, labels, stats):
                                export_iteration_pointcloud(arr, elems, labels, step_dir, f"step_{iteration}")
                                if self.open3d_live_var.get():
                                    start_open3d_live_preview()
                                    live_preview_push_open3d(arr, scene.scene_id, f"Round {iteration}", 1.2)

                            self.log(f"  Starting pipeline: {n_iters} iterations, {max_tries} tries/round", "INFO")

                            final_code, total_usage, trace, dialog_history = run_full_pipeline(
                                designer_system_prompt=designer_sys,
                                scene_text=scene_text,
                                model_name=model,
                                n_dialog_iterations=n_iters,
                                prev_code=None,
                                profile_fns=profile_fns,
                                scene=scene,
                                max_codegen_tries=max_tries,
                                log_fn=lambda m: self.log(m, "DEBUG"),
                                dialog_callback=dialog_callback,
                                step_callback=step_callback,
                            )

                            # Execute final code and export
                            from ..core.execution import run_generated_code
                            pts = run_generated_code(final_code, profile_fns)
                            arr, elems, labels = points_to_numpy(pts, scene.allowed_ifc)
                            stats = compute_stats(arr)
                            ok, issues = constraint_check(scene, arr, elems, stats)

                            if self.open3d_live_var.get():
                                start_open3d_live_preview()
                                live_preview_push_open3d(arr, scene.scene_id, "Final", 1.2)

                            if ok:
                                exported = export_pointcloud(arr, elems, labels, out_dir, "final")
                                disc_logger.save_json()
                                self.log(f"  SUCCESS \u2014 {stats['num_points']} pts. Exported: {', '.join(k for k, v in exported.items() if v)}", "SUCCESS")
                                # Save final code
                                with open(os.path.join(out_dir, "final_code.py"), "w", encoding="utf-8") as f:
                                    f.write(final_code)
                                append_run_meta([
                                    time.strftime("%Y-%m-%dT%H:%M:%S"), run_id,
                                    profile.profile_id, scene.scene_id, pv[0], seed,
                                    n_iters, True, stats["num_points"],
                                    total_usage["prompt_tokens"],
                                    total_usage["completion_tokens"],
                                    total_usage["total_tokens"],
                                    out_dir,
                                ])
                            else:
                                self.log(f"  Constraint check failed: {issues}", "WARN")

        except Exception as exc:
            self.log(f"Error: {exc}", "ERROR")
        finally:
            self.set_running(False)
            self.refresh_results()
            self.log("Run finished.", "END")
