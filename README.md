# SYN3D-LLM

![Bridge Generation Iterations](GIF.gif)

A framework for generating synthetic 3D point cloud data using Large Language Models (LLMs) via an iterative **Designer–Critic** self-discussion loop.

## Framework Overview

The workflow begins with a **user prompt** describing the scene to generate. The **Designer** LLM receives this prompt combined with a system prompt that enforces:
- Strict code generation constraints (output schema, Python only, no external libraries)
- Available helper functions (e.g. `box_surface`, `plane_surface`, `cylinder_surface`)
- IFC element type restrictions per scene template

The Designer produces an initial code draft. This is not executed immediately — it is first reviewed by the **Critic** LLM, which checks constraints, geometric plausibility, and IFC label correctness. The Critic's feedback is passed back to the Designer for the next iteration.

By default, the framework runs **2 Designer–Critic iterations** after an initial draft (3 rounds total). Each round allows up to **3 retries** to produce executable code. After all iterations, the final point cloud is executed and exported.

## Exports

Each generated point cloud is saved in three formats:
- **`.e57`** — industry-standard laser scan format (requires `pye57`)
- **`.xyz`** — plain-text XYZ coordinates
- **`.ply`** — polygon file format with per-class colour

All Designer–Critic discussions are automatically logged to:
- **`.JSON`** — full per-run discussion log (`syn3d_runs/logs/<run_id>/discussion_<run_id>.json`)
- **`.CSV`** — per-turn summary table (`syn3d_runs/discussion_log.csv`)

## Scene Templates

The framework can generate various infrastructure and building types. By default, the following templates are included:
- **Apartments**: Rectangular, L-shape, and Courtyard layouts.
- **High-rise buildings**: Multi-story structures with load-bearing walls and slabs.
- **Industrial halls**: Large open spaces with repeating columns and beams.
- **Bridges**: Infrastructure featuring bridge decks, piers, and foundations.
- **Railway tracks**: Dual-rail tracks with sleepers and track beds.
- **Streets**: Road intersections equipped with traffic signs and traffic lights.

Each scene is strictly bound to an IFC ontology, dictating which semantic elements (e.g., `IfcWall`, `IfcSlab`, `IfcColumn`) the LLM is allowed and required to use. Every individual object instance is explicitly assigned a semantic string label (e.g., "Bridge Pier").

## The Experiment Matrix

SYN3D-LLM is designed for high-throughput scientific evaluation. Instead of generating scenes one by one, the framework executes a multi-dimensional **Experiment Matrix**. This matrix systematically combinations the following variables:

1. **Geometry Profiles**: The level of mathematical assistance provided to the LLM (e.g., `G0_NoHelper` forces the LLM to write raw loop arithmetic, while `G3_Round` provides cylinder generation functions).
2. **Scenes**: The architectural templates listed above.
3. **Prompt Variants (PV)**: Semantic variations of the same scene (e.g., `PV1_Standard` vs. `PV2_Complex` irregular structures).
4. **Seeds**: Multiple independent runs per configuration to measure LLM stability and variance.

When you click **Run ALL** in the GUI, the framework loops through every combination of `Profile × Scene × Variant × Seed`. Results are stored hierarchically in `syn3d_runs/data/RunID/Profile/Scene/PV/Seed/` to facilitate large-scale ablation studies.

## GUI

A Tkinter-based GUI provides access to all settings (model, iterations, geometry profile), scene selection, system prompt viewing, and real-time logging of the Designer–Critic discussion.
An Open3D live preview window updates after each iteration.

## Installation

```bash
pip install -r requirements.txt
# Optional: pip install pye57  (for .e57 export)
```

## Usage

```bash
python main.py
```

1. Enter your OpenAI API key and click **Apply**.
2. Select a scene template and geometry profile.
3. Set the number of iterations (default: 2).
4. Click **Run ONE** for a single seed or **Run ALL** for the full experiment matrix.
5. Results are saved to `syn3d_runs/data/`.

## Project Structure

```
SYN3DLLM/
├── main.py                  # Entry point
├── requirements.txt
├── LICENSE
├── src/
│   ├── config.py            # API client, directory config
│   ├── core/
│   │   ├── generator.py     # Designer/Critic loop, prompt construction
│   │   ├── execution.py     # Sandboxed code execution, constraint checking
│   │   ├── geometry.py      # Helper functions (box_surface, plane_surface, …)
│   │   └── scenes.py        # Scene templates, IFC ontology
│   ├── ui/
│   │   ├── app.py           # Tkinter GUI
│   │   └── theme.py         # Dark mode styling
│   └── utils/
│       ├── file_io.py       # Export (.e57/.xyz/.ply), CSV/JSON discussion logging
│       └── visualization.py # Open3D live preview
├── syn3d_runs/              # Generated outputs (gitignored)
│   ├── data/                # Point cloud files
│   ├── logs/                # Discussion JSON logs
│   ├── runs_metadata.csv    # Per-run summary
│   └── discussion_log.csv   # Per-turn Designer/Critic log
└── tests/
    └── test_basics.py
```

## Citation

If you use this framework or the generated datasets in your research, please cite our paper:

```bibtex
@article{syn3dllm,
  title={SYN3D-LLM: SYNTHETIC 3D POINT CLOUDS GENERATED THROUGH LLM DISCUSSION},
  author={},
  journal={},
  year={}
}
```
