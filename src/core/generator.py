import time
import threading
from typing import Optional, Dict, List, Tuple, Any

from ..config import get_client
from ..core.scenes import SceneSpec
from ..core.geometry import GeometryProfile
from ..core.execution import (
    run_generated_code,
    points_to_numpy,
    compute_stats,
    constraint_check,
    detect_string_indexing_bug,
)

# ============================================================
# System Prompts
# ============================================================

DESIGNER_SYSTEM_PROMPT = """You generate Python code that creates synthetic 3D point clouds.

Rules:
- Do NOT import anything.
- Do NOT define any new geometry helper functions.
- Return ONLY valid Python code (no Markdown, no comments, no explanations).
- Define exactly ONE function named build_scene() that returns a list of points.

Output schema — each point is a dict with EXACT keys:
{
  "x": float,
  "y": float,
  "z": float,
  "element": "<IFC type name string>",
  "label": "<short object description>"
}

CRITICAL LABELING RULE:
- Ensure the `element` and `label` accurately match the true geometric and structural nature of the object.
- For example: If you build a column, use `IfcColumn` and a label like "Column". If you build a bridge deck, use `IfcSlab` and a label like "Bridge Deck". 
- Do NOT mislabel objects (e.g., do not label a vertical support as a "Wall" if it is structurally a "Column").

CRITICAL — helper functions return plain Python lists of [x, y, z] coordinates:
- NEVER index helper output as pts[i]["x"] or pts["x"] — this will crash.
- Always unpack: for x, y, z in pts: ...
- Or index numerically: x = pt[0], y = pt[1], z = pt[2].
""".strip()

CRITIC_SYSTEM_PROMPT = """You are the Critic in an iterative Designer-Critic framework for generating executable point cloud code.

You receive:
- USER_PROMPT: the scene description the Designer was given.
- DRAFT_CODE: the Python code the Designer produced.

Your task:
- Check for violations of code constraints (imports, extra helper definitions, wrong output schema).
- Check that the `element` and `label` assigned to points accurately reflect their true geometry and structural role (e.g., vertical supports must be Columns, not Walls). Flag any semantic mismatches.
- Identify missing or incorrect IFC element types for the scene.
- Discuss what structural or geometric aspects can be improved.
- Suggest concrete, actionable changes for the Designer to apply in the next round.
- Do NOT provide a numeric score. Do NOT rewrite the code yourself.

Your response MUST follow this structure exactly:

CONSTRAINTS_CHECK:
- PASS/FAIL: <overall>
- Issues:
  - <issue or NONE>

IMPROVEMENT_DISCUSSION:
- <observation and reasoning>

NEXT_CHANGES_FOR_DESIGNER:
- <concrete action item>
""".strip()


def _geom_contract_text(profile: GeometryProfile) -> str:
    if profile.profile_id == "G0_NoHelper":
        return (
            "Geometry availability:\n"
            "NO helper functions are available.\n"
            "You must sample points manually using arithmetic loops.\n"
        )
    items = []
    if profile.provide_box_surface:
        items.append("box_surface(x_min, x_max, y_min, y_max, z_min, z_max, density=0.1)")
    if profile.provide_minimal:
        items.append("plane_surface(x_min, x_max, y_min, y_max, z, density=0.1)")
    if profile.provide_round:
        items.append("cylinder_surface(cx, cy, z_min, z_max, radius=0.15, density=0.05, n_theta=36)")
    if profile.provide_specialized:
        items.append("rail_pair(x0, x1, y_center, gauge=1.435, z=0.05, density=0.05)")
        items.append("sign_post(x, y, z0=0.0, z1=2.2, radius=0.05, density=0.05)")
        items.append("traffic_light(x, y, z0=0.0, z1=3.0, density=0.05)")
    return (
        "Geometry availability:\n"
        "You MAY use ONLY the following pre-defined helper functions (each returns a list of [x, y, z] lists):\n"
        + "\n".join(f"  - {it}" for it in items)
        + "\nDo NOT define new helper functions.\n"
    )


def build_designer_system_prompt(scene: SceneSpec, profile: GeometryProfile, density_hint: float) -> str:
    allowed = ", ".join(scene.allowed_ifc)
    required = ", ".join(scene.required_ifc)
    blueprint = "\n".join(f"  - {b}" for b in scene.blueprint)
    return (
        DESIGNER_SYSTEM_PROMPT
        + "\n\nIFC_TYPES_FOR_THIS_SCENE:\n"
        + f"  Allowed types (use ONLY these): {allowed}\n"
        + f"  Required types (include at least one of each): {required}\n"
        + "\n\nSCENE_BLUEPRINT:\n"
        + blueprint
        + "\n\nDENSITY_HINT:\n"
        + f"  Use point spacing around {density_hint:.3f} metres.\n"
        + "\n\n"
        + _geom_contract_text(profile)
        + "\nREGRESSION_GUARD:\n"
        + "  Preserve all correct parts from the previous version.\n"
        + "  Apply only the Critic's requested changes. Do not shrink the scene.\n"
    )


def build_designer_user_message(scene_text: str, iteration: int,
                                critic_feedback: Optional[str],
                                prev_code: Optional[str]) -> str:
    parts = [f"USER_PROMPT:\n{scene_text}", f"ITERATION: {iteration}"]
    if prev_code:
        parts.append(f"PREVIOUS_CODE (improve this; do not copy blindly):\n{prev_code}")
    if critic_feedback:
        parts.append(f"CRITIC_FEEDBACK_TO_APPLY:\n{critic_feedback}")
    return "\n\n".join(parts)


def build_critic_user_message(scene_text: str, draft_code: str) -> str:
    return (
        f"USER_PROMPT:\n{scene_text}\n\n"
        f"DRAFT_CODE:\n{draft_code}"
    )


def make_scene_text(scene: SceneSpec, prompt_variant_text: str) -> str:
    blueprint = "\n".join(f"  - {x}" for x in scene.blueprint)
    variant_section = f"\nPROMPT_VARIANT:\n{prompt_variant_text}\n" if prompt_variant_text else ""
    return f"{scene.prompt}{variant_section}\nBLUEPRINT:\n{blueprint}\n"


# ============================================================
# OpenAI API calls
# ============================================================

def call_openai(system_prompt: str, user_content: str, model_name: str, temperature: float) -> Tuple[str, Dict[str, int]]:
    client = get_client()
    if client is None:
        raise RuntimeError("OpenAI client is not configured. Enter an API key in the GUI.")
    max_tries = 6
    base_sleep = 1.0
    last_exc = None
    for attempt in range(1, max_tries + 1):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=temperature,
            )
            text = resp.choices[0].message.content or ""
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            if getattr(resp, "usage", None):
                usage["prompt_tokens"] = int(getattr(resp.usage, "prompt_tokens", 0) or 0)
                usage["completion_tokens"] = int(getattr(resp.usage, "completion_tokens", 0) or 0)
                usage["total_tokens"] = int(getattr(resp.usage, "total_tokens", 0) or 0)
            return text, usage
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            transient = any(k in msg for k in ("connection", "timeout", "temporarily", "timed out"))
            if (not transient) or (attempt == max_tries):
                raise
            sleep_s = min(base_sleep * (2 ** (attempt - 1)), 20.0) + 0.1 * attempt
            time.sleep(sleep_s)
    raise last_exc


def quick_api_ping(model_name: str) -> Tuple[bool, str]:
    try:
        text, _ = call_openai("You are a connectivity check.", "Reply with OK.", model_name, 0.0)
        return True, text
    except Exception as exc:
        return False, str(exc)


# ============================================================
# Code extraction
# ============================================================

def extract_code(raw_text: str) -> str:
    if "```" in raw_text:
        parts = raw_text.split("```")
        candidates = [p for p in parts if "def build_scene" in p]
        if candidates:
            raw_text = max(candidates, key=len)
    idx = raw_text.find("def build_scene")
    if idx < 0:
        raise RuntimeError("No 'def build_scene' found in LLM output.")
    return raw_text[idx:].strip()


# ============================================================
# Designer retry loop (retries Designer only on execution failure)
# ============================================================

def generate_code_with_retries(
    designer_system_prompt: str,
    scene_text: str,
    model_name: str,
    iteration: int,
    critic_feedback: Optional[str],
    prev_code: Optional[str],
    profile_fns: Dict[str, Any],
    scene: SceneSpec,
    max_tries: int,
    log_fn,
    step_callback=None,
) -> Tuple[str, Dict[str, int], List[Dict[str, Any]], int]:
    """
    Runs the Designer up to max_tries times until executable+valid code is produced.
    The Critic is NOT called here; this is pure Designer retry logic.

    Returns: (code, usage_sum, trace_entries, tries_used)
    """
    usage_sum = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    trace_entries: List[Dict[str, Any]] = []
    last_error = ""
    feedback = critic_feedback

    for attempt in range(1, max_tries + 1):
        user_msg = build_designer_user_message(scene_text, iteration, feedback, prev_code)

        raw, usage = call_openai(designer_system_prompt, user_msg, model_name, temperature=0.6)
        usage_sum["prompt_tokens"] += usage["prompt_tokens"]
        usage_sum["completion_tokens"] += usage["completion_tokens"]
        usage_sum["total_tokens"] += usage["total_tokens"]

        trace_entries.append({
            "ts": int(time.time()),
            "role": "designer",
            "iteration": iteration,
            "attempt": attempt,
            "model": model_name,
            "user_message": user_msg,
            "raw_response": raw,
            "usage": dict(usage),
        })

        try:
            code = extract_code(raw)
        except RuntimeError as exc:
            last_error = str(exc)
            log_fn(f"    [Retry {attempt}/{max_tries}] Code extraction failed: {last_error}")
            feedback = (critic_feedback or "") + f"\n\nEXTRACTION_ERROR: {last_error}\nEnsure you output ONLY the build_scene() function."
            continue

        if detect_string_indexing_bug(code):
            last_error = "String-key indexing on helper output detected (pts[i]['x'] etc.)."
            log_fn(f"    [Retry {attempt}/{max_tries}] Static check failed: {last_error}")
            feedback = (critic_feedback or "") + f"\n\nSTATIC_ERROR: {last_error}\nHelper functions return [x,y,z] lists. Always unpack: for x, y, z in pts."
            trace_entries.append({"ts": int(time.time()), "role": "exec_check",
                                  "iteration": iteration, "attempt": attempt,
                                  "ok": False, "phase": "static_check", "error": last_error})
            continue

        try:
            pts = run_generated_code(code, profile_fns)
            arr, elems, labels = points_to_numpy(pts, scene.allowed_ifc)
            stats = compute_stats(arr)
            ok, issues = constraint_check(scene, arr, elems, stats)

            trace_entries.append({"ts": int(time.time()), "role": "exec_check",
                                  "iteration": iteration, "attempt": attempt,
                                  "ok": bool(ok), "phase": "exec_and_constraint",
                                  "stats": stats, "constraint_issues": list(issues)})

            if ok:
                if step_callback:
                    step_callback(iteration, code, arr, elems, labels, stats)
                return code, usage_sum, trace_entries, attempt

            last_error = "Constraint check failed: " + (", ".join(issues) if issues else "unknown")
            log_fn(f"    [Retry {attempt}/{max_tries}] {last_error}")
            feedback = (critic_feedback or "") + f"\n\nCONSTRAINT_FAILURE: {last_error}\nEnsure non-empty points and all required IFC types are present."
            prev_code = code

        except Exception as exc:
            last_error = str(exc)
            log_fn(f"    [Retry {attempt}/{max_tries}] Runtime error: {last_error}")
            feedback = (critic_feedback or "") + f"\n\nRUNTIME_ERROR: {last_error}\nSimplify loops, avoid undefined variables, strictly follow the output dict schema."
            trace_entries.append({"ts": int(time.time()), "role": "exec_check",
                                  "iteration": iteration, "attempt": attempt,
                                  "ok": False, "phase": "runtime_exception", "error": last_error})
            prev_code = code

    raise RuntimeError(f"Failed to produce executable code after {max_tries} tries. Last error: {last_error}")


# ============================================================
# Full Designer → Critic dialog round
# ============================================================

def run_dialog_round(
    designer_system_prompt: str,
    scene_text: str,
    model_name: str,
    iteration: int,
    critic_feedback: Optional[str],
    prev_code: Optional[str],
    profile_fns: Dict[str, Any],
    scene: SceneSpec,
    max_codegen_tries: int,
    log_fn,
    step_callback=None,
) -> Tuple[str, str, Dict[str, int], List[Dict[str, Any]]]:
    """
    Executes one full Designer→Critic round:
      1. Designer generates executable code (with up to max_codegen_tries retries).
      2. Critic reviews the code and produces feedback.

    Returns: (code, critic_feedback_text, usage_sum, trace_entries)
    """
    log_fn(f"  [Round {iteration}] Designer generating code...")
    code, usage_d, trace_d, tries_used = generate_code_with_retries(
        designer_system_prompt=designer_system_prompt,
        scene_text=scene_text,
        model_name=model_name,
        iteration=iteration,
        critic_feedback=critic_feedback,
        prev_code=prev_code,
        profile_fns=profile_fns,
        scene=scene,
        max_tries=max_codegen_tries,
        log_fn=log_fn,
        step_callback=step_callback,
    )
    log_fn(f"  [Round {iteration}] Designer OK ({tries_used} tries). Calling Critic...")

    critic_user_msg = build_critic_user_message(scene_text, code)
    critic_raw, usage_c = call_openai(CRITIC_SYSTEM_PROMPT, critic_user_msg, model_name, temperature=0.4)
    critic_text = critic_raw.strip()

    usage_sum = {
        "prompt_tokens": usage_d["prompt_tokens"] + usage_c["prompt_tokens"],
        "completion_tokens": usage_d["completion_tokens"] + usage_c["completion_tokens"],
        "total_tokens": usage_d["total_tokens"] + usage_c["total_tokens"],
    }

    trace_critic = {
        "ts": int(time.time()),
        "role": "critic",
        "iteration": iteration,
        "model": model_name,
        "user_message": critic_user_msg,
        "raw_response": critic_raw,
        "feedback_text": critic_text,
        "usage": dict(usage_c),
    }

    log_fn(f"  [Round {iteration}] Critic responded.")
    return code, critic_text, usage_sum, trace_d + [trace_critic]


# ============================================================
# Backward-compat wrapper used by app.py worker
# ============================================================

def run_full_pipeline(
    designer_system_prompt: str,
    scene_text: str,
    model_name: str,
    n_dialog_iterations: int,
    prev_code: Optional[str],
    profile_fns: Dict[str, Any],
    scene: SceneSpec,
    max_codegen_tries: int,
    log_fn,
    dialog_callback=None,
    step_callback=None,
) -> Tuple[str, Dict[str, int], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Runs the full SYN3D-LLM pipeline:
      - 1 initial draft (no critic feedback)
      - n_dialog_iterations subsequent Designer→Critic rounds

    dialog_callback(iteration, role, text): called after each Designer/Critic turn for live GUI logging.

    Returns: (final_code, total_usage, all_trace_entries, dialog_history)
    """
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    all_trace: List[Dict[str, Any]] = []
    dialog_history: List[Dict[str, Any]] = []

    critic_feedback: Optional[str] = None
    current_code: Optional[str] = prev_code

    total_rounds = 1 + n_dialog_iterations

    for i in range(1, total_rounds + 1):
        code, critic_text, usage, trace = run_dialog_round(
            designer_system_prompt=designer_system_prompt,
            scene_text=scene_text,
            model_name=model_name,
            iteration=i,
            critic_feedback=critic_feedback,
            prev_code=current_code,
            profile_fns=profile_fns,
            scene=scene,
            max_codegen_tries=max_codegen_tries,
            log_fn=log_fn,
            step_callback=step_callback,
        )

        for k in total_usage:
            total_usage[k] += usage[k]
        all_trace.extend(trace)

        designer_entry = next((t for t in trace if t.get("role") == "designer" and t.get("iteration") == i), None)
        dialog_history.append({
            "iteration": i,
            "role": "designer",
            "text": designer_entry["raw_response"] if designer_entry else "",
            "code": code,
        })
        dialog_history.append({
            "iteration": i,
            "role": "critic",
            "text": critic_text,
        })

        if dialog_callback:
            designer_raw = designer_entry["raw_response"] if designer_entry else code
            dialog_callback(i, "designer", designer_raw)
            dialog_callback(i, "critic", critic_text)

        current_code = code
        critic_feedback = critic_text

    return current_code, total_usage, all_trace, dialog_history
