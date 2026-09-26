#!/usr/bin/env python3
"""Validate, aggregate, plot, and typeset the prompt-specificity ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import zlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
PERSONAS = ("level-0", "level-1", "level-2", "level-3")
PERSONA_LABELS = {"level-0": "NT", "level-1": "L1", "level-2": "L2", "level-3": "L3"}
THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.I | re.S)
THINK_FIELD = re.compile(r"\bThinking\s*:\s*(.*?)(?=\bAnswer\s*:|\Z)", re.I | re.S)
WORD = re.compile(r"\b[\w']+\b", re.UNICODE)
MENTAL_STATE = re.compile(
    r"\b(?:think|believ|know|knew|known|feel|felt|want|intend|remember|guess|"
    r"understand|realiz|notic|imagin|pretend|expect|hop)(?:s|es|ed|ing|er|ation)?\b",
    re.I,
)

# ---------------------------------------------------------------------------
# Shared plot styling, matched to the paper's other accuracy figures
# (gray dashed baseline vs. blue solid conditions, thin horizontal
# gridlines only, sans-serif titles with the model name on its own line,
# and a bottom-centered legend).
# ---------------------------------------------------------------------------
PLOT_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": "#D9D9D9",
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "axes.edgecolor": "#000000",
    "axes.labelcolor": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "font.size": 9,
}

# "Full profile" reads as the more-specific / higher-signal condition, so it
# gets the solid dark-teal line; "Role only" is a dashed warm orange, giving
# a two-color palette instead of a black/gray-vs-color contrast.
LINE_STYLE = {
    "full_profile": {"label": "Full profile", "color": "#0072B2", "linestyle": "-", "marker": "o"},
    "roleplay_only": {"label": "Role only", "color": "#E69F00", "linestyle": "-", "marker": "s"},
}


class InclusionError(ValueError):
    """A selected run is not eligible for paper reporting."""


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise InclusionError(f"malformed JSON at {path}:{line_number}") from exc
    return rows


def transport_errors(run_dir: Path, manifest: dict[str, Any]) -> int:
    if "transport_error_count" in manifest:
        return int(manifest["transport_error_count"])
    error_path = run_dir / "errors.jsonl"
    if not error_path.exists():
        return 0
    return sum(
        not str(row.get("error", "")).lower().startswith("parse failed:")
        for row in read_jsonl(error_path)
    )


def infer_prompt_mode(record: dict[str, Any]) -> str:
    if record.get("prompt_mode") in {"full_profile", "roleplay_only"}:
        return str(record["prompt_mode"])
    prompt = record.get("prompt", [])
    system = "\n".join(
        str(message.get("content", "")) for message in prompt if message.get("role") == "system"
    )
    return "full_profile" if "COGNITIVE STYLE:" in system and "RESPONSE RULES:" in system else "roleplay_only"


def validate_run(spec: dict[str, Any], root: Path, expected_count: int, expected_tasks: int | set[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = root / spec["path"]
    if not run_dir.is_dir():
        raise InclusionError(f"{spec['id']}: missing run directory {run_dir}")
    manifest = read_json(run_dir / "manifest.json")
    if manifest.get("status") != "complete":
        raise InclusionError(f"{spec['id']}: manifest.status is {manifest.get('status')!r}, not 'complete'")
    if manifest.get("model") != spec["model"]:
        raise InclusionError(f"{spec['id']}: model mismatch ({manifest.get('model')!r})")
    if int(manifest.get("completed_record_count", -1)) != expected_count:
        raise InclusionError(f"{spec['id']}: manifest does not report {expected_count} completed records")
    if transport_errors(run_dir, manifest):
        raise InclusionError(f"{spec['id']}: run contains transport/server failures")

    decoding = manifest.get("decoding", {})
    if bool(decoding.get("enable_thinking")) != bool(spec["thinking"]):
        raise InclusionError(f"{spec['id']}: native-thinking setting mismatch")
    expected_sampler = (
        {"temperature": 1.0, "top_p": 0.95, "max_tokens": 4096}
        if spec["thinking"] else {"temperature": 0.2, "top_p": 0.5, "max_tokens": 512}
    )
    for name, expected in expected_sampler.items():
        if decoding.get(name) != expected:
            raise InclusionError(f"{spec['id']}: {name}={decoding.get(name)!r}, expected {expected!r}")
    if decoding.get("seed") != 42:
        raise InclusionError(f"{spec['id']}: seed is not 42")
    if "max_model_len" in spec and manifest.get("max_model_len") != spec["max_model_len"]:
        raise InclusionError(
            f"{spec['id']}: max_model_len={manifest.get('max_model_len')!r}, "
            f"expected {spec['max_model_len']}"
        )
    if spec["thinking"]:
        expected_top_k = 20 if spec["model"].startswith("Qwen/") else 64
        if decoding.get("top_k") != expected_top_k:
            raise InclusionError(f"{spec['id']}: top_k is not the model-specific recommended value")

    records = read_jsonl(run_dir / "responses.jsonl")
    if len(records) != expected_count:
        raise InclusionError(f"{spec['id']}: found {len(records)} records, expected {expected_count}")
    keys = [row.get("key") for row in records]
    if any(not key for key in keys) or len(set(keys)) != len(keys):
        raise InclusionError(f"{spec['id']}: missing or duplicate resumable item keys")
    if {row.get("model") for row in records} != {spec["model"]}:
        raise InclusionError(f"{spec['id']}: response model identity mismatch")
    if {infer_prompt_mode(row) for row in records} != {spec["prompt_mode"]}:
        raise InclusionError(f"{spec['id']}: prompt content/mode mismatch")
    if {row.get("persona") for row in records} != set(PERSONAS):
        raise InclusionError(f"{spec['id']}: all four personas are not represented")
    tasks = {row.get("task") for row in records}
    expected_task_set = expected_tasks if isinstance(expected_tasks, set) else None
    expected_task_count = len(expected_task_set) if expected_task_set is not None else expected_tasks
    if len(tasks) != expected_task_count or None in tasks or (expected_task_set is not None and tasks != expected_task_set):
        raise InclusionError(f"{spec['id']}: task set does not match the expected {expected_task_count} tasks")
    per_persona = {p: sum(row.get("persona") == p for row in records) for p in PERSONAS}
    if set(per_persona.values()) != {expected_count // len(PERSONAS)}:
        raise InclusionError(f"{spec['id']}: unbalanced persona records {per_persona}")
    return manifest, records


def visible_thinking(record: dict[str, Any]) -> str:
    """Extract the public Thinking field; never use provider-native reasoning."""
    visible = THINK_BLOCK.sub(" ", str(record.get("raw_response") or ""))
    visible = re.sub(r"<think\b[^>]*>.*\Z", " ", visible, flags=re.I | re.S)
    match = THINK_FIELD.search(visible)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    visible = re.sub(r"\bRole\s*:.*?(?=\bThinking\s*:|\bAnswer\s*:|\Z)", " ", visible, flags=re.I | re.S)
    visible = re.sub(r"\bAnswer\s*:\s*\[\[?\d+\]?\]", " ", visible, flags=re.I)
    return re.sub(r"\s+", " ", visible).strip()


def style_metrics(text: str) -> dict[str, float]:
    words = WORD.findall(text)
    sentences = [part for part in re.split(r"[.!?\n]+", text) if WORD.search(part)]
    return {
        "length_chars": float(len(text)),
        "mlu": float(len(words) / max(1, len(sentences))),
        "zlib_bytes": float(len(zlib.compress(text.encode("utf-8")))),
        "mental_state_per_100": float(100 * len(MENTAL_STATE.findall(text)) / max(1, len(words))),
    }


def percentile(values: list[float], proportion: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * proportion
    lower = math.floor(position); upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def task_stratified_bootstrap(records: list[dict[str, Any]], iterations: int, seed: int) -> tuple[float, float]:
    groups: dict[str, list[float]] = {}
    for task in sorted({str(row["task"]) for row in records}):
        groups[task] = [float(bool(row.get("is_correct"))) for row in records if row["task"] == task]
    rng = random.Random(seed)
    draws = []
    for _ in range(iterations):
        sampled = [rng.choice(values) for values in groups.values() for _ in values]
        draws.append(statistics.fmean(sampled))
    return percentile(draws, 0.025), percentile(draws, 0.975)


def paired_task_bootstrap(pairs: list[tuple[str, float]], iterations: int, seed: int) -> tuple[float, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for task, difference in pairs:
        groups[task].append(difference)
    rng = random.Random(seed)
    draws = []
    for _ in range(iterations):
        sampled = [rng.choice(values) for values in groups.values() for _ in values]
        draws.append(statistics.fmean(sampled))
    return percentile(draws, 0.025), percentile(draws, 0.975)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    # Contrast exports mix accuracy rows with metric-specific style rows.  Keep
    # the first-seen column order while retaining fields introduced later.
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(name for name in row if name not in fieldnames)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def render_table(rows: list[dict[str, Any]], path: Path) -> None:
    by_condition: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_condition[row["run_id"]][row["persona"]] = row
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\caption{\rev{Prompt-specificity ablation. Accuracies are percentages; gaps are NT minus L3. Native thinking uses each model's recommended sampler. Private native-reasoning text is excluded from style metrics.}}",
        r"\label{tab:prompt_ablation}", r"\scriptsize", r"\setlength{\tabcolsep}{3.1pt}",
        r"\begin{tabular}{lllrrrrrrrrrr}", r"\toprule",
        r"Model & Configuration & Prompt & NT & L1 & L2 & L3 & Acc. gap & Invalid & Length gap & MLU gap & Complexity gap & MS-rate gap \\", r"\midrule",
    ]
    for run_id in sorted(by_condition):
        group = by_condition[run_id]
        nt, l3 = group["level-0"], group["level-3"]
        config = "Native thinking" if nt["thinking"] else "Direct"
        prompt = "Full profile" if nt["prompt_mode"] == "full_profile" else "Role only"
        values = [
            latex_escape(nt["display_model"]), config, prompt,
            *(f"{group[p]['accuracy'] * 100:.2f}" for p in PERSONAS),
            f"{(nt['accuracy'] - l3['accuracy']) * 100:.2f}",
            str(sum(int(group[p]["invalid_count"]) for p in PERSONAS)),
            f"{nt['length_chars'] - l3['length_chars']:.1f}", f"{nt['mlu'] - l3['mlu']:.2f}",
            f"{nt['zlib_bytes'] - l3['zlib_bytes']:.1f}", f"{nt['mental_state_per_100'] - l3['mental_state_per_100']:.2f}",
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def make_figure(rows: list[dict[str, Any]], ci_rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {(r["run_id"], r["persona"]): r for r in rows}
    ci = {(r["run_id"], r["persona"]): r for r in ci_rows}

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(5.2, 5.0), sharex=True, sharey=True)
        models = ("Qwen3.6-27B", "Gemma-4-26B-A4B-it")
        for col, model in enumerate(models):
            for row_index, thinking in enumerate((False, True)):
                ax = axes[row_index, col]
                for mode in ("full_profile", "roleplay_only"):
                    style = LINE_STYLE[mode]
                    run_ids = {r["run_id"] for r in rows if r["display_model"] == model and r["thinking"] == thinking and r["prompt_mode"] == mode}
                    if len(run_ids) != 1:
                        raise InclusionError(f"figure cell {model}/{thinking}/{mode} has {len(run_ids)} runs")
                    run_id = next(iter(run_ids))
                    values = [lookup[(run_id, p)]["accuracy"] * 100 for p in PERSONAS]
                    lows = [ci[(run_id, p)]["ci_low"] * 100 for p in PERSONAS]
                    highs = [ci[(run_id, p)]["ci_high"] * 100 for p in PERSONAS]
                    yerr = [[value - low for value, low in zip(values, lows)], [high - value for value, high in zip(values, highs)]]
                    ax.errorbar(
                        range(4), values, yerr=yerr,
                        label=style["label"], color=style["color"], linestyle=style["linestyle"],
                        marker=style["marker"], markersize=4.5, lw=1.4, capsize=2.5,
                        markerfacecolor=style["color"], markeredgecolor=style["color"],
                    )
                # Two-line title: condition on top, model name on its own line, matching
                # the "Accuracy by Category and ASD Level (Model)" convention.
                condition = "Native thinking" if thinking else "Direct"
                ax.set_title(f"{condition}\n{model}", fontsize=9, fontweight="bold", color="#222222")
                ax.set_ylim(0, 100)
                ax.set_yticks(range(0, 101, 20))
                ax.set_xticks(range(4), [PERSONA_LABELS[p] for p in PERSONAS])
                ax.grid(True, axis="y")
                for spine in ("left", "bottom"):
                    ax.spines[spine].set_color("#000000")
                if col == 0:
                    ax.set_ylabel("Accuracy (%)")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.legend(
            handles, labels, loc="lower center",
            ncol=2, frameon=False, fontsize=8.5, title_fontsize=8.5,
            bbox_to_anchor=(0.5, 0.0),
        )
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


def make_gap_figure(rows: list[dict[str, Any]], prompt_mode: str, path: Path) -> None:
    """Render paired NT-minus-L3 gaps for direct and native inference.

    Each figure fixes prompt specificity (full profile or role only).  The four
    x-axis conditions therefore compare model family and inference configuration
    without conflating either with prompt mode.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_panels = (
        ("accuracy", "NT–L3 Accuracy Gap", "Percentage points", 100.0),
        ("length_chars", "Thinking Length Gap", "Characters", 1.0),
        ("mlu", "Thinking MLU Gap", "MLU", 1.0),
        ("zlib_bytes", "Thinking Complexity Gap", "zlib bytes", 1.0),
    )
    conditions = (
        ("Qwen3.6-27B", False, "Qwen\nDirect", "#4D4D4D"),
        ("Qwen3.6-27B", True, "Qwen\nNative", "#0072B2"),
        ("Gemma-4-26B-A4B-it", False, "Gemma\nDirect", "#999999"),
        ("Gemma-4-26B-A4B-it", True, "Gemma\nNative", "#E69F00"),
    )
    lookup = {
        (row["display_model"], bool(row["thinking"]), row.get("metric", "accuracy")): row
        for row in rows
        if row["prompt_mode"] == prompt_mode
        and row["contrast"] in {"NT_minus_L3", "NT_minus_L3_style"}
    }
    expected = {(model, thinking, metric) for model, thinking, _, _ in conditions for metric, _, _, _ in metric_panels}
    if set(lookup) != expected:
        missing = expected - set(lookup)
        extra = set(lookup) - expected
        raise InclusionError(f"gap figure inputs are incomplete; missing={missing}, extra={extra}")

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.4), constrained_layout=True)
        fig.get_layout_engine().set(w_pad=0.08, h_pad=0.10, wspace=0.10, hspace=0.16)
        positions = list(range(len(conditions)))
        for axis, (metric, title, ylabel, scale) in zip(axes.flat, metric_panels):
            selected = [lookup[(model, thinking, metric)] for model, thinking, _, _ in conditions]
            estimates = [scale * float(row["estimate"]) for row in selected]
            lows = [scale * float(row["ci_low"]) for row in selected]
            highs = [scale * float(row["ci_high"]) for row in selected]
            axis.bar(positions, estimates, color=[color for _, _, _, color in conditions], width=0.62, alpha=0.85, zorder=2)
            axis.errorbar(
                positions, estimates,
                yerr=[[estimate - low for estimate, low in zip(estimates, lows)],
                      [high - estimate for estimate, high in zip(estimates, highs)]],
                fmt="none", ecolor="#1a1a1a", elinewidth=0.9, capsize=2.6, capthick=0.9, zorder=3,
            )
            axis.axhline(0, color="#666666", lw=0.7, zorder=1)
            axis.set_title(title, fontsize=8.5, pad=3)
            axis.set_ylabel(ylabel, fontsize=7.8, labelpad=2)
            axis.set_xticks(positions, [label for _, _, label, _ in conditions])
            axis.tick_params(axis="x", labelsize=7.0)
            axis.grid(True, axis="y")

        fig.suptitle("Full profile" if prompt_mode == "full_profile" else "Role only", fontsize=9.5)
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "analysis/prompt_ablation_runs.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/analysis/prompt_ablation")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    config = read_json(args.config)
    all_runs = []
    failures = []
    for spec in config["runs"]:
        try:
            manifest, records = validate_run(spec, ROOT, config["expected_records_per_run"], set(config.get("expected_tasks", [])) or config["expected_task_count"])
            all_runs.append((spec, manifest, records))
        except (OSError, InclusionError, json.JSONDecodeError) as exc:
            failures.append(str(exc))
    if failures:
        raise SystemExit("Selected-run validation failed; no paper artifacts were written:\n- " + "\n- ".join(failures))

    canonical_items = {
        (row["dataset"], row["item_id"])
        for row in all_runs[0][2]
        if row["persona"] == PERSONAS[0]
    }
    for spec, _, records in all_runs[1:]:
        items = {(row["dataset"], row["item_id"]) for row in records if row["persona"] == PERSONAS[0]}
        if items != canonical_items:
            raise InclusionError(f"{spec['id']}: dataset/item keys differ from the other selected runs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    ci_rows: list[dict[str, Any]] = []
    for run_index, (spec, _, records) in enumerate(all_runs):
        for record in records:
            record["_style"] = style_metrics(visible_thinking(record))
        for persona in PERSONAS:
            selected = [r for r in records if r["persona"] == persona]
            style = {name: statistics.fmean(r["_style"][name] for r in selected) for name in ("length_chars", "mlu", "zlib_bytes", "mental_state_per_100")}
            row = {"run_id": spec["id"], "display_model": spec["display_model"], "model": spec["model"], "thinking": spec["thinking"], "prompt_mode": spec["prompt_mode"], "persona": persona, "n": len(selected), "correct_count": sum(bool(r.get("is_correct")) for r in selected), "invalid_count": sum(r.get("predicted_answer") in (None, "None") for r in selected), **style}
            row["accuracy"] = row["correct_count"] / row["n"]
            aggregate.append(row)
            low, high = task_stratified_bootstrap(selected, args.bootstrap_resamples, args.seed + 100 * run_index + PERSONAS.index(persona))
            ci_rows.append({"run_id": spec["id"], "persona": persona, "resamples": args.bootstrap_resamples, "seed": args.seed + 100 * run_index + PERSONAS.index(persona), "ci_low": low, "ci_high": high})
            for task in sorted({r["task"] for r in selected}):
                subset = [r for r in selected if r["task"] == task]
                task_rows.append({"run_id": spec["id"], "persona": persona, "task": task, "n": len(subset), "correct_count": sum(bool(r.get("is_correct")) for r in subset), "accuracy": statistics.fmean(bool(r.get("is_correct")) for r in subset)})

    contrasts = []
    for spec, _, records in all_runs:
        nt = {(r["dataset"], r["item_id"]): r for r in records if r["persona"] == "level-0"}
        l3 = {(r["dataset"], r["item_id"]): r for r in records if r["persona"] == "level-3"}
        if nt.keys() != l3.keys():
            raise InclusionError(f"NT/L3 paired keys differ for {spec['id']}")
        pairs = [(str(nt[key]["task"]), int(bool(nt[key].get("is_correct"))) - int(bool(l3[key].get("is_correct")))) for key in nt]
        point = statistics.fmean(value for _, value in pairs)
        low, high = paired_task_bootstrap(pairs, args.bootstrap_resamples, args.seed + 1000 + len(contrasts))
        contrasts.append({"contrast": "NT_minus_L3", "display_model": spec["display_model"], "thinking": spec["thinking"], "prompt_mode": spec["prompt_mode"], "persona": "all", "n": len(pairs), "estimate": point, "ci_low": low, "ci_high": high})
        for metric in ("length_chars", "mlu", "zlib_bytes"):
            style_pairs = [
                (str(nt[key]["task"]), float(nt[key]["_style"][metric]) - float(l3[key]["_style"][metric]))
                for key in nt
            ]
            point = statistics.fmean(value for _, value in style_pairs)
            low, high = paired_task_bootstrap(style_pairs, args.bootstrap_resamples, args.seed + 1000 + len(contrasts))
            contrasts.append({"contrast": "NT_minus_L3_style", "metric": metric, "display_model": spec["display_model"], "thinking": spec["thinking"], "prompt_mode": spec["prompt_mode"], "persona": "all", "n": len(style_pairs), "estimate": point, "ci_low": low, "ci_high": high})
    for model in sorted({r["display_model"] for r in aggregate}):
        for thinking in (False, True):
            full_spec = next(s for s, _, _ in all_runs if s["display_model"] == model and s["thinking"] == thinking and s["prompt_mode"] == "full_profile")
            role_spec = next(s for s, _, _ in all_runs if s["display_model"] == model and s["thinking"] == thinking and s["prompt_mode"] == "roleplay_only")
            full_records = next(rs for s, _, rs in all_runs if s is full_spec); role_records = next(rs for s, _, rs in all_runs if s is role_spec)
            for persona in PERSONAS:
                f = {r["key"]: r for r in full_records if r["persona"] == persona}; q = {r["key"]: r for r in role_records if r["persona"] == persona}
                if f.keys() != q.keys(): raise InclusionError(f"paired keys differ for {model}/{thinking}/{persona}")
                pairs = [(str(f[key]["task"]), int(bool(f[key].get("is_correct"))) - int(bool(q[key].get("is_correct")))) for key in f]
                point = statistics.fmean(value for _, value in pairs)
                low, high = paired_task_bootstrap(pairs, args.bootstrap_resamples, args.seed + 2000 + len(contrasts))
                contrasts.append({"contrast": "full_profile_minus_role_only", "display_model": model, "thinking": thinking, "prompt_mode": "paired", "persona": persona, "n": len(f), "estimate": point, "ci_low": low, "ci_high": high})

    write_csv(args.output_dir / "aggregate_metrics.csv", aggregate)
    write_csv(args.output_dir / "task_accuracy.csv", task_rows)
    write_csv(args.output_dir / "bootstrap_intervals.csv", ci_rows)
    write_csv(args.output_dir / "paired_contrasts.csv", contrasts)
    write_csv(args.output_dir / "style_gap_contrasts.csv", [row for row in contrasts if row["contrast"] == "NT_minus_L3_style"])
    render_table(aggregate, ROOT / "paper/Tables/prompt_ablation.tex")
    make_figure(aggregate, ci_rows, ROOT / "paper/Figures/prompt_ablation_accuracy.pdf")
    make_gap_figure(contrasts, "full_profile", args.output_dir / "prompt_ablation_gaps_full_profile")
    make_gap_figure(contrasts, "roleplay_only", args.output_dir / "prompt_ablation_gaps_roleplay_only")
    hashes = {spec["id"]: hashlib.sha256((ROOT / spec["path"] / "responses.jsonl").read_bytes()).hexdigest() for spec, _, _ in all_runs}
    manifest = {"status": "complete", "analysis_id": config["analysis_id"], "created_at": datetime.now(timezone.utc).isoformat(), "inclusion_config": str(args.config.relative_to(ROOT)), "run_response_sha256": hashes, "bootstrap_resamples": args.bootstrap_resamples, "bootstrap_seed": args.seed, "private_reasoning_policy": "Qwen <think> blocks stripped; Gemma raw_reasoning ignored; only visible Thinking field analyzed", "style_metrics": {"length_chars": "Unicode character count", "mlu": "regex word count divided by nonempty sentence units", "zlib_bytes": "RFC 1950 zlib-compressed UTF-8 byte length", "mental_state_per_100": "fixed regex lexicon matches per 100 regex words"}, "outputs": ["aggregate_metrics.csv", "task_accuracy.csv", "bootstrap_intervals.csv", "paired_contrasts.csv", "style_gap_contrasts.csv", "paper/Tables/prompt_ablation.tex", "paper/Figures/prompt_ablation_accuracy.pdf", "prompt_ablation_gaps_full_profile.pdf", "prompt_ablation_gaps_full_profile.png", "prompt_ablation_gaps_roleplay_only.pdf", "prompt_ablation_gaps_roleplay_only.png"]}
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote validated ablation artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
