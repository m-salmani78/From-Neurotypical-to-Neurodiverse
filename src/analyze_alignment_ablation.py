#!/usr/bin/env python3
"""Strict paired analysis for the Llama-3 preference-alignment ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from plot_style import (
    METHODS, PERSONAS, PERSONA_LABELS, METHOD_COLORS,
    TEXT_WIDTH_IN, use_paper_style, style_axis,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analyze_prompt_ablation import (  # noqa: E402
    InclusionError,
    percentile,
    style_metrics,
    visible_thinking,
)


PERSONAS = ("level-0", "level-1", "level-2", "level-3")
PERSONA_LABELS = {"level-0": "NT", "level-1": "L1", "level-2": "L2", "level-3": "L3"}
# SimPO is deliberately excluded from the final cohort: its tokenizer cannot
# be instantiated in the validated cluster environment, so no valid responses
# exist for either prompt condition. Infrastructure failures are never treated
# as incorrect model answers.
METHODS = ("SFT", "DPO", "ORPO", "KTO")
PROMPT_MODES = ("full_profile", "roleplay_only")
STYLE_GAP_METRICS = (
    ("length_chars", "Length Gap"),
    ("mlu", "MLU Gap"),
    ("zlib_bytes", "Complexity Gap"),
)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise InclusionError(f"malformed JSON at {path}:{line_number}") from exc
    return records


def validate_condition(
    spec: dict[str, Any],
    model_spec: dict[str, Any],
    config: dict[str, Any],
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = root / spec["path"]
    if not run_dir.is_dir():
        raise InclusionError(f"{spec['method']}/{spec['prompt_mode']}: missing {run_dir}")
    manifest = read_json(run_dir / "manifest.json")
    expected = config["decoding"]
    checks = {
        "status": "complete",
        "model": model_spec["model"],
        "alignment_method": spec["method"],
        "prompt_mode": spec["prompt_mode"],
        "completed_record_count": config["expected_records_per_condition"],
        "transport_error_count": 0,
        "max_model_len": expected["max_model_len"],
    }
    for field, value in checks.items():
        if manifest.get(field) != value:
            raise InclusionError(
                f"{spec['method']}/{spec['prompt_mode']}: manifest {field}="
                f"{manifest.get(field)!r}, expected {value!r}"
            )
    if not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("model_revision") or "")):
        raise InclusionError(f"{spec['method']}/{spec['prompt_mode']}: missing immutable model revision")
    job_manifests = [read_json(path) for path in sorted(run_dir.parent.glob("job-*.json"))]
    successful_jobs = [
        job for job in job_manifests
        if job.get("status") == "complete"
        and job.get("exit_status") == 0
        and job.get("model") == model_spec["model"]
        and job.get("model_revision") == manifest.get("model_revision")
    ]
    if not successful_jobs:
        raise InclusionError(
            f"{spec['method']}/{spec['prompt_mode']}: no matching zero-exit job manifest"
        )
    for field in ("temperature", "top_p", "seed", "max_tokens", "enable_thinking"):
        if manifest.get("decoding", {}).get(field) != expected[field]:
            raise InclusionError(f"{spec['method']}/{spec['prompt_mode']}: decoding {field} mismatch")

    preflight_path = Path(str(manifest.get("tokenizer_preflight") or ""))
    if not preflight_path.is_absolute():
        preflight_path = root / preflight_path
    if not preflight_path.is_file():
        raise InclusionError(f"{spec['method']}/{spec['prompt_mode']}: tokenizer preflight is missing")
    preflight = read_json(preflight_path)
    if preflight.get("bos_count") != 1:
        raise InclusionError(f"{spec['method']}/{spec['prompt_mode']}: BOS count is not one")
    if preflight.get("tokenizer_revision") != manifest.get("tokenizer_revision"):
        raise InclusionError(f"{spec['method']}/{spec['prompt_mode']}: tokenizer revision mismatch")
    if not preflight.get("chat_template_sha256"):
        raise InclusionError(f"{spec['method']}/{spec['prompt_mode']}: chat-template hash is missing")

    records = read_jsonl(run_dir / "responses.jsonl")
    expected_count = config["expected_records_per_condition"]
    if len(records) != expected_count:
        raise InclusionError(f"{spec['method']}/{spec['prompt_mode']}: {len(records)} records, expected {expected_count}")
    keys = [record.get("key") for record in records]
    if any(not key for key in keys) or len(keys) != len(set(keys)):
        raise InclusionError(f"{spec['method']}/{spec['prompt_mode']}: missing or duplicate keys")
    if {record.get("model") for record in records} != {model_spec["model"]}:
        raise InclusionError(f"{spec['method']}/{spec['prompt_mode']}: response model mismatch")
    if {record.get("prompt_mode") for record in records} != {spec["prompt_mode"]}:
        raise InclusionError(f"{spec['method']}/{spec['prompt_mode']}: response prompt-mode mismatch")
    if {record.get("persona") for record in records} != set(config["expected_personas"]):
        raise InclusionError(f"{spec['method']}/{spec['prompt_mode']}: persona set mismatch")
    if {record.get("task") for record in records} != set(config["expected_tasks"]):
        raise InclusionError(f"{spec['method']}/{spec['prompt_mode']}: task set mismatch")
    per_persona = {persona: sum(r["persona"] == persona for r in records) for persona in PERSONAS}
    if set(per_persona.values()) != {expected_count // 4}:
        raise InclusionError(f"{spec['method']}/{spec['prompt_mode']}: unbalanced personas {per_persona}")
    if any(str(record.get("error") or "").startswith("request failed:") for record in records):
        raise InclusionError(f"{spec['method']}/{spec['prompt_mode']}: transport failure in responses")
    return manifest, records


def item_map(records: list[dict[str, Any]], persona: str) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(record["dataset"]), str(record["item_id"])): record
        for record in records
        if record["persona"] == persona
    }


def paired_values(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    persona: str,
) -> list[tuple[str, float]]:
    left_map, right_map = item_map(left, persona), item_map(right, persona)
    if left_map.keys() != right_map.keys():
        raise InclusionError(f"paired item keys differ for persona {persona}")
    return [
        (
            str(left_map[key]["task"]),
            float(bool(left_map[key].get("is_correct")))
            - float(bool(right_map[key].get("is_correct"))),
        )
        for key in left_map
    ]


def stratified_bootstrap(
    groups: dict[str, list[float]], iterations: int, seed: int
) -> tuple[float, float]:
    """Task-stratified bootstrap using bounded NumPy arrays, not Python draws."""
    if not groups or any(not values for values in groups.values()):
        raise ValueError("bootstrap requires one or more observations per task")
    try:
        import numpy as np
    except ImportError:
        # The job environment always installs NumPy; retain a dependency-light
        # deterministic fallback for unit tests and external reuse.
        import random
        rng = random.Random(seed)
        draws = [
            statistics.fmean(
                rng.choice(values)
                for values in groups.values()
                for _ in values
            )
            for _ in range(iterations)
        ]
        return percentile(draws, 0.025), percentile(draws, 0.975)
    rng = np.random.default_rng(seed)
    total = np.zeros(iterations, dtype=float)
    denominator = sum(len(values) for values in groups.values())
    for task in sorted(groups):
        values = np.asarray(groups[task], dtype=float)
        indices = rng.integers(0, len(values), size=(iterations, len(values)))
        total += values[indices].sum(axis=1)
    return tuple(float(value) for value in np.quantile(total / denominator, (0.025, 0.975), method="linear"))


def task_bootstrap(records: list[dict[str, Any]], iterations: int, seed: int) -> tuple[float, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for record in records:
        groups[str(record["task"])].append(float(bool(record.get("is_correct"))))
    return stratified_bootstrap(groups, iterations, seed)


def paired_bootstrap(pairs: list[tuple[str, float]], iterations: int, seed: int) -> tuple[float, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for task, value in pairs:
        groups[task].append(value)
    return stratified_bootstrap(groups, iterations, seed)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: Any, path: Path) -> None:
    """Write a vector PDF and a 300-dpi PNG with the same figure geometry."""
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=300)


def compact_summary_rows(
    aggregate: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One machine-readable paper-table row for every method × prompt condition."""
    values = {(row["method"], row["prompt_mode"], row["persona"]): row for row in aggregate}
    cis = {(row["method"], row["prompt_mode"], row["persona"]): row for row in intervals}
    gaps = {
        (row["method"], row["prompt_mode"]): row
        for row in contrasts
        if row["contrast"] == "NT_minus_L3"
    }
    rows = []
    for method in METHODS:
        for mode in PROMPT_MODES:
            row: dict[str, Any] = {"method": method, "prompt_mode": mode}
            for persona in PERSONAS:
                metric = values[(method, mode, persona)]
                interval = cis[(method, mode, persona)]
                short = PERSONA_LABELS[persona].lower()
                row[f"{short}_accuracy"] = metric["accuracy"]
                row[f"{short}_ci_low"] = interval["ci_low"]
                row[f"{short}_ci_high"] = interval["ci_high"]
            gap = gaps[(method, mode)]
            row["nt_l3_gap"] = gap["estimate"]
            row["nt_l3_gap_ci_low"] = gap["ci_low"]
            row["nt_l3_gap_ci_high"] = gap["ci_high"]
            row["invalid_count"] = sum(
                int(values[(method, mode, persona)]["invalid_count"])
                for persona in PERSONAS
            )
            row["n"] = sum(int(values[(method, mode, persona)]["n"]) for persona in PERSONAS)
            rows.append(row)
    return rows


def format_percent_ci(value: float, low: float, high: float) -> str:
    return f"{100 * value:.1f} [{100 * low:.1f}, {100 * high:.1f}]"


def write_latex_summary_table(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a compact manuscript-ready table; CSV retains unrounded values."""
    labels = {"full_profile": "Full profile", "roleplay_only": "Role only"}
    lines = [
        "% Generated by src/analyze_alignment_ablation.py; values are percent [95\\% CI].",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Accuracy by persona for the Llama-3 preference-alignment ablation. Invalid responses are counted as incorrect.}",
        "\\label{tab:llama-alignment-ablation}",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Method & Prompt & NT & L1 & L2 & L3 & NT--L3 & Invalid \\\\",
        "\\midrule",
    ]
    for row in rows:
        cells = [row["method"], labels[row["prompt_mode"]]]
        for short in ("nt", "l1", "l2", "l3"):
            cells.append(format_percent_ci(
                float(row[f"{short}_accuracy"]),
                float(row[f"{short}_ci_low"]),
                float(row[f"{short}_ci_high"]),
            ))
        cells.append(format_percent_ci(
            float(row["nt_l3_gap"]),
            float(row["nt_l3_gap_ci_low"]),
            float(row["nt_l3_gap_ci_high"]),
        ))
        cells.append(str(row["invalid_count"]))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def add_ci(
    rows: list[dict[str, Any]],
    *,
    contrast: str,
    method: str,
    prompt_mode: str,
    persona: str,
    pairs: list[tuple[str, float]],
    iterations: int,
    seed: int,
    metric: str = "accuracy",
) -> None:
    point = statistics.fmean(value for _, value in pairs)
    low, high = paired_bootstrap(pairs, iterations, seed)
    positive = sum(value > 0 for _, value in pairs)
    negative = sum(value < 0 for _, value in pairs)
    discordant = positive + negative
    if discordant:
        tail = sum(math.comb(discordant, index) for index in range(min(positive, negative) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    else:
        p_value = 1.0
    rows.append({
        "contrast": contrast,
        "metric": metric,
        "method": method,
        "prompt_mode": prompt_mode,
        "persona": persona,
        "n": len(pairs),
        "estimate": point,
        "ci_low": low,
        "ci_high": high,
        "p_value": p_value,
        "p_holm": "",
        "bootstrap_resamples": iterations,
        "bootstrap_seed": seed,
    })


def apply_holm(rows: list[dict[str, Any]]) -> None:
    """Holm-adjust planned method comparisons within each contrast family."""
    families: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        families[(row["contrast"], row["prompt_mode"], row["persona"])].append(row)
    for family in families.values():
        ordered = sorted(family, key=lambda row: float(row["p_value"]))
        running = 0.0
        count = len(ordered)
        for rank, row in enumerate(ordered):
            adjusted = min(1.0, (count - rank) * float(row["p_value"]))
            running = max(running, adjusted)
            row["p_holm"] = running


def make_accuracy_figure(
    aggregate: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
    path: Path,
) -> None:
    """Plot each alignment checkpoint with full-profile and role-only curves."""
    import matplotlib.pyplot as plt
    use_paper_style()
 
    values = {(r["method"], r["prompt_mode"], r["persona"]): r for r in aggregate}
    cis = {(r["method"], r["prompt_mode"], r["persona"]): r for r in intervals}

    # The visual encoding follows the direct-versus-native-thinking figures:
    # prompt specificity is encoded by line style, while each panel isolates a
    # single alignment checkpoint.
    mode_styles = {
        "full_profile": {"label": "Full profile", "color": "#0072B2", "linestyle": "-", "marker": "o"},
        "roleplay_only": {"label": "Role only", "color": "#E69F00", "linestyle": "-", "marker": "s"},
    }
    # Draw the less-specific condition first so that the full-profile curve,
    # its markers, and its confidence intervals remain visible when they overlap.
    draw_order = ("roleplay_only", "full_profile")
    fig, axes = plt.subplots(2, 2, figsize=(5.0, 4.6), sharex=True, sharey=True)
    x = list(range(len(PERSONAS)))
    legend_handles: dict[str, Any] = {}
    for axis, method in zip(axes.flat, METHODS):
        for mode in draw_order:
            style = mode_styles[mode]
            ys = [100 * values[(method, mode, persona)]["accuracy"] for persona in PERSONAS]
            lows = [100 * cis[(method, mode, persona)]["ci_low"] for persona in PERSONAS]
            highs = [100 * cis[(method, mode, persona)]["ci_high"] for persona in PERSONAS]
            handle = axis.errorbar(
                x, ys,
                yerr=[[y - low for y, low in zip(ys, lows)], [high - y for y, high in zip(ys, highs)]],
                color=style["color"], linestyle=style["linestyle"], marker=style["marker"],
                markersize=4.2, lw=1.3, capsize=2.2, capthick=0.9, elinewidth=0.9,
                label=style["label"], zorder=2 if mode == "roleplay_only" else 3,
            )
            if method == METHODS[0]:
                legend_handles[mode] = handle
        axis.set_title(method, pad=4)
        axis.set_xticks(x, [PERSONA_LABELS[p] for p in PERSONAS])
        axis.set_xlim(-0.35, len(PERSONAS) - 0.65)
        axis.set_ylim(0, 100)
        style_axis(axis)

    for axis in axes[:, 0]:
        axis.set_ylabel("Accuracy (%)")
    # for axis in axes[-1, :]:
    #     axis.set_xlabel("Prompted role")
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.0, h_pad=1.3)
    fig.legend([legend_handles[mode] for mode in PROMPT_MODES],
               [mode_styles[mode]["label"] for mode in PROMPT_MODES], ncol=2,
               frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               columnspacing=1.5, handletextpad=0.5)
    save_figure(fig, path)
    plt.close(fig)


def make_gap_figure(
    contrasts: list[dict[str, Any]],
    mode: str,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt
    use_paper_style()
 
    lookup = {
        (row["method"], row["prompt_mode"], row.get("metric", "accuracy")): row
        for row in contrasts
        if row["contrast"] in {"NT_minus_L3", "NT_minus_L3_style"}
    }
 
    # constrained_layout (not tight_layout) actually reserves space for
    # suptitle + per-axis titles + wrapped y-labels instead of overlapping them.
    fig, axes = plt.subplots(
        2, 2,
        figsize=(5.0, 4.6),
        constrained_layout=True,
    )
    fig.get_layout_engine().set(w_pad=0.06, h_pad=0.06, wspace=0.08, hspace=0.10)
 
    # Short axis labels; the full metric name goes in the per-panel title only,
    # so nothing has to wrap onto a second line and collide with the title above it.
    short_ylabels = {
        "accuracy": "PP",
        "length_chars": "Chars",
        "mlu": "MLU",
        "zlib_bytes": "Bytes",
    }
 
    def draw_bars(axis, metric: str, title: str, scale: float = 1.0) -> None:
        rows = [lookup[(method, mode, metric)] for method in METHODS]
        estimates = [scale * row["estimate"] for row in rows]
        lows = [scale * row["ci_low"] for row in rows]
        highs = [scale * row["ci_high"] for row in rows]
        positions = list(range(len(METHODS)))
        # axis.bar(
        #     positions, estimates,
        #     color=[METHOD_COLORS[m] for m in METHODS],
        #     width=0.6, alpha=0.85, edgecolor="#2b2b2b", linewidth=0.6, zorder=2,
        # )
        axis.bar(
            positions, estimates,
            color=[METHOD_COLORS[m] for m in METHODS],
            width=0.6, alpha=0.85, zorder=2,
        )
        axis.errorbar(
            positions, estimates,
            yerr=[[e - low for e, low in zip(estimates, lows)],
                  [high - e for e, high in zip(estimates, highs)]],
            fmt="none", ecolor="#1a1a1a", elinewidth=0.9, capsize=2.6, capthick=0.9, zorder=3,
        )
        axis.set_title(title, fontsize=8.3, pad=3)
        axis.set_ylabel(short_ylabels[metric], fontsize=7.8, labelpad=2)
        axis.axhline(0, color="#666666", lw=0.7, zorder=1)
        axis.set_xticks(range(len(METHODS)), METHODS)
        style_axis(axis)
 
    draw_bars(axes.flat[0], "accuracy", "NT\u2013L3 Accuracy Gap", scale=100.0)
    for axis, (metric, label) in zip(axes.flat[1:], STYLE_GAP_METRICS):
        draw_bars(axis, metric, label)
 
    fig.suptitle("Full profile" if mode == "full_profile" else "Role only", fontsize=9.5)
    save_figure(fig, path)
    plt.close(fig)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "analysis/alignment_ablation_runs.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/analysis/alignment_ablation")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    config = read_json(args.config)
    models = {entry["method"]: entry for entry in config["models"]}
    if tuple(models) != METHODS:
        raise SystemExit(f"model methods/order must be {METHODS}")
    expected_cells = {(method, mode) for method in METHODS for mode in PROMPT_MODES}
    actual_cells = {(entry["method"], entry["prompt_mode"]) for entry in config["conditions"]}
    if actual_cells != expected_cells or len(config["conditions"]) != len(expected_cells):
        raise SystemExit(
            f"inclusion manifest must contain each of the {len(expected_cells)} "
            "method × prompt cells exactly once"
        )

    included: dict[tuple[str, str], tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    failures = []
    for spec in config["conditions"]:
        try:
            included[(spec["method"], spec["prompt_mode"])] = validate_condition(
                spec, models[spec["method"]], config, ROOT
            )
        except (OSError, InclusionError, json.JSONDecodeError) as exc:
            failures.append(str(exc))
    if failures:
        raise SystemExit("Alignment-run validation failed; no artifacts were written:\n- " + "\n- ".join(failures))

    canonical_items = None
    for (method, mode), (_, records) in included.items():
        keys = set(item_map(records, "level-0"))
        if canonical_items is None:
            canonical_items = keys
        elif keys != canonical_items:
            raise InclusionError(f"{method}/{mode}: dataset-item set differs from other conditions")
    for method in METHODS:
        revisions = {included[(method, mode)][0]["model_revision"] for mode in PROMPT_MODES}
        tokenizer_revisions = {included[(method, mode)][0]["tokenizer_revision"] for mode in PROMPT_MODES}
        if len(revisions) != 1 or len(tokenizer_revisions) != 1:
            raise InclusionError(f"{method}: full and role-only runs did not use identical revisions")

    aggregate = []
    task_rows = []
    intervals = []
    for method_index, method in enumerate(METHODS):
        for mode_index, mode in enumerate(PROMPT_MODES):
            _, records = included[(method, mode)]
            for record in records:
                record["_style"] = style_metrics(visible_thinking(record))
            for persona_index, persona in enumerate(PERSONAS):
                selected = [record for record in records if record["persona"] == persona]
                correct = sum(bool(record.get("is_correct")) for record in selected)
                invalid = sum(record.get("predicted_answer") in (None, "None") for record in selected)
                style = {
                    metric: statistics.fmean(record["_style"][metric] for record in selected)
                    for metric in ("length_chars", "mlu", "zlib_bytes", "mental_state_per_100")
                }
                aggregate.append({
                    "method": method,
                    "group": models[method]["group"],
                    "model": models[method]["model"],
                    "prompt_mode": mode,
                    "persona": persona,
                    "n": len(selected),
                    "correct_count": correct,
                    "invalid_count": invalid,
                    "format_compliance": (len(selected) - invalid) / len(selected),
                    "accuracy": correct / len(selected),
                    **style,
                })
                bootstrap_seed = args.seed + 1000 * method_index + 100 * mode_index + persona_index
                low, high = task_bootstrap(selected, args.bootstrap_resamples, bootstrap_seed)
                intervals.append({
                    "method": method, "prompt_mode": mode, "persona": persona,
                    "n": len(selected), "ci_low": low, "ci_high": high,
                    "bootstrap_resamples": args.bootstrap_resamples, "bootstrap_seed": bootstrap_seed,
                })
                for task in config["expected_tasks"]:
                    subset = [record for record in selected if record["task"] == task]
                    task_rows.append({
                        "method": method, "prompt_mode": mode, "persona": persona,
                        "task": task, "n": len(subset),
                        "correct_count": sum(bool(record.get("is_correct")) for record in subset),
                        "invalid_count": sum(record.get("predicted_answer") in (None, "None") for record in subset),
                        "accuracy": statistics.fmean(bool(record.get("is_correct")) for record in subset),
                    })

    contrasts: list[dict[str, Any]] = []
    gap_distributions: list[dict[str, Any]] = []
    contrast_seed = args.seed + 100_000
    for method in METHODS:
        for mode in PROMPT_MODES:
            records = included[(method, mode)][1]
            nt, l3 = item_map(records, "level-0"), item_map(records, "level-3")
            pairs = [(str(nt[key]["task"]), float(bool(nt[key].get("is_correct"))) - float(bool(l3[key].get("is_correct")))) for key in nt]
            add_ci(contrasts, contrast="NT_minus_L3", method=method, prompt_mode=mode, persona="NT-L3", pairs=pairs, iterations=args.bootstrap_resamples, seed=contrast_seed)
            contrast_seed += 1
            for metric, _ in STYLE_GAP_METRICS:
                style_pairs = []
                for key in nt:
                    difference = float(nt[key]["_style"][metric]) - float(l3[key]["_style"][metric])
                    style_pairs.append((str(nt[key]["task"]), difference))
                    gap_distributions.append({
                        "method": method,
                        "prompt_mode": mode,
                        "metric": metric,
                        "dataset": key[0],
                        "item_id": key[1],
                        "task": str(nt[key]["task"]),
                        "gap": difference,
                    })
                add_ci(
                    contrasts,
                    contrast="NT_minus_L3_style",
                    metric=metric,
                    method=method,
                    prompt_mode=mode,
                    persona="NT-L3",
                    pairs=style_pairs,
                    iterations=args.bootstrap_resamples,
                    seed=contrast_seed,
                )
                contrast_seed += 1
        for persona in PERSONAS:
            pairs = paired_values(included[(method, "full_profile")][1], included[(method, "roleplay_only")][1], persona)
            add_ci(contrasts, contrast="full_profile_minus_role_only", method=method, prompt_mode="paired", persona=persona, pairs=pairs, iterations=args.bootstrap_resamples, seed=contrast_seed)
            contrast_seed += 1
    for method in METHODS[1:]:
        for mode in PROMPT_MODES:
            for persona in PERSONAS:
                pairs = paired_values(included[(method, mode)][1], included[("SFT", mode)][1], persona)
                add_ci(contrasts, contrast="method_minus_SFT", method=method, prompt_mode=mode, persona=persona, pairs=pairs, iterations=args.bootstrap_resamples, seed=contrast_seed)
                contrast_seed += 1

        component_maps = {}
        for candidate in (method, "SFT"):
            for mode in PROMPT_MODES:
                records = included[(candidate, mode)][1]
                component_maps[(candidate, mode, "level-0")] = item_map(records, "level-0")
                component_maps[(candidate, mode, "level-3")] = item_map(records, "level-3")
        keys = canonical_items or set()
        did_pairs = []
        for key in keys:
            def score(candidate: str, mode: str, persona: str) -> float:
                return float(bool(component_maps[(candidate, mode, persona)][key].get("is_correct")))
            method_prompt = (score(method, "full_profile", "level-0") - score(method, "full_profile", "level-3")) - (score(method, "roleplay_only", "level-0") - score(method, "roleplay_only", "level-3"))
            sft_prompt = (score("SFT", "full_profile", "level-0") - score("SFT", "full_profile", "level-3")) - (score("SFT", "roleplay_only", "level-0") - score("SFT", "roleplay_only", "level-3"))
            task = str(component_maps[(method, "full_profile", "level-0")][key]["task"])
            did_pairs.append((task, method_prompt - sft_prompt))
        add_ci(contrasts, contrast="alignment_x_prompt_DiD_on_NT_L3_gap", method=method, prompt_mode="paired", persona="NT-L3", pairs=did_pairs, iterations=args.bootstrap_resamples, seed=contrast_seed)
        contrast_seed += 1

    apply_holm(contrasts)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "aggregate_metrics.csv", aggregate)
    write_csv(args.output_dir / "task_accuracy.csv", task_rows)
    write_csv(args.output_dir / "bootstrap_intervals.csv", intervals)
    write_csv(args.output_dir / "paired_contrasts.csv", contrasts)
    style_gaps = [row for row in contrasts if row["contrast"] == "NT_minus_L3_style"]
    write_csv(args.output_dir / "style_gap_contrasts.csv", style_gaps)
    write_csv(args.output_dir / "style_gap_distributions.csv", gap_distributions)
    table_rows = compact_summary_rows(aggregate, intervals, contrasts)
    write_csv(args.output_dir / "alignment_summary_table.csv", table_rows)
    write_latex_summary_table(args.output_dir / "alignment_summary_table.tex", table_rows)
    make_accuracy_figure(aggregate, intervals, args.output_dir / "alignment_accuracy_prompt_modes")
    make_gap_figure(contrasts, "full_profile", args.output_dir / "alignment_gaps_full_profile")
    make_gap_figure(contrasts, "roleplay_only", args.output_dir / "alignment_gaps_roleplay_only")

    hashes = {
        f"{method}:{mode}": hashlib.sha256((ROOT / next(spec["path"] for spec in config["conditions"] if spec["method"] == method and spec["prompt_mode"] == mode) / "responses.jsonl").read_bytes()).hexdigest()
        for method in METHODS for mode in PROMPT_MODES
    }
    record_count = sum(len(records) for _, records in included.values())
    analysis_manifest = {
        "status": "complete",
        "analysis_id": config["analysis_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "condition_count": len(included),
        "record_count": record_count,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.seed,
        "interpretation": "checkpoint-specific SFT-only versus post-SFT preference-optimization comparison; not a training-objective causal estimate",
        "actor_critic": False,
        "native_thinking": False,
        "responses_sha256": hashes,
        "outputs": [
            "aggregate_metrics.csv", "task_accuracy.csv", "bootstrap_intervals.csv",
            "paired_contrasts.csv", "style_gap_contrasts.csv", "style_gap_distributions.csv",
            "alignment_summary_table.csv", "alignment_summary_table.tex",
            "alignment_accuracy_prompt_modes.pdf", "alignment_accuracy_prompt_modes.png",
            "alignment_gaps_full_profile.pdf", "alignment_gaps_full_profile.png",
            "alignment_gaps_roleplay_only.pdf", "alignment_gaps_roleplay_only.png",
        ],
    }
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps(analysis_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Validated and analyzed {record_count:,} responses in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
