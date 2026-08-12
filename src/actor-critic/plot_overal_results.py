import argparse
import json, os, sys, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ───────────────────────── CLI ─────────────────────────
parser = argparse.ArgumentParser(description="Plot accuracy before/after actor-critic (mean ± SE)")
parser.add_argument("--model_name", required=True)
parser.add_argument("--results_dir", default="../../results/actor_critic_results")
args = parser.parse_args()

MODEL_NAME = args.model_name
RESULT_DIR = os.path.join(args.results_dir, MODEL_NAME)
LEVELS = [0, 1, 2, 3]
CATEGORY_ORDER = ["Neurotypical", "Level 1", "Level 2", "Level 3"]

# ──────────────────── pull helper once ───────────────────
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # utils/
)
from utils.prompts import extract_answer


def acc(series) -> float:
    """helper: returns percent accuracy"""
    return series.mean() * 100.0


def get_all_counters(level, step, model):
    """Return all counter file paths for a given level and step."""
    step_dir = f"{RESULT_DIR}/level-{level}/step{step}/"
    pattern = f"{step_dir}{model}_level-{level}_step-{step}_*.jsonl"
    return sorted(glob.glob(pattern))


def get_all_critic_counters(level, model):
    step_dir = f"{RESULT_DIR}/level-{level}/step2-parsed/"
    pattern = f"{step_dir}{model}_level-{level}_critic_*.jsonl"
    return sorted(glob.glob(pattern))


def get_counter_from_filename(filename):
    # Extracts the counter from the filename
    return filename.split("_")[-1].split(".")[0]


# ──────────────── Aggregate results across counters ────────────────
all_rows = []
for level in LEVELS:
    # Find all available counters for this level (using step1 as reference)
    step1_files = get_all_counters(level, 1, MODEL_NAME)
    critic_files = get_all_critic_counters(level, MODEL_NAME)
    step3_files = get_all_counters(level, 3, MODEL_NAME)

    # Map counter to files
    counters = set()
    for f in step1_files:
        counters.add(get_counter_from_filename(f))
    for f in critic_files:
        counters.add(get_counter_from_filename(f))
    for f in step3_files:
        counters.add(get_counter_from_filename(f))

    for counter in sorted(counters):
        step1_file = f"{RESULT_DIR}/level-{level}/step1/{MODEL_NAME}_level-{level}_step-1_{counter}.jsonl"
        critic_file = f"{RESULT_DIR}/level-{level}/step2-parsed/{MODEL_NAME}_level-{level}_critic_{counter}.jsonl"
        step3_file = f"{RESULT_DIR}/level-{level}/step3/{MODEL_NAME}_level-{level}_step-3_{counter}.jsonl"

        if not os.path.exists(critic_file):
            continue

        # 1️⃣ which items were revised?
        revised_mask = {}
        with open(critic_file, encoding="utf-8") as f:
            for i, line in enumerate(f):
                should_revise = json.loads(line).get("revise", "").lower().strip()
                assert should_revise in ("yes", "no")
                revised_mask[i] = should_revise == "yes"

        # 2️⃣ load predictions + ground truth
        if not os.path.exists(step1_file):
            continue
        step1_preds = {
            i: json.loads(l) for i, l in enumerate(open(step1_file, encoding="utf-8"))
        }
        step3_preds = {}
        if os.path.exists(step3_file):
            step3_preds = {
                i: json.loads(l)
                for i, l in enumerate(open(step3_file, encoding="utf-8"))
            }

        # map revision-id ⭢ original-id
        rev2orig, orig2rev = {}, {}
        rev_id = 0
        for orig_id, was_revised in revised_mask.items():
            if was_revised:
                rev2orig[rev_id] = orig_id
                orig2rev[orig_id] = rev_id
                rev_id += 1

        # 3️⃣ build one row per item
        for item_id, gd in step1_preds.items():
            try:
                first_ans, _ = extract_answer(gd["predict"])
            except:
                first_ans = 0
            revised_ans = None
            is_revised = revised_mask.get(item_id, False)
            if is_revised:
                try:
                    revised_ans, _ = extract_answer(
                        step3_preds[orig2rev[item_id]]["predict"]
                    )
                except Exception:
                    pass  # leave as None

            final_ans = revised_ans if revised_ans is not None else first_ans
            label = gd["label"].strip()
            all_rows.append(
                dict(
                    level=level,
                    initial_corr=first_ans == label,
                    final_corr=final_ans == label,
                    is_revised=is_revised,
                    counter=counter,
                )
            )

# Nothing to plot?
if not all_rows:
    sys.exit("No data found – aborting.")

df = pd.DataFrame(all_rows)

# ──────────────── Compute mean and standard error ────────────────
means = {"initial": [], "final": []}
std_errs = {"initial": [], "final": []}

for lvl in LEVELS:
    lvl_df = df[df.level == lvl]
    # Group by counter, compute accuracy per counter
    if lvl_df.empty:
        means["initial"].append(np.nan)
        means["final"].append(np.nan)
        std_errs["initial"].append(0)
        std_errs["final"].append(0)
        continue
    grouped = lvl_df.groupby("counter")
    initial_accs = grouped["initial_corr"].mean() * 100.0
    final_accs = grouped["final_corr"].mean() * 100.0
    means["initial"].append(initial_accs.mean())
    means["final"].append(final_accs.mean())
    std_errs["initial"].append(initial_accs.sem())
    std_errs["final"].append(final_accs.sem())

# ───────────────────────── PLOT ──────────────────────────
x = np.arange(len(CATEGORY_ORDER))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 8), facecolor="white")
ax.set_facecolor("white")

blue = "#3498db"
red = "#e57373"

# Plot bars with error bars
a1 = ax.bar(
    x - width / 2,
    means["initial"],
    width,
    yerr=std_errs["initial"],
    label="Initial Accuracy",
    color=blue,
    capsize=8,
)
a2 = ax.bar(
    x + width / 2,
    means["final"],
    width,
    yerr=std_errs["final"],
    label="Post-Actor-Critic Accuracy",
    color=red,
    capsize=8,
)

# value labels
for rect in a1:
    height = rect.get_height()
    ax.text(
        rect.get_x() + rect.get_width() / 2.0,
        height + 0.6,
        f"{height:.1f}%",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=blue,
    )
for rect in a2:
    height = rect.get_height()
    ax.text(
        rect.get_x() + rect.get_width() / 2.0,
        height + 0.6,
        f"{height:.1f}%",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color=red,
    )

ax.set_title(
    f"LLM Response Accuracy Before and After Actor-Critic ({MODEL_NAME})",
    fontsize=14,
    fontweight="bold",
    pad=18,
)
ax.set_ylabel("Total Accuracy (%)", fontsize=14)
ax.set_xlabel("Cognitive Perspective", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(CATEGORY_ORDER, fontsize=13)

ax.set_xlim(-0.6, len(CATEGORY_ORDER) - 0.4)
ax.set_ylim(20, 80)
ax.set_yticks(range(0, 101, 10))
ax.set_axisbelow(True)
ax.grid(True, axis="y", color="#dddddd", linewidth=1)

for spine in ax.spines.values():
    spine.set_color("black")

legend = ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False, fontsize=14
)
legend.get_texts()[0].set_color(blue)
legend.get_texts()[1].set_color(red)

plt.tight_layout()
plt.subplots_adjust(bottom=0.14)
output_dir = os.path.join(RESULT_DIR, "figures")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"{MODEL_NAME}_accuracy_by_level.png")
plt.savefig(output_path, dpi=300)
plt.show()
print(f"✓ saved figure to {MODEL_NAME}_accuracy_by_level.png")
