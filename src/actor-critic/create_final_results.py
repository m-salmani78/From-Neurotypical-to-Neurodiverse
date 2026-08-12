import argparse
import json, os, sys, glob
import numpy as np
import pandas as pd

# ───────────────────────── CLI ─────────────────────────
parser = argparse.ArgumentParser(description="Create final results combining step1 (non-revised) and step3 (revised)")
parser.add_argument("--model_name", required=True)
parser.add_argument("--results_dir", default="../../results/actor_critic_results")
args = parser.parse_args()

MODEL_NAME = args.model_name
RESULT_DIR = os.path.join(args.results_dir, MODEL_NAME)
LEVELS = [0, 1, 2, 3]

# ──────────────────── pull helper once ───────────────────
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # utils/
)
from utils.prompts import extract_answer


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


# ──────────────── Create final results for each level and counter ────────────────
for level in LEVELS:
    print(f"Processing level {level}...")
    
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
        print(f"  Processing counter {counter}...")
        
        step1_file = f"{RESULT_DIR}/level-{level}/step1/{MODEL_NAME}_level-{level}_step-1_{counter}.jsonl"
        critic_file = f"{RESULT_DIR}/level-{level}/step2-parsed/{MODEL_NAME}_level-{level}_critic_{counter}.jsonl"
        step3_file = f"{RESULT_DIR}/level-{level}/step3/{MODEL_NAME}_level-{level}_step-3_{counter}.jsonl"

        if not os.path.exists(critic_file):
            print(f"    Skipping - no critic file found: {critic_file}")
            continue

        # 1️⃣ which items were revised?
        revised_mask = {}
        with open(critic_file, encoding="utf-8") as f:
            for i, line in enumerate(f):
                should_revise = json.loads(line).get("revise", "").lower().strip()
                assert should_revise in ("yes", "no"), f"Invalid revise value: {should_revise}"
                revised_mask[i] = should_revise == "yes"

        # 2️⃣ load predictions from step1 and step3
        if not os.path.exists(step1_file):
            print(f"    Skipping - no step1 file found: {step1_file}")
            continue
            
        step1_preds = {}
        with open(step1_file, encoding="utf-8") as f:
            for i, line in enumerate(f):
                step1_preds[i] = json.loads(line)

        step3_preds = {}
        if os.path.exists(step3_file):
            with open(step3_file, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    step3_preds[i] = json.loads(line)

        # map revision-id ⭢ original-id
        rev2orig, orig2rev = {}, {}
        rev_id = 0
        for orig_id, was_revised in revised_mask.items():
            if was_revised:
                rev2orig[rev_id] = orig_id
                orig2rev[orig_id] = rev_id
                rev_id += 1

        # 3️⃣ Create final results
        final_results = []
        
        for item_id in sorted(step1_preds.keys()):
            is_revised = revised_mask.get(item_id, False)
            
            if is_revised and item_id in orig2rev:
                # Use revised prediction from step3
                rev_id = orig2rev[item_id]
                if rev_id in step3_preds:
                    final_result = {
                        "predict": step3_preds[rev_id]["predict"],
                        "is_revised": True,
                        "original_id": item_id,
                        "revision_id": rev_id
                    }
                else:
                    # Fallback to step1 if step3 doesn't have the revision
                    print(f"    Warning: No step3 prediction for revised item {item_id}, using step1")
                    final_result = {
                        "predict": step1_preds[item_id]["predict"],
                        "is_revised": False,
                        "original_id": item_id,
                    }
            else:
                # Use original prediction from step1
                final_result = {
                    "predict": step1_preds[item_id]["predict"],
                    "is_revised": False,
                    "original_id": item_id,
                }
            
            final_result["label"] = step1_preds[item_id]["label"].strip()
            final_results.append(final_result)

        # 4️⃣ Save final results
        final_dir = f"{RESULT_DIR}/level-{level}/step-final/"
        os.makedirs(final_dir, exist_ok=True)
        
        final_file = f"{final_dir}{MODEL_NAME}_level-{level}_final_{counter}.jsonl"
        
        with open(final_file, 'w', encoding='utf-8') as f:
            for result in final_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"    ✓ Saved {len(final_results)} items to {final_file}")

print("✓ Final results creation completed!")