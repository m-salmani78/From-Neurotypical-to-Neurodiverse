import json
import tempfile
import unittest
from pathlib import Path

from src.analyze_alignment_ablation import (
    InclusionError,
    STYLE_GAP_METRICS,
    add_ci,
    apply_holm,
    compact_summary_rows,
    main,
    paired_values,
    validate_condition,
    write_latex_summary_table,
)
from src.run_model_baseline import remove_transport_failures_for_resume


ROOT = Path(__file__).resolve().parents[1]


class AlignmentRunTests(unittest.TestCase):
    def make_condition(self, root: Path):
        run = root / "condition"
        run.mkdir()
        revision = "a" * 40
        preflight = root / "tokenizer_preflight.json"
        preflight.write_text(json.dumps({
            "bos_count": 1,
            "tokenizer_revision": revision,
            "chat_template_sha256": "b" * 64,
        }))
        manifest = {
            "status": "complete",
            "model": "princeton-nlp/test",
            "model_revision": revision,
            "tokenizer_revision": revision,
            "tokenizer_preflight": str(preflight),
            "alignment_method": "SFT",
            "prompt_mode": "full_profile",
            "completed_record_count": 4,
            "transport_error_count": 0,
            "max_model_len": 8192,
            "decoding": {
                "temperature": 0.2,
                "top_p": 0.5,
                "seed": 42,
                "max_tokens": 512,
                "enable_thinking": False,
            },
        }
        (run / "manifest.json").write_text(json.dumps(manifest))
        (root / "job-1.json").write_text(json.dumps({
            "status": "complete",
            "exit_status": 0,
            "model": "princeton-nlp/test",
            "model_revision": revision,
        }))
        records = []
        for index, persona in enumerate(("level-0", "level-1", "level-2", "level-3")):
            records.append({
                "key": f"{persona}|data.json|item-1",
                "model": "princeton-nlp/test",
                "persona": persona,
                "prompt_mode": "full_profile",
                "dataset": "data.json",
                "item_id": "item-1",
                "task": "Task",
                "raw_response": "Thinking: brief. Answer: [[1]]",
                "predicted_answer": "A",
                "is_correct": index < 2,
                "error": None,
            })
        (run / "responses.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
        spec = {"method": "SFT", "prompt_mode": "full_profile", "path": "condition"}
        model = {"method": "SFT", "model": "princeton-nlp/test", "group": "SFT-only"}
        config = {
            "expected_records_per_condition": 4,
            "expected_personas": ["level-0", "level-1", "level-2", "level-3"],
            "expected_tasks": ["Task"],
            "decoding": {
                "temperature": 0.2,
                "top_p": 0.5,
                "seed": 42,
                "max_tokens": 512,
                "max_model_len": 8192,
                "enable_thinking": False,
            },
        }
        return run, preflight, records, spec, model, config

    def test_valid_condition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, _, spec, model, config = self.make_condition(root)
            manifest, records = validate_condition(spec, model, config, root)
            self.assertEqual(manifest["model_revision"], "a" * 40)
            self.assertEqual(len(records), 4)

    def test_rejects_duplicate_transport_and_bad_bos(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, preflight, records, spec, model, config = self.make_condition(root)
            records[-1]["key"] = records[0]["key"]
            (run / "responses.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
            with self.assertRaisesRegex(InclusionError, "duplicate"):
                validate_condition(spec, model, config, root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, _, _, spec, model, config = self.make_condition(root)
            manifest = json.loads((run / "manifest.json").read_text())
            manifest["transport_error_count"] = 1
            (run / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(InclusionError, "transport_error_count"):
                validate_condition(spec, model, config, root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, preflight, _, spec, model, config = self.make_condition(root)
            payload = json.loads(preflight.read_text())
            payload["bos_count"] = 2
            preflight.write_text(json.dumps(payload))
            with self.assertRaisesRegex(InclusionError, "BOS"):
                validate_condition(spec, model, config, root)

    def test_paired_values_require_identical_items(self):
        left = [
            {"persona": "level-0", "dataset": "d", "item_id": "1", "task": "t", "is_correct": True}
        ]
        right = [
            {"persona": "level-0", "dataset": "d", "item_id": "1", "task": "t", "is_correct": False}
        ]
        self.assertEqual(paired_values(left, right, "level-0"), [("t", 1.0)])
        right[0]["item_id"] = "2"
        with self.assertRaisesRegex(InclusionError, "paired item keys"):
            paired_values(left, right, "level-0")

    def test_resume_retries_only_transport_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            responses = root / "responses.jsonl"
            errors = root / "errors.jsonl"
            rows = [
                {"key": "ok", "error": None},
                {"key": "parse", "error": "parse failed: bad format"},
                {"key": "retry", "error": "request failed: timeout"},
            ]
            responses.write_text("".join(json.dumps(row) + "\n" for row in rows))
            errors.write_text("".join(json.dumps(row) + "\n" for row in rows[1:]))
            self.assertEqual(remove_transport_failures_for_resume(responses, errors), 1)
            kept = [json.loads(line)["key"] for line in responses.read_text().splitlines()]
            self.assertEqual(kept, ["ok", "parse"])
            error_keys = [json.loads(line)["key"] for line in errors.read_text().splitlines()]
            self.assertEqual(error_keys, ["parse"])
            audit = [json.loads(line) for line in (root / "retry_audit.jsonl").read_text().splitlines()]
            self.assertEqual([row["key"] for row in audit], ["retry"])
            self.assertIn("removed_for_transport_retry_at", audit[0])

    def test_paired_test_and_holm_are_reproducible(self):
        first = []
        second = []
        pairs = [("task", 1.0), ("task", 1.0), ("task", 0.0)]
        add_ci(first, contrast="method_minus_SFT", method="DPO", prompt_mode="full_profile", persona="level-0", pairs=pairs, iterations=100, seed=42)
        add_ci(second, contrast="method_minus_SFT", method="DPO", prompt_mode="full_profile", persona="level-0", pairs=pairs, iterations=100, seed=42)
        self.assertEqual(first, second)
        first.append({**first[0], "method": "KTO", "p_value": 0.2})
        apply_holm(first)
        self.assertGreaterEqual(first[0]["p_holm"], first[0]["p_value"])
        self.assertGreaterEqual(first[1]["p_holm"], first[1]["p_value"])

    def test_summary_rows_and_latex_rounding_match_source_metrics(self):
        methods = ("SFT", "DPO", "ORPO", "KTO")
        modes = ("full_profile", "roleplay_only")
        personas = ("level-0", "level-1", "level-2", "level-3")
        aggregate, intervals, contrasts = [], [], []
        for method_index, method in enumerate(methods):
            for mode_index, mode in enumerate(modes):
                for persona_index, persona in enumerate(personas):
                    accuracy = 0.10 + method_index / 100 + mode_index / 1000 + persona_index / 10000
                    aggregate.append({
                        "method": method, "prompt_mode": mode, "persona": persona,
                        "accuracy": accuracy, "invalid_count": persona_index, "n": 460,
                    })
                    intervals.append({
                        "method": method, "prompt_mode": mode, "persona": persona,
                        "ci_low": accuracy - 0.01, "ci_high": accuracy + 0.01,
                    })
                contrasts.append({
                    "contrast": "NT_minus_L3", "method": method, "prompt_mode": mode,
                    "estimate": 0.02, "ci_low": 0.01, "ci_high": 0.03,
                })
        rows = compact_summary_rows(aggregate, intervals, contrasts)
        self.assertEqual(len(rows), 8)
        row = next(row for row in rows if row["method"] == "KTO" and row["prompt_mode"] == "roleplay_only")
        self.assertAlmostEqual(row["nt_accuracy"], 0.131)
        self.assertEqual(row["invalid_count"], 6)
        self.assertEqual(row["n"], 1840)
        with tempfile.TemporaryDirectory() as directory:
            table = Path(directory) / "table.tex"
            write_latex_summary_table(table, rows)
            text = table.read_text()
            self.assertIn("13.1 [12.1, 14.1]", text)
            self.assertIn("\\begin{table*}", text)

    def test_main_rejects_missing_condition_without_writing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads((ROOT / "analysis/alignment_ablation_runs.json").read_text())
            for condition in config["conditions"]:
                condition["path"] = f"missing/{condition['method']}/{condition['prompt_mode']}"
            config_path = root / "runs.json"
            config_path.write_text(json.dumps(config))
            output = root / "output"
            with self.assertRaisesRegex(SystemExit, "KTO/full_profile"):
                main(["--config", str(config_path), "--output-dir", str(output), "--bootstrap-resamples", "10"])
            self.assertFalse(output.exists())


class AlignmentLauncherTests(unittest.TestCase):
    def test_array_contains_exact_models_and_two_direct_prompt_runs(self):
        script = (ROOT / "scripts/run_alignment_ablation.sh").read_text()
        for suffix in ("SFT", "SFT-DPO", "SFT-ORPO", "SFT-KTO"):
            self.assertIn(f'princeton-nlp/Llama-3-Base-8B-{suffix}', script)
        self.assertNotIn("Llama-3-Base-8B-SFT-SimPO", script)
        self.assertIn("#SBATCH --array=0-3%4", script)
        self.assertIn('--output-dir "${FULL_OUTPUT}"', script)
        self.assertIn('--roleplay-only --output-dir "${ROLE_OUTPUT}"', script)
        self.assertNotIn("--enable-thinking", script)
        self.assertIn("bos_count != 1", script)
        self.assertIn("--revision", script)
        self.assertIn('vllm_runtime != "0.23.0"', script)
        self.assertIn("len(vllm_distributions) != 1", script)
        self.assertIn("Refusing mixed-revision resume", script)
        self.assertIn("Refusing mixed-configuration resume", script)
        self.assertIn('ids = ids["input_ids"]', script)
        self.assertIn("use_fast=False", script)
        self.assertIn("--tokenizer-mode slow", script)

    def test_inclusion_manifest_has_eight_unique_cells_without_simpo(self):
        config = json.loads((ROOT / "analysis/alignment_ablation_runs.json").read_text())
        cells = {(row["method"], row["prompt_mode"]) for row in config["conditions"]}
        self.assertEqual(len(config["conditions"]), 8)
        self.assertEqual(len(cells), 8)
        self.assertNotIn("SimPO", {row["method"] for row in config["models"]})
        self.assertEqual(config["expected_records_per_condition"], 1840)
        self.assertFalse(config["decoding"]["enable_thinking"])

    def test_plotted_style_gaps_are_the_three_selected_metrics(self):
        self.assertEqual(
            tuple(metric for metric, _ in STYLE_GAP_METRICS),
            ("length_chars", "mlu", "zlib_bytes"),
        )


if __name__ == "__main__":
    unittest.main()
