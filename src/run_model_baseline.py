#!/usr/bin/env python3
"""Run the paper's four baseline persona prompts through a vLLM server."""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.prompts import format_prompt  # noqa: E402


PERSONAS = ("level-0", "level-1", "level-2", "level-3")
PERSONA_NAMES = {
    "level-0": "Neurotypical",
    "level-1": "ASD Level 1",
    "level-2": "ASD Level 2",
    "level-3": "ASD Level 3",
}
INDEX_TO_OPTION = {index: chr(ord("A") + index - 1) for index in range(1, 9)}
ANSWER_PATTERN = re.compile(r"Answer\s*:\s*\[\[(\d+)\]\]", re.IGNORECASE)
FALLBACK_ANSWER_PATTERN = re.compile(r"\[\[(\d+)\]\]")


@dataclass(frozen=True)
class DatasetItem:
    dataset: str
    dataset_index: int
    item_id: str
    task: str
    story: str
    question: str
    options: list[str]
    gold_answer: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def unwrap_english(value: Any, field: str) -> Any:
    """Normalize either {en: ...} or the repository's [{en: ...}, ...] shape.

    A dataset record is the experiment unit. Eighteen Desire records contain a
    second aligned question/options/answer set; the baseline's 460-record
    protocol consistently uses the first set.
    """
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{field} cannot be empty")
        value = value[0]
    if not isinstance(value, dict) or "en" not in value:
        raise ValueError(f"{field} does not contain an English value")
    return value["en"]


def normalize_answer(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            raise ValueError("answer cannot be empty")
        value = value[0]
    answer = str(value).strip().upper()
    if answer not in set(INDEX_TO_OPTION.values()):
        raise ValueError(f"unsupported gold answer: {answer!r}")
    return answer


def normalize_item(raw: dict[str, Any], dataset: str, index: int) -> DatasetItem:
    story = unwrap_english(raw.get("story"), "story")
    question = unwrap_english(raw.get("question"), "question")
    options = unwrap_english(raw.get("options"), "options")
    if not isinstance(story, str) or not isinstance(question, str):
        raise ValueError("story and question must be strings")
    if not isinstance(options, list) or len(options) not in (2, 4, 8):
        raise ValueError(f"options must have 2, 4, or 8 entries; got {len(options)}")
    if not all(isinstance(option, str) and option.strip() for option in options):
        raise ValueError("every option must be a non-empty string")

    gold_answer = normalize_answer(raw.get("answer"))
    if ord(gold_answer) - ord("A") >= len(options):
        raise ValueError(f"gold answer {gold_answer} is outside {len(options)} options")

    item_id = str(raw.get("id") or f"{Path(dataset).stem}-{index:04d}")
    return DatasetItem(
        dataset=dataset,
        dataset_index=index,
        item_id=item_id,
        task=str(raw.get("task") or Path(dataset).stem),
        story=story,
        question=question,
        options=[option.strip() for option in options],
        gold_answer=gold_answer,
    )


def load_datasets(data_dir: Path, limit: int = 0) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    paths = sorted(data_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no JSON datasets found under {data_dir}")

    for path in paths:
        with path.open(encoding="utf-8") as handle:
            raw_items = json.load(handle)
        if not isinstance(raw_items, list):
            raise ValueError(f"{path} must contain a JSON array")
        normalized = [normalize_item(raw, path.name, index) for index, raw in enumerate(raw_items)]
        if limit > 0:
            normalized = normalized[:limit]
        items.extend(normalized)
    return items


def parse_answer(text: str, option_count: int) -> tuple[int, str]:
    matches = ANSWER_PATTERN.findall(text) or FALLBACK_ANSWER_PATTERN.findall(text)
    if not matches:
        raise ValueError("no answer in [[index]] format")
    index = int(matches[-1])
    if index == 0:
        return index, "None"
    if index < 1 or index > option_count:
        raise ValueError(f"answer index {index} is outside 1..{option_count}")
    return index, INDEX_TO_OPTION[index]


def make_key(persona: str, item: DatasetItem) -> str:
    return f"{persona}|{item.dataset}|{item.item_id}"


def read_completed_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = record.get("key")
            if not key:
                raise ValueError(f"missing key in {path}:{line_number}")
            if key in keys:
                raise ValueError(f"duplicate completed key in {path}: {key}")
            keys.add(key)
    return keys


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def atomic_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def remove_transport_failures_for_resume(
    responses_path: Path, errors_path: Path
) -> int:
    """Remove only transient request failures so --resume retries their keys.

    Parse/format failures are valid model outputs under the paper protocol and
    therefore remain in the response file as incorrect answers.
    """
    records = load_response_records(responses_path)
    transient_keys = {
        record.get("key")
        for record in records
        if str(record.get("error") or "").startswith("request failed:")
    }
    transient_keys.discard(None)
    if not transient_keys:
        return 0
    audit_path = responses_path.parent / "retry_audit.jsonl"
    for record in records:
        if record.get("key") in transient_keys:
            append_jsonl(
                audit_path,
                {**record, "removed_for_transport_retry_at": utc_now()},
            )
    atomic_jsonl(
        responses_path,
        (record for record in records if record.get("key") not in transient_keys),
    )
    if errors_path.exists():
        error_records = load_response_records(errors_path)
        atomic_jsonl(
            errors_path,
            (record for record in error_records if record.get("key") not in transient_keys),
        )
    return len(transient_keys)


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("vllm", "torch", "transformers", "openai"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def command_output(command: Sequence[str]) -> str | None:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def build_manifest(args: argparse.Namespace, item_count: int) -> dict[str, Any]:
    return {
        "status": "running",
        "started_at": utc_now(),
        "model": args.model,
        "model_revision": os.environ.get("MODEL_REVISION") or None,
        "tokenizer_revision": os.environ.get("TOKENIZER_REVISION") or None,
        "tokenizer_preflight": os.environ.get("TOKENIZER_PREFLIGHT") or None,
        "alignment_method": os.environ.get("ALIGNMENT_METHOD") or None,
        "base_url": args.base_url,
        "max_model_len": int(os.environ["MAX_MODEL_LEN"]) if os.environ.get("MAX_MODEL_LEN") else None,
        "vllm_reasoning_parser": os.environ.get("VLLM_REASONING_PARSER") or None,
        "text_only_server": os.environ.get("VLLM_TEXT_ONLY", "0") == "1",
        "dataset_item_count": item_count,
        "persona_count": len(PERSONAS),
        "expected_request_count": item_count * len(PERSONAS),
        "personas": PERSONA_NAMES,
        "prompt_mode": "roleplay_only" if args.roleplay_only else "full_profile",
        "excluded_prompt_fields": (
            ["speech", "cognitive_style", "response_rules"] if args.roleplay_only else []
        ),
        "decoding": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "presence_penalty": args.presence_penalty,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "enable_thinking": args.enable_thinking,
        },
        "concurrency": args.concurrency,
        "request_retries": args.request_retries,
        "format_retries": 1,
        "limit_per_dataset": args.limit or None,
        "packages": package_versions(),
        "python": sys.version,
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "gpu": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "note": (
            "Persona labels are heuristic prompt conditions and are not validated clinical "
            "representations of autistic people or support needs."
        ),
    }


def response_text(response: Any) -> tuple[str, str | None, str | None]:
    choice = response.choices[0]
    message = choice.message
    content = message.content
    reasoning = getattr(message, "reasoning", None) or getattr(
        message, "reasoning_content", None
    )
    if content is None:
        content = reasoning
    if content is None:
        raise ValueError("server returned neither content nor reasoning")
    return (
        str(content),
        str(reasoning) if reasoning is not None else None,
        getattr(choice, "finish_reason", None),
    )


async def validate_served_model(client: Any, expected_model: str) -> None:
    """Fail before evaluation if the endpoint does not serve the requested model."""
    response = await client.models.list()
    served = {str(model.id) for model in response.data}
    if expected_model not in served:
        raise RuntimeError(
            f"vLLM model mismatch: requested {expected_model!r}, served {sorted(served)!r}"
        )


async def request_completion(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    seed: int,
    max_tokens: int,
    request_retries: int,
    option_count: int,
    enable_thinking: bool = False,
    top_k: int | None = None,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
) -> tuple[str, int | None, str | None, str | None, int, str | None, str | None]:
    """Return text, parsed fields, retry/error information, and native reasoning."""
    retry_count = 0
    current_messages = list(messages)
    final_text = ""
    final_reasoning: str | None = None
    finish_reason: str | None = None

    for format_attempt in range(2):
        response = None
        for transport_attempt in range(request_retries + 1):
            try:
                extra_body: dict[str, Any] = {
                    "chat_template_kwargs": {"enable_thinking": enable_thinking},
                }
                if top_k is not None:
                    extra_body["top_k"] = top_k
                if min_p is not None:
                    extra_body["min_p"] = min_p
                if repetition_penalty is not None:
                    extra_body["repetition_penalty"] = repetition_penalty
                request_options: dict[str, Any] = {
                    "model": model,
                    "messages": current_messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "seed": seed,
                    "max_tokens": max_tokens,
                    "extra_body": extra_body,
                }
                if presence_penalty is not None:
                    request_options["presence_penalty"] = presence_penalty
                response = await client.chat.completions.create(
                    **request_options,
                )
                break
            except Exception as exc:  # OpenAI exposes several transport exception types.
                if transport_attempt >= request_retries:
                    return (
                        final_text,
                        None,
                        None,
                        finish_reason,
                        retry_count,
                        f"request failed: {exc}",
                        final_reasoning,
                    )
                retry_count += 1
                await asyncio.sleep(min(2**transport_attempt, 8))

        try:
            final_text, final_reasoning, finish_reason = response_text(response)
            answer_index, predicted_answer = parse_answer(final_text, option_count)
            return (
                final_text,
                answer_index,
                predicted_answer,
                finish_reason,
                retry_count,
                None,
                final_reasoning,
            )
        except ValueError as exc:
            if format_attempt == 1:
                return (
                    final_text,
                    None,
                    None,
                    finish_reason,
                    retry_count,
                    f"parse failed: {exc}",
                    final_reasoning,
                )
            retry_count += 1
            current_messages = [
                *messages,
                {"role": "assistant", "content": final_text},
                {
                    "role": "user",
                    "content": (
                        "Return the same response in the required format and end with exactly "
                        "Answer: [[<option index>]]."
                    ),
                },
            ]

    raise AssertionError("unreachable")


async def evaluate_one(
    client: Any,
    semaphore: asyncio.Semaphore,
    write_lock: asyncio.Lock,
    item: DatasetItem,
    persona: str,
    args: argparse.Namespace,
    responses_path: Path,
    errors_path: Path,
) -> None:
    messages = format_prompt(
        persona,
        item.story,
        {"en": item.question},
        {"en": item.options},
        roleplay_only=args.roleplay_only,
    )
    started = time.monotonic()
    async with semaphore:
        text, index, predicted, finish_reason, retries, error, reasoning = await request_completion(
            client,
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            max_tokens=args.max_tokens,
            request_retries=args.request_retries,
            option_count=len(item.options),
            enable_thinking=args.enable_thinking,
            top_k=args.top_k,
            min_p=args.min_p,
            presence_penalty=args.presence_penalty,
            repetition_penalty=args.repetition_penalty,
        )
    latency = time.monotonic() - started
    key = make_key(persona, item)
    record = {
        "key": key,
        "model": args.model,
        "persona": persona,
        "persona_name": PERSONA_NAMES[persona],
        "prompt_mode": "roleplay_only" if args.roleplay_only else "full_profile",
        **asdict(item),
        "prompt": messages,
        "raw_response": text,
        "raw_reasoning": reasoning,
        "answer_index": index,
        "predicted_answer": predicted,
        "is_correct": predicted == item.gold_answer if predicted is not None else False,
        "finish_reason": finish_reason,
        "latency_seconds": round(latency, 4),
        "retry_count": retries,
        "error": error,
        "completed_at": utc_now(),
    }

    async with write_lock:
        append_jsonl(responses_path, record)
        if error:
            append_jsonl(
                errors_path,
                {
                    "key": key,
                    "persona": persona,
                    "dataset": item.dataset,
                    "item_id": item.item_id,
                    "error": error,
                    "completed_at": record["completed_at"],
                },
            )


def load_response_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def aggregate_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["persona"], record["task"])].append(record)

    rows: list[dict[str, Any]] = []
    for (persona, task), values in sorted(groups.items()):
        valid = [value for value in values if value.get("predicted_answer") is not None]
        correct = sum(bool(value.get("is_correct")) for value in values)
        rows.append(
            {
                "persona": persona,
                "persona_name": PERSONA_NAMES[persona],
                "task": task,
                "n_total": len(values),
                "n_valid": len(valid),
                "n_invalid": len(values) - len(valid),
                "n_correct": correct,
                "accuracy": correct / len(values) if values else 0.0,
                "valid_accuracy": (
                    sum(bool(value.get("is_correct")) for value in valid) / len(valid)
                    if valid
                    else 0.0
                ),
            }
        )
    return rows


def write_summaries(output_dir: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = aggregate_records(records)
    csv_path = output_dir / "accuracy_by_task.csv"
    fieldnames = [
        "persona",
        "persona_name",
        "task",
        "n_total",
        "n_valid",
        "n_invalid",
        "n_correct",
        "accuracy",
        "valid_accuracy",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    overall: dict[str, dict[str, Any]] = {}
    for persona in PERSONAS:
        values = [record for record in records if record["persona"] == persona]
        valid = [value for value in values if value.get("predicted_answer") is not None]
        correct = sum(bool(value.get("is_correct")) for value in values)
        overall[persona] = {
            "persona_name": PERSONA_NAMES[persona],
            "n_total": len(values),
            "n_valid": len(valid),
            "n_invalid": len(values) - len(valid),
            "n_correct": correct,
            "accuracy": correct / len(values) if values else 0.0,
            "valid_accuracy": (
                sum(bool(value.get("is_correct")) for value in valid) / len(valid)
                if valid
                else 0.0
            ),
        }

    summary = {
        "generated_at": utc_now(),
        "record_count": len(records),
        "overall": overall,
        "by_task": rows,
    }
    atomic_json(output_dir / "accuracy_summary.json", summary)
    return summary


async def run(args: argparse.Namespace, client: Any | None = None) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_path = output_dir / "responses.jsonl"
    errors_path = output_dir / "errors.jsonl"
    manifest_path = output_dir / "manifest.json"

    items = load_datasets(args.data_dir.resolve(), args.limit)
    manifest = build_manifest(args, len(items))
    atomic_json(manifest_path, manifest)

    if responses_path.exists() and responses_path.stat().st_size > 0 and not args.resume:
        raise FileExistsError(
            f"{responses_path} already contains results; use --resume or a new --output-dir"
        )
    if args.resume:
        removed = remove_transport_failures_for_resume(responses_path, errors_path)
        if removed:
            print(f"Removed {removed} transient failure record(s) for retry.")
    completed = read_completed_keys(responses_path) if args.resume else set()
    work = [
        (item, persona)
        for item in items
        for persona in PERSONAS
        if make_key(persona, item) not in completed
    ]
    print(
        f"Loaded {len(items)} items; {len(completed)} completed keys; "
        f"submitting {len(work)} requests."
    )

    validate_endpoint = client is None
    if client is None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("openai is not installed; run scripts/setup_qwen36_env.sh") from exc
        client = AsyncOpenAI(
            base_url=args.base_url,
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            timeout=args.request_timeout,
        )
    if validate_endpoint:
        try:
            await validate_served_model(client, args.model)
        except Exception as exc:
            manifest.update(
                {
                    "status": "failed",
                    "completed_at": utc_now(),
                    "failure": f"model endpoint validation failed: {exc}",
                }
            )
            atomic_json(manifest_path, manifest)
            raise

    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    tasks = [
        asyncio.create_task(
            evaluate_one(
                client,
                semaphore,
                write_lock,
                item,
                persona,
                args,
                responses_path,
                errors_path,
            )
        )
        for item, persona in work
    ]
    if tasks:
        for completed_count, future in enumerate(asyncio.as_completed(tasks), start=1):
            await future
            if completed_count % 50 == 0 or completed_count == len(tasks):
                print(f"Completed {completed_count}/{len(tasks)} new requests.", flush=True)

    records = load_response_records(responses_path)
    keys = [record["key"] for record in records]
    if len(keys) != len(set(keys)):
        raise RuntimeError("responses.jsonl contains duplicate experiment keys")
    summary = write_summaries(output_dir, records)

    transport_error_count = sum(
        str(record.get("error") or "").startswith("request failed:") for record in records
    )
    manifest.update(
        {
            "status": "failed" if transport_error_count else "complete",
            "completed_at": utc_now(),
            "completed_record_count": len(records),
            "error_record_count": sum(bool(record.get("error")) for record in records),
            "transport_error_count": transport_error_count,
        }
    )
    atomic_json(manifest_path, manifest)
    if transport_error_count:
        raise RuntimeError(
            f"evaluation had {transport_error_count} transport/server failures; "
            "see errors.jsonl"
        )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "Data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.5)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--min-p", type=float)
    parser.add_argument("--presence-penalty", type=float)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable the model chat template's native thinking mode.",
    )
    parser.add_argument(
        "--roleplay-only",
        action="store_true",
        help=(
            "Omit speech, cognitive_style, and response_rules from persona prompts and "
            "use a neutral response-format instruction."
        ),
    )
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Maximum items per dataset; 0 means all.")
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")
    if args.top_k is not None and args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.min_p is not None and not 0 <= args.min_p <= 1:
        parser.error("--min-p must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
