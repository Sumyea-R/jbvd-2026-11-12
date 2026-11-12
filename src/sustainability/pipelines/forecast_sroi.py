import argparse
import json
import os
import re
import time
from pathlib import Path
import glob
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import openai
from openai import OpenAI
from ..prompts.sroi import PROMPTS
from ..utils.paths import find_repo_root, get_outputs_root, get_questions_path, get_results_root
from ..utils.run_log import utc_timestamp, write_run_manifest
from ..validation.output_validation import (
    atomic_write_json,
    build_topic_criteria_len,
    validate_sroi_payload,
)


def slugify_model_name(model: str) -> str:
    """Convert model ids like openai/gpt-5.2 into safe folder/log name fragments."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_").lower()
    return slug


def parse_args() -> argparse.Namespace:
    """Expose model and path configuration so forecasts for different providers stay separated."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="openai/gpt-5.2",
        help="OpenRouter model id to use for forecasting.",
    )
    parser.add_argument(
        "--output-subdir",
        default=None,
        help="Optional subdirectory under private/outputs/. Defaults to forecast_<model-slug>.",
    )
    parser.add_argument(
        "--logs-subdir",
        default=None,
        help="Optional subdirectory under private/results/logs/. Defaults to <model-slug>.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit for number of business plans to process.",
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        default=None,
        help=(
            "Optional explicit markdown files to process. Paths may be absolute, "
            "relative to the current working directory, or relative to the repo root."
        ),
    )
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=60.0,
        help="Seconds to sleep after rate limits and successful files. Defaults to the batch-run delay.",
    )
    parser.add_argument(
        "--latest-version-only",
        action="store_true",
        help="Keep only the highest _V# file per startup before forecasting.",
    )
    return parser.parse_args()


def parse_startup_and_version(stem: str) -> Optional[Tuple[str, int]]:
    match = re.match(r"^(?P<startup>.+)_V(?P<version>\d+)$", stem)
    if not match:
        return None
    return match.group("startup"), int(match.group("version"))


def select_latest_version_files(bp_files: List[str]) -> List[str]:
    """Choose the highest version per startup; keep unmatched filenames unchanged."""
    grouped: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    passthrough: List[str] = []

    for bp_file in bp_files:
        stem = Path(bp_file).stem
        parsed = parse_startup_and_version(stem)
        if parsed is None:
            passthrough.append(bp_file)
            continue
        startup, version = parsed
        grouped[startup].append((version, bp_file))

    selected = list(passthrough)
    for _, versioned in grouped.items():
        versioned.sort(key=lambda item: item[0])
        selected.append(versioned[-1][1])
    return sorted(selected, key=lambda path: Path(path).name.lower())


def resolve_input_files(
    input_files: Optional[List[str]],
    repo_root: Path,
    data_folder: Path,
) -> List[str]:
    if not input_files:
        return sorted(glob.glob(str(data_folder / "*.md")))

    resolved_files: List[str] = []
    seen: set[str] = set()
    for raw_path in input_files:
        candidate = Path(raw_path).expanduser()
        candidates = [candidate] if candidate.is_absolute() else [Path.cwd() / candidate, repo_root / candidate]

        resolved_path: Optional[Path] = None
        for option in candidates:
            option = option.resolve()
            if option.exists():
                resolved_path = option
                break
        if resolved_path is None:
            raise FileNotFoundError(f"Input markdown file not found: {raw_path}")
        if resolved_path.suffix.lower() != ".md":
            raise ValueError(f"Input file must be markdown (.md): {resolved_path}")

        resolved_str = str(resolved_path)
        if resolved_str not in seen:
            resolved_files.append(resolved_str)
            seen.add(resolved_str)

    return resolved_files


def sleep_if_needed(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped


def cleanup_common_json_issues(text: str) -> str:
    # Repair lines like: \"...\", that appear as standalone list items.
    return re.sub(r'(?m)^(\s*)\\\"(.*?)\\\"(\s*,?\s*)$', r'\1"\2"\3', text)


def looks_like_topic_payload(obj: Any) -> bool:
    if not isinstance(obj, list) or not obj:
        return False
    return all(isinstance(item, dict) and "Topic" in item for item in obj)


def parse_first_json_fragment(text: str) -> Any:
    decoder = json.JSONDecoder()
    for start_char in ("[", "{"):
        idx = text.find(start_char)
        if idx == -1:
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            return obj
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("No JSON fragment found", text, 0)


def parse_topic_payload_from_text(text: str) -> List[Dict[str, Any]]:
    text = cleanup_common_json_issues(strip_code_fences(text))

    try:
        obj = json.loads(text, strict=False)
        if isinstance(obj, dict) and "Topic" in obj:
            return [obj]
        if looks_like_topic_payload(obj):
            return obj
    except json.JSONDecodeError:
        pass

    blocks = re.findall(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    block_results: List[Dict[str, Any]] = []
    for block in blocks:
        block = cleanup_common_json_issues(block)
        try:
            obj = json.loads(block, strict=False)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "Topic" in obj:
            block_results.append(obj)
        elif looks_like_topic_payload(obj):
            block_results.extend(obj)
    if block_results:
        return block_results

    decoder = json.JSONDecoder(strict=False)
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "Topic" in obj:
            block_results.append(obj)
    if block_results:
        return block_results

    obj = parse_first_json_fragment(text)
    if isinstance(obj, dict) and "Topic" in obj:
        return [obj]
    if looks_like_topic_payload(obj):
        return obj
    raise ValueError("Could not parse a valid topic payload.")


def main():
    args = parse_args()
    if args.rate_limit_delay < 0:
        raise ValueError("--rate-limit-delay must be non-negative.")

    repo_root = find_repo_root(Path(__file__).resolve())
    data_folder = repo_root / "private" / "plans"
    model_slug = slugify_model_name(args.model)
    output_subdir = args.output_subdir or f"forecast_{model_slug}"
    logs_subdir = args.logs_subdir or model_slug
    output_folder = get_outputs_root(repo_root) / output_subdir
    logs_dir = get_results_root(repo_root) / "logs" / logs_subdir

    bp_files = resolve_input_files(args.input_files, repo_root, data_folder)

    if not bp_files:
        source = "--input-files" if args.input_files else str(data_folder)
        print(f"No business plan files found from {source}")
        return

    if args.latest_version_only:
        bp_files = select_latest_version_files(bp_files)

    questions_path = get_questions_path(repo_root, "sustainability_forecasting_questions.json")
    if not questions_path.exists():
        raise FileNotFoundError(f"Missing questions file: {questions_path}")

    with open(questions_path, "r", encoding="utf-8") as f:
        topic_and_assessment_criteria = json.load(f)
    topic_criteria_len = build_topic_criteria_len(topic_and_assessment_criteria)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set in your environment")

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    output_folder.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "timestamp_utc": utc_timestamp(),
        "pipeline": "forecast_sroi",
        "model": args.model,
        "input_folder": str(data_folder),
        "input_files": bp_files if args.input_files else None,
        "output_folder": str(output_folder),
        "logs_folder": str(logs_dir),
        "latest_version_only": args.latest_version_only,
        "rate_limit_delay": args.rate_limit_delay,
        "total_files": len(bp_files),
        "processed": 0,
        "succeeded": 0,
        "failed": 0,
        "skipped_existing": 0,
        "records": [],
    }

    selected_files = bp_files[: args.max_files] if args.max_files is not None else bp_files
    run_manifest["total_files"] = len(selected_files)

    for bp_file in selected_files:
        run_manifest["processed"] += 1
        base_name = os.path.splitext(os.path.basename(bp_file))[0]
        json_output_path = output_folder / f"{base_name}.json"

        if json_output_path.exists():
            print(f"Output file {json_output_path} already exists. Skipping {bp_file}.")
            run_manifest["skipped_existing"] += 1
            run_manifest["records"].append(
                {"file": os.path.basename(bp_file), "status": "skipped_existing"}
            )
            continue

        print(f"Processing file: {bp_file}")
        with open(bp_file, "r", encoding="utf-8") as f:
            business_plan = f.read()

        prompt = PROMPTS["sroi_with_confidence"].format(
            topic_and_assessment_criteria=topic_and_assessment_criteria,
            business_plan=business_plan
        )

        max_attempts = 3
        write_success = False
        failure_reason = ""
        for attempt in range(1, max_attempts + 1):
            try:
                response = client.responses.create(
                    model=args.model,
                    instructions="You are an expert evaluator in sustainability assessment based on the startup's business plan.",
                    input=prompt,
                    temperature=0
                )
                responses = response.output_text
                parsed_payload = parse_topic_payload_from_text(responses)
                is_valid, reason = validate_sroi_payload(parsed_payload, topic_criteria_len)
                if not is_valid:
                    print(f"Validation failed for {base_name} (attempt {attempt}/{max_attempts}): {reason}")
                    failure_reason = reason
                else:
                    atomic_write_json(json_output_path, parsed_payload)
                    write_success = True
                    run_manifest["records"].append(
                        {
                            "file": os.path.basename(bp_file),
                            "status": "succeeded",
                            "attempts": attempt,
                        }
                    )
                    break
            except openai.RateLimitError:
                print("Rate limit reached. Input token more than 30000.")
                sleep_if_needed(args.rate_limit_delay)
                failure_reason = "rate_limit"
            except Exception as e:
                print(f"Failed to parse/validate output for {base_name} (attempt {attempt}/{max_attempts}): {e}")
                failure_reason = str(e)

            if attempt < max_attempts:
                time.sleep(5 * attempt)

        if not write_success:
            print(f"Failed to generate complete output for {base_name} after {max_attempts} attempts.")
            run_manifest["failed"] += 1
            run_manifest["records"].append(
                {
                    "file": os.path.basename(bp_file),
                    "status": "failed",
                    "attempts": max_attempts,
                    "reason": failure_reason,
                }
            )
            continue
        run_manifest["succeeded"] += 1

        sleep_if_needed(args.rate_limit_delay)

    manifest_path = write_run_manifest(logs_dir, output_subdir, run_manifest)
    print(f"Run manifest written: {manifest_path}")


if __name__ == "__main__":
    main()
