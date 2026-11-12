"""Run the isolated criterion evaluation without changing the existing CLI."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version as package_version
import json
from pathlib import Path
import re
import sys

import pandas as pd

from .data import prepare_data
from .evaluation import evaluate

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[2]
WORKFLOW = "sustainability_criterion_evaluation_v1"


def load_configuration(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    runs = config.get("configurations", [])
    if not runs:
        raise ValueError("Configuration must define at least one model configuration.")
    identifiers = set()
    for run in runs:
        for key in ("id", "label", "provider", "method", "input_dir"):
            if not isinstance(run.get(key), str) or not run[key].strip():
                raise ValueError(f"Configuration has missing or invalid {key}: {run}")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", run["id"]):
            raise ValueError(f"Unsafe configuration ID: {run['id']}")
        if run["id"] in identifiers:
            raise ValueError(f"Duplicate configuration ID: {run['id']}")
        if run["method"] not in {"full_context", "rag"}:
            raise ValueError(f"Unsupported method: {run['method']}")
        identifiers.add(run["id"])
    comparisons = set()
    by_id = {run["id"]: run for run in runs}
    for pair in config.get("comparisons", []):
        if not re.fullmatch(r"[a-z][a-z0-9_]*", pair.get("id", "")):
            raise ValueError("Comparison IDs must use lowercase letters, numbers and underscores.")
        if pair["id"] in comparisons:
            raise ValueError(f"Duplicate comparison ID: {pair['id']}")
        comparisons.add(pair["id"])
        for key, method in (("full_context", "full_context"), ("rag", "rag")):
            if pair.get(key) not in by_id or by_id[pair[key]]["method"] != method:
                raise ValueError(f"Invalid {key} reference in comparison {pair['id']}")
        if not isinstance(pair.get("label"), str) or not pair["label"].strip():
            raise ValueError(f"Missing comparison label: {pair['id']}")
    config.setdefault("comparisons", [])
    return config


def coverage_table(aligned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (config_id, question_id), group in aligned.groupby(
        ["config_id", "question_id"], sort=False
    ):
        eligible = group["eligible"].astype(bool)
        rows.append({
            "config_id": config_id,
            "question_id": question_id,
            "total_startups": len(group),
            "two_expert_ratings": int(group[["expert_1", "expert_2"]].notna().all(axis=1).sum()),
            "valid_model_ratings": int(group["model_rating"].notna().sum()),
            "eligible_startups": int(eligible.sum()),
            "excluded_startups": int((~eligible).sum()),
        })
    return pd.DataFrame(rows)


def input_inventory(repo_root: Path, questions: Path, experts: Path, config_path: Path,
                    config: dict) -> list[dict]:
    paths = {questions, config_path, *PACKAGE_DIR.glob("*.py"), *PACKAGE_DIR.glob("*.json")}
    paths.update(p for p in experts.glob("*.xlsx") if not p.name.startswith("~$"))
    for run in config["configurations"]:
        paths.update((repo_root / run["input_dir"]).glob("*.json"))
    inventory = []
    for path in sorted({p.resolve() for p in paths}):
        try:
            label = str(path.relative_to(repo_root))
        except ValueError:
            label = str(path)
        inventory.append({"path": label, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return inventory


def check_output_directory(output: Path) -> None:
    """Only reuse destinations previously created by this workflow."""
    if not output.exists():
        return
    if not output.is_dir():
        raise ValueError(f"Output destination is not a directory: {output}")
    if not any(output.iterdir()):
        return
    manifest_path = output / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Refusing to overwrite an existing unrelated directory: {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("workflow") != WORKFLOW:
        raise ValueError(f"Output directory belongs to a different workflow: {output}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE_DIR / "config.json")
    parser.add_argument("--expert-dir", "--experts-dir", dest="expert_dir", type=Path,
                        default=REPO_ROOT / "private/expert_rating")
    parser.add_argument("--questions", "--rubric", dest="questions", type=Path,
                        default=REPO_ROOT / "src/sustainability/config/sustainability_forecasting_questions_category.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "private/evaluation")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.bootstrap < 2:
        parser.error("--bootstrap must be at least 2")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_configuration(args.config)
    output = args.output_dir.resolve()
    check_output_directory(output)
    # Load plotting before writing output so a missing dependency fails cleanly.
    from .plotting import generate_figures
    from .reporting import generate_report

    manifest = {
        "workflow": WORKFLOW,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "configuration": config,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "confidence_level": 0.95,
        "minimum_ci_startups": 5,
        "python": sys.version,
        "packages": {name: package_version(name) for name in ("numpy", "pandas", "openpyxl", "matplotlib")},
        "inputs": input_inventory(REPO_ROOT, args.questions, args.expert_dir, args.config, config),
    }
    print("Reading criterion ratings and validating source data...", flush=True)
    data = prepare_data(REPO_ROOT, args.questions, args.expert_dir, config)
    if len(data["catalog"]) != 37:
        raise ValueError("This workflow requires the 37-criterion rubric.")
    print(f"Evaluating {len(config['configurations'])} configurations with {args.bootstrap} bootstrap resamples...", flush=True)
    results, comparisons = evaluate(data["aligned_records"], data["catalog"], config,
                                    bootstrap=args.bootstrap, seed=args.seed)
    expected = 37 * len(config["configurations"])
    if len(results) != expected or len(comparisons) != 37 * len(config["comparisons"]):
        raise RuntimeError("Incomplete criterion result grid.")

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tables = {
        "criterion_catalog.csv": data["catalog"],
        "expert_ratings.csv": data["expert_ratings"],
        "model_ratings.csv": data["model_ratings"],
        "aligned_records.csv": data["aligned_records"],
        "data_diagnostics.csv": data["diagnostics"],
        "forecast_selections.csv": data["selections"],
        "coverage_diagnostics.csv": coverage_table(data["aligned_records"]),
        "excluded_records.csv": data["aligned_records"].loc[~data["aligned_records"]["eligible"].astype(bool)],
        "criterion_results.csv": results,
        "full_context_rag_comparisons.csv": comparisons,
    }
    for pair in config["comparisons"]:
        tables[f"{pair['id']}_ranked_deltas.csv"] = comparisons.loc[
            comparisons["comparison_id"] == pair["id"]
        ].sort_values(["rank", "question_id"], na_position="last")
    for filename, frame in tables.items():
        frame.to_csv(output / filename, index=False)
    print("Rendering figures and generating the report...", flush=True)
    figures = generate_figures(data["catalog"], results, comparisons, config, output)
    report = generate_report(data["catalog"], results, comparisons, tables["coverage_diagnostics.csv"],
                             data["diagnostics"], config, output, args.bootstrap, args.seed)
    manifest.update({
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "result_rows": len(results),
        "comparison_rows": len(comparisons),
        "artifacts": list(tables) + [str(path.relative_to(output)) for path in figures] + [report.name],
    })
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(results)} criterion results, {len(comparisons)} comparisons and {len(figures)} figure files to {output}", flush=True)


if __name__ == "__main__":
    main()
