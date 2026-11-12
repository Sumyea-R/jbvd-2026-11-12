"""Check shareable exports and derive headline summaries without private data."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "results"
CONFIGS = ["gpt_full_context", "gpt_rag", "gemini_full_context", "gemini_rag"]


def check_results():
    catalog = pd.read_csv(DATA / "criterion_catalog.csv")
    results = pd.read_csv(DATA / "criterion_results.csv")
    pairs = pd.read_csv(DATA / "full_context_rag_comparisons.csv")
    expected = {f"Q{i:03d}" for i in range(1, 38)}
    if len(catalog) != 37 or set(catalog.question_id) != expected:
        raise ValueError("The catalog must contain each of the 37 criteria once.")
    if len(results) != 148 or results.duplicated(["config_id", "question_id"]).any():
        raise ValueError("Expected one result for each criterion/configuration.")
    if set(results.config_id) != set(CONFIGS):
        raise ValueError("Unexpected model configuration IDs.")
    if len(pairs) != 74 or pairs.duplicated(["comparison_id", "question_id"]).any():
        raise ValueError("Expected 37 paired comparisons per model family.")
    if set(pairs.comparison_id) != {"gpt", "gemini"}:
        raise ValueError("Unexpected paired-comparison families.")
    for _, part in results.groupby("config_id"):
        if set(part.question_id) != expected:
            raise ValueError("A configuration is missing criterion results.")
    for _, part in pairs.groupby("comparison_id"):
        if set(part.question_id) != expected:
            raise ValueError("A family is missing paired comparisons.")
    for name in ("pearson", "spearman", "mae", "rmse", "bias",
                 "human_human_mae", "model_human_mae", "hrmae"):
        if not np.isfinite(results[name]).all():
            raise ValueError(f"Non-finite result in {name}.")
    if not results[["pearson", "spearman"]].abs().le(1).all().all():
        raise ValueError("A correlation is outside [-1, 1].")
    if not results[["mae", "rmse", "human_human_mae", "model_human_mae"]].ge(0).all().all():
        raise ValueError("An absolute-error metric is negative.")
    if not np.allclose(pairs.delta_mae, pairs.rag_mae - pairs.full_mae):
        raise ValueError("Paired differences do not equal RAG MAE minus full-context MAE.")
    if not np.allclose(results.hrmae, results.human_human_mae / results.model_human_mae):
        raise ValueError("HRMAE ratios disagree with the two exported disagreement metrics.")
    if not (results.n.between(1, 244).all() and pairs.n.between(1, 244).all()):
        raise ValueError("An exported usable-record count is outside the study cohort.")
    summary = {"criterion_results_rows": len(results), "paired_comparison_rows": len(pairs),
               "usable_records_range": [int(results.n.min()), int(results.n.max())],
               "configurations": {}, "method_comparisons": {}}
    for config in CONFIGS:
        part = results[results.config_id == config]
        summary["configurations"][config] = {
            "mae_below_one": int((part.mae < 1).sum()),
            "hrmae_above_one": int((part.hrmae > 1).sum()),
            "hrmae_interval_above_one": int((part.hrmae_ci_low > 1).sum()),
        }
    for family in ("gpt", "gemini"):
        part = pairs[pairs.comparison_id == family]
        summary["method_comparisons"][family] = {
            "full_context_lower_mae": int((part.delta_mae > 0).sum()),
            "rag_lower_mae": int((part.delta_mae < 0).sum()),
            "equal_mae": int((part.delta_mae == 0).sum()),
            "interval_favors_full_context": int((part.delta_mae_ci_low > 0).sum()),
            "interval_favors_rag": int((part.delta_mae_ci_high < 0).sum()),
        }
    return summary


if __name__ == "__main__":
    print(json.dumps(check_results(), indent=2))
