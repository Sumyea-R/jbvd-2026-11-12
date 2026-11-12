"""Criterion-level accuracy, human disagreement, and paired bootstrap estimates.

Every metric for a configuration/criterion uses the same complete startup cohort.
Bootstrap rows retain their individual expert ratings and model ratings together.
The only dependencies are NumPy and pandas; tied Spearman ranks use average ranks.
"""

from __future__ import annotations

import hashlib
from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np
import pandas as pd


METRICS = (
    "pearson",
    "spearman",
    "mae",
    "rmse",
    "bias",
    "human_human_mae",
    "model_human_mae",
    "hrmae",
)
MIN_BOOTSTRAP_STARTUPS = 5
_RATINGS = ["expert_1", "expert_2", "model_rating"]
_KEYS = ["startup", "question_id", "config_id"]
_METADATA = ["topic", "category", "question_text"]


def _row_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson correlations of rows, with constant inputs explicitly undefined."""
    x_centered = x - x.mean(axis=1, keepdims=True)
    y_centered = y - y.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        np.sum(x_centered**2, axis=1) * np.sum(y_centered**2, axis=1)
    )
    result = np.full(x.shape[0], np.nan)
    np.divide(
        np.sum(x_centered * y_centered, axis=1),
        denominator,
        out=result,
        where=denominator > 0,
    )
    return np.clip(result, -1.0, 1.0)


def _batch_metrics(
    expert_1: np.ndarray, expert_2: np.ndarray, model: np.ndarray
) -> dict[str, np.ndarray]:
    """Return one value per row of (replicates, startups) arrays."""
    consensus = (expert_1 + expert_2) / 2.0
    error = model - consensus
    human = np.mean(np.abs(expert_1 - expert_2), axis=1)
    model_human = np.mean(
        (np.abs(model - expert_1) + np.abs(model - expert_2)) / 2.0,
        axis=1,
    )
    hrmae = np.full(len(human), np.nan)
    np.divide(human, model_human, out=hrmae, where=model_human > 0)
    model_ranks = pd.DataFrame(model).rank(axis=1, method="average").to_numpy()
    consensus_ranks = (
        pd.DataFrame(consensus).rank(axis=1, method="average").to_numpy()
    )
    return {
        "pearson": _row_correlation(model, consensus),
        "spearman": _row_correlation(model_ranks, consensus_ranks),
        "mae": np.mean(np.abs(error), axis=1),
        "rmse": np.sqrt(np.mean(error**2, axis=1)),
        "bias": np.mean(error, axis=1),
        "human_human_mae": human,
        "model_human_mae": model_human,
        "hrmae": hrmae,
    }


def compute_metrics(
    expert_1: np.ndarray, expert_2: np.ndarray, model: np.ndarray
) -> dict[str, float]:
    """Calculate the eight metrics for already aligned, complete startup rows.

    Inputs must be equally sized finite one-dimensional arrays on the 1–5 scale.
    Empty inputs return N/A for every metric. Undefined correlations and HRMAE
    are represented as ``numpy.nan``, never infinity.
    """
    arrays = [np.asarray(values, dtype=float) for values in (expert_1, expert_2, model)]
    if any(values.ndim != 1 for values in arrays):
        raise ValueError("Rating arrays must be one-dimensional.")
    if len({len(values) for values in arrays}) != 1:
        raise ValueError("All rating arrays must contain the same startup rows.")
    if any(
        not np.all(np.isfinite(values) & (values >= 1) & (values <= 5))
        for values in arrays
    ):
        raise ValueError("Metrics require complete, finite ratings in the range 1–5.")
    if not len(arrays[0]):
        return dict.fromkeys(METRICS, float("nan"))
    return {
        name: float(values[0])
        for name, values in _batch_metrics(*(a[None, :] for a in arrays)).items()
    }


def _rng(seed: int, *identifiers: str) -> np.random.Generator:
    """Stable streams independent of Python hash randomization and input order."""
    key = "\x00".join([str(seed), *identifiers]).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "little"))


def _interval(values: np.ndarray) -> tuple[float, float, int]:
    valid = values[np.isfinite(values)]
    if not len(valid):
        return float("nan"), float("nan"), 0
    low, high = np.percentile(valid, [2.5, 97.5])
    return float(low), float(high), len(valid)


def _status(n: int) -> str:
    if n == 0:
        return "no_eligible_startups"
    return "ok" if n >= MIN_BOOTSTRAP_STARTUPS else "insufficient_startups"


def _metric_status(name: str, value: float, n: int) -> str:
    if n == 0:
        return "no_eligible_startups"
    if np.isfinite(value):
        return "ok"
    if name in ("pearson", "spearman"):
        return "insufficient_startups_for_correlation" if n < 2 else "constant_input"
    if name == "hrmae":
        return "zero_model_human_disagreement"
    return "undefined"


def _complete_rows(aligned: pd.DataFrame) -> pd.DataFrame:
    required = [*_KEYS, *_RATINGS, "version", "eligible"]
    missing = sorted(set(required).difference(aligned.columns))
    if missing:
        raise ValueError(f"Aligned records are missing required columns: {missing}")
    frame = aligned.copy()
    if frame[_KEYS].isna().any().any():
        raise ValueError("Aligned startup, criterion, and configuration identifiers cannot be missing.")
    frame[_KEYS] = frame[_KEYS].astype(str)
    if frame.duplicated(_KEYS).any():
        raise ValueError("Aligned records contain duplicate startup/criterion/configuration rows.")
    # An absent/unknown/string eligibility flag must never silently become True.
    mask = frame["eligible"].map(
        lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
    )
    for column in _RATINGS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        mask &= np.isfinite(frame[column]) & frame[column].between(1, 5)
    return frame.loc[mask].sort_values(_KEYS).reset_index(drop=True)


def _version_key(value: Any) -> str | None:
    if pd.isna(value):
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        numeric = Decimal(value)
        if not numeric.is_finite():
            return None
        return str(numeric.normalize())
    except InvalidOperation:
        return value


def _evaluate_configuration(
    frame: pd.DataFrame,
    criterion: dict[str, Any],
    configuration: dict[str, Any],
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    n = len(frame)
    row = {
        **criterion,
        "config_id": configuration["id"],
        "label": configuration["label"],
        "provider": configuration["provider"],
        "method": configuration["method"],
        "n": n,
        "status": _status(n),
        "n_boot_requested": bootstrap,
    }
    ratings = frame[_RATINGS].to_numpy(dtype=float)
    values = compute_metrics(*ratings.T)
    distributions: dict[str, np.ndarray] = {}
    if n >= MIN_BOOTSTRAP_STARTUPS and bootstrap:
        indices = _rng(
            seed, "configuration", configuration["id"], criterion["question_id"]
        ).integers(0, n, size=(bootstrap, n))
        sampled = ratings[indices]
        distributions = _batch_metrics(sampled[:, :, 0], sampled[:, :, 1], sampled[:, :, 2])
    for name, value in values.items():
        low, high, valid = _interval(distributions.get(name, np.array([])))
        row.update(
            {
                name: value,
                f"{name}_ci_low": low,
                f"{name}_ci_high": high,
                f"{name}_n_boot_valid": valid,
                f"{name}_status": _metric_status(name, value, n),
            }
        )
    return row


def _evaluate_comparison(
    full: pd.DataFrame,
    rag: pd.DataFrame,
    criterion: dict[str, Any],
    comparison: dict[str, Any],
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    joined = full.merge(rag, on="startup", suffixes=("_full", "_rag"), validate="one_to_one")
    full_versions = joined["version_full"].map(_version_key)
    rag_versions = joined["version_rag"].map(_version_key)
    matching_versions = full_versions.notna() & full_versions.eq(rag_versions)
    matching_experts = (
        joined["expert_1_full"].eq(joined["expert_1_rag"])
        & joined["expert_2_full"].eq(joined["expert_2_rag"])
    )
    paired = joined.loc[matching_versions & matching_experts].sort_values("startup")
    n = len(paired)
    row = {
        **criterion,
        "comparison_id": comparison["id"],
        "label": comparison["label"],
        "full_context": comparison["full_context"],
        "rag": comparison["rag"],
        "n": n,
        "status": _status(n),
        "n_full_eligible": len(full),
        "n_rag_eligible": len(rag),
        "n_common_startups": len(joined),
        "n_version_mismatch": int((~matching_versions).sum()),
        "n_expert_mismatch": int((matching_versions & ~matching_experts).sum()),
        "n_boot_requested": bootstrap,
        "full_mae": float("nan"),
        "rag_mae": float("nan"),
        "delta_mae": float("nan"),
        "delta_mae_ci_low": float("nan"),
        "delta_mae_ci_high": float("nan"),
        "delta_mae_n_boot_valid": 0,
        "interpretation": "unavailable",
    }
    if not n:
        return row
    consensus = (
        paired["expert_1_full"].to_numpy() + paired["expert_2_full"].to_numpy()
    ) / 2.0
    full_errors = np.abs(paired["model_rating_full"].to_numpy() - consensus)
    rag_errors = np.abs(paired["model_rating_rag"].to_numpy() - consensus)
    row["full_mae"] = float(full_errors.mean())
    row["rag_mae"] = float(rag_errors.mean())
    # Algebraically the difference of MAEs, calculated before averaging so
    # cancellation in two rounded means does not split mathematically tied ranks.
    row["delta_mae"] = float((rag_errors - full_errors).mean())
    row["interpretation"] = (
        "full_context_lower_mae" if row["delta_mae"] > 0
        else "rag_lower_mae" if row["delta_mae"] < 0
        else "equal_mae"
    )
    if n >= MIN_BOOTSTRAP_STARTUPS and bootstrap:
        indices = _rng(
            seed, "comparison", comparison["id"], criterion["question_id"]
        ).integers(0, n, size=(bootstrap, n))
        # One shared startup index resamples E1/E2/Full/RAG jointly. Computing
        # paired per-startup differences is algebraically the same operation.
        distribution = (rag_errors - full_errors)[indices].mean(axis=1)
        low, high, valid = _interval(distribution)
        row.update(
            delta_mae_ci_low=low,
            delta_mae_ci_high=high,
            delta_mae_n_boot_valid=valid,
        )
    return row


def evaluate(
    aligned_records: pd.DataFrame,
    catalog: pd.DataFrame,
    config: dict[str, Any],
    bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every catalog criterion, preserving rows without eligible data.

    ``config`` contains ``configurations`` with id/label/provider/method and
    ``comparisons`` with id/label/full_context/rag configuration IDs. Comparison
    MAEs and intervals use common complete startups with equal known plan
    versions and equal expert pairs. Rankings descend by delta MAE (RAG−Full),
    with shared minimum ranks for ties and missing ranks for unavailable data.

    ``bootstrap=0`` disables confidence intervals. Fewer than five startups
    always suppresses intervals while retaining all defined point estimates.
    Input rows marked ineligible, missing ratings, or outside 1–5 are excluded.
    Duplicate startup/criterion/configuration records are an explicit error.
    """
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, (int, np.integer)) or bootstrap < 0:
        raise ValueError("bootstrap must be a non-negative integer.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if "question_id" not in catalog or catalog["question_id"].isna().any():
        raise ValueError("Catalog requires nonmissing question_id values.")
    if catalog["question_id"].astype(str).duplicated().any():
        raise ValueError("Catalog question_id values must be unique.")
    configurations = config["configurations"]
    configuration_ids = [item["id"] for item in configurations]
    if len(set(configuration_ids)) != len(configuration_ids):
        raise ValueError("Configuration IDs must be unique.")
    comparisons = config.get("comparisons", [])
    if len({item["id"] for item in comparisons}) != len(comparisons):
        raise ValueError("Comparison IDs must be unique.")
    for comparison in comparisons:
        if not {comparison["full_context"], comparison["rag"]} <= set(configuration_ids):
            raise ValueError("Comparisons must refer to configured configuration IDs.")
    complete = _complete_rows(aligned_records)
    groups = {
        key: rows for key, rows in complete.groupby(["config_id", "question_id"], sort=False)
    }
    empty = complete.iloc[:0]
    result_rows = []
    comparison_rows = []
    for entry in catalog.to_dict("records"):
        criterion = {
            "question_id": str(entry["question_id"]),
            **{field: entry.get(field, "") for field in _METADATA},
        }
        question_id = criterion["question_id"]
        for configuration in configurations:
            frame = groups.get((configuration["id"], question_id), empty)
            result_rows.append(
                _evaluate_configuration(frame, criterion, configuration, bootstrap, seed)
            )
        for comparison in comparisons:
            full = groups.get((comparison["full_context"], question_id), empty)
            rag = groups.get((comparison["rag"], question_id), empty)
            comparison_rows.append(
                _evaluate_comparison(full, rag, criterion, comparison, bootstrap, seed)
            )
    results = pd.DataFrame(result_rows)
    comparisons_frame = pd.DataFrame(comparison_rows)
    if not comparisons_frame.empty:
        comparisons_frame["rank"] = (
            comparisons_frame.groupby("comparison_id")["delta_mae"]
            .rank(method="min", ascending=False, na_option="keep")
            .astype("Int64")
        )
        comparisons_frame = comparisons_frame.sort_values(
            ["comparison_id", "rank", "question_id"], na_position="last"
        ).reset_index(drop=True)
    return results, comparisons_frame
