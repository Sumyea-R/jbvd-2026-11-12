"""Generate a report from calculated results, without hardcoded conclusions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _number(value: float, digits: int = 3) -> str:
    return "N/A" if pd.isna(value) else f"{value:.{digits}f}"


def _interval(row: pd.Series, metric: str) -> str:
    point = _number(row[metric])
    low, high = row[f"{metric}_ci_low"], row[f"{metric}_ci_high"]
    if pd.isna(low) or pd.isna(high):
        return f"{point} [N/A]"
    return f"{point} [{low:.3f}, {high:.3f}]"


def _table(headers: list[str], rows: list[list]) -> str:
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(cell(x) for x in row) + " |" for row in rows)
    return "\n".join(lines)


def generate_report(catalog: pd.DataFrame, results: pd.DataFrame, comparisons: pd.DataFrame,
                    coverage: pd.DataFrame, diagnostics: pd.DataFrame, config: dict,
                    output_dir: Path, bootstrap: int, seed: int) -> Path:
    lines = [
        "# Sustainability criterion evaluation",
        "",
        f"This evaluation compares {len(catalog)} individual sustainability criteria across "
        f"{len(config['configurations'])} configurations. Ratings retain their original 1–5 scale. "
        "Each criterion is evaluated across startups; no composite scores or confidence weights are used.",
        "",
        "## Method and interpretation",
        "",
        "The expert consensus is C = (E1 + E2) / 2. Pearson and Spearman measure model agreement "
        "with consensus across startups. MAE = mean(|M − C|); RMSE = sqrt(mean((M − C)²)); "
        "bias = mean(M − C), so negative bias indicates lower model scores.",
        "",
        "Human–human MAE = mean(|E1 − E2|). Model–human MAE = "
        "mean((|M − E1| + |M − E2|) / 2). **HRMAE = human–human MAE / model–human MAE**: "
        "values above 1 indicate lower model–expert disagreement than expert–expert disagreement; "
        "values below 1 indicate greater disagreement. This ratio is the inverse of the old composite "
        "evaluation's `relative_mae_vs_human`. HRMAE 1.20 implies 1 − 1/1.20 = 16.7% lower disagreement.",
        "",
        "Every metric for a criterion/configuration uses the same complete (E1, E2, M) records. "
        "Missing ratings are excluded without imputation. Confidence values are retained in the "
        "extracted data but do not affect scores or inclusion.",
        "",
        f"Confidence intervals use {bootstrap:,} startup bootstrap resamples, seed {seed}, "
        "and the 2.5th and 97.5th percentiles. Complete startup records are sampled together. "
        "At least five eligible startups are required for an interval; smaller samples retain point "
        "estimates and are flagged. Constant inputs make correlations undefined; zero model–human MAE "
        "makes HRMAE undefined. Intervals use finite replicates; valid replicate counts are exported "
        "for every metric. Intervals are pointwise, without a multiple-comparison adjustment.",
        "",
        "Full Context/RAG comparisons use the common complete (E1, E2, Full, RAG) cohort for each "
        "criterion, with matching business-plan versions. ΔMAE = MAE_RAG − MAE_Full is bootstrapped "
        "using the same sampled startup indices for both methods. Positive values favor Full Context; "
        "negative values favor RAG. The method-specific MAEs in these comparisons can differ from "
        "the primary results because their eligible cohorts can differ.",
        "",
        "## Coverage and data limitations",
        "",
        "Expert workbooks do not identify business-plan versions. Following the existing evaluation "
        "convention, each configuration uses its latest available startup version. Selected files and "
        "versions are recorded in [forecast selections](forecast_selections.csv). This does not establish "
        "that the expert saw the same version.",
        "",
        "Legacy 35-question forecasts originally omitted **Q013 (social impact KPIs)** and **Q019 "
        "(environmental impact KPIs)**. An explicit mapping preserves subsequent question positions "
        "in files still using that layout. This evaluation reads the supplied forecast files, including "
        "any merged supplemental criterion ratings, and makes no model calls. Ambiguous topic lengths "
        "are quarantined; valid topics are retained. "
        "Expert wording variants use reviewed aliases, and out-of-rubric questions are excluded.",
        "",
    ]
    coverage_rows = []
    for run in config["configurations"]:
        subset = results[results.config_id == run["id"]]
        counts = subset.n.astype(int)
        coverage_rows.append([run["label"], f"{counts.min()}–{counts.max()}",
                              int((counts >= 5).sum()), int(((counts > 0) & (counts < 5)).sum()),
                              int((counts == 0).sum())])
    lines += [_table(["Configuration", "Eligible N range", "Criteria N ≥ 5", "Criteria N 1–4", "Criteria N = 0"], coverage_rows), "",
              "See [coverage by criterion](coverage_diagnostics.csv), [excluded records](excluded_records.csv), "
              "and [source diagnostics](data_diagnostics.csv) for exact exclusions. Counts below refer to "
              "diagnostic events, which may cover multiple criteria or occur more than once per startup.", ""]
    if not diagnostics.empty and "reason" in diagnostics:
        lines += [_table(["Diagnostic reason", "Events"],
                          [[reason, int(n)] for reason, n in diagnostics.reason.value_counts().items()]), ""]
    lines += ["## Criterion heatmaps", "",
              "Cells show point estimates. Asterisks mark N < 5; gray cells are unavailable. "
              "Eligible counts differ across criteria and configurations, so use the paired tables "
              "below when comparing Full Context with RAG.", "",
              "![Criterion MAE heatmap](figures/mae_heatmap.png)", "",
              "![Criterion HRMAE heatmap](figures/hrmae_heatmap.png)", "",
              "## Model disagreement relative to experts", "",
              "Each point represents one criterion. Points below y = x have lower model–human "
              "disagreement than human–human disagreement. These are point-estimate comparisons; "
              "uncertainty is available in the result tables.", "",
              "![Human disagreement versus model error](figures/human_disagreement_vs_model_error.png)", ""]
    for run in config["configurations"]:
        subset = results[(results.config_id == run["id"]) & (results.n >= 5)]
        valid_ratio = subset[np.isfinite(subset.hrmae)]
        below = int((valid_ratio.hrmae > 1).sum())
        lines.append(f"- **{run['label']}**: {below}/{len(valid_ratio)} criteria with N ≥ 5 and defined "
                     "HRMAE have lower model–human disagreement than human–human disagreement.")
    lines += ["", "## Paired Full Context/RAG differences", "",
              "Ranking is descending ΔMAE: the largest Full Context advantages appear first. "
              "Intervals excluding zero indicate a consistent direction under the paired bootstrap; "
              "intervals containing zero leave that direction uncertain. These exploratory comparisons "
              "do not establish why a method performs differently.", ""]
    for pair in config["comparisons"]:
        subset = comparisons[comparisons.comparison_id == pair["id"]].sort_values(
            ["rank", "question_id"], na_position="last")
        valid = subset[subset.n >= 5]
        finite = valid[np.isfinite(valid.delta_mae)]
        full = int((finite.delta_mae > 0).sum())
        rag = int((finite.delta_mae < 0).sum())
        full_ci = int((finite.delta_mae_ci_low > 0).sum())
        rag_ci = int((finite.delta_mae_ci_high < 0).sum())
        lines += [f"### {pair['label']}", "",
                  f"Among {len(finite)} criteria with N ≥ 5 and defined paired differences, point "
                  f"estimates favor Full Context for {full}, RAG for {rag}, and tie for "
                  f"{len(finite) - full - rag}. Pointwise 95% intervals exclude zero in favor of "
                  f"Full Context for {full_ci} and RAG for {rag_ci}.", "",
                  f"![{pair['label']} ranked MAE differences](figures/delta_mae_{pair['id']}.png)", "",
                  _table(["Rank", "Criterion", "N", "Full MAE", "RAG MAE", "ΔMAE [95% CI]", "Status"], [
                      [_number(row["rank"], 0), row.question_id, int(row.n), _number(row.full_mae),
                       _number(row.rag_mae), _interval(row, "delta_mae"), row.status]
                      for _, row in subset.iterrows()
                  ]), ""]
    lines += ["## Detailed criterion estimates", "",
              "Entries are point estimates [95% confidence interval]. N/A means undefined or "
              "insufficient data. CSV exports retain full numerical precision and per-metric statuses.", ""]
    for run in config["configurations"]:
        subset = results[results.config_id == run["id"]].sort_values("question_id")
        lines += [f"### {run['label']}", "",
                  _table(["Criterion", "N", "Pearson", "Spearman", "MAE", "RMSE", "Bias"], [
                      [row.question_id, int(row.n)] + [_interval(row, metric) for metric in
                       ("pearson", "spearman", "mae", "rmse", "bias")]
                      for _, row in subset.iterrows()
                  ]), "",
                  _table(["Criterion", "Human–human MAE", "Model–human MAE", "HRMAE"], [
                      [row.question_id] + [_interval(row, metric) for metric in
                       ("human_human_mae", "model_human_mae", "hrmae")]
                      for _, row in subset.iterrows()
                  ]), ""]
    lines += ["## Criterion catalog", "",
              _table(["ID", "Topic", "Category", "Question"], [
                  [row.question_id, row.topic, row.category, row.question_text]
                  for _, row in catalog.iterrows()
              ]), "", "## Reproducible outputs", "",
              "- [All criterion metrics and confidence intervals](criterion_results.csv)",
              "- [All paired method comparisons](full_context_rag_comparisons.csv)",
              "- [Aligned evaluation records](aligned_records.csv)",
              "- [Individual expert ratings](expert_ratings.csv) and [model ratings](model_ratings.csv)",
              "- [Input hashes, configuration and dependency versions](run_manifest.json)",
              "", "PNG figures have matching SVG files in `figures/` for vector export.", ""]
    path = output_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
