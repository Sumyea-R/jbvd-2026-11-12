# Criterion-level sustainability evaluation

Public-package usage, confidentiality limits, and private-source commands are
described in [the evaluation guide](../../../docs/evaluation.md). The runner
writes individual records as well as aggregates. Keep its entire output
directory under ignored `private/`; do not publish it.

This independent workflow compares the 37 raw sustainability criterion ratings
against two experts. It reads existing forecasts and workbooks, generates no
forecasts, and does not use or modify the composite evaluation workflow.

## Run

From the repository root, using the project's Python environment:

```bash
.venv/bin/python -m pip install -r src/sustainability/criterion_evaluation/requirements.txt
PYTHONPATH=src .venv/bin/python -m sustainability.criterion_evaluation \
  --bootstrap 2000 \
  --seed 42 \
  --output-dir private/evaluation
```

If the environment is activated, `python3` can replace `.venv/bin/python`.
No model API keys or network calls are needed for evaluation after dependencies
are installed. The runner leaves all source data unchanged. It only reuses a
nonempty output directory if its manifest identifies this workflow.

Options:

| Option | Default / behavior |
| --- | --- |
| `--config` | The adjacent `config.json`; selects configurations and method pairs |
| `--expert-dir` / `--experts-dir` | Repository `private/expert_rating/` directory |
| `--questions` / `--rubric` | Current `sustainability_forecasting_questions_category.json` |
| `--output-dir` | Repository `private/evaluation/` |
| `--bootstrap` | 2,000 resamples; minimum 2 |
| `--seed` | 42; nonnegative integer |

Model `input_dir` values in configuration are relative to the repository root
(absolute paths also work). Explicit command-line paths are relative to the
working directory. The rubric must contain exactly 37 unique criterion texts
in the established order, with the social and environmental KPI questions at
Q013 and Q019. Alternate rubric paths must preserve these identities, because
the historical mapping and reviewed aliases refer to these IDs.

The default configurations are:

| Configuration | Forecast folder under `private/outputs/` |
| --- | --- |
| GPT Full Context | `forecast_gpt_5.2` |
| GPT Question-level RAG | `forecast_question_rag_openai_gpt_5_2` |
| Gemini Full Context | `forecast_gemini_3.1_pro` |
| Gemini Question-level RAG | `forecast_question_rag_google_gemini_3_1_pro_preview` |

`config.json` contains `configurations` (each with `id`, `label`, `provider`,
`method`, and `input_dir`) and `comparisons` (each with `id`, `label`,
`full_context`, and `rag`, the latter two referencing configuration IDs).
IDs use lowercase letters, numbers and underscores, beginning with a letter.
Methods are `full_context` or `rag`. Earlier topic-level RAG and Claude runs
are not included in the default evaluation.

## Data mapping

- The rubric defines IDs Q001–Q037, question text, topic and category. All 37
  questions are retained, including scaling questions.
- Expert workbook sheets identify startups; columns B, C and D hold question
  text, rating and confidence. `question_aliases.json` lists reviewed wording
  variants. There is no automatic fuzzy matching. Out-of-rubric questions and
  missing/invalid ratings appear in diagnostics.
- Expert identities retain their workbook sources; Expert 1/2 are consistent
  labels within a startup, not two globally fixed people. Both experts must
  supply the criterion rating. Duplicate observations or more than two experts
  are excluded as ambiguous rather than averaged.
- Forecasts use the highest `_V<number>` file per startup/configuration.
  Invalid latest files are diagnosed without silently using an earlier file.
  Full Context JSON is a topic list; RAG JSON contains `topics`, with ratings
  under `stage2_scoring`.
- Legacy 35-question forecasts omit Q013 and Q019. Their topic positions are
  mapped explicitly; later scores do not slide onto the missing questions.
  RAG criterion text is validated when present. Unknown or malformed topic
  lengths are quarantined, retaining valid ratings in other topics.
- Ratings must be finite values on the 1–5 scale. Confidence is metadata only:
  missing or low confidence never changes a valid rating or its inclusion.

Expert workbooks have no business-plan version field. Latest-model selection
preserves the evaluation convention. The authors confirmed that consistent
plan versions were distributed to experts and models; this distribution is not
encoded in the workbook format. Full/RAG comparisons additionally require identical
selected model versions and expert ratings.

## Statistics

Every criterion/configuration uses complete `(E1, E2, M)` startup records, with
consensus `C = (E1 + E2) / 2`. The same cohort supplies all eight metrics:

| Field | Meaning |
| --- | --- |
| `pearson` | Pearson correlation of M and C |
| `spearman` | Pearson correlation of average tied ranks of M and C |
| `mae` | mean(abs(M − C)) |
| `rmse` | sqrt(mean((M − C)²)) |
| `bias` | mean(M − C); negative means lower model ratings |
| `human_human_mae` | mean(abs(E1 − E2)) |
| `model_human_mae` | mean((abs(M − E1) + abs(M − E2)) / 2) |
| `hrmae` | human_human_mae / model_human_mae |

HRMAE above 1 indicates lower model–human disagreement than expert–expert
disagreement; below 1 indicates greater disagreement. The ratio reverses the
old composite evaluator's `relative_mae_vs_human` field. HRMAE 1.20 means
approximately 16.7% lower disagreement, using `1 − 1 / HRMAE`.

Each criterion is bootstrapped independently across startups. All ratings in
a record are sampled together. Confidence intervals use the 2.5th and 97.5th
percentiles of finite replicates, with at least five eligible startups. Stable
random streams depend on seed, criterion ID and configuration/comparison ID;
source row order does not change results. All eight metrics have `*_ci_low`,
`*_ci_high`, `*_n_boot_valid` and `*_status` fields. No multiple-comparison
adjustment is applied to these pointwise intervals.

Overall status is `ok` for N ≥ 5, `insufficient_startups` for N 1–4, and
`no_eligible_startups` for N = 0. Point estimates remain available below five
where mathematically defined. Constant inputs yield undefined correlations.
Zero model–human MAE yields undefined HRMAE, including perfect agreement of all
three ratings. Undefined values are empty CSV cells and N/A in the report.

For each method pair, `delta_mae = rag_mae − full_mae` uses only their common
complete startup records with matching versions and experts. Both method MAEs
are recalculated on this cohort. Their per-startup error differences are
resampled together to obtain the paired interval. Positive delta favors Full
Context; negative delta favors RAG. Ranking is descending delta, ties share
the minimum rank, and missing differences are listed last without a rank.

## Outputs and modules

| Output | Contents |
| --- | --- |
| `report.md` | Methods, coverage, data-derived findings, figures and detailed tables |
| `criterion_catalog.csv` | Canonical 37-criterion catalog |
| `expert_ratings.csv`, `model_ratings.csv` | Individual observations, confidence, mapping status and source references |
| `aligned_records.csv` | One startup/criterion/configuration row, including experts, model, version and eligibility |
| `forecast_selections.csv` | Latest-file selection and available versions |
| `data_diagnostics.csv` | Source parsing, mapping and validation events |
| `coverage_diagnostics.csv`, `excluded_records.csv` | Eligible counts and reasons for excluding records |
| `criterion_results.csv` | 148 rows for the four defaults, with all metrics and intervals |
| `full_context_rag_comparisons.csv` | 74 paired comparisons with coverage and mismatch counts |
| `gpt_ranked_deltas.csv`, `gemini_ranked_deltas.csv` | 37 ranked rows per default method pair |
| `figures/` | MAE/HRMAE heatmaps, four scatter panels, two delta plots; PNG and SVG |
| `run_manifest.json` | Configuration, seed, dependency versions, input/source hashes and artifact inventory |

Heatmaps preserve all 37 rows; missing values are gray/NA and N < 5 is marked.
The HRMAE color scale is centered at 1. Scatter axes show human–human MAE and
model–human MAE with the equal-disagreement diagonal. Rankings and their plots
use paired cohorts; primary heatmap cells can have different sample sizes.

The new modules are independent of the existing evaluator:

- `data.py`: `prepare_data(repo_root, questions_path, experts_dir, config)`
  returns in-memory catalog, observation, alignment and diagnostic tables.
- `evaluation.py`: `evaluate(aligned_records, catalog, config, bootstrap, seed)`
  returns all criterion results and paired method comparisons.
- `plotting.py`: `generate_figures(catalog, results, comparisons, config, output_dir)`
  returns the saved figure paths.
- `reporting.py`: builds the Markdown report from calculated tables.
- `__main__.py`: validates configuration, orchestrates the stages and saves outputs.

## Verify

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover \
  -s src/sustainability/criterion_evaluation/tests -v
```

Tests cover formula direction, tied ranks, missing/ambiguous ratings, historical
question alignment, forecast formats, version selection, deterministic paired
resampling, unequal method coverage, low sample sizes and undefined statistics.
Running the full command above additionally exercises the current workbooks
and all four forecast folders; inspect its report and figures alongside the
coverage diagnostics.
