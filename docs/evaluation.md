# Criterion-level evaluation

## Public and restricted reproduction

The public research package contains the evaluation implementation and
aggregate results. It does not contain the 244 business plans, individual
expert ratings, individual model outputs, plan text, or retrieved passages.
These source records are confidential.

The public `criterion_results.csv` and `full_context_rag_comparisons.csv` files
can be used to inspect reported estimates and regenerate figures and summary
tables. Aggregate CSVs cannot recover the individual ratings, independently
recalculate their correlations or error metrics, or rerun the venture-level
bootstrap. Those operations require the restricted source records. Running
the supplied offline tests verifies the implementation on synthetic inputs;
it does not reproduce the study's empirical estimates.

## Installation and offline checks

From the repository root, in your Python environment:

```bash
python -m pip install -r requirements-evaluation.txt
PYTHONPATH=src python -m unittest discover \
  -s src/sustainability/criterion_evaluation/tests -v
```

The tests create synthetic ratings and workbooks in temporary directories.
Their venture and expert names are artificial. They cover metric definitions,
tied ranks, missing and ambiguous observations, explicit criterion mapping,
forecast formats, version selection, low sample sizes, undefined statistics,
and deterministic paired resampling. No API key or network call is used after
the dependencies are installed.

## Full evaluation with authorized private records

Only users who already have authorization to process the source records can
run the full empirical evaluation. Keep their copies under ignored `private/`.
Do not commit this directory or upload its outputs as supplementary material.

`src/sustainability/criterion_evaluation/config.json` expects these model
directories beneath `private/outputs/`:

| Configuration | Directory name |
| --- | --- |
| GPT full context | `forecast_gpt_5.2` |
| GPT criterion RAG | `forecast_question_rag_openai_gpt_5_2` |
| Gemini full context | `forecast_gemini_3.1_pro` |
| Gemini criterion RAG | `forecast_question_rag_google_gemini_3_1_pro_preview` |

Each model directory contains JSON forecasts named
`<venture>_V<number>.json`. Full-context JSON is a list of topic blocks with
`Topic`, `ratings`, and optional confidence metadata. Criterion-RAG JSON has
a `topics` list with criterion texts and ratings under `stage2_scoring`.

Place expert workbooks in `private/expert_rating/`. Each workbook represents
one expert; sheet names identify ventures. Columns B, C, and D contain criterion
text, rating, and optional confidence. Text is matched to the public rubric and
reviewed `question_aliases.json`; fuzzy matching is not used. Both experts'
valid ratings are required. Duplicate observations or more than two experts
are diagnosed and excluded as ambiguous.

Run from the repository root:

```bash
PYTHONPATH=src python -m sustainability.criterion_evaluation \
  --bootstrap 2000 --seed 42
```

The default output is `private/evaluation/`. To use other authorized local
sources, create an untracked configuration with appropriate `input_dir` values:

```bash
PYTHONPATH=src python -m sustainability.criterion_evaluation \
  --config private/evaluation_config.json \
  --expert-dir private/expert_rating \
  --questions src/sustainability/config/sustainability_forecasting_questions_category.json \
  --output-dir private/evaluation --bootstrap 2000 --seed 42
```

Configuration `input_dir` values are relative to the repository root; absolute
paths also work. Other explicit paths are relative to the working directory.
Sources are read without alteration. A nonempty output directory is reused
only if its manifest identifies this evaluation workflow.

The parser selects the highest `_V<number>` forecast per venture/configuration.
Invalid latest files are diagnosed, without silently falling back to older
ones. Expert workbooks have no machine-readable plan-version field. The authors
confirmed that the study distributed consistent plan versions to experts and
models; this consistency is established by data collection. Paired comparisons
additionally require matching selected model versions and expert ratings.
Compatibility handling for older layouts does not create or infer ratings.

## Calculations

Each configuration/criterion uses the same complete venture records for its
metrics. Let `E1` and `E2` be the expert ratings, `M` the model rating, and
`C = (E1 + E2) / 2` the arithmetic-mean reference. Ratings remain on the 1–5
scale. Confidence is metadata and never weights or filters valid ratings.

| Field | Definition |
| --- | --- |
| `pearson` | Pearson correlation of `M` and `C` |
| `spearman` | Correlation of average tied ranks of `M` and `C` |
| `mae` | Mean of `abs(M - C)` |
| `rmse` | Square root of mean squared `M - C` |
| `bias` | Mean of `M - C`; negative means lower model ratings |
| `human_human_mae` | Mean of `abs(E1 - E2)` |
| `model_human_mae` | Mean of `(abs(M - E1) + abs(M - E2)) / 2` |
| `hrmae` | `human_human_mae / model_human_mae` |

HRMAE above one means lower model–expert than expert–expert disagreement.
Its denominator is not MAE against expert consensus. Correlation describes
association rather than equality of score levels.

Bootstrap resampling keeps each venture's expert and model ratings together.
Study settings are 2,000 resamples, seed 42, and the 2.5th and 97.5th percentiles
of finite replicates for pointwise 95% intervals. At least five eligible ventures
are needed for an interval. Stable random streams depend on seed, criterion ID,
and configuration/comparison ID. These exploratory intervals are not adjusted
for multiple comparisons.

Paired comparisons use common complete records with matching versions and
experts. `delta_mae = rag_mae - full_mae`; positive favors full context and
negative favors RAG. Both methods use the same sampled ventures in each
replicate. An interval containing zero identifies no clear direction.
Constant inputs yield undefined correlations; zero model–expert MAE gives an
undefined HRMAE. Undefined values appear as empty CSV cells and N/A in reports.

## Outputs and privacy

The full runner writes aggregate results together with `expert_ratings.csv`,
`model_ratings.csv`, `aligned_records.csv`, diagnostics, excluded observations,
forecast selections, and an input/run manifest. These files can identify
ventures or experts and must remain private. Never publish or archive the
runner's complete output directory. The committed aggregate results were
selected separately for sharing; renaming a file does not anonymize it.

## Provenance

Source commit: `7b7ef52ac1a8e98901278e26502711a8d4567990`. Statistical
calculations and data-alignment logic are unchanged. Adaptations set default
paths to `private/`, use explicit synthetic test names, and clarify the public
documentation. The separate criterion-forecasting patch workflow is not needed
by this evaluator and is not included.
