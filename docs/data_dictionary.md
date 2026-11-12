# Shareable aggregate exports

All three CSV files are UTF-8, comma-delimited, with a header row. They contain
criterion/configuration aggregates and rubric wording, not venture IDs, expert
IDs, plan text, model evidence, or individual ratings. The original exports are
preserved without rounding or removing columns. Their hashes are included in
the snapshot's provenance and checksum files.

## `criterion_catalog.csv` — 37 rows, 5 columns

| Column | Meaning |
| --- | --- |
| `question_id` | Stable criterion ID, Q001–Q037 |
| `topic` | Rubric topic |
| `category` | Sustainability category used in reporting |
| `question_text` | Operational criterion wording |
| `topic_index` | Zero-based topic position in the operational rubric |

## `criterion_results.csv` — 148 rows, 51 columns

One row per criterion and assessment configuration. The four `config_id`
values are `gpt_full_context`, `gpt_rag`, `gemini_full_context`, and
`gemini_rag`. Raw scores use the original 1–5 rating scale.

| Column or field group | Meaning |
| --- | --- |
| `question_id`, `topic`, `category`, `question_text` | Criterion descriptors matching the catalog |
| `config_id` | Configuration identifier |
| `label` | Human-readable configuration label |
| `provider` | Model family, GPT or Gemini |
| `method` | `full_context` or `rag` |
| `n` | Usable venture records for that criterion/configuration; the export retains values from 236 to 242 |
| `status` | Evaluation status for the row |
| `n_boot_requested` | Requested venture-level bootstrap resamples, 2,000 in these exports |
| `pearson` | Linear correlation between model ratings and the mean of the two expert ratings |
| `spearman` | Correlation of ranks between model ratings and the expert mean, with tied ranks |
| `mae` | Mean absolute difference between model rating and expert mean, in rating points |
| `rmse` | Root mean squared difference from expert mean, in rating points |
| `bias` | Mean signed difference, model minus expert mean; negative values indicate lower model scores |
| `human_human_mae` | Mean absolute difference between the two experts |
| `model_human_mae` | Average absolute difference between the model and each of the two experts |
| `hrmae` | `human_human_mae / model_human_mae`; values above 1 indicate lower model–expert disagreement |

For **each of the eight metrics** above, four additional columns are included:

| Suffix | Meaning |
| --- | --- |
| `_ci_low` | 2.5th percentile of the valid bootstrap estimates |
| `_ci_high` | 97.5th percentile of the valid bootstrap estimates |
| `_n_boot_valid` | Number of bootstrap resamples giving a defined estimate for that metric |
| `_status` | Metric-specific status |

Resampling keeps the complete venture record together. The intervals are
pointwise, not adjusted for the set of criterion comparisons. A point estimate
is the value calculated on the observed eligible sample; an interval describes
its uncertainty under the specified resampling procedure. HRMAE above 1 in the
observed data and an interval entirely above 1 are different summaries.

## `full_context_rag_comparisons.csv` — 74 rows, 24 columns

One row per criterion and model family, based on a paired comparison using the
same eligible venture records for the two algorithms.

| Column or field group | Meaning |
| --- | --- |
| `question_id`, `topic`, `category`, `question_text` | Criterion descriptors |
| `comparison_id` | `gpt` or `gemini` |
| `label` | Human-readable model-family label |
| `full_context`, `rag` | Configuration IDs being compared |
| `n` | Records used in the paired comparison |
| `status` | Pair-comparison status |
| `n_full_eligible`, `n_rag_eligible` | Eligible record counts for each algorithm before pairing |
| `n_common_startups` | Number of ventures shared by the two eligible input sets |
| `n_version_mismatch` | Aggregate count rejected by the document-version matching guard |
| `n_expert_mismatch` | Aggregate count rejected by the paired expert-reference matching guard |
| `n_boot_requested` | Requested paired venture-level bootstrap resamples |
| `full_mae`, `rag_mae` | MAE for each algorithm on the paired records |
| `delta_mae` | `rag_mae - full_mae`; positive values favor full context |
| `delta_mae_ci_low`, `delta_mae_ci_high` | Paired 95% percentile-bootstrap interval for the difference |
| `delta_mae_n_boot_valid` | Number of defined paired bootstrap differences |
| `interpretation` | Direction of the observed difference: full context lower, RAG lower, or equal MAE |
| `rank` | Original export's ordering index for the method-difference report |

Zero denotes equal MAE. An interval entirely above zero supports lower
full-context error; an interval entirely below zero supports lower RAG error.
When an interval includes zero, the comparison does not establish a clear
direction. The `interpretation` field describes the **point estimate**, not an
automatic conclusion from the confidence interval.

## Reproduction limits

The aggregate files allow validation of reported counts and regeneration of
the manuscript tables and numerical figures. They cannot reconstruct the
underlying venture-level scores or regenerate the bootstrap distributions.
The restricted-input evaluation module is supplied for readers with authorized
compatible records; its synthetic unit tests provide examples of the formats.
