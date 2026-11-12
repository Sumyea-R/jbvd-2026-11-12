# AI assisted assessment of venture sustainability impact potential

Source materials for **AI Assisted Assessment of Sustainability Impact
Potential for Early Stage Ventures**, by Sumyea Rahman.

## What can be reproduced

| Task | Included inputs | Private records or model calls needed? |
| --- | --- | --- |
| Check reported summaries and regenerate three LaTeX tables and four numerical figures | Three aggregate CSVs in `data/results/` | No |
| Inspect metric definitions and run synthetic statistical tests | Evaluation module and synthetic tests | No |
| Recalculate the study's correlations, errors, confidence intervals, and method comparisons from individual observations | Evaluation scripts only | Restricted expert ratings and model outputs are required |
| Generate new assessments | Two algorithms, prompts, rubric, and a fictional Markdown plan | The scoring steps require a provider key and live API calls |

The 244 original business plans, venture-linked expert assessments, raw model
outputs, retrieved passages, and historical logs are excluded for privacy and
commercial confidentiality. Aggregate exports contain 236–242 usable records
per criterion/configuration; the `n` columns preserve these reported counts.
Synthetic examples are not part of the study dataset. See
[`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md).

## Recreate the aggregate results

Use Python 3.10 or later from the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/reproduce_manuscript.py
```

This offline command checks the exports, prints the headline counts, and writes
tables to `build/tables/` and numerical figures as PDF and SVG to
`build/figures/`. It makes no LLM calls and does not recalculate the exported
confidence intervals. The manuscript's author-edited annotation and workflow
diagrams are preserved in the LaTeX project rather than regenerated here.

## Statistical tests and restricted-data analysis

```bash
PYTHONPATH=src python -m unittest discover \
  -s src/sustainability/criterion_evaluation/tests -p 'test_*.py' -v
```

These are synthetic tests of metric definitions, cohort alignment, bootstrap
resampling, and input handling. See [`docs/evaluation.md`](docs/evaluation.md)
for the evaluation command and required confidential input formats. The
default evaluation output remains under the ignored `private/` directory;
record-level intermediates are not public result exports.

## Assessment algorithms

See [`docs/inference.md`](docs/inference.md) for installation and commands for
both model families and both algorithms. The supplied pipelines use the
OpenRouter endpoint through the OpenAI Python client. Their environment
variable `OPENAI_API_KEY` therefore expects an OpenRouter key.

The original prompts, scoring logic, retrieval settings, and statistical
calculations are retained. Only file locations and archive portability were
adapted in the method and evaluation code. Live model calls were not made while
preparing this package. New calls are not guaranteed to reproduce historical
model responses.

## Files and interpretation

| Path | Contents |
| --- | --- |
| `data/results/criterion_catalog.csv` | 37 criterion identifiers, topics, categories, and wording |
| `data/results/criterion_results.csv` | 148 criterion/configuration rows, including correlations, errors, bias, expert-disagreement measures, and bootstrap intervals |
| `data/results/full_context_rag_comparisons.csv` | 74 paired comparisons, two model families by 37 criteria |
| `src/sustainability/pipelines/` | Full-context and criterion-level RAG inference |
| `src/sustainability/prompts/` | Scoring instructions |
| `src/sustainability/config/` | Rubric JSON files |
| `src/sustainability/criterion_evaluation/` | Alignment, metric computation, bootstrap evaluation, reporting, plotting, and tests |
| `scripts/` | Verification and manuscript table/figure regeneration from aggregates |
| `examples/synthetic_business_plan.md` | A clearly fictional input-format example |

Human-relative MAE is expert–expert MAE divided by model–expert MAE; values
above one indicate lower model–expert disagreement. The paired difference is
RAG MAE minus full-context MAE; positive values favor full context. Bootstrap
intervals are pointwise and exploratory. See
[`docs/data_dictionary.md`](docs/data_dictionary.md) for all exported fields.

