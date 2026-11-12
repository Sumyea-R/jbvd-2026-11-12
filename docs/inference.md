# Assessment algorithms and prompts

The release contains the full-context and criterion-level hybrid RAG algorithms
used to generate rubric-based assessments. The code was selected from
`Sumyea-R/sustainabilty_assessment` at commit
`7b7ef52ac1a8e98901278e26502711a8d4567990`. The original 244 business plans,
individual expert assessments, model responses, retrieved passages, and run
logs are not included. The supplied example is synthetic, not a study venture.

## Install

Use Python 3.10 or later, from the repository root:

```bash
python -m pip install -r requirements-inference.txt
python -m pip install --no-deps -e .
```

The inference requirements are compatible lower bounds. They do not specify
the exact package versions used in the original experiments. RAG also needs
the `sentence-transformers/all-MiniLM-L6-v2` embedding model; its first use may
download model files. The BM25 implementation is included in the RAG script.

## Provider configuration

Both scripts use the OpenAI Python client with the OpenRouter endpoint
`https://openrouter.ai/api/v1`. Despite its environment-variable name,
`OPENAI_API_KEY` must contain an **OpenRouter API key** for these scripts.
Set that variable in the local environment without putting the key in source
files. Native OpenAI or Google API credentials are not interchangeable here.

The reported model IDs are `openai/gpt-5.2` and
`google/gemini-3.1-pro-preview`. Use `--model` to choose the provider model.
The code sends temperature zero, which reduces sampling randomness without
guaranteeing identical repeated responses. Availability and provider behavior
can change after the experiments.

## Full-context prompting

```bash
python -m sustainability.pipelines.forecast_sroi \
  --model openai/gpt-5.2 \
  --input-files examples/synthetic_business_plan.md \
  --max-files 1 \
  --output-subdir example_full_context_gpt
```

For Gemini, run the same command with
`--model google/gemini-3.1-pro-preview` and a different `--output-subdir`.
The complete Markdown plan and assessment criteria are placed in the prompt.
The JSON response contains criterion ratings, confidence scores, evidence,
missing criteria, and a narrative. Validation checks the topic structure and
rating counts before saving a response.

## Criterion-level hybrid RAG

```bash
python -m sustainability.pipelines.forecast_sroi_rag_question \
  --mode full \
  --model openai/gpt-5.2 \
  --input-files examples/synthetic_business_plan.md \
  --max-files 1 \
  --retrieval-top-k 5 \
  --neighbor-window 1 \
  --output-subdir example_rag_gpt
```

For Gemini, change `--model` and the output subdirectory as above. The original
RAG script supports three modes:

| Mode | Behavior | External scoring call |
| --- | --- | --- |
| `full` | Retrieve passages and score each topic | Yes |
| `retrieve` | Save criterion-specific retrieved passages | No |
| `score` | Score a previously saved retrieval output | Yes |

`--mode score` additionally requires `--retrieval-subdir` pointing to a saved
retrieval output under `private/outputs/`, or an absolute directory. The selected input
Markdown files are still used to identify the corresponding saved JSON files.
Retrieval-only mode does not need an API key, but it may need an embedding-model
download unless the model is already cached locally.

The default RAG settings preserve paragraph and heading context, target
1,400-character chunks with a nominal maximum of 1,800 characters and
one-paragraph overlap, combine BM25 (`k1=1.5`, `b=0.75`) with dense similarity,
and use reciprocal rank fusion (`k=60`). Five fused base hits are retained
before one-neighbor expansion. An individual long paragraph is not split by
the chunking routine, so it may exceed the nominal chunk size. Scoring is done
per topic using the raw retrieved evidence for each criterion.

## Inputs, outputs, and restricted material

Use `--input-files` to select only documents you are authorized to process.
Without that flag, the release scripts look for `private/plans/*.md`; this
release does not contain that dataset. A scoring command makes real provider
calls and can incur usage charges. The example commands above are examples for
the user to run; preparing this release made no model calls.

Outputs are written under `private/outputs/`, and local run manifests under
`private/results/logs/`. These files can contain input filenames, paths, quoted plan
content, and model explanations. They are excluded by the release's ignore
rules and are not part of its shareable research results. Do not add them to a
public repository when processing confidential documents.

The aggregate results supplied with the release reproduce the published
summaries and plots. The synthetic example illustrates input format and how
to invoke the scripts; it does not reproduce the 244-plan empirical study.

## Source selection and changes

| Included area | Purpose |
| --- | --- |
| `src/sustainability/pipelines/forecast_sroi.py` | Full-context assessment |
| `src/sustainability/pipelines/forecast_sroi_rag_question.py` | Criterion-level RAG assessment |
| `src/sustainability/prompts/sroi.py` | Original full-context prompt templates |
| `src/sustainability/prompts/sroi_rag_question.py` | Original RAG scoring prompt |
| `src/sustainability/config/*.json` | Assessment criteria and category mapping |
| `src/sustainability/validation/output_validation.py` | Output validation and atomic saves |
| `src/sustainability/utils/run_log.py` | Local run-manifest support |
| `src/sustainability/utils/paths.py` | Local paths and repository-root detection |

Only file locations and archive portability have changed in these upstream
method files. In `utils/paths.py`, repository-root detection also accepts a
`pyproject.toml` file beside `src/sustainability`, allowing downloaded source
archives without a `.git` directory to work. Its default output and log roots
are now under `private/`. In each pipeline, the default input directory is
`private/plans` instead of `data/processed`. The entire `private/` directory is
excluded from version control. These changes do not alter prompts, retrieval,
model calls, selection logic, scoring, or numerical evaluation.

Legacy composite-score evaluation, the demonstration web app, and generated
bytecode are omitted because they are not required by these two algorithms.
