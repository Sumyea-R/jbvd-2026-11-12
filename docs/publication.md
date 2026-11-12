# Publication and deposit

This package accompanies *AI Assisted Assessment of Sustainability Impact Potential
for Early Stage Ventures*. It is prepared for sharing; no public repository or
archival DOI has been created by this preparation.

The code, prompts, operational criterion catalog, and aggregate result CSVs are
research materials suitable for repository deposit. Elsevier's definition of
research data includes software, code, algorithms, methods, and raw or processed
data files. A journal supplement and a repository deposit serve complementary
purposes: the supplement accompanies the article, while an archived version
provides a persistent, independently citable record.

## Files to submit

Upload `JBVD_Research_Materials.zip` as Supplementary Material S1 at the same time
as the manuscript. Its three aggregate CSVs are `criterion_catalog.csv`,
`criterion_results.csv`, and `full_context_rag_comparisons.csv`; the catalog
defines the 37 criteria, and the other two files report criterion-level metrics
and paired algorithm comparisons. The ZIP also contains the assessment and
evaluation source code, prompts, documentation, and regeneration scripts.

Use this caption, which matches the manuscript:

> S1. JBVD_Research_Materials.zip — Source code, scoring prompts, criterion
> definitions, aggregate criterion-level results and paired method comparisons,
> and instructions for regenerating tables and figures. Confidential business
> plans and venture-level records are excluded.

The methodology, evaluation, and results sections cite S1. The ZIP is a frozen
copy: if it is updated before submission, use the same final version for the
journal supplement and archival deposit.

## Create the public source repository

The existing `Sumyea-R/sustainabilty_assessment` repository and its history contain
restricted research inputs. Keep that repository private. The preparation branch
and draft pull request are review locations, not public data deposits.

Create a separate empty GitHub repository, for example
`jbvd-sustainability-assessment`, and populate it only with the contents of
`JBVD_Research_Materials.zip`. Initialize a new Git history; do not push the
existing repository's branches, tags, history, or `.git` directory. The prepared
GitHub connection can update an existing repository but cannot create this new
repository, so creating it is a remaining owner action.

Choose the reuse terms for the code, documentation, and aggregate CSVs before
public release. This preparation assigns no new license. Zenodo requires license
metadata and defaults to CC BY; select the intended terms explicitly rather than
accepting that default accidentally. A software license and a data/documentation
license may differ.

## Archive the submitted version

Two practical routes are available:

1. Upload the final ZIP directly to Zenodo or Mendeley Data, describe it using the
   metadata below, and publish the record. This route does not require a public
   GitHub repository.
2. After creating the clean public GitHub repository, enable its Zenodo
   integration and create a `v1.0.0` release containing the submitted package.
   Zenodo archives GitHub releases and issues a DOI.

Use the DOI of the specific submitted version in the manuscript's data
availability section, add a data/software reference to the bibliography, and
provide the verified repository/DOI link during submission. A private GitHub URL,
draft pull request, proposed repository name, or pending DOI is not a public
dataset link. The source comment in `sections/declarations.tex` marks where the
verified deposit citation must be added.

Suggested deposit metadata:

| Field | Value |
| --- | --- |
| Title | AI Assisted Assessment of Sustainability Impact Potential for Early Stage Ventures: Code and Aggregate Results |
| Creators | Sumyea Rahman; Jan Kratzer |
| Affiliation | Technical University Berlin |
| Version | v1.0.0 |
| Preparation date | 12 November 2026 |
| Description | Assessment and evaluation source code, scoring prompts, operational esSROI criterion catalog, aggregated criterion-level results for four GPT/Gemini algorithm configurations, paired full-context/RAG comparisons, and regeneration instructions. Original business plans and venture-level records are excluded. |
| License | Select the authors' intended code/data terms before publication; none is assigned by this package preparation. |
| Publication date | Use the actual archive publication date. |
| Related manuscript | AI Assisted Assessment of Sustainability Impact Potential for Early Stage Ventures; add its published DOI when one exists. |

Do not enter a journal DOI, public repository URL, publication date, or license
that has not actually been established.

## Restricted data and reproducibility

The 244 original business plans cannot be shared publicly because they contain
personal and confidential venture information. This package also excludes
venture-linked expert ratings, model predictions, supporting passages, and run
logs. It contains aggregate outputs rather than the individual records from
which they were calculated.

Readers can inspect the algorithms and regenerate the reported summaries and
plots from the released aggregates. Recomputing correlations, errors, bootstrap
intervals, or paired differences from individual assessments requires the
restricted records. Re-running LLM assessment additionally requires permitted
input documents and access to the relevant model service. No access-on-request
arrangement is promised by this package.

## Primary guidance

- Elsevier research data definition and repository sharing:
  https://www.elsevier.com/researcher/author/tools-and-resources/research-data
- Elsevier Option C and the unavailable-data statement:
  https://www.elsevier.com/researcher/author/tools-and-resources/research-data/data-guidelines
- GitHub releases archived with Zenodo:
  https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content
- Zenodo upload and DOI workflow:
  https://help.zenodo.org/docs/get-started/quickstart/
- Zenodo version-specific records:
  https://help.zenodo.org/docs/deposit/manage-versions/
- Zenodo license fields and mixed-license uploads:
  https://help.zenodo.org/docs/deposit/describe-records/licenses/
- GitHub license guidance:
  https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository
