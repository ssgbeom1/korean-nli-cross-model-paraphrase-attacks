# korean-nli-cross-model-paraphrase-attacks

Clean reproducibility package for the revised Korean NLI cross-model paraphrase-attack evaluation.

## Scope

This repository keeps the data, scripts, and result tables needed to inspect or rerun the revised experiments:

- KLUE-NLI sampled source set
- LLM-generated and baseline paraphrase attack files
- BERTScore-filtered valid attack files
- 2209-sample LLM shared-valid set
- 4x4 generator-target evaluation outputs
- Back Translation comparison outputs
- 600-case LLM human-validation package and 180-case Back Translation human-validation package
- PAWS-X Korean/English external benchmark sample and outputs
- Korean linguistic feature analysis
- statistical summaries, integrity checks, and reviewer-response evidence tables

Temporary logs, debug dumps, bytecode caches, API keys, and one-off run records are not part of the clean package.

## Repository Layout

```text
code/
  analysis/                   Current analysis, sampling, verification, and summary scripts
  analysis/_legacy_archive/   Superseded analysis scripts kept for audit history
  pipeline/                   Data loading, generation, validation, shared-set, and evaluation pipeline
  shell/                      PowerShell and shell pipeline runners
  utils/                      Shared API and metric helpers
data/
  01_sampled_source/          Sampled KLUE-NLI validation source set
  02_generated_attacks/       Generated LLM and baseline attack files
  03_validated_attacks/       BERTScore-filtered attack files
  04_shared_valid_set/        Shared-valid subsets and shared ids
  05_external_benchmark/      PAWS-X Korean/English benchmark samples
results/
  01_cross_model_evaluation/  4x4 evaluation outputs and original prediction caches
  02_backtranslation_evaluation/
                              Back Translation target-evaluation outputs
  03_summary_tables/          ASR, CI, human-adjusted, PAWS-X, and manuscript summary tables
  05_human_evaluation/        Human-validation packages and aggregation outputs
  06_external_benchmark/      PAWS-X model outputs
  06_integrity_checks/        Integrity verification reports
```

## Current Evaluation State

The LLM shared-valid set contains 2209 samples.

Integrity verification passed with:

- original prediction caches OK: 4/4
- 4x4 generator-target cells OK: 16/16
- rows per cell: 2209
- missing `pred_original`: 0
- missing `pred_attacked`: 0
- duplicate ids: 0
- prediction labels restricted to 0/1/2
- derived correctness and attack-success columns consistent with predictions

Primary outputs:

```text
results/03_summary_tables/4x4_analysis_summary.txt
results/03_summary_tables/human_adjusted_asr_summary.txt
results/03_summary_tables/bt_vs_llm_human_adjusted_comparison.csv
results/03_summary_tables/pawsx_external_benchmark_summary.csv
results/03_summary_tables/korean_linguistic_feature_summary.txt
results/06_integrity_checks/4x4_integrity_summary.txt
```

## Environment

Install dependencies:

```bash
pip install -r code/requirements.txt
```

Create a local `.env` file from `.env.example` and fill in API credentials only if you need to rerun API-based generation or evaluation.

```bash
cp .env.example .env
```

The `.env` file is intentionally ignored by Git.

## Main Pipeline

The core pipeline is organized as:

```text
code/pipeline/01_load_data.py
code/pipeline/02_generate_llm_attacks.py
code/pipeline/03_generate_baselines.py
code/pipeline/04_validate_attacks.py
code/pipeline/05_build_shared_valid.py
code/pipeline/06_evaluate_attacks.py
```

PowerShell runners:

```powershell
powershell -ExecutionPolicy Bypass -File .\code\shell\run_4x4_reeval_gemini31.ps1
powershell -ExecutionPolicy Bypass -File .\code\shell\run_bt_evaluation.ps1
powershell -ExecutionPolicy Bypass -File .\code\shell\run_pawsx_evaluation.ps1
```

## Key Analysis Scripts

```text
code/analysis/01_prepare_bt_shared.py
code/analysis/02_verify_4x4_integrity.py
code/analysis/03_generate_4x4_analysis.py
code/analysis/04_statistical_analysis_summary.py
code/analysis/05_calculate_quality_metrics.py
code/analysis/06_generate_quality_figure.py
code/analysis/07_sample_human_validation.py
code/analysis/08_make_llm_human_validation_exact_balanced.py
code/analysis/09_sample_bt_human_validation.py
code/analysis/10_make_bt_human_validation_exact_balanced.py
code/analysis/11_aggregate_human_validation.py
code/analysis/12_compute_human_adjusted_asr.py
code/analysis/13_write_bt_llm_comparison_table.py
code/analysis/14_prepare_pawsx_from_parquet.py
code/analysis/15_evaluate_pawsx_external_benchmark.py
code/analysis/16_summarize_pawsx_external_benchmark.py
code/analysis/17_korean_linguistic_feature_analysis.py
code/analysis/18_generate_paper_figures.py
code/analysis/19_write_manuscript_revision_evidence.py
```

## Verification

To re-run the 4x4 integrity check:

```bash
python code/analysis/02_verify_4x4_integrity.py --expected_rows 2209
```

To refresh manuscript evidence tables after result updates:

```bash
python code/analysis/19_write_manuscript_revision_evidence.py
```

## Notes

Dataset-derived files and model outputs may be subject to the terms of the original data sources and API providers. Verify redistribution requirements before public release.
