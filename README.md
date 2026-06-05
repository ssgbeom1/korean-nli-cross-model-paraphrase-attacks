# Korean NLI Cross-Model Paraphrase Robustness

Reproducibility package for the paper:

> **Cross-Model Adversarial Robustness Evaluation of LLMs on Korean Natural Language Inference**

This repository contains the code, data splits, model outputs, analysis tables, and figures used for a Korean NLI robustness study based on cross-model, meaning-preserving paraphrase attacks.

## Overview

We evaluate four API-served LLMs as both paraphrase generators and NLI target models:

- Gemini 3.1 Pro Preview
- GPT-5.2
- Claude Sonnet 4.5
- HyperCLOVA X HCX-005

The main experiment uses a **4 x 4 generator-target design** on the KLUE-NLI validation split. Each model generates Korean paraphrases of the hypothesis sentence, and each target model is evaluated on the original and paraphrased premise-hypothesis pairs.

The package also includes Back Translation comparison, expanded human label-invariance validation, PAWS-X auxiliary evaluation, Korean linguistic feature analysis, statistical summaries, and integrity checks.

## Headline Results

- Shared valid LLM set: **2,209 samples**
- LLM inference rows: **35,344**
- Overall pooled ASR: **6.05%**
- LLM human validation: **600 attack-success cases**
- Back Translation human validation: **180 attack-success cases**
- Pooled adjusted ASR on the LLM-BT comparison subset:
  - LLM paraphrases: **5.60%**
  - Back Translation: **4.88%**

Main qualitative findings:

- Cross-model transferability is asymmetric.
- HyperCLOVA X shows the highest generator-side ASR and the highest target-side vulnerability in this evaluation setting.
- Neutral-labeled samples are more vulnerable than Entailment or Contradiction.
- LLM-generated paraphrases preserve labels more reliably than Back Translation.
- Korean particle-count and spacing/token-count changes are significantly associated with attack success.

## Repository Layout

```text
code/
  pipeline/                   Data loading, generation, filtering, shared-set construction, evaluation
  analysis/                   Analysis, validation sampling, statistical summaries, figures
  shell/                      PowerShell and shell runners
  utils/                      Shared API, parsing, cache, and metric helpers

data/
  01_sampled_source/          KLUE-NLI validation sample
  02_generated_attacks/       Generated LLM and baseline paraphrases
  03_validated_attacks/       BERTScore-filtered attack files
  04_shared_valid_set/        Shared valid subsets and IDs
  05_external_benchmark/      PAWS-X Korean/English samples

results/
  01_cross_model_evaluation/  4 x 4 evaluation outputs and original-prediction caches
  02_backtranslation_evaluation/
                              Back Translation target-evaluation outputs
  03_summary_tables/          Main ASR, adjusted ASR, PAWS-X, linguistic, and statistical summaries
  05_human_evaluation/        Human-validation sampling metadata and aggregate outputs
  06_external_benchmark/      PAWS-X model outputs
  06_integrity_checks/        Integrity verification reports

figures/                      Paper figures generated from the analysis outputs
tools/                        Utility scripts for integrity/cache synchronization
```

## Key Files

Primary manuscript-facing outputs:

```text
results/03_summary_tables/4x4_analysis_summary.txt
results/03_summary_tables/human_adjusted_asr_summary.txt
results/03_summary_tables/bt_vs_llm_human_adjusted_comparison.csv
results/03_summary_tables/pawsx_external_benchmark_summary.csv
results/03_summary_tables/korean_linguistic_feature_summary.txt
results/06_integrity_checks/4x4_integrity_summary.txt
```

Primary figures:

```text
figures/fig_pipeline.png
figures/fig_asr_heatmap.png
figures/fig_gen_tgt_asr.png
figures/fig_label_transition.png
figures/fig_quality_comparison.png
figures/fig_adjusted_asr.png
figures/fig_korean_linguistic_features.png
```

## Environment

Install dependencies:

```bash
pip install -r code/requirements.txt
```

Create a local environment file only if you need to rerun API-based generation or evaluation:

```bash
cp .env.example .env
```

The `.env` file is intentionally ignored by Git.

## Reproducing the Pipeline

The pipeline is organized as:

```text
code/pipeline/01_load_data.py
code/pipeline/02_generate_llm_attacks.py
code/pipeline/03_generate_baselines.py
code/pipeline/04_validate_attacks.py
code/pipeline/05_build_shared_valid.py
code/pipeline/06_evaluate_attacks.py
```

Convenience runners:

```powershell
powershell -ExecutionPolicy Bypass -File .\code\shell\run_pipeline.ps1
powershell -ExecutionPolicy Bypass -File .\code\shell\run_4x4_reeval_gemini31.ps1
powershell -ExecutionPolicy Bypass -File .\code\shell\run_bt_evaluation.ps1
powershell -ExecutionPolicy Bypass -File .\code\shell\run_pawsx_evaluation.ps1
```

Note that full regeneration requires access to the corresponding API providers and may produce different outputs if provider-side models change.

## Verification

Run the 4 x 4 integrity check:

```bash
python code/analysis/02_verify_4x4_integrity.py --expected_rows 2209
```

Expected final check:

```text
Original caches OK: 4/4
4x4 cells OK: 16/16
Cell original-cache mismatches: 0
Fatal problems: 0
```

Refresh manuscript evidence summaries:

```bash
python code/analysis/19_write_manuscript_revision_evidence.py
```

## Human Validation

The public package includes validation sampling metadata and aggregate results. Raw per-rater annotation sheets and private answer keys are intentionally excluded from Git tracking. The manuscript reports:

- LLM label-invariance validation: 600 sampled attack-success cases
- Back Translation label-invariance validation: 180 sampled attack-success cases
- Three independent native Korean-speaking evaluators per case

## Notes on Data and Model Outputs

Dataset-derived files and model outputs may be subject to the terms of the original data sources and API providers. Please verify redistribution and reuse requirements before using these files outside research reproducibility or review contexts.

## Citation

If you use this repository, please cite the associated manuscript:

```bibtex
@misc{kim2026korean_nli_cross_model_paraphrase,
  title  = {Cross-Model Adversarial Robustness Evaluation of LLMs on Korean Natural Language Inference},
  author = {Kim, Beomseok and Choi, Hoansuk and Yoo, Namhyun and Yang, Jinhong},
  year   = {2026},
  note   = {Reproducibility package}
}
```
