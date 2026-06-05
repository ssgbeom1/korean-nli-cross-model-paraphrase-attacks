# Manuscript Revision Evidence Summary

## Core 4x4 Evaluation

- 4x4 evaluation rows: 35,344; original-correct denominator: 31,896; attack successes: 1,931.
- Overall raw ASR: 6.05% [5.80%, 6.32%].

## Human Validation

- LLM human validation: 552/600 label-preserving (92.00%); resolved LPR 94.52%.
- BT human validation: 111/180 label-preserving (61.67%); resolved LPR 67.27%.
- Label agreement: LLM Fleiss' kappa=0.8425; BT Fleiss' kappa=0.7042.

## BT vs LLM After Human Adjustment

- LLM paraphrase: raw ASR 6.14 [5.88, 6.42], human LPR 92.00 [89.55, 93.91], adjusted ASR 5.60 [5.35, 5.86].
- Back Translation: raw ASR 7.67 [7.10, 8.29], human LPR 61.67 [54.39, 68.46], adjusted ASR 4.88 [4.25, 5.51].

## PAWS-X External Benchmark

- Gemini: accuracy 83.30% (1000/1000 valid; parse failures 0).
- GPT: accuracy 80.10% (1000/1000 valid; parse failures 0).
- Claude Sonnet: accuracy 79.30% (1000/1000 valid; parse failures 0).
- HyperCLOVA X: accuracy 70.50% (1000/1000 valid; parse failures 0).

## Korean-Specific Linguistic Features

- particle_count_changed: present-ASR 6.50%, odds ratio 1.315, Holm p=1.476e-06.
- spacing_token_count_changed: present-ASR 6.42%, odds ratio 1.296, Holm p=1.663e-05.
- word_order_shift: present-ASR 7.09%, odds ratio 1.201, Holm p=0.1152.
- sentence_ending_changed: present-ASR 6.47%, odds ratio 1.114, Holm p=0.1152.
- negation_modality_changed: present-ASR 5.71%, odds ratio 0.932, Holm p=0.7922.
- number_changed: present-ASR 5.72%, odds ratio 0.940, Holm p=0.7975.

## Recommended Manuscript Inserts

- Add Algorithm 1 for the 4x4 paraphrase attack evaluation protocol.
- Add a Statistical Analysis subsection describing Wilson CI, bootstrap CI, Holm correction, and human-adjusted ASR.
- Add an Expanded Human Validation subsection with LLM 600 and BT 180 annotation designs.
- Add a BT Baseline subsection explaining why raw BT ASR must be human-adjusted.
- Add an External PAWS-X Validation subsection as auxiliary paraphrase benchmark evidence.
- Add a Threat Model subsection separating instability, robustness failure, and security-relevant vulnerability.
- Add a Korean Linguistic Feature Analysis subsection with feature-level ASR and odds-ratio results.
