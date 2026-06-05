# Back Translation Human Validation Guideline

Each row is one unique Back Translation attack-success paraphrase unit.
Duplicate units across target models have been removed.

## Task

Judge only the NLI relation between `premise` and `backtranslated_hypothesis`.
Do not re-label the original premise-hypothesis pair. The original gold label is stored
only in the answer key and will be used during aggregation.

The `original_hypothesis` column is provided only to help identify semantic drift.

## Label Values

- 0 = Entailment
- 1 = Neutral
- 2 = Contradiction
- U = Unclear or not reliably judgeable

## Columns to Fill

- `paraphrased_pair_label`: NLI label for premise + backtranslated hypothesis.
- `semantic_drift_type`: choose one taxonomy label below.
- `fluency_defect`: Y if the backtranslation has a grammar, formatting, or readability defect; otherwise N.
- `annotator_confidence`: high, medium, or low.

## Semantic Drift Taxonomy

- no_semantic_drift
- specificity_shift
- entity_number_time_change
- negation_modality_scope_drift
- causality_temporal_relation_drift
- fluency_grammar_defect
- ambiguous_label
- other


Note: This exact-balanced package contains 180 annotation rows with 179 unique source BT units. One duplicate source unit is included to preserve the 15-per-target-label sampling layout.
