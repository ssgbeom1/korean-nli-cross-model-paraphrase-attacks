# Human Validation Annotation Guideline

Each row is one unique paraphrase unit sampled from 4x4 attack-success cases.
Duplicate target-model instances are removed before annotation. A separate mapping file
links each annotation unit to all generator-target attack-success instances represented by it.

## Task

Judge only the relation between `premise` and `paraphrased_hypothesis`.
Do not re-label the original premise-hypothesis pair. The original gold label is stored
only in the answer key and will be used during aggregation.

The `original_hypothesis` column is provided only to help identify semantic drift.

## Label Values

- 0 = Entailment
- 1 = Neutral
- 2 = Contradiction
- U = Unclear or not reliably judgeable

## Columns to Fill

- `paraphrased_pair_label`: NLI label for premise + paraphrased hypothesis.
- `semantic_drift_type`: choose one taxonomy label below.
- `fluency_defect`: Y if the paraphrase has a grammar, formatting, or readability defect; otherwise N.
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

Use `no_semantic_drift` when the paraphrase preserves the original hypothesis meaning.
Use a drift label when the paraphrase changes meaning, narrows or broadens the claim,
changes entities/numbers/time, changes negation/modality/scope, changes causality or
temporal relation, is grammatically defective, or is ambiguous.


Sampling note: This exact-balanced package contains 600 annotation rows with 597 unique source paraphrase units. Three GPT-Contradiction source units are repeated to preserve the 50-per-generator-label annotation layout.

