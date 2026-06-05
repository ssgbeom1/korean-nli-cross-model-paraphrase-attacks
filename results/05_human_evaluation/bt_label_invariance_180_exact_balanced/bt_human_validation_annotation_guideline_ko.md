# BT Human Validation 평가 지침

각 행은 Back Translation attack-success 사례에서 뽑은 고유 문항입니다.
동일한 BT 문장이 여러 target model에서 attack-success가 된 경우에도 한 번만 평가합니다.

## 평가자가 할 일

평가자는 `premise`와 `backtranslated_hypothesis`의 NLI 관계만 판단합니다.
`original_hypothesis`는 의미 변화 여부를 참고하기 위한 문장입니다.
평가자는 original pair의 label을 다시 판단하지 않습니다.

## 채워야 하는 열

`paraphrased_pair_label`

- 0 = Entailment
- 1 = Neutral
- 2 = Contradiction
- U = 판단 불가 또는 애매함

`semantic_drift_type`

- no_semantic_drift
- specificity_shift
- entity_number_time_change
- negation_modality_scope_drift
- causality_temporal_relation_drift
- fluency_grammar_defect
- ambiguous_label
- other

`fluency_defect`

- Y = 문법, 형식, 가독성 문제가 있음
- N = 특별한 문제가 없음

`annotator_confidence`

- high
- medium
- low

평가자에게 answer key, model prediction, target model 정보는 제공하지 않습니다.


Note: This exact-balanced package contains 180 annotation rows with 179 unique source BT units. One duplicate source unit is included to preserve the 15-per-target-label sampling layout.
