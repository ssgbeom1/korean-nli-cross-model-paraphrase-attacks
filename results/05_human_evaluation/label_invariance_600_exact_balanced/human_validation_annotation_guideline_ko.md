# 최종 Human Validation 평가 지침

각 행은 최종 4x4 attack-success 사례에서 뽑은 고유 paraphrase 문항입니다.
동일한 paraphrase가 여러 target model에서 attack-success가 된 경우에도 평가자는 한 번만 평가합니다.

## 평가자가 할 일

평가자는 `premise`와 `paraphrased_hypothesis`의 NLI 관계만 판단합니다.

`original_hypothesis`는 paraphrase가 원래 가설의 의미를 보존했는지 참고하기 위한 문장입니다.
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

## 중요 원칙

평가자에게 answer key는 제공하지 않습니다.
정답 label은 나중에 연구자가 `human_validation_answer_key.csv`와 비교해서 label preservation 여부를 계산합니다.

평가자는 모델 예측 결과나 attack success 여부를 보지 않고, 문장 관계만 독립적으로 판단해야 합니다.


Sampling note: This exact-balanced package contains 600 annotation rows with 597 unique source paraphrase units. Three GPT-Contradiction source units are repeated to preserve the 50-per-generator-label annotation layout.

