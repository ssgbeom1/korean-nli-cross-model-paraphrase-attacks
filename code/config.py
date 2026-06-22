from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# The actual API prompts are intentionally written in Korean because the
# experiments evaluate Korean NLI. The English comments below are non-runtime
# renderings for international readability and do not alter the experiment.
#
# English rendering: You are a meaning-preserving paraphrase generator for
# robustness evaluation. Rewrite the given Korean sentence by changing only its
# surface form while preserving its truth-conditional and entailment meaning.
# Allowed changes include lexical substitution, structural reordering, and
# harmless expansions. Do not change facts, quantities, names, negation,
# causality, scope, or conditions. Output exactly one paraphrased sentence.
PARAPHRASE_SYSTEM_PROMPT = """당신은 NLP 시스템의 강건성(robustness) 평가를 위한 의미 보존 패러프레이즈 생성기입니다.
주어진 문장의 의미(참/거짓 관계 및 함의)는 바꾸지 말고, 표면 형태만 바꾼 한국어 문장으로 다시 작성하세요.

허용되는 변환 예시:
1) 어휘 교체: 동의어, 격식체, 한자어, 고유어 등으로 교체
2) 구조 변경: 어순 위치, 능동/수동 전환, 수식어 위치 변경
3) 무해한 확장: 의미를 바꾸지 않는 부사, 관용적 표현, 형식구 추가

금지 사항:
- 사실관계, 수치, 인명/지명, 부정 여부, 인과관계, 범위/조건을 바꾸지 말 것
- 새 정보 추가 또는 기존 정보 삭제 금지
- 두 문장 이상 출력 금지

출력 규칙: 패러프레이즈 문장 1개만 출력"""

PARAPHRASE_USER_TEMPLATE = "원본 문장: {text}\n\n패러프레이즈:"

# English rendering: This is a natural language inference task. Classify the
# relation between the premise and hypothesis as 0 entailment, 1 neutral, or
# 2 contradiction. Output only one digit, without explanation or extra text.
NLI_EVAL_PROMPT = """다음은 자연어 추론(NLI) 문제입니다.
Premise(전제)와 Hypothesis(가설)의 관계를 0/1/2 중 하나로 분류하세요.

라벨 정의(Label definitions):
0 = entailment (전제가 가설을 함의)
1 = neutral (중립)
2 = contradiction (모순)

규칙(Rules):
- 출력은 반드시 숫자 0 또는 1 또는 2 한 글자만
- 설명/단어/기호/줄바꿈 금지

Premise: {premise}
Hypothesis: {hypothesis}
Label:"""


DEFAULT_MODELS = {
    "clova": os.getenv("CLOVA_MODEL", "HCX-005"),
    "gemini": os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview"),
    "openai": os.getenv("OPENAI_MODEL", "gpt-5.2"),
    "sonnet": os.getenv("SONNET_MODEL", "claude-sonnet-4-5"),
}

GENERATORS = ["clova", "gemini", "openai", "sonnet"]
TARGETS = ["clova", "gemini", "openai", "sonnet"]

GENERATION_PARAMS = {
    "clova": dict(temperature=0.7, max_tokens=256, top_p=0.8, repeat_penalty=1.1),
    "gemini": dict(
        temperature=0.7,
        max_tokens=int(os.getenv("GEMINI_GENERATION_MAX_TOKENS", "256")),
        thinking=os.getenv("GEMINI_GENERATION_THINKING", "low"),
    ),
    "openai": dict(temperature=0.7, max_tokens=256),
    "sonnet": dict(temperature=0.7, max_tokens=256),
}

EVAL_PARAMS = {
    "clova": dict(temperature=0.0, max_tokens=16, repeat_penalty=1.1),
    "gemini": dict(temperature=0.0, max_tokens=256, thinking="low"),
    "openai": dict(temperature=0.0, max_tokens=16),
    "sonnet": dict(temperature=0.0, max_tokens=16),
}

BERTSCORE_THRESHOLD = 0.8
BERTSCORE_LANG = "ko"

DATA_DIR = "data"
RAW_DIR = f"{DATA_DIR}/raw"
ATTACK_DIR = f"{DATA_DIR}/attacks"
VALIDATED_DIR = f"{DATA_DIR}/validated"
SHARED_DIR = f"{DATA_DIR}/validated/shared"

RESULTS_DIR = "results"
MATRIX_DIR = f"{RESULTS_DIR}/shared_matrix"
BT_EVAL_DIR = f"{RESULTS_DIR}/bt_eval"
ASR_DIR = f"{RESULTS_DIR}/asr_analysis"

LABEL_NAMES = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
LABEL_SHORT = {0: "Ent", 1: "Neu", 2: "Con"}

SAMPLED_SOURCE_DIR = "data/01_sampled_source"
GENERATED_ATTACK_DIR = "data/02_generated_attacks"
VALIDATED_ATTACK_DIR = "data/03_validated_attacks"
SHARED_VALID_DIR = "data/04_shared_valid_set"

CROSS_MODEL_EVAL_DIR = "results/01_cross_model_evaluation"
CROSS_MODEL_CACHE_DIR = f"{CROSS_MODEL_EVAL_DIR}/original_predictions"
BACKTRANSLATION_EVAL_DIR = "results/02_backtranslation_evaluation"
SUMMARY_TABLE_DIR = "results/03_summary_tables"
QUALITY_RESULT_DIR = "results/04_quality_analysis"
HUMAN_EVAL_RESULT_DIR = "results/05_human_evaluation"

CROSS_MODEL_MATRIX_FILENAME = "cross_model_asr_matrix.csv"
GENERATOR_TARGET_ASR_CI_FILENAME = "generator_target_asr_ci.csv"
GENERATOR_TARGET_ASR_CI_TEXT_FILENAME = "generator_target_asr_ci_summary.txt"
LABELWISE_ASR_FILENAME = "labelwise_asr_summary.csv"
TRANSITION_SUMMARY_FILENAME = "label_transition_summary.csv"
TRANSITION_SUMMARY_TEXT_FILENAME = "label_transition_summary.txt"
BT_VS_LLM_FILENAME = "backtranslation_vs_llm_summary.csv"
BT_VS_LLM_TEXT_FILENAME = "backtranslation_vs_llm_summary.txt"
QUALITY_SUMMARY_FILENAME = "paraphrase_quality_summary.csv"
QUALITY_SUMMARY_TEXT_FILENAME = "paraphrase_quality_summary.txt"
QUALITY_TABLE_FILENAME = "table_paraphrase_quality.tex"
ORIGINAL_CACHE_SUMMARY_FILENAME = "original_prediction_cache_summary.csv"

QUALITY_HUMAN_EVAL_DIR = f"{HUMAN_EVAL_RESULT_DIR}/quality"
LABEL_INVARIANCE_DIR = f"{HUMAN_EVAL_RESULT_DIR}/label_invariance"
EQUAL_CONDITION_DIR = f"{HUMAN_EVAL_RESULT_DIR}/equal_condition"

MODEL_FILE_KEYS = {
    "clova": "hyperclova_x",
    "gemini": "gemini",
    "openai": "gpt",
    "sonnet": "claude_sonnet",
}

BASELINE_FILE_KEYS = {
    "backtranslation": "backtranslation",
    "bert_attack": "bert_attack",
    "eda": "eda",
}


def model_file_key(model_name: str) -> str:
    return MODEL_FILE_KEYS[model_name]


def baseline_file_key(method_name: str) -> str:
    return BASELINE_FILE_KEYS[method_name]


def sampled_source_filename(n: int = 3000) -> str:
    return f"klue_nli_validation_sample_{n}.csv"


def llm_attack_filename(generator: str, n: int = 3000) -> str:
    return f"{model_file_key(generator)}_attacks_{n}.csv"


def baseline_attack_filename(method: str, n: int = 3000) -> str:
    return f"{baseline_file_key(method)}_attacks_{n}.csv"


def llm_validated_filename(generator: str, threshold_tag: str = "bert80") -> str:
    return f"{model_file_key(generator)}_valid_{threshold_tag}.csv"


def baseline_validated_filename(method: str, threshold_tag: str = "bert80") -> str:
    return f"{baseline_file_key(method)}_valid_{threshold_tag}.csv"


def shared_valid_filename(generator: str, threshold_tag: str = "bert80") -> str:
    return f"{model_file_key(generator)}_shared_valid_{threshold_tag}.csv"


def backtranslation_shared_valid_filename(threshold_tag: str = "bert80") -> str:
    return f"backtranslation_shared_valid_{threshold_tag}.csv"


def shared_valid_ids_filename() -> str:
    return "shared_valid_ids.csv"


def cross_model_generator_dirname(generator: str) -> str:
    return f"{model_file_key(generator)}_as_generator"


def target_output_filename(target: str) -> str:
    return f"to_{model_file_key(target)}.csv"


def original_prediction_cache_filename(target: str) -> str:
    return f"{model_file_key(target)}_original_predictions.csv"
