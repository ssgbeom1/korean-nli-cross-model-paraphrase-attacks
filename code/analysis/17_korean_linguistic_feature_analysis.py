from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    CROSS_MODEL_EVAL_DIR,
    GENERATORS,
    LABEL_NAMES,
    SHARED_VALID_DIR,
    SUMMARY_TABLE_DIR,
    TARGETS,
    cross_model_generator_dirname,
    model_file_key,
    shared_valid_filename,
    target_output_filename,
)
from utils.original_cache import apply_original_cache  # noqa: E402


PARTICLES = [
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "에서",
    "으로",
    "로",
    "와",
    "과",
    "도",
    "만",
    "의",
]
NEGATION_MODALITY_TERMS = [
    "아니",
    "않",
    "못",
    "없",
    "전혀",
    "결코",
    "불가능",
    "가능",
    "해야",
    "수 있다",
    "수 없다",
]
SENTENCE_ENDINGS = [
    "습니다",
    "습니까",
    "했다",
    "한다",
    "였다",
    "이다",
    "다",
    "요",
    "죠",
    "까",
    "네",
    "음",
    "함",
]


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda idx: p_values[idx])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p_values[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def tokens(text: str) -> list[str]:
    return [token for token in str(text).strip().split() if token]


def token_set(text: str) -> set[str]:
    return set(tokens(text))


def count_terms(text: str, terms: list[str]) -> int:
    value = str(text)
    return sum(value.count(term) for term in terms)


def number_signature(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\d+(?:[.,]\d+)?", str(text)))


def sentence_ending(text: str) -> str:
    stripped = str(text).strip().rstrip(".!?。")
    if not stripped:
        return ""
    last_token = stripped.split()[-1]
    for ending in sorted(SENTENCE_ENDINGS, key=len, reverse=True):
        if last_token.endswith(ending):
            return ending
    return last_token[-1:]


def lexical_substitution_rate(original: str, attacked: str) -> float:
    original_set = token_set(original)
    attacked_set = token_set(attacked)
    union = original_set | attacked_set
    if not union:
        return 0.0
    return 1.0 - (len(original_set & attacked_set) / len(union))


def word_order_shift(original: str, attacked: str) -> int:
    original_tokens = tokens(original)
    attacked_tokens = tokens(attacked)
    if len(original_tokens) < 2 or len(attacked_tokens) < 2:
        return 0
    shared = [token for token in original_tokens if token in set(attacked_tokens)]
    if len(shared) < 2:
        return 0
    attacked_positions = {token: idx for idx, token in enumerate(attacked_tokens)}
    projected = [attacked_positions[token] for token in shared if token in attacked_positions]
    return int(projected != sorted(projected))


def extract_features(original: str, attacked: str) -> dict[str, float | int]:
    original_tokens = tokens(original)
    attacked_tokens = tokens(attacked)
    return {
        "particle_count_changed": int(count_terms(original, PARTICLES) != count_terms(attacked, PARTICLES)),
        "negation_modality_changed": int(
            count_terms(original, NEGATION_MODALITY_TERMS) != count_terms(attacked, NEGATION_MODALITY_TERMS)
        ),
        "number_changed": int(number_signature(original) != number_signature(attacked)),
        "sentence_ending_changed": int(sentence_ending(original) != sentence_ending(attacked)),
        "spacing_token_count_changed": int(len(original_tokens) != len(attacked_tokens)),
        "word_order_shift": word_order_shift(original, attacked),
        "lexical_substitution_rate": lexical_substitution_rate(original, attacked),
    }


def load_rows() -> pd.DataFrame:
    frames = []
    for generator in GENERATORS:
        generator_key = model_file_key(generator)
        text_path = Path(SHARED_VALID_DIR) / shared_valid_filename(generator)
        if not text_path.exists():
            continue
        text_df = pd.read_csv(text_path, encoding="utf-8-sig")[
            ["id", "hypothesis", "attacked_hypothesis", "bert_score"]
        ].copy()
        for target in TARGETS:
            target_key = model_file_key(target)
            eval_path = Path(CROSS_MODEL_EVAL_DIR) / cross_model_generator_dirname(generator) / target_output_filename(target)
            if not eval_path.exists():
                continue
            eval_df = pd.read_csv(eval_path, encoding="utf-8-sig")
            eval_df = apply_original_cache(eval_df, CROSS_MODEL_EVAL_DIR, target_key)
            merged = eval_df.merge(text_df, on="id", how="left", validate="many_to_one")
            merged["generator_key"] = generator_key
            merged["target_key"] = target_key
            frames.append(merged)
    if not frames:
        raise FileNotFoundError("No cross-model rows found.")
    out = pd.concat(frames, ignore_index=True)
    for column in ["label", "original_correct", "attack_success"]:
        out[column] = out[column].astype(int)
    out["label_name"] = out["label"].map(LABEL_NAMES)
    return out


def binary_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    correct = df[df["original_correct"] == 1].copy()
    feature_cols = [
        "particle_count_changed",
        "negation_modality_changed",
        "number_changed",
        "sentence_ending_changed",
        "spacing_token_count_changed",
        "word_order_shift",
    ]
    rows = []
    p_values = []
    for feature in feature_cols:
        present = correct[correct[feature] == 1]
        absent = correct[correct[feature] == 0]
        a = int(present["attack_success"].sum())
        b = len(present) - a
        c = int(absent["attack_success"].sum())
        d = len(absent) - c
        odds_ratio, p_value = fisher_exact([[a, b], [c, d]]) if len(present) and len(absent) else (np.nan, 1.0)
        p_values.append(float(p_value))
        for value, subset in [(1, present), (0, absent)]:
            n = len(subset)
            successes = int(subset["attack_success"].sum())
            lo, hi = wilson_ci(successes, n)
            rows.append(
                {
                    "feature": feature,
                    "feature_present": value,
                    "n_original_correct": n,
                    "n_attack_success": successes,
                    "asr_pct": successes / n * 100 if n else 0.0,
                    "asr_ci_lower_pct": lo * 100,
                    "asr_ci_upper_pct": hi * 100,
                    "odds_ratio_present_vs_absent": odds_ratio,
                    "fisher_p": p_value,
                }
            )
    adjusted = holm_adjust(p_values)
    feature_to_holm = dict(zip(feature_cols, adjusted))
    out = pd.DataFrame(rows)
    out["holm_p"] = out["feature"].map(feature_to_holm)
    return out


def label_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    correct = df[df["original_correct"] == 1].copy()
    feature_cols = [
        "particle_count_changed",
        "negation_modality_changed",
        "number_changed",
        "sentence_ending_changed",
        "spacing_token_count_changed",
        "word_order_shift",
    ]
    rows = []
    for label, label_df in correct.groupby("label", sort=True):
        for feature in feature_cols:
            present = label_df[label_df[feature] == 1]
            absent = label_df[label_df[feature] == 0]
            present_success = int(present["attack_success"].sum())
            absent_success = int(absent["attack_success"].sum())
            present_asr = present_success / len(present) * 100 if len(present) else 0.0
            absent_asr = absent_success / len(absent) * 100 if len(absent) else 0.0
            rows.append(
                {
                    "label": int(label),
                    "label_name": LABEL_NAMES[int(label)],
                    "feature": feature,
                    "n_present": len(present),
                    "success_present": present_success,
                    "asr_present_pct": present_asr,
                    "n_absent": len(absent),
                    "success_absent": absent_success,
                    "asr_absent_pct": absent_asr,
                    "asr_difference_pct_points": present_asr - absent_asr,
                }
            )
    return pd.DataFrame(rows)


def lexical_summary(df: pd.DataFrame) -> pd.DataFrame:
    correct = df[df["original_correct"] == 1].copy()
    correct["lexical_substitution_bin"] = pd.cut(
        correct["lexical_substitution_rate"],
        bins=[-0.001, 0.25, 0.50, 0.75, 1.0],
        labels=["0-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00"],
    )
    rows = []
    for value, subset in correct.groupby("lexical_substitution_bin", observed=False):
        n = len(subset)
        successes = int(subset["attack_success"].sum())
        lo, hi = wilson_ci(successes, n)
        rows.append(
            {
                "lexical_substitution_bin": str(value),
                "n_original_correct": n,
                "n_attack_success": successes,
                "asr_pct": successes / n * 100 if n else 0.0,
                "asr_ci_lower_pct": lo * 100,
                "asr_ci_upper_pct": hi * 100,
                "mean_lexical_substitution_rate": subset["lexical_substitution_rate"].mean() if n else 0.0,
            }
        )
    return pd.DataFrame(rows)


def write_summary(path: Path, binary_summary: pd.DataFrame, lexical: pd.DataFrame) -> None:
    present = binary_summary[binary_summary["feature_present"] == 1].sort_values(
        "asr_pct", ascending=False
    )
    lines = [
        "Korean Linguistic Feature Analysis",
        "==================================",
        "",
        "Binary feature ASR when feature is present:",
    ]
    for _, row in present.iterrows():
        lines.append(
            f"- {row.feature}: ASR={row.asr_pct:.2f}% "
            f"({int(row.n_attack_success)}/{int(row.n_original_correct)}), "
            f"OR={row.odds_ratio_present_vs_absent:.3f}, Holm p={row.holm_p:.4g}"
        )
    lines.extend(["", "Lexical substitution bins:"])
    for _, row in lexical.iterrows():
        lines.append(
            f"- {row.lexical_substitution_bin}: ASR={row.asr_pct:.2f}% "
            f"({int(row.n_attack_success)}/{int(row.n_original_correct)})"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Korean-specific transformation features and ASR.")
    parser.add_argument("--output_dir", default=SUMMARY_TABLE_DIR)
    args = parser.parse_args()

    df = load_rows()
    feature_df = pd.DataFrame(
        [extract_features(row["hypothesis"], row["attacked_hypothesis"]) for _, row in df.iterrows()]
    )
    enriched = pd.concat([df.reset_index(drop=True), feature_df], axis=1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched_path = output_dir / "korean_linguistic_features_by_row.csv"
    summary_path = output_dir / "korean_linguistic_feature_summary.csv"
    label_summary_path = output_dir / "korean_linguistic_feature_label_summary.csv"
    lexical_path = output_dir / "korean_linguistic_lexical_substitution_summary.csv"
    text_path = output_dir / "korean_linguistic_feature_summary.txt"

    binary = binary_feature_summary(enriched)
    label_summary = label_feature_summary(enriched)
    lexical = lexical_summary(enriched)

    enriched.to_csv(enriched_path, index=False, encoding="utf-8-sig")
    binary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    label_summary.to_csv(label_summary_path, index=False, encoding="utf-8-sig")
    lexical.to_csv(lexical_path, index=False, encoding="utf-8-sig")
    write_summary(text_path, binary, lexical)

    print(f"Rows analyzed: {len(enriched):,}")
    print(f"Saved: {summary_path}")
    print(binary[binary['feature_present'] == 1].sort_values('asr_pct', ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
