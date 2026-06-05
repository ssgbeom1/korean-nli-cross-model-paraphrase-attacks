from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DISPLAY = {
    "llm": "LLM paraphrase",
    "bt": "Back Translation",
}


def fmt_ci(row: pd.Series, point: str, lower: str, upper: str) -> str:
    return f"{row[point]:.2f} [{row[lower]:.2f}, {row[upper]:.2f}]"


def make_latex(df: pd.DataFrame) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Back Translation and LLM paraphrase comparison after human validation.}",
        "\\label{tab:bt_llm_human_adjusted}",
        "\\begin{tabular}{lrrrr}",
        "\\hline",
        "Method & Raw ASR (\\%) & Human LPR (\\%) & Adjusted ASR (\\%) & Resolved Adj. ASR (\\%) \\\\",
        "\\hline",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['method']} & {row['raw_asr_ci']} & {row['human_lpr_ci']} & "
            f"{row['adjusted_asr_ci']} & {row['resolved_adjusted_asr_ci']} \\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def make_text(df: pd.DataFrame) -> str:
    lines = [
        "BT vs LLM Human-Adjusted Comparison",
        "===================================",
        "",
        "The comparison uses the BT-intersection subset for raw ASR.",
        "Primary adjusted ASR treats U/TIE human-validation outcomes as not label-preserving.",
        "",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"- {row['method']}: raw ASR={row['raw_asr_ci']}, "
            f"LPR={row['human_lpr_ci']}, adjusted ASR={row['adjusted_asr_ci']}, "
            f"resolved adjusted ASR={row['resolved_adjusted_asr_ci']}"
        )
    bt = df[df["condition_type"] == "bt"].iloc[0]
    llm = df[df["condition_type"] == "llm"].iloc[0]
    lines.extend(
        [
            "",
            "Interpretation:",
            (
                f"- BT has higher raw ASR ({bt.raw_asr_pct:.2f}% vs {llm.raw_asr_pct:.2f}%), "
                f"but lower human label preservation ({bt.lpr_all_pct:.2f}% vs {llm.lpr_all_pct:.2f}%)."
            ),
            (
                f"- After human adjustment, LLM paraphrases are higher "
                f"({llm.adjusted_asr_all_pct:.2f}% vs {bt.adjusted_asr_all_pct:.2f}%)."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a manuscript-ready BT vs LLM comparison table.")
    parser.add_argument(
        "--adjusted",
        default="results/03_summary_tables/human_adjusted_asr_bt_vs_llm_intersection.csv",
    )
    parser.add_argument(
        "--lpr",
        default="results/03_summary_tables/human_validation_lpr_overall.csv",
    )
    parser.add_argument("--output_dir", default="results/03_summary_tables")
    args = parser.parse_args()

    adjusted = pd.read_csv(args.adjusted, encoding="utf-8-sig")
    lpr = pd.read_csv(args.lpr, encoding="utf-8-sig")
    merged = adjusted.merge(lpr, left_on="group_value", right_on="condition_type", how="left", validate="one_to_one")
    merged = merged.rename(columns={"condition_type_x": "adjusted_condition_type", "condition_type_y": "condition_type"})
    merged["condition_type"] = merged["group_value"]
    merged["method"] = merged["condition_type"].map(DISPLAY)
    merged["raw_asr_ci"] = merged.apply(lambda row: fmt_ci(row, "raw_asr_pct", "raw_asr_ci_lower_pct", "raw_asr_ci_upper_pct"), axis=1)
    merged["human_lpr_ci"] = merged.apply(
        lambda row: fmt_ci(row, "lpr_all_pct", "lpr_all_ci_lower_pct", "lpr_all_ci_upper_pct"),
        axis=1,
    )
    merged["human_lpr_resolved_ci"] = merged.apply(
        lambda row: fmt_ci(row, "lpr_resolved_pct", "lpr_resolved_ci_lower_pct", "lpr_resolved_ci_upper_pct"),
        axis=1,
    )
    merged["adjusted_asr_ci"] = merged.apply(
        lambda row: fmt_ci(row, "adjusted_asr_all_pct", "adjusted_asr_all_ci_lower_pct", "adjusted_asr_all_ci_upper_pct"),
        axis=1,
    )
    merged["resolved_adjusted_asr_ci"] = merged.apply(
        lambda row: fmt_ci(row, "adjusted_asr_resolved_pct", "adjusted_asr_resolved_ci_lower_pct", "adjusted_asr_resolved_ci_upper_pct"),
        axis=1,
    )
    merged["interpretation"] = merged["condition_type"].map(
        {
            "llm": "Higher semantic preservation; adjusted ASR remains close to raw ASR.",
            "bt": "Higher raw ASR, but many flips are attributable to semantic drift.",
        }
    )
    merged = merged.sort_values("condition_type", ascending=False)

    columns = [
        "condition_type",
        "method",
        "raw_n_original_correct",
        "raw_attack_success",
        "raw_asr_ci",
        "n_total",
        "n_preserved",
        "n_unresolved",
        "human_lpr_ci",
        "human_lpr_resolved_ci",
        "adjusted_asr_ci",
        "resolved_adjusted_asr_ci",
        "interpretation",
    ]
    out = merged[columns].copy()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / "bt_vs_llm_human_adjusted_comparison.csv", index=False, encoding="utf-8-sig")
    (output_dir / "bt_vs_llm_human_adjusted_comparison.tex").write_text(make_latex(out), encoding="utf-8")
    (output_dir / "bt_vs_llm_human_adjusted_comparison.txt").write_text(make_text(merged), encoding="utf-8")

    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
