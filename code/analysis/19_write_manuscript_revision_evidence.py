from __future__ import annotations

from pathlib import Path

import pandas as pd


SUMMARY_DIR = Path("results/03_summary_tables")
TARGET_DISPLAY = {
    "gemini": "Gemini",
    "openai": "GPT",
    "sonnet": "Claude Sonnet",
    "clova": "HyperCLOVA X",
}


def pct(value: float) -> str:
    return f"{value:.2f}%"


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(SUMMARY_DIR / name, encoding="utf-8-sig")


def main() -> None:
    four_by_four = read_csv("generator_target_asr_ci.csv")
    human_lpr = read_csv("human_validation_lpr_overall.csv")
    agreement = read_csv("human_validation_agreement_summary.csv")
    adjusted = read_csv("human_adjusted_asr_summary.csv")
    bt_llm = read_csv("bt_vs_llm_human_adjusted_comparison.csv")
    pawsx = read_csv("pawsx_external_benchmark_summary.csv")
    korean = read_csv("korean_linguistic_feature_summary.csv")

    overall_4x4 = four_by_four[(four_by_four["category"] == "Overall") & (four_by_four["name"] == "Total")].iloc[0]
    llm_lpr = human_lpr[human_lpr["condition_type"] == "llm"].iloc[0]
    bt_lpr = human_lpr[human_lpr["condition_type"] == "bt"].iloc[0]
    llm_agreement = agreement[
        (agreement["condition_type"] == "llm") & (agreement["field"] == "paraphrased_pair_label")
    ].iloc[0]
    bt_agreement = agreement[
        (agreement["condition_type"] == "bt") & (agreement["field"] == "paraphrased_pair_label")
    ].iloc[0]
    bt_llm_rows = {row["condition_type"]: row for _, row in bt_llm.iterrows()}
    paws_all = pawsx[pawsx["language"] == "all"].sort_values("accuracy_pct", ascending=False)
    korean_present = korean[korean["feature_present"] == 1].sort_values("holm_p")

    md = [
        "# Manuscript Revision Evidence Summary",
        "",
        "## Core 4x4 Evaluation",
        "",
        (
            f"- 4x4 evaluation rows: {int(overall_4x4['rows']):,}; "
            f"original-correct denominator: {int(overall_4x4['n_original_correct']):,}; "
            f"attack successes: {int(overall_4x4['n_attack_success']):,}."
        ),
        (
            f"- Overall raw ASR: {pct(overall_4x4['asr_pct'])} "
            f"[{pct(overall_4x4['wilson_ci_lower_pct'])}, {pct(overall_4x4['wilson_ci_upper_pct'])}]."
        ),
        "",
        "## Human Validation",
        "",
        (
            f"- LLM human validation: {int(llm_lpr.n_preserved)}/{int(llm_lpr.n_total)} label-preserving "
            f"({pct(llm_lpr.lpr_all_pct)}); resolved LPR {pct(llm_lpr.lpr_resolved_pct)}."
        ),
        (
            f"- BT human validation: {int(bt_lpr.n_preserved)}/{int(bt_lpr.n_total)} label-preserving "
            f"({pct(bt_lpr.lpr_all_pct)}); resolved LPR {pct(bt_lpr.lpr_resolved_pct)}."
        ),
        (
            f"- Label agreement: LLM Fleiss' kappa={llm_agreement.fleiss_kappa:.4f}; "
            f"BT Fleiss' kappa={bt_agreement.fleiss_kappa:.4f}."
        ),
        "",
        "## BT vs LLM After Human Adjustment",
        "",
    ]
    for key in ["llm", "bt"]:
        row = bt_llm_rows[key]
        md.append(
            f"- {row['method']}: raw ASR {row['raw_asr_ci']}, human LPR {row['human_lpr_ci']}, "
            f"adjusted ASR {row['adjusted_asr_ci']}."
        )
    md.extend(
        [
            "",
            "## PAWS-X External Benchmark",
            "",
        ]
    )
    for _, row in paws_all.iterrows():
        md.append(
            f"- {TARGET_DISPLAY.get(row.target, row.target)}: accuracy {pct(row.accuracy_pct)} "
            f"({int(row.n_valid_pred)}/{int(row.n_rows)} valid; parse failures {int(row.parse_failure)})."
        )
    md.extend(
        [
            "",
            "## Korean-Specific Linguistic Features",
            "",
        ]
    )
    for _, row in korean_present.head(6).iterrows():
        md.append(
            f"- {row.feature}: present-ASR {pct(row.asr_pct)}, "
            f"odds ratio {row.odds_ratio_present_vs_absent:.3f}, Holm p={row.holm_p:.4g}."
        )
    md.extend(
        [
            "",
            "## Recommended Manuscript Inserts",
            "",
            "- Add Algorithm 1 for the 4x4 paraphrase attack evaluation protocol.",
            "- Add a Statistical Analysis subsection describing Wilson CI, bootstrap CI, Holm correction, and human-adjusted ASR.",
            "- Add an Expanded Human Validation subsection with LLM 600 and BT 180 annotation designs.",
            "- Add a BT Baseline subsection explaining why raw BT ASR must be human-adjusted.",
            "- Add an External PAWS-X Validation subsection as auxiliary paraphrase benchmark evidence.",
            "- Add a Threat Model subsection separating instability, robustness failure, and security-relevant vulnerability.",
            "- Add a Korean Linguistic Feature Analysis subsection with feature-level ASR and odds-ratio results.",
            "",
        ]
    )

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    (SUMMARY_DIR / "manuscript_revision_evidence_summary.md").write_text("\n".join(md), encoding="utf-8")

    evidence_rows = [
        {
            "reviewer_comment": "R1-1 pseudocode/code",
            "response_action": "Add Algorithm 1 and point to pipeline scripts.",
            "evidence_files": "code/pipeline/*.py; code/analysis/02_verify_4x4_integrity.py",
        },
        {
            "reviewer_comment": "R1-2 statistical analysis",
            "response_action": "Report ASR CI, bootstrap pairwise tests, adjusted ASR, and feature odds ratios.",
            "evidence_files": "results/03_summary_tables/generator_target_asr_ci.csv; statistical_analysis_pairwise.csv; human_adjusted_asr_summary.csv",
        },
        {
            "reviewer_comment": "R2-1 human validation and paraphrase errors",
            "response_action": "Use LLM 600 and BT 180 human annotation with drift taxonomy and LPR.",
            "evidence_files": "results/03_summary_tables/human_validation_*.csv; human_validation_summary.txt",
        },
        {
            "reviewer_comment": "R2-2 external paraphrase benchmark",
            "response_action": "Add PAWS-X Korean/English auxiliary validation.",
            "evidence_files": "results/03_summary_tables/pawsx_external_benchmark_summary.csv",
        },
        {
            "reviewer_comment": "R2-3 vulnerability vs quality issue",
            "response_action": "Add threat model and interpret BT raw ASR through human-adjusted ASR.",
            "evidence_files": "results/03_summary_tables/bt_vs_llm_human_adjusted_comparison.csv",
        },
        {
            "reviewer_comment": "R2-4 Korean specificity",
            "response_action": "Add Korean linguistic feature analysis.",
            "evidence_files": "results/03_summary_tables/korean_linguistic_feature_summary.csv",
        },
    ]
    pd.DataFrame(evidence_rows).to_csv(
        SUMMARY_DIR / "reviewer_response_evidence_map.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(SUMMARY_DIR / "manuscript_revision_evidence_summary.md")
    print(SUMMARY_DIR / "reviewer_response_evidence_map.csv")


if __name__ == "__main__":
    main()
