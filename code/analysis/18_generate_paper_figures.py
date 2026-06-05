from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_DIR = ROOT / "results" / "03_summary_tables"
FIGURE_DIR = ROOT / "figures"


DISPLAY_NAMES = {
    "claude_sonnet": "Claude",
    "gemini": "Gemini",
    "gpt": "GPT",
    "hyperclova_x": "HyperCLOVA X",
    "sonnet": "Claude",
    "openai": "GPT",
    "clova": "HyperCLOVA X",
    "llm": "LLM",
    "bt": "Back Translation",
}

MODEL_ORDER = ["claude_sonnet", "gemini", "gpt", "hyperclova_x"]
MODEL_LABELS = [DISPLAY_NAMES[x] for x in MODEL_ORDER]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def save_figure(fig: plt.Figure, name: str, formats: tuple[str, ...] = ("png",)) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(FIGURE_DIR / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    boxes = [
        ("KLUE-NLI\nvalidation\nN=3,000", 0.05, 0.58, 0.14, 0.22, "#E8F0FE"),
        ("4 LLM\nparaphrase\ngenerators", 0.24, 0.58, 0.14, 0.22, "#FCE8E6"),
        ("BERTScore\nprefilter\nthreshold=0.80", 0.43, 0.58, 0.14, 0.22, "#E6F4EA"),
        ("Shared valid set\nN=2,209", 0.62, 0.58, 0.14, 0.22, "#FFF4E5"),
        ("Synchronized\noriginal-cache\nper target", 0.81, 0.58, 0.14, 0.22, "#F3E8FD"),
        ("4x4 NLI\nevaluation\n35,344 rows", 0.24, 0.17, 0.14, 0.22, "#E8F0FE"),
        ("ASR, CI,\nlabel transition,\nstatistics", 0.43, 0.17, 0.14, 0.22, "#E6F4EA"),
        ("Human validation\nLLM 600\nBT 180", 0.62, 0.17, 0.14, 0.22, "#FFF4E5"),
        ("Adjusted ASR\nPAWS-X 1,000\nKorean features", 0.81, 0.17, 0.14, 0.22, "#FCE8E6"),
    ]

    for text, x, y, w, h, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=0.9,
            facecolor=color,
            edgecolor="#4A5568",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", linespacing=1.15)

    arrow_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]
    centers = [(x + w / 2, y + h / 2) for _, x, y, w, h, _ in boxes]
    for src, dst in arrow_pairs:
        x1, y1 = centers[src]
        x2, y2 = centers[dst]
        if src == 4 and dst == 5:
            start, end = (x1, y1 - 0.13), (x2, y2 + 0.13)
            connection = "angle3,angleA=-90,angleB=180"
        else:
            start = (x1 + 0.075, y1) if x2 > x1 else (x1 - 0.075, y1)
            end = (x2 - 0.075, y2) if x2 > x1 else (x2 + 0.075, y2)
            connection = "arc3,rad=0"
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=0.9,
                color="#374151",
                connectionstyle=connection,
            )
        )

    save_figure(fig, "fig_pipeline")


def fig_asr_heatmap() -> None:
    matrix = pd.read_csv(SUMMARY_DIR / "cross_model_asr_matrix_pct.csv", index_col=0)
    matrix = matrix.loc[MODEL_ORDER, MODEL_ORDER]
    values = matrix.values

    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    im = ax.imshow(values, cmap="YlOrRd", vmin=0, vmax=max(13.5, values.max()))
    ax.set_xticks(np.arange(len(MODEL_LABELS)), labels=MODEL_LABELS, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(MODEL_LABELS)), labels=MODEL_LABELS)
    ax.set_xlabel("Target model")
    ax.set_ylabel("Generator model")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if values[i, j] >= 8.5 else "#1F2937"
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color=color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("ASR (%)")
    save_figure(fig, "fig_asr_heatmap")


def fig_gen_tgt_asr() -> None:
    df = pd.read_csv(SUMMARY_DIR / "generator_target_asr_ci.csv")
    gen = df[df["category"].eq("Generator")].copy()
    tgt = df[df["category"].eq("Target")].copy()
    gen["display"] = gen["name"].map(DISPLAY_NAMES)
    tgt["display"] = tgt["name"].map(DISPLAY_NAMES)
    gen_order = ["HyperCLOVA X", "Gemini", "Claude", "GPT"]
    tgt_order = ["HyperCLOVA X", "GPT", "Claude", "Gemini"]
    gen = gen.set_index("display").loc[gen_order].reset_index()
    tgt = tgt.set_index("display").loc[tgt_order].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
    colors = ["#B42318", "#1D4ED8", "#047857", "#7C3AED"]
    for ax, sub, x_label in zip(axes, [gen, tgt], ["Generator model", "Target model"]):
        x = np.arange(len(sub))
        y = sub["asr_pct"].to_numpy()
        lo = y - sub["bootstrap_ci_lower_pct"].to_numpy()
        hi = sub["bootstrap_ci_upper_pct"].to_numpy() - y
        ax.bar(x, y, color=colors, edgecolor="#374151", linewidth=0.6)
        ax.errorbar(x, y, yerr=[lo, hi], fmt="none", ecolor="#111827", elinewidth=0.9, capsize=3)
        ax.set_xticks(x, labels=sub["display"], rotation=25, ha="right")
        ax.set_xlabel(x_label)
        ax.set_ylabel("ASR (%)")
        ax.yaxis.grid(True, color="#E5E7EB", linewidth=0.7)
        upper = sub["bootstrap_ci_upper_pct"].to_numpy()
        for idx, (val, top) in enumerate(zip(y, upper)):
            ax.text(idx, top + 0.22, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ymax = max(gen["bootstrap_ci_upper_pct"].max(), tgt["bootstrap_ci_upper_pct"].max()) + 0.9
    for ax in axes:
        ax.set_ylim(0, ymax)
    save_figure(fig, "fig_gen_tgt_asr")


def fig_label_transition() -> None:
    df = pd.read_csv(SUMMARY_DIR / "label_transition_summary.csv")
    labels = ["Entailment", "Neutral", "Contradiction"]
    mat = pd.DataFrame(0, index=labels, columns=labels, dtype=float)
    counts = pd.DataFrame("", index=labels, columns=labels)
    for _, row in df.iterrows():
        mat.loc[row["gold_name"], row["pred_name"]] = row["percentage"]
        counts.loc[row["gold_name"], row["pred_name"]] = f"{int(row['count'])}\n({row['percentage']:.1f}%)"

    fig, ax = plt.subplots(figsize=(4.1, 3.25))
    im = ax.imshow(mat.values, cmap="Blues", vmin=0, vmax=90)
    ax.set_xticks(np.arange(3), labels=labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(3), labels=labels)
    ax.set_xlabel("Post-attack prediction")
    ax.set_ylabel("Gold label")
    for i in range(3):
        for j in range(3):
            if i == j:
                ax.text(j, i, "--", ha="center", va="center", color="#374151")
            else:
                color = "white" if mat.values[i, j] > 55 else "#111827"
                ax.text(j, i, counts.iloc[i, j], ha="center", va="center", color=color, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Share within gold label (%)")
    save_figure(fig, "fig_label_transition")


def fig_adjusted_asr() -> None:
    df = pd.read_csv(SUMMARY_DIR / "human_adjusted_asr_bt_vs_llm_intersection.csv")
    df = df[df["group_name"].eq("condition_type")].copy()
    df["display"] = df["group_value"].map(DISPLAY_NAMES)
    df = df.set_index("display").loc[["LLM", "Back Translation"]].reset_index()

    x = np.arange(len(df))
    width = 0.34
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    raw = df["raw_asr_pct"].to_numpy()
    adj = df["adjusted_asr_all_pct"].to_numpy()
    raw_err = np.vstack(
        [
            raw - df["raw_asr_ci_lower_pct"].to_numpy(),
            df["raw_asr_ci_upper_pct"].to_numpy() - raw,
        ]
    )
    adj_err = np.vstack(
        [
            adj - df["adjusted_asr_all_ci_lower_pct"].to_numpy(),
            df["adjusted_asr_all_ci_upper_pct"].to_numpy() - adj,
        ]
    )
    ax.bar(x - width / 2, raw, width, label="Raw ASR", color="#9CA3AF", edgecolor="#374151", linewidth=0.6)
    ax.bar(x + width / 2, adj, width, label="Human-adjusted ASR", color="#2563EB", edgecolor="#374151", linewidth=0.6)
    ax.errorbar(x - width / 2, raw, yerr=raw_err, fmt="none", ecolor="#111827", elinewidth=0.9, capsize=3)
    ax.errorbar(x + width / 2, adj, yerr=adj_err, fmt="none", ecolor="#111827", elinewidth=0.9, capsize=3)
    ax.set_xticks(x, labels=df["display"])
    ax.set_ylabel("ASR (%)")
    ax.set_ylim(0, 10.2)
    ax.yaxis.grid(True, color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2)
    label_offset = 0.62
    for idx, val in enumerate(raw):
        ax.text(idx - width / 2, val + label_offset, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    for idx, val in enumerate(adj):
        ax.text(idx + width / 2, val + label_offset, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    save_figure(fig, "fig_adjusted_asr")


def fig_pawsx_external() -> None:
    df = pd.read_csv(SUMMARY_DIR / "pawsx_external_benchmark_summary.csv")
    sub = df[df["language"].isin(["ko", "en"])].copy()
    sub["display"] = sub["target"].map(DISPLAY_NAMES)
    order = ["Gemini", "GPT", "Claude", "HyperCLOVA X"]
    ko = sub[sub["language"].eq("ko")].set_index("display").loc[order]
    en = sub[sub["language"].eq("en")].set_index("display").loc[order]

    x = np.arange(len(order))
    width = 0.36
    fig, ax = plt.subplots(figsize=(5.1, 3.2))
    ax.bar(x - width / 2, ko["accuracy_pct"], width, label="Korean", color="#2563EB", edgecolor="#374151", linewidth=0.6)
    ax.bar(x + width / 2, en["accuracy_pct"], width, label="English", color="#F59E0B", edgecolor="#374151", linewidth=0.6)
    ax.set_xticks(x, labels=order, rotation=20, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False, loc="upper right")
    for idx, val in enumerate(ko["accuracy_pct"]):
        ax.text(idx - width / 2, val + 1.2, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    for idx, val in enumerate(en["accuracy_pct"]):
        ax.text(idx + width / 2, val + 1.2, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    save_figure(fig, "fig_pawsx_external")


def fig_korean_linguistic_features() -> None:
    df = pd.read_csv(SUMMARY_DIR / "korean_linguistic_feature_summary.csv")
    names = {
        "particle_count_changed": "Particle-count change",
        "spacing_token_count_changed": "Spacing/token-count change",
        "sentence_ending_changed": "Sentence-ending change",
        "word_order_shift": "Word-order shift",
        "negation_modality_changed": "Negation/modality change",
        "number_changed": "Number change",
    }
    order = list(names)
    present = df[df["feature_present"].eq(1)].set_index("feature").loc[order]
    absent = df[df["feature_present"].eq(0)].set_index("feature").loc[order]

    y = np.arange(len(order))
    width = 0.35
    fig, ax = plt.subplots(figsize=(4.9, 3.95))
    ax.barh(
        y + width / 2,
        absent["asr_pct"],
        width,
        label="Absent",
        color="#D1D5DB",
        edgecolor="#374151",
        linewidth=0.5,
    )
    colors = ["#B42318" if p < 0.05 else "#6B7280" for p in present["holm_p"]]
    ax.barh(
        y - width / 2,
        present["asr_pct"],
        width,
        label="Present",
        color=colors,
        edgecolor="#374151",
        linewidth=0.5,
    )
    ax.set_yticks(y, labels=[names[key] for key in order])
    ax.invert_yaxis()
    ax.set_xlabel("ASR (%)")
    ax.set_xlim(0, 8.6)
    ax.margins(y=0.08)
    ax.xaxis.grid(True, color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    for idx, feature in enumerate(order):
        row = present.loc[feature]
        star = "*" if row["holm_p"] < 0.05 else ""
        xpos = max(absent.loc[feature, "asr_pct"], row["asr_pct"]) + 0.28
        ax.text(
            xpos,
            idx,
            f"OR={row['odds_ratio_present_vs_absent']:.2f}{star}",
            ha="left",
            va="center",
            fontsize=8,
        )
    ax.text(
        0.99,
        -0.14,
        "* Holm-adjusted p < 0.05",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#374151",
    )
    save_figure(fig, "fig_korean_linguistic_features")


def main() -> None:
    setup_style()
    fig_pipeline()
    fig_asr_heatmap()
    fig_gen_tgt_asr()
    fig_label_transition()
    fig_adjusted_asr()
    fig_pawsx_external()
    fig_korean_linguistic_features()
    print(f"Saved paper figures to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
