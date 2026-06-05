from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = ROOT / "results" / "03_summary_tables" / "quality_metrics" / "paraphrase_quality_generated.csv"
FIGURE_DIR = ROOT / "figures"


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


def main() -> None:
    setup_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SUMMARY_PATH)
    order = ["Gemini", "GPT", "Claude Sonnet", "HyperCLOVA X", "Back Translation", "BERT-Attack", "EDA"]
    df = df.set_index("method").loc[order].reset_index()

    labels = ["Gemini", "GPT", "Claude", "HCX", "BT", "BERT-\nAttack", "EDA"]
    x = np.arange(len(df))
    colors = ["#2563EB", "#2563EB", "#2563EB", "#2563EB", "#F59E0B", "#6B7280", "#6B7280"]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.25))

    axes[0].bar(x, df["ppl_mean"], color=colors, edgecolor="#374151", linewidth=0.6)
    axes[0].set_xticks(x, labels=labels, rotation=28, ha="right")
    axes[0].set_ylabel("PPL mean")
    axes[0].yaxis.grid(True, color="#E5E7EB", linewidth=0.7)
    axes[0].text(0.01, 0.96, "(a)", transform=axes[0].transAxes, ha="left", va="top", fontsize=9, weight="bold")
    for idx, val in enumerate(df["ppl_mean"]):
        axes[0].text(idx, val + 30, f"{val:.0f}", ha="center", va="bottom", fontsize=7)

    axes[1].bar(x, df["defect_rate"], color=colors, edgecolor="#374151", linewidth=0.6)
    axes[1].set_xticks(x, labels=labels, rotation=28, ha="right")
    axes[1].set_ylabel("Defect rate (%)")
    axes[1].set_xlabel("")
    axes[1].set_ylim(0, max(10, float(df["defect_rate"].max()) * 1.25))
    axes[1].yaxis.grid(True, color="#E5E7EB", linewidth=0.7)
    axes[1].text(0.01, 0.96, "(b)", transform=axes[1].transAxes, ha="left", va="top", fontsize=9, weight="bold")
    for idx, val in enumerate(df["defect_rate"]):
        axes[1].text(idx, val + 0.25, f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout(w_pad=1.6)
    fig.savefig(FIGURE_DIR / "fig_quality_comparison.png", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(FIGURE_DIR / "fig_quality_comparison.png")


if __name__ == "__main__":
    main()
