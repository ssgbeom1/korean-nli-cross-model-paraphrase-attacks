from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize_group(path: Path, df: pd.DataFrame, language: str, target: str) -> dict:
    valid = df[df["pred"].notna()].copy()
    rows = {
        "language": language,
        "target": target,
        "file": str(path),
        "n_rows": len(df),
        "n_valid_pred": len(valid),
        "parse_failure": len(df) - len(valid),
        "accuracy_pct": valid["correct"].mean() * 100 if len(valid) else 0.0,
    }
    for label in [0, 1]:
        subset = valid[valid["label"].astype(int) == label]
        rows[f"label_{label}_n"] = len(subset)
        rows[f"label_{label}_accuracy_pct"] = subset["correct"].mean() * 100 if len(subset) else 0.0
    return rows


def summarize_file(path: Path) -> list[dict]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    target = str(df["target"].iloc[0]) if len(df) else path.stem
    rows = [summarize_group(path, df, "all", target)]
    for language, group in df.groupby("language", sort=True):
        rows.append(summarize_group(path, group, str(language), target))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PAWS-X external benchmark LLM outputs.")
    parser.add_argument("--input_dir", default="results/06_external_benchmark/pawsx")
    parser.add_argument("--output", default="results/03_summary_tables/pawsx_external_benchmark_summary.csv")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("pawsx_ko_en_*.csv"))
    if not files:
        raise FileNotFoundError(f"No result CSVs found in {input_dir}")
    rows = []
    for path in files:
        rows.extend(summarize_file(path))
    summary = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False, encoding="utf-8-sig")
    text = output.with_suffix(".txt")
    lines = [
        "PAWS-X External Benchmark Summary",
        "=================================",
        "",
        "Task: binary semantic equivalence on balanced PAWS-X Korean/English samples.",
        "",
    ]
    all_rows = summary[summary["language"].eq("all")].sort_values("accuracy_pct", ascending=False)
    for _, row in all_rows.iterrows():
        lines.append(
            f"- {row.target}: accuracy={row.accuracy_pct:.2f}% "
            f"({int(row.n_valid_pred)}/{int(row.n_rows)} valid, parse failures={int(row.parse_failure)})"
        )
    lines.extend(["", "By language:"])
    for _, row in summary[~summary["language"].eq("all")].sort_values(["language", "accuracy_pct"], ascending=[True, False]).iterrows():
        lines.append(
            f"- {row.language}/{row.target}: accuracy={row.accuracy_pct:.2f}% "
            f"({int(row.n_valid_pred)}/{int(row.n_rows)} valid)"
        )
    text.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Saved: {output}")
    print(f"Saved: {text}")


if __name__ == "__main__":
    main()
