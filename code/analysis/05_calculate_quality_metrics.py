from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import GENERATED_ATTACK_DIR, SHARED_VALID_DIR, SUMMARY_TABLE_DIR


MODEL_NAME = "skt/kogpt2-base-v2"
METHOD_FILES = {
    "Gemini": "gemini_attacks_3000.csv",
    "GPT": "gpt_attacks_3000.csv",
    "Claude Sonnet": "claude_sonnet_attacks_3000.csv",
    "HyperCLOVA X": "hyperclova_x_attacks_3000.csv",
    "Back Translation": "backtranslation_attacks_3000.csv",
    "BERT-Attack": "bert_attack_attacks_3000.csv",
    "EDA": "eda_attacks_3000.csv",
}
LLM_METHODS = {"Gemini", "GPT", "Claude Sonnet", "HyperCLOVA X"}


def surface_defects(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return ["empty"]

    value = text.strip()
    defects: list[str] = []

    lowered = value.lower()
    if lowered.startswith("error") or "traceback" in lowered:
        defects.append("api_error")
    if "[unk]" in lowered or "<unk>" in lowered:
        defects.append("unk_token")
    if re.search(r"(.)\1{9,}", value):
        defects.append("char_repetition")

    words = value.split()
    for idx in range(len(words) - 2):
        if words[idx] == words[idx + 1] == words[idx + 2]:
            defects.append("word_repetition")
            break

    if len(value.replace(" ", "")) < 6 or len(words) < 2:
        defects.append("too_short")
    if value.endswith((",", ";", ":", "-", "(", "[", "{")):
        defects.append("dangling_end")
    if "paraphrase:" in lowered or "label:" in lowered:
        defects.append("prompt_echo")

    return sorted(set(defects))


class KoGPT2Perplexity:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
        added_pad_token = False
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif self.tokenizer.pad_token is None and self.tokenizer.unk_token is not None:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        elif self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            added_pad_token = True
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        if added_pad_token:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()

    @torch.no_grad()
    def score(self, texts: list[str], batch_size: int = 32) -> list[float]:
        out: list[float] = []
        for start in tqdm(range(0, len(texts), batch_size), desc="PPL batches"):
            batch = texts[start : start + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous().bool()

            flat_loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view(shift_labels.size())
            token_counts = shift_mask.sum(dim=1).clamp(min=1)
            seq_loss = (flat_loss * shift_mask).sum(dim=1) / token_counts
            ppl = torch.exp(seq_loss).detach().cpu().numpy().tolist()
            out.extend(float(min(value, 10000.0)) if math.isfinite(value) else 10000.0 for value in ppl)
        return out


def read_generated(method: str) -> pd.DataFrame:
    path = Path(GENERATED_ATTACK_DIR) / METHOD_FILES[method]
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "attacked_hypothesis" not in df.columns:
        raise ValueError(f"Missing attacked_hypothesis column: {path}")
    return df


def build_scope_frames(scope: str) -> dict[str, pd.DataFrame]:
    frames = {method: read_generated(method) for method in METHOD_FILES}

    if scope == "generated":
        return frames

    if scope != "shared":
        raise ValueError(f"Unsupported scope: {scope}")

    shared_ids = set(pd.read_csv(Path(SHARED_VALID_DIR) / "shared_valid_ids.csv")["id"].tolist())
    bt_valid = pd.read_csv(Path(SHARED_VALID_DIR) / "backtranslation_shared_valid_bert80.csv", encoding="utf-8-sig")
    bt_ids = set(bt_valid["id"].tolist())

    scoped: dict[str, pd.DataFrame] = {}
    for method, df in frames.items():
        if method in LLM_METHODS:
            scoped[method] = df[df["id"].isin(shared_ids)].copy()
        elif method == "Back Translation":
            scoped[method] = df[df["id"].isin(bt_ids)].copy()
        else:
            scoped[method] = df.copy()
    return scoped


def summarize_method(method: str, df: pd.DataFrame, calculator: KoGPT2Perplexity, batch_size: int) -> dict[str, object]:
    texts = df["attacked_hypothesis"].fillna("").astype(str).tolist()
    ppl = calculator.score(texts, batch_size=batch_size)
    defects = [surface_defects(text) for text in texts]
    defect_flags = [bool(items) for items in defects]
    defect_types: dict[str, int] = {}
    for items in defects:
        for item in items:
            defect_types[item] = defect_types.get(item, 0) + 1

    return {
        "method": method,
        "n_samples": len(texts),
        "ppl_mean": float(np.mean(ppl)),
        "ppl_std": float(np.std(ppl)),
        "ppl_median": float(np.median(ppl)),
        "ppl_q1": float(np.quantile(ppl, 0.25)),
        "ppl_q3": float(np.quantile(ppl, 0.75)),
        "defect_count": int(sum(defect_flags)),
        "defect_rate": float(sum(defect_flags) / len(defect_flags) * 100 if defect_flags else 0.0),
        "defect_types": "; ".join(f"{key}:{value}" for key, value in sorted(defect_types.items())),
    }


def write_latex(summary: pd.DataFrame, output_path: Path) -> None:
    order = ["Gemini", "GPT", "Claude Sonnet", "HyperCLOVA X", "Back Translation", "BERT-Attack", "EDA"]
    rows = summary.set_index("method").loc[[item for item in order if item in set(summary["method"])]].reset_index()
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Paraphrase quality comparison. Defect denotes surface-form defects, not semantic invalidity.}",
        "\\label{tab:quality}",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "Method & N & PPL Mean & PPL Median & Defect (\\%) \\\\",
        "\\hline",
    ]
    for _, row in rows.iterrows():
        lines.append(
            f"{row['method']} & {int(row['n_samples'])} & {float(row['ppl_mean']):.2f} & "
            f"{float(row['ppl_median']):.2f} & {float(row['defect_rate']):.2f} \\\\"
        )
        if row["method"] == "HyperCLOVA X":
            lines.append("\\hline")
        if row["method"] == "EDA":
            lines.append("\\hline")
    lines.extend(["\\end{tabular}", "\\end{table}"])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate latest PPL and surface-defect metrics.")
    parser.add_argument("--scope", choices=["generated", "shared", "both"], default="both")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default=f"{SUMMARY_TABLE_DIR}/quality_metrics")
    args = parser.parse_args()

    scopes = ["generated", "shared"] if args.scope == "both" else [args.scope]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calculator = KoGPT2Perplexity()

    for scope in scopes:
        print(f"\n=== Quality metrics: {scope} ===")
        frames = build_scope_frames(scope)
        rows = []
        for method, df in frames.items():
            print(f"{method}: {len(df)} rows")
            rows.append(summarize_method(method, df, calculator, args.batch_size))
        summary = pd.DataFrame(rows).sort_values("ppl_mean")
        csv_path = output_dir / f"paraphrase_quality_{scope}.csv"
        tex_path = output_dir / f"table_paraphrase_quality_{scope}.tex"
        txt_path = output_dir / f"paraphrase_quality_{scope}.txt"
        summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
        write_latex(summary, tex_path)
        txt_lines = [
            f"Paraphrase quality summary ({scope})",
            "=" * 44,
            "",
        ]
        for _, row in summary.iterrows():
            txt_lines.append(
                f"- {row['method']}: n={int(row['n_samples'])}, "
                f"PPL mean={float(row['ppl_mean']):.2f}, "
                f"PPL median={float(row['ppl_median']):.2f}, "
                f"defect={float(row['defect_rate']):.2f}%"
            )
        txt_path.write_text("\n".join(txt_lines), encoding="utf-8")
        print("\n".join(txt_lines))


if __name__ == "__main__":
    main()
