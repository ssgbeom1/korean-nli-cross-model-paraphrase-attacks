import argparse
import sys
from pathlib import Path
import pandas as pd
from datasets import load_dataset
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SAMPLED_SOURCE_DIR, sampled_source_filename

def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    df = df[df['label'].isin([0, 1, 2])].copy()
    labels = [0, 1, 2]
    base = n // len(labels)
    rem = n % len(labels)
    per_label = {lab: base for lab in labels}
    for lab in labels[:rem]:
        per_label[lab] += 1
    parts = []
    for lab in labels:
        sub = df[df['label'] == lab]
        if len(sub) < per_label[lab]:
            raise ValueError(f'Not enough samples for label={lab}. need={per_label[lab]}, available={len(sub)}')
        parts.append(sub.sample(n=per_label[lab], random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out

def main():
    dataset_split = 'validation'
    ap = argparse.ArgumentParser(description='KLUE-NLI 데이터 로드 및 샘플링')
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=str, default=str(Path(SAMPLED_SOURCE_DIR) / sampled_source_filename()))
    args = ap.parse_args()
    print('Loading KLUE-NLI from HuggingFace...')
    ds = load_dataset('klue', 'nli')
    if dataset_split not in ds:
        raise ValueError(f"Split '{dataset_split}' not found. Available: {list(ds.keys())}")
    df = pd.DataFrame(ds[dataset_split])
    df = df[['premise', 'hypothesis', 'label']].copy()
    df_sample = stratified_sample(df, n=args.n, seed=args.seed)
    df_sample.insert(0, 'id', range(len(df_sample)))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_sample.to_csv(out_path, index=False, encoding='utf-8-sig')
    dist = df_sample['label'].value_counts().sort_index().to_dict()
    print(f'Saved: {out_path} | n={len(df_sample)}, split={dataset_split}')
    print(f'Label distribution: {dist}')
if __name__ == '__main__':
    main()
