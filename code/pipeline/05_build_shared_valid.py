import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import GENERATORS, SHARED_VALID_DIR, VALIDATED_ATTACK_DIR, llm_validated_filename, shared_valid_filename, shared_valid_ids_filename

def main():
    paths = {g: Path(VALIDATED_ATTACK_DIR) / llm_validated_filename(g) for g in GENERATORS}
    dfs = {}
    for k, p in paths.items():
        if not p.exists():
            print(f'[SKIP] {p} not found')
            continue
        dfs[k] = pd.read_csv(p, encoding='utf-8-sig')
    ids_sets = {k: set(df['id'].tolist()) for k, df in dfs.items()}
    shared_ids = set.intersection(*ids_sets.values())
    out_dir = Path(SHARED_VALID_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'id': sorted(shared_ids)}).to_csv(out_dir / shared_valid_ids_filename(), index=False, encoding='utf-8-sig')
    for k, df in dfs.items():
        out = df[df['id'].isin(shared_ids)].copy().sort_values('id')
        out.to_csv(out_dir / shared_valid_filename(k), index=False, encoding='utf-8-sig')
    print(f'shared_ids = {len(shared_ids)}')
    for k, s in ids_sets.items():
        print(f'  {k}: valid={len(s)} -> shared={len(shared_ids)}')
    print(f'Saved to: {out_dir}')
if __name__ == '__main__':
    main()
