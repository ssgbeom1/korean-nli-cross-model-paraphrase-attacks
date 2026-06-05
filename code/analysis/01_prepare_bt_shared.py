from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import GENERATED_ATTACK_DIR, SHARED_VALID_DIR, VALIDATED_ATTACK_DIR, backtranslation_shared_valid_filename, baseline_attack_filename, baseline_validated_filename, shared_valid_ids_filename

def find_input_file(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError('Could not find a Back Translation input file.')

def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare the Back Translation shared-valid file.')
    parser.add_argument('--input', default='', help='Back Translation CSV. If omitted, the script checks the standard attack and validated paths.')
    parser.add_argument('--shared_ids', default=str(Path(SHARED_VALID_DIR) / shared_valid_ids_filename()), help='Shared-id CSV created by the shared-valid step.')
    parser.add_argument('--output', default=str(Path(SHARED_VALID_DIR) / backtranslation_shared_valid_filename()), help='Output CSV for the shared Back Translation set.')
    args = parser.parse_args()
    candidates = []
    if args.input:
        candidates.append(Path(args.input))
    candidates.extend([Path(VALIDATED_ATTACK_DIR) / baseline_validated_filename('backtranslation'), Path(GENERATED_ATTACK_DIR) / baseline_attack_filename('backtranslation')])
    bt_file = find_input_file(candidates)
    shared_file = Path(args.shared_ids)
    output_file = Path(args.output)
    if not shared_file.exists():
        raise FileNotFoundError(f'Shared-id file not found: {shared_file}')
    print('=' * 60)
    print('Prepare Back Translation Shared Set')
    print('=' * 60)
    bt = pd.read_csv(bt_file, encoding='utf-8-sig')
    shared_ids = pd.read_csv(shared_file, encoding='utf-8-sig')['id'].astype(str).tolist()
    bt['id'] = bt['id'].astype(str)
    if 'bert_score' in bt.columns:
        bt = bt[bt['bert_score'] >= 0.8].copy()
    bt_shared = bt[bt['id'].isin(shared_ids)].copy().sort_values('id')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    bt_shared.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f'Input:      {bt_file}')
    print(f'Shared ids: {len(shared_ids)}')
    print(f'Output:     {output_file}')
    print(f'Rows:       {len(bt_shared)}')
if __name__ == '__main__':
    main()
