"""Install a local fixed-risk CSV for the archived experiment scripts."""
from pathlib import Path
import argparse
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=Path)
    args = parser.parse_args()
    data = pd.read_csv(args.csv, dtype={'patient_id': str})
    columns = ['cohort', 'patient_id', 'os_days', 'os_event', 'p_Clinical', 'p_Gene', 'p_Pathway']
    missing = set(columns) - set(data.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if data[columns].isna().any().any() or data.duplicated(['cohort', 'patient_id']).any():
        raise ValueError('The fixed-risk input must have complete values and unique cohort/patient keys.')
    if not data.os_event.isin([0, 1]).all() or (data.os_days < 0).any():
        raise ValueError('Use nonnegative follow-up and binary death indicators.')
    if not data[['p_Clinical', 'p_Gene', 'p_Pathway']].apply(lambda s: s.between(0, 1).all()).all():
        raise ValueError('Fixed risks must lie between zero and one.')
    for name in ['benchmark/fixed_predictions.csv', 'sensitivity/inputs/fixed_predictions.csv']:
        shutil.copyfile(args.csv, ROOT / name)
    print('Local fixed-risk input installed.')


if __name__ == '__main__':
    main()
