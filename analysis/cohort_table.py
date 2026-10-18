"""Calculate Table I from the local fixed-risk input."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmark'))
from calibration import weights, km_risk, TAU


def main():
    data = pd.read_csv(ROOT / 'benchmark/fixed_predictions.csv')
    rows = []
    for cohort, group in data.groupby('cohort', sort=False):
        time, event = group.os_days.to_numpy(), group.os_event.to_numpy()
        y, w = weights(time, event)
        cases, controls = y == 1, (y == 0) & (w > 0)
        row = {'cohort': cohort, 'n': len(group), 'deaths': int(cases.sum()),
               'early_censoring': int(((event == 0) & (time < TAU)).sum()),
               'km_percent': 100 * km_risk(time, event, np.ones(len(group), dtype=int))}
        for model in ['Clinical', 'Gene', 'Pathway']:
            p = group['p_' + model].to_numpy()
            differences = p[cases, None] - p[None, controls]
            pair_weights = w[cases, None] * w[None, controls]
            row['auc_' + model] = float((pair_weights * ((differences > 0) + .5 * (differences == 0))).sum() / pair_weights.sum())
        rows.append(row)
    out = ROOT / 'results/table1_cohorts.csv'
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print('Table I saved to results/table1_cohorts.csv')


if __name__ == '__main__':
    main()
