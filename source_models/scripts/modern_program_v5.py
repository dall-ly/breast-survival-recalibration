"""XGBoost Cox/AFT on five clinical variables plus 50 Hallmark rank means."""
import modern_baseline_v5 as mb
from modern_baseline_v5 import load_data, sub, ROOT, SEED, TIMES, outcome, strata_of, evaluate
import itertools
import json
import time
import hashlib
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

RUN = mb.RUN / 'programs'
RUN.mkdir(parents=True, exist_ok=True)
GMT = RUN.parents[1] / 'resources/MSigDB_Hallmark_2020.gmt'


def dump(name, value):
    (RUN / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                      default=lambda v: v.item() if hasattr(v, 'item') else str(v)), encoding='utf-8')


def candidate_grid():
    rows = []
    for objective, depth, reg, trees in itertools.product(['cox', 'aft'], [1, 2, 3], [1., 10.], [100, 300]):
        params = dict(objective=objective, genes=50, depth=depth, reg=reg, trees=trees)
        if objective == 'aft':
            params.update(distribution='normal', scale=1.)
        rows.append(params)
    return rows


def make_sets(symbols):
    lookup = {str(g): i for i, g in enumerate(symbols)}
    sets = {}
    for line in GMT.read_text(encoding='utf-8').splitlines():
        fields = line.split('\t')
        indices = [lookup[g] for g in dict.fromkeys(fields[2:]) if g in lookup]
        assert len(indices) >= 10
        sets[fields[0]] = np.array(indices)
    assert len(sets) == 50
    return sets


def program_data(a, sets):
    return {**a, 'ranks': np.column_stack([np.mean(a['ranks'][:, indices], axis=1, dtype=float) for indices in sets.values()])}


def fit_program(a, params, symbols):
    sets = make_sets(symbols)
    data = program_data(a, sets)
    fit = mb.fit_model(data, params, np.arange(50))
    return {'fit': fit, 'gene_sets': sets, 'shared_genes': list(symbols), 'parameters': params}


def predict_program(fit, b, times=TIMES):
    return mb.predict_fit(fit['fit'], program_data(b, fit['gene_sets']), times)


def select(a, symbols, seed, label):
    grid = candidate_grid()
    folds = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(a['t'], strata_of(a)))
    sets = make_sets(symbols)
    data = program_data(a, sets)
    prepared = [(sub(data, tr), sub(data, va)) for tr, va in folds]
    rows = []
    membership = []
    for fold, (tr, va) in enumerate(prepared, 1):
        membership.extend({'fold': fold, 'role': role, 'patient_id': p}
                          for role, d in [('train', tr), ('validation', va)] for p in d['id'])
    pd.DataFrame(membership).to_csv(RUN / (label + '_fold_membership.csv'), index=False)
    for i, params in enumerate(grid):
        scores = []
        for tr, va in prepared:
            fit = mb.fit_model(tr, params, np.arange(50))
            scores.append(evaluate(va, mb.predict_fit(fit, va))[2])
        rows.append({'candidate': i, 'model': 'XGB_' + params['objective'].upper() + '_Hallmark',
                     'parameters': params, 'cv_auc': float(np.mean([m['auc_ipcw'] for m in scores])),
                     'cv_brier': float(np.mean([m['brier_ipcw'] for m in scores])), 'fold_metrics': scores})
    choices = {name: sorted([r for r in rows if r['model'] == name], key=lambda r: (-r['cv_auc'], r['cv_brier']))[0]
               for name in sorted(set(r['model'] for r in rows))}
    dump(label + '_candidates.json', {'training_ids': a['id'].tolist(), 'records': rows})
    pd.DataFrame([{k: v for k, v in row.items() if k not in ['parameters', 'fold_metrics']} | row['parameters'] for row in rows]).to_csv(RUN / (label + '_candidates.csv'), index=False)
    dump(label + '_selected.json', choices)
    print(label, 'selected', {n: round(v['cv_auc'], 5) for n, v in choices.items()}, flush=True)
    return choices, folds


def main():
    started = time.time()
    scan, meta, dev, test, strata, repeats, genes = load_data()
    a = sub(scan, dev)
    symbols = genes.Hugo_Symbol.tolist()
    dump('protocol.json', {'features': 'Five clinical variables and 50 mean within-sample rank Hallmark scores',
                           'gmt_sha256': hashlib.sha256(GMT.read_bytes()).hexdigest(),
                           'grid': candidate_grid(), 'aft_distribution': 'Normal, scale=1.0 fixed default; not chosen from prior external results',
                           'selection': 'Mean 3-fold development IPCW 3-year AUC, Brier tie-break',
                           'nested': 'Three outer folds seed SEED+101, three inner folds seed SEED+fold',
                           'external_selection': False, 'xgboost_version': mb.xgb.__version__})
    selected, folds = select(a, symbols, SEED, 'development')
    frames = []
    for fold, (tr, va) in enumerate(folds, 1):
        aa, bb = sub(a, tr), sub(a, va)
        ps = {n: predict_program(fit_program(aa, v['parameters'], symbols), bb) for n, v in selected.items()}
        frames.append(mb.prediction_frame(bb, ps, {'fold': fold}))
    pd.concat(frames).to_csv(RUN / 'development_oof_predictions.csv', index=False)
    dump('evaluation_freeze.json', {'selected': selected, 'frozen_unix_time': time.time(), 'training_ids': a['id'].tolist()})
    models = {n: fit_program(a, v['parameters'], symbols) for n, v in selected.items()}
    joblib.dump(models, RUN / 'models.joblib')
    replay = joblib.load(RUN / 'models.joblib')
    rows = []
    checks = []
    for cohort, b in [('SCANB_holdout', sub(scan, test)), ('METABRIC_external', meta)]:
        ps = {n: predict_program(fit, b) for n, fit in models.items()}
        for name, p in ps.items():
            error = float(np.max(np.abs(p - predict_program(replay[name], b))))
            assert error == 0
            checks.append({'model': name, 'cohort': cohort, 'replay_error': error,
                           'train_eval_overlap': len(set(a['id']) & set(b['id']))})
            rows.extend({'model': name, 'cohort': cohort, **m} for m in evaluate(b, p))
        mb.prediction_frame(b, ps).to_csv(RUN / ('predictions_' + cohort + '.csv'), index=False)
    pd.DataFrame(rows).to_csv(RUN / 'metrics.csv', index=False)
    dump('replay_audit.json', checks)
    outer_frames, outer_metrics = [], []
    for fold, (tr, va) in enumerate(StratifiedKFold(3, shuffle=True, random_state=SEED + 101).split(a['t'], strata_of(a)), 1):
        aa, bb = sub(a, tr), sub(a, va)
        choice, _ = select(aa, symbols, SEED + fold, 'nested' + str(fold))
        ps = {n: predict_program(fit_program(aa, v['parameters'], symbols), bb) for n, v in choice.items()}
        winner = max(choice, key=lambda n: (choice[n]['cv_auc'], -choice[n]['cv_brier']))
        ps['XGB_Hallmark_inner_selected'] = ps[winner]
        outer_frames.append(mb.prediction_frame(bb, ps, {'outer_fold': fold}))
        outer_metrics.extend({'outer_fold': fold, 'model': n, **m} for n, p in ps.items() for m in evaluate(bb, p))
        pd.concat(outer_frames).to_csv(RUN / 'nested_outer_predictions.csv', index=False)
        pd.DataFrame(outer_metrics).to_csv(RUN / 'nested_outer_metrics.csv', index=False)
    from survival_experiments import km_censor, metric
    pooled = pd.concat(outer_frames)
    pooled_rows = []
    for year, df in pooled.groupby('year'):
        for name in ps:
            pooled_rows.append({'year': year, 'model': name, **metric(df.os_days.to_numpy(), df.os_event.to_numpy(), df[name].to_numpy(), km_censor(df.os_days.to_numpy(), df.os_event.to_numpy()), year * 365)})
    pd.DataFrame(pooled_rows).to_csv(RUN / 'nested_pooled_metrics.csv', index=False)
    dump('execution.json', {'completed': True, 'seconds': time.time() - started, 'nested_completed': True})
    print(pd.DataFrame(rows).query('year == 3')[['cohort', 'model', 'auc_ipcw', 'brier_ipcw']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
