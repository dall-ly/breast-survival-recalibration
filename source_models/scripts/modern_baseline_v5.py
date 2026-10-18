"""XGBoost survival baselines under the fixed SCAN-B research protocol.

Official XGBoost objectives, not a reproduction of Transcriptome Transformer.
All variable screening and candidate selection are fitted on training folds.
External outcomes are used only after development choices have been serialized.
"""
from survival_experiments import load_data, sub, ROOT, SEED, TIMES, outcome, strata_of, feature_order, evaluate
import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import ndtr, expit
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

RUN = ROOT / 'runs/gse96058_os_v5_expansion/modern_baseline'
RUN.mkdir(parents=True, exist_ok=True)


def dump(name, value):
    (RUN / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                      default=lambda v: v.item() if hasattr(v, 'item') else str(v)), encoding='utf-8')


def candidate_grid():
    rows = []
    for genes, depth, reg, trees in itertools.product([0, 30, 100], [1, 3], [1., 10.], [120, 350]):
        rows.append(dict(objective='cox', genes=genes, depth=depth, reg=reg, trees=trees))
    for genes, depth, trees, distribution, scale in itertools.product(
            [0, 30, 100], [1, 3], [120, 350], ['normal', 'logistic'], [1., 1.5]):
        rows.append(dict(objective='aft', genes=genes, depth=depth, reg=10., trees=trees,
                         distribution=distribution, scale=scale))
    return rows


def transform(a, indices):
    # Native XGBoost missing-value handling for clinical variables; no test-statistic fitting.
    return np.asarray(np.column_stack([a['clinical'], a['ranks'][:, indices]]), dtype=np.float32)


def breslow_fit(lp, a):
    y = outcome(a)
    center = float(np.median(lp))
    relative = np.exp(lp - center)
    times = np.unique(y['time'][y['event']])
    increments = np.array([np.sum((y['time'] == t) & y['event']) / relative[y['time'] >= t].sum()
                           for t in times])
    return {'times': times, 'cumhaz': np.cumsum(increments), 'center': center}


def fit_model(a, params, indices=None):
    if indices is None:
        indices = feature_order(a)[:params['genes']] if params['genes'] else np.array([], dtype=int)
    x = transform(a, indices)
    y = outcome(a)
    dm = xgb.DMatrix(x)
    settings = dict(objective='survival:' + params['objective'], eta=.03, max_depth=params['depth'],
                    reg_lambda=params['reg'], min_child_weight=10., tree_method='hist', max_bin=64,
                    nthread=2, seed=SEED, subsample=1., colsample_bytree=1.)
    if params['objective'] == 'cox':
        dm.set_label(np.where(y['event'], y['time'], -y['time']))
        settings['eval_metric'] = 'cox-nloglik'
    else:
        dm.set_float_info('label_lower_bound', y['time'])
        dm.set_float_info('label_upper_bound', np.where(y['event'], y['time'], np.inf))
        settings.update(aft_loss_distribution=params['distribution'],
                        aft_loss_distribution_scale=params['scale'], eval_metric='aft-nloglik',
                        base_score=float(np.median(y['time'])))
    history = {}
    booster = xgb.train(settings, dm, num_boost_round=params['trees'], evals=[(dm, 'train')],
                        evals_result=history, verbose_eval=False)
    margin = booster.predict(dm, output_margin=True).astype(float)
    fit = {'model': booster, 'gene_indices': indices, 'parameters': params,
           'training_ids': np.asarray(a['id']), 'training_loss': history}
    if params['objective'] == 'cox':
        fit['breslow'] = breslow_fit(margin, a)
    assert np.isfinite(margin).all()
    assert np.isfinite(np.asarray(next(iter(history['train'].values())))).all()
    return fit


def predict_fit(fit, b, times=TIMES):
    """Predict absolute death risks at requested days. b requires clinical and ranks only."""
    margin = fit['model'].predict(xgb.DMatrix(transform(b, fit['gene_indices'])), output_margin=True).astype(float)
    params = fit['parameters']
    if params['objective'] == 'cox':
        base = fit['breslow']
        index = np.searchsorted(base['times'], times, side='right') - 1
        hazard = np.where(index < 0, 0., base['cumhaz'][np.maximum(index, 0)])
        p = -np.expm1(-np.exp(margin - base['center'])[:, None] * hazard[None, :])
    else:
        z = (np.log(np.asarray(times))[None, :] - margin[:, None]) / params['scale']
        p = ndtr(z) if params['distribution'] == 'normal' else expit(z)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert np.all(np.diff(p, axis=1) >= -1e-12)
    return p


def family(params):
    return 'XGB_' + params['objective'].upper() + ('_clinical' if params['genes'] == 0 else '_fusion')


def select(a, seed, label):
    grid = candidate_grid()
    config_hash = hashlib.sha256(json.dumps(grid, sort_keys=True).encode()).hexdigest()
    cachefile = RUN / (label + '_candidates.json')
    if cachefile.exists():
        cache = json.loads(cachefile.read_text(encoding='utf-8'))
        assert cache['grid_hash'] == config_hash
        assert cache['training_ids'] == a['id'].tolist()
        records = cache['records']
    else:
        records = []
    folds = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(a['t'], strata_of(a)))
    prepared = []
    membership = []
    for fold, (tr, va) in enumerate(folds, 1):
        train, valid = sub(a, tr), sub(a, va)
        prepared.append((train, valid, feature_order(train)))
        membership.extend({'fold': fold, 'role': role, 'patient_id': p}
                          for role, data in [('train', train), ('validation', valid)] for p in data['id'])
    pd.DataFrame(membership).to_csv(RUN / (label + '_fold_membership.csv'), index=False)
    for i, params in enumerate(grid):
        if any(row['candidate'] == i for row in records):
            continue
        scores = []
        for train, valid, order in prepared:
            fit = fit_model(train, params, order[:params['genes']])
            scores.append(evaluate(valid, predict_fit(fit, valid))[2])
        row = {'candidate': i, 'model': family(params), 'parameters': params,
               'cv_auc': float(np.mean([s['auc_ipcw'] for s in scores])),
               'cv_brier': float(np.mean([s['brier_ipcw'] for s in scores])), 'fold_metrics': scores}
        records.append(row)
        dump(label + '_candidates.json', {'grid_hash': config_hash, 'training_ids': a['id'].tolist(), 'records': records})
        if (i + 1) % 12 == 0:
            print(label, 'finished', i + 1, '/', len(grid), flush=True)
    df = pd.DataFrame([{k: v for k, v in row.items() if k not in ['parameters', 'fold_metrics']} | row['parameters']
                       for row in records])
    df.to_csv(RUN / (label + '_candidates.csv'), index=False)
    choices = {name: sorted([r for r in records if r['model'] == name], key=lambda r: (-r['cv_auc'], r['cv_brier']))[0]
               for name in sorted(set(r['model'] for r in records))}
    dump(label + '_selected.json', choices)
    return choices, prepared


def prediction_frame(b, predictions, extra=None):
    frames = []
    for j, t in enumerate(TIMES):
        frame = pd.DataFrame({'patient_id': b['id'], 'year': int(t / 365), 'os_days': b['t'], 'os_event': b['e'],
                              **{n: p[:, j] for n, p in predictions.items()}})
        for key, value in (extra or {}).items():
            frame[key] = value
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nested', action='store_true')
    args = parser.parse_args()
    started = time.time()
    scan, meta, dev, test, strata, repeats, genes = load_data()
    a = sub(scan, dev)
    dump('protocol.json', {'endpoint': 'All-cause overall survival; administrative training truncation at 2190 days',
                           'selection': 'Mean 3-fold development IPCW 3-year AUC, Brier tie-break',
                           'data_split': 'Existing 2618 development / 655 SCAN-B research holdout / 1978 METABRIC external',
                           'features': 'Five clinical columns with native missing-value handling; 0/30/100 ranks of genes selected on training survival residual score',
                           'grid': candidate_grid(), 'external_selection': False,
                           'xgboost_version': xgb.__version__, 'threads': 2,
                           'not_TxT': 'Official XGBoost Cox/AFT implementations; not TxT reproduction or adaptation',
                           'history': 'Continuing study using previously inspected holdout/external cohorts',
                           'nested_outer_seed': SEED + 101})
    choices, prepared = select(a, SEED, 'development')
    oof = []
    for fold, (tr, va, order) in enumerate(prepared, 1):
        predictions = {name: predict_fit(fit_model(tr, choice['parameters'], order[:choice['parameters']['genes']]), va)
                       for name, choice in choices.items()}
        oof.append(prediction_frame(va, predictions, {'fold': fold}))
    pd.concat(oof).to_csv(RUN / 'development_oof_predictions.csv', index=False)
    # This file is finalized before any holdout/external metric is computed.
    dump('evaluation_freeze.json', {'choices': choices, 'development_ids': a['id'].tolist(),
                                    'gene_universe': genes.Hugo_Symbol.tolist(), 'frozen_unix_time': time.time()})
    fits = {name: fit_model(a, choice['parameters']) for name, choice in choices.items()}
    for fit in fits.values():
        fit['shared_genes'] = genes.Hugo_Symbol.tolist()
    joblib.dump(fits, RUN / 'models.joblib')
    reloaded = joblib.load(RUN / 'models.joblib')
    metrics = []
    audit = []
    for cohort, b in [('SCANB_holdout', sub(scan, test)), ('METABRIC_external', meta)]:
        predictions = {}
        for name, fit in fits.items():
            p = predict_fit(fit, b)
            predictions[name] = p
            replay = float(np.max(np.abs(p - predict_fit(reloaded[name], b))))
            assert replay == 0.
            audit.append({'cohort': cohort, 'model': name, 'replay_max_error': replay,
                          'train_eval_patient_overlap': len(set(a['id']) & set(b['id'])),
                          'finite': bool(np.isfinite(p).all()), 'monotone': bool((np.diff(p, axis=1) >= -1e-12).all())})
            metrics.extend({'cohort': cohort, 'model': name, **m} for m in evaluate(b, p))
        prediction_frame(b, predictions).to_csv(RUN / ('predictions_' + cohort + '.csv'), index=False)
    pd.DataFrame(metrics).to_csv(RUN / 'metrics.csv', index=False)
    dump('replay_audit.json', audit)
    if args.nested:
        outer_frames = []
        outer_metrics = []
        for fold, (tr, va) in enumerate(StratifiedKFold(3, shuffle=True, random_state=SEED + 101).split(a['t'], strata_of(a)), 1):
            train, valid = sub(a, tr), sub(a, va)
            selected, _ = select(train, SEED + fold, 'nested' + str(fold))
            ps = {name: predict_fit(fit_model(train, choice['parameters']), valid) for name, choice in selected.items()}
            best = max(selected, key=lambda n: (selected[n]['cv_auc'], -selected[n]['cv_brier']))
            ps['XGB_inner_selected'] = ps[best]
            outer_frames.append(prediction_frame(valid, ps, {'outer_fold': fold}))
            outer_metrics.extend({'outer_fold': fold, 'model': n, **m} for n, p in ps.items() for m in evaluate(valid, p))
            pd.concat(outer_frames).to_csv(RUN / 'nested_outer_predictions.csv', index=False)
            pd.DataFrame(outer_metrics).to_csv(RUN / 'nested_outer_metrics.csv', index=False)
        pooled = pd.concat(outer_frames)
        pooled_metrics = []
        from survival_experiments import metric, km_censor
        for year, data in pooled.groupby('year'):
            for name in ps:
                met = metric(data.os_days.to_numpy(), data.os_event.to_numpy(), data[name].to_numpy(),
                             km_censor(data.os_days.to_numpy(), data.os_event.to_numpy()), year * 365)
                pooled_metrics.append({'year': year, 'model': name, **met})
        pd.DataFrame(pooled_metrics).to_csv(RUN / 'nested_pooled_metrics.csv', index=False)
    dump('execution.json', {'completed': True, 'seconds': time.time() - started, 'nested_completed': args.nested})
    print(pd.DataFrame(metrics).query('year == 3')[['cohort', 'model', 'auc_ipcw', 'brier_ipcw']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
