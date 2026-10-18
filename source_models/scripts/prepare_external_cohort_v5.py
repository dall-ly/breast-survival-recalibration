"""Prepare outcome-independent GPL570 transport inputs and locked external fits.

Run from the parent workspace with the project's survival Python environment.
The original 18,150-gene model is never silently applied to a different rank
universe. GPL570 adaptation uses the exact measured Entrez-ID intersection and
refits fixed source-only recipes. External outcomes are not used for fitting.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/gse96058_os_v5_expansion/external_cohort'
RAW = RUN / 'raw'
V1 = ROOT / 'runs/gse96058_os_v1'
V3 = ROOT / 'runs/gse96058_os_v3_survival'

SOURCES = {
    'GSE42568_matrix': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE42nnn/GSE42568/matrix/GSE42568_series_matrix.txt.gz',
    'GPL570_annotation': 'https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz',
    'GSE42568_geo': 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE42568',
    'GSE42568_publication': 'https://doi.org/10.1093/carcin/bgt208',
    'GSE20685_geo': 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE20685',
    'GSE20685_matrix': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE20nnn/GSE20685/matrix/GSE20685_series_matrix.txt.gz',
    'GSE20685_publication': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC3094326/',
}


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                   default=lambda v: v.item() if hasattr(v, 'item') else str(v)), encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_matrix(path, expression=True):
    """Parse exact GEO column order; characteristics may vary by sample row."""
    meta, series, lines = [], {}, []
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.startswith('!series_matrix_table_begin'):
                break
            row = next(csv.reader([line], delimiter='\t'))
            if not row:
                continue
            if row[0].startswith('!Sample'):
                meta.append(row)
            elif row[0].startswith('!Series'):
                series.setdefault(row[0], []).extend(row[1:])
            lines.append(line)
        if expression:
            table = f.read().split('!series_matrix_table_end')[0]
            expr = pd.read_csv(io.StringIO(table), sep='\t', index_col=0)
        else:
            expr = None
    ids = next(row[1:] for row in meta if row[0] == '!Sample_geo_accession')
    records = [{'sample_id': x} for x in ids]
    for row in meta:
        assert len(row) == len(ids) + 1, (row[0], len(row), len(ids))
        for record, value in zip(records, row[1:]):
            if row[0] == '!Sample_characteristics_ch1':
                if ':' in value:
                    key, val = value.split(':', 1)
                    assert key.strip() not in record, key
                    record[key.strip()] = val.strip()
            elif row[0] in ('!Sample_title', '!Sample_source_name_ch1', '!Sample_platform_id'):
                record[row[0].replace('!Sample_', '')] = value
    data = pd.DataFrame(records)
    assert data.sample_id.is_unique
    if expression:
        assert expr.columns.tolist() == ids
        assert expr.index.is_unique
        assert np.isfinite(expr.to_numpy()).all()
    return data, expr, series, ''.join(lines)


def read_annotation():
    path = RAW / 'GPL570.annot.gz'
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        content = f.read()  # Reading through EOF also validates gzip CRC.
    table = content.split('!platform_table_begin\n')[1].split('!platform_table_end')[0]
    ann = pd.read_csv(io.StringIO(table), sep='\t', dtype=str, keep_default_na=False)
    ann['entrez_ids'] = ann['Gene ID'].map(lambda s: sorted(set(x.strip() for x in s.split('///') if x.strip())))
    ann['single_entrez'] = ann.entrez_ids.map(len).eq(1)
    ann['entrez_id'] = ann.entrez_ids.map(lambda x: x[0] if len(x) == 1 else '')
    ann[['ID', 'Gene symbol', 'Gene ID', 'single_entrez', 'entrez_id']].to_csv(RUN / 'gpl570_probe_mapping_audit.csv', index=False)
    return ann


def clean_clinical(data, cohort):
    out = pd.DataFrame({'sample_id': data.sample_id, 'cohort': cohort})
    num = lambda col: pd.to_numeric(data[col], errors='coerce')
    if cohort == 'GSE42568':
        out['is_primary_cancer'] = data['tissue'].eq('breast cancer')
        out['age'] = num('age')
        out['size_original_cm'] = num('size')
        out['size'] = out.size_original_cm * 10.0
        out['grade'] = num('grade')
        out['node_positive'] = num('lymph node status')
        out['er_positive'] = num('er_status')
        out['os_days'] = num('overall survival time_days')
        out['os_event'] = num('overall survival event')
    elif cohort == 'GSE20685':
        out['is_primary_cancer'] = data['tissue'].eq('primary breast cancer')
        out['age'] = num('age at diagnosis')
        out['size'] = np.nan  # T stage is not a measured diameter.
        out['grade'] = np.nan
        out['er_positive'] = np.nan
        out['n_stage'] = num('n_stage')
        out['node_positive'] = np.where(out.n_stage.notna(), (out.n_stage > 0).astype(float), np.nan)
        out['t_stage'] = num('t_stage')
        out['m_stage'] = num('m_stage')
        out['os_years_original'] = num('follow_up_duration (years)')
        out['os_days'] = out.os_years_original * 365.0
        out['os_event'] = num('event_death')
    else:
        raise ValueError(cohort)
    out['os_eligible'] = out.is_primary_cancer & out.os_days.gt(0) & out.os_event.isin([0, 1])
    out['exclusion_reason'] = np.select([~out.is_primary_cancer, ~out.os_days.gt(0), ~out.os_event.isin([0, 1])],
                                     ['normal_or_nonprimary_tissue', 'missing_or_nonpositive_os_time', 'invalid_or_missing_os_event'], default='')
    for col in ['node_positive', 'er_positive']:
        assert out[col].dropna().isin([0, 1]).all(), col
    assert out.grade.dropna().isin([1, 2, 3]).all()
    out.to_csv(RUN / f'{cohort}_clinical_all.csv', index=False)
    eligible = out.loc[out.os_eligible].reset_index(drop=True)
    eligible.to_csv(RUN / f'{cohort}_clinical_eligible.csv', index=False)
    summary = {'samples_total': len(out), 'primary_cancer': int(out.is_primary_cancer.sum()),
               'os_eligible': len(eligible), 'events_all_followup': int(eligible.os_event.sum()),
               'events_by_3y': int(((eligible.os_event == 1) & (eligible.os_days <= 1095)).sum()),
               'censored_before_3y': int(((eligible.os_event == 0) & (eligible.os_days < 1095)).sum()),
               'followup_days_range': [float(eligible.os_days.min()), float(eligible.os_days.max())],
               'missing_clinical': eligible[['age', 'size', 'grade', 'node_positive', 'er_positive']].isna().sum().to_dict(),
               'ER_positive': int(eligible.er_positive.eq(1).sum()), 'node_positive': int(eligible.node_positive.eq(1).sum())}
    return out, eligible, summary


def prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    protocol = {
        'date': '2026-10-16', 'endpoint': 'all-cause OS; original deposited event definitions',
        'external_outcomes_for_selection': False,
        'primary_external_cohort': 'GSE42568',
        'primary_cohort_choice_basis': 'All five required clinical predictors and OS available in official GEO metadata; selection before model assessment.',
        'expression': 'Deposited log2 GC-RMA signals; no target-outcome correction or batch fitting.',
        'mapping': 'GEO GPL570 annotation Aug 09 2016; unique Entrez ID only. Exclude missing/multiple distinct Entrez IDs. Match model Entrez IDs to avoid historical symbol aliases.',
        'probe_collapse': 'Arithmetic mean of all eligible log2 probes for each Entrez gene, within each sample. No outcome, variance, or survival-based probe selection.',
        'rank_universe': 'Intersection of 18,150 source genes with uniquely mapped GPL570 genes. Identical within-sample average rank / universe size for source, target, and technical repeats.',
        'clinical': 'GSE42568 tumor diameter cm multiplied by 10 to mm; age years; deposited ER and node binary codes checked against series totals. Missing values retained for source-trained imputer.',
        'GSE20685_role': 'Backup molecular / reduced clinical validation only; deposited GEO has no grade, ER, or measured continuous tumor diameter. Do not invent these from molecular subtype/T stage.',
        'validation_rule': 'Freeze source-only model recipes before external prediction; no external gene screening, tuning, threshold choice, or recalibration.',
        'overlap': {'GSE42568': 'Irish St Vincent hospital cases 1993-1997; SCAN-B is Swedish and METABRIC UK/Canada. No known shared recruitment source. GEO IDs checked disjoint; anonymized identifiers cannot definitively exclude every cross-study duplicate.',
                    'GSE20685': 'Koo Foundation Sun Yat-Sen Cancer Center, Taiwan, 1991-2004; different recruitment setting from source/METABRIC.'},
        'sources': SOURCES,
    }
    dump(RUN / 'external_preparation_protocol.json', protocol)
    ann = read_annotation()
    genes = pd.read_csv(V1 / 'shared_gene_map.csv')
    genes['entrez_key'] = genes.Entrez_Gene_Id.astype(str)
    single = ann.loc[ann.single_entrez].copy()
    mask = genes.entrez_key.isin(single.entrez_id).to_numpy()
    np.save(RUN / 'shared_gene_mask.npy', mask)
    mapped = genes.loc[mask].copy()
    mapped['source_shared_index'] = np.flatnonzero(mask)
    mapped['transport_index'] = np.arange(len(mapped))
    mapped['n_probes'] = mapped.entrez_key.map(single.groupby('entrez_id').size())
    mapped.to_csv(RUN / 'transport_gene_map.csv', index=False)
    genes.loc[~mask].to_csv(RUN / 'unmeasured_source_genes.csv', index=False)
    summaries = {}
    path = RAW / 'GSE42568_series_matrix.txt.gz'
    data, expr, series, header = read_matrix(path)
    data.to_csv(RUN / 'GSE42568_metadata_raw.csv', index=False)
    (RUN / 'GSE42568_series_header.txt').write_text(header, encoding='utf-8')
    dump(RUN / 'GSE42568_series_metadata.json', series)
    all_clin, clin, summary = clean_clinical(data, 'GSE42568')
    assert len(clin) == 104 and len(data) == 121
    assert clin.er_positive.eq(1).sum() == 67 and clin.er_positive.eq(0).sum() == 34
    assert clin.node_positive.eq(1).sum() == 59 and clin.node_positive.eq(0).sum() == 45
    assert clin['size'].min() == 6 and clin['size'].max() == 80
    aligned = single.set_index('ID').reindex(expr.index)
    valid = aligned.single_entrez.fillna(False).to_numpy(dtype=bool)
    group_ids = aligned.loc[valid, 'entrez_id'].to_numpy()
    collapsed = expr.iloc[np.flatnonzero(valid)].groupby(group_ids, sort=False).mean()
    target = collapsed.loc[mapped.entrez_key, clin.sample_id].to_numpy().T
    assert target.shape == (len(clin), len(mapped))
    assert np.isfinite(target).all()
    ranks = (rankdata(target, axis=1, method='average') / target.shape[1]).astype(np.float32)
    np.save(RUN / 'GSE42568_expression_transport.npy', target.astype(np.float32))
    np.save(RUN / 'GSE42568_ranks_transport.npy', ranks)
    summary.update({'matrix_probes': len(expr), 'transport_genes': len(mapped), 'full_model_rank_universe_available': bool(mask.all()),
                    'all_v3_selected30_genes_measured': bool(pd.read_csv(V3 / 'Surv_consistency_genes.csv').Entrez_Gene_Id.astype(str).isin(single.entrez_id).all())})
    summaries['GSE42568'] = summary
    backup = RAW / 'GSE20685_series_matrix.txt.gz'
    if not backup.exists():
        backup = RAW / 'GSE20685_series_matrix.partial.gz'
    if backup.exists() and backup.stat().st_size:
        # A transport interruption can leave a gzip with a complete metadata header.
        # Header-only extraction is separately marked; no incomplete expression read.
        data_b, _, series_b, header_b = read_matrix(backup, expression=False)
        data_b.to_csv(RUN / 'GSE20685_metadata_raw.csv', index=False)
        (RUN / 'GSE20685_series_header.txt').write_text(header_b, encoding='utf-8')
        dump(RUN / 'GSE20685_series_metadata.json', series_b)
        _, _, summaries['GSE20685'] = clean_clinical(data_b, 'GSE20685')
        try:
            with gzip.open(backup, 'rb') as f:
                for _ in iter(lambda: f.read(1024 * 1024), b''):
                    pass
            complete = True
        except (EOFError, OSError):
            complete = False
        summaries['GSE20685']['expression_download_complete'] = complete
        summaries['GSE20685']['expression_prepared'] = False
    scan_ids = set(pd.read_csv(V1 / 'scanb_clinical_clean.csv').sample_id.astype(str))
    met_ids = set(pd.read_csv(V1 / 'metabric_clinical_os.csv').patient_id.astype(str))
    summary['sample_id_overlap_SCANB'] = len(set(clin.sample_id) & scan_ids)
    summary['sample_id_overlap_METABRIC'] = len(set(clin.sample_id) & met_ids)
    # Restricting old ranks preserves each patient's within-universe order/ties;
    # re-ranking exactly implements the common-gene rank universe without refitting.
    for name, file in [('SCANB', 'scanb_shared_ranks.npy'), ('METABRIC', 'metabric_shared_ranks.npy')]:
        original = np.load(V1 / file, mmap_mode='r')
        transformed = (rankdata(original[:, mask], axis=1, method='average') / mask.sum()).astype(np.float32)
        np.save(RUN / f'{name}_ranks_transport.npy', transformed)
        assert np.isfinite(transformed).all()
    manifest = {'created': '2026-10-16', 'primary_complete_inputs': {}, 'backup_inputs': {}, 'summaries': summaries,
                'universe': {'original': len(genes), 'measured': int(mask.sum()), 'unmeasured': int((~mask).sum()),
                             'single_entrez_probes': int(ann.single_entrez.sum()), 'mapped_target_probes': int(mapped.n_probes.sum())},
                'source_outcome_model_tuning': False, 'external_prediction_performed': False}
    for path in [RAW / 'GPL570.annot.gz', RAW / 'GSE42568_series_matrix.txt.gz']:
        manifest['primary_complete_inputs'][path.name] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    if backup.exists():
        manifest['backup_inputs'][backup.name] = {'sha256': digest(backup), 'bytes': backup.stat().st_size,
                                                 'complete': summaries['GSE20685']['expression_download_complete'],
                                                 'use': 'metadata-only; expression unavailable if incomplete'}
    dump(RUN / 'preparation_audit.json', manifest)
    pd.DataFrame([{'cohort': k, **{a: b for a, b in v.items() if not isinstance(b, dict)}} for k, v in summaries.items()]).to_csv(RUN / 'cohort_summary.csv', index=False)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def transport_experiment(evaluate_external=False):
    """Fixed source recipes, then one external assessment and source bridges."""
    import joblib
    import program_ranking_v5 as pr
    from survival_experiments import baseline_predict

    work = RUN / 'locked_transport'
    work.mkdir(parents=True, exist_ok=True)
    pr.WORK = work  # Keep Hallmark coverage and outputs isolated from root runs.
    selected_path = ROOT / 'runs/gse96058_os_v5_expansion/program_ranking/selection_frozen.json'
    source_selection = json.loads(selected_path.read_text(encoding='utf-8'))
    expected = {
        'genes30': {'representation': 'genes30', 'alpha': .3, 'ranking_weight': 0., 'gamma': .3},
        'hallmark': {'representation': 'hallmark', 'alpha': .3, 'ranking_weight': 1., 'gamma': .3},
        'hybrid': {'representation': 'hybrid', 'alpha': .3, 'ranking_weight': 0., 'gamma': .3},
        'hallmark_context': {'representation': 'hallmark_context', 'alpha': .3, 'ranking_weight': .3, 'gamma': .3},
    }
    for family, cfg in expected.items():
        assert source_selection['selected'][family] == cfg, (family, source_selection)
    scan, meta, dev, test, strata, repeats, genes = pr.load_data()
    mask = np.load(RUN / 'shared_gene_mask.npy')
    scan['ranks'] = np.load(RUN / 'SCANB_ranks_transport.npy')
    met_clin = pd.read_csv(V1 / 'metabric_clinical_os.csv')
    meta['ranks'] = np.load(RUN / 'METABRIC_ranks_transport.npy')[met_clin.os_eligible.to_numpy()]
    repeat_ids = list(repeats)
    repeat_array = np.stack([repeats[x][mask] for x in repeat_ids])
    repeat_array = (rankdata(repeat_array, axis=1, method='average') / mask.sum()).astype(np.float32)
    repeats = dict(zip(repeat_ids, repeat_array))
    symbols = genes.loc[mask, 'Hugo_Symbol'].tolist()
    a = pr.sub(scan, dev)
    bscan = pr.sub(scan, test)
    clin = pd.read_csv(RUN / 'GSE42568_clinical_eligible.csv')
    external = {'clinical': clin[['age', 'size', 'grade', 'node_positive', 'er_positive']].to_numpy(),
                'ranks': np.load(RUN / 'GSE42568_ranks_transport.npy'), 'id': clin.sample_id.to_numpy(),
                't': clin.os_days.to_numpy(), 'e': clin.os_event.to_numpy().astype(int)}
    for name, ids in [('source_training', a['id']), ('source_holdout', bscan['id']), ('external_GSE42568', external['id'])]:
        pd.DataFrame({'sample_id': ids}).to_csv(work / f'{name}_ids.csv', index=False)
    assert not (set(a['id']) & set(bscan['id']))
    assert not (set(a['id']) & set(external['id']))
    frozen = {'frozen_utc': datetime.now(timezone.utc).isoformat(),
              'source_selection': source_selection, 'source_selection_sha256': digest(selected_path),
              'mode': 'fixed-recipe refit on platform intersection; no model selection in external cohort',
              'parameters': expected, 'clinical_RSF': {'genes': 0, 'leaf': 30, 'features': 'sqrt'},
              'fusion_RSF': {'genes': 20, 'leaf': 30, 'features': 1.0},
              'source_train_n': len(a['id']), 'source_holdout_n': len(bscan['id']),
              'target_n': len(external['id']), 'universe_n': int(mask.sum()),
              'gmt_sha256': digest(pr.GMT),
              'ids_sha256': {p.name: digest(p) for p in work.glob('*_ids.csv')},
              'choice_on_target_outcomes': False,
              'transport_recipe': 'Re-screen genes and refit source-fold preprocessors with identical fixed hyperparameters using the measured gene universe.',
              'comparison_scope': 'All four source-frozen representation families plus clinical and fusion RSF; report all, do not select a winner using the new target cohort.'}
    dump(work / 'external_model_lock.json', frozen)
    sets = pr.gene_sets(symbols)
    # Fit/predict covariates are separated from target outcomes; dummy targets
    # below are not used by prepare/fit, and true target values enter evaluate only.
    masked_target = {**external, 't': np.ones(len(clin)), 'e': np.zeros(len(clin), dtype=int)}
    prep = pr.prepare(a, masked_target, repeats, sets, symbols)
    assert set(prep['repeat_ids']).issubset(set(a['id']))
    assert not set(prep['repeat_ids']) & set(bscan['id'])
    models, statuses, target_predictions = {}, {}, {}
    for rep, cfg in expected.items():
        target_predictions[rep], models[rep], statuses[rep] = pr.fit(prep, cfg)
        print('Transport fitted', rep, statuses[rep], flush=True)
    baselines = {'clinical_RSF': prep['anchor'],
                 'fusion_RSF': pr.baseline_fit(a, 'rsf', frozen['fusion_RSF'])}
    for name, model in baselines.items():
        target_predictions[name] = baseline_predict(model, masked_target)
    artifact = {'models': models, 'baselines': baselines, 'gene_symbols': symbols,
                'gene_mask_original': mask, 'source_training_ids': a['id'], 'parameters': expected,
                'repeat_training_ids': prep['repeat_ids'], 'freeze': frozen}
    joblib.dump(artifact, work / 'transport_models.joblib', compress=3)
    dump(work / 'fit_status.json', statuses)
    if not evaluate_external:
        dump(work / 'fit_complete_pending_evaluation.json', {'completed_utc': datetime.now(timezone.utc).isoformat(),
                'model_sha256': digest(work / 'transport_models.joblib'), 'external_outcomes_evaluated': False,
                'note': 'Awaiting additional source-frozen Hallmark XGBoost recipe before joint external assessment.'})
        print('Source-only transport models saved. External assessment is paused.', flush=True)
        return
    evaluate_saved_models()


def fit_modern_transport():
    import joblib
    import modern_program_v5 as mp
    work = RUN / 'locked_transport'
    artifact = joblib.load(work / 'transport_models.joblib')
    selection = ROOT / 'runs/gse96058_os_v5_expansion/modern_baseline/programs/development_selected.json'
    freeze_path = selection.parent / 'evaluation_freeze.json'
    selected = json.loads(selection.read_text(encoding='utf-8'))
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    assert freeze['selected'] == selected
    assert set(freeze['training_ids']) == set(artifact['source_training_ids'])
    clinical = pd.read_csv(V1 / 'scanb_clinical_clean.csv')
    ix = pd.Index(clinical.sample_id).get_indexer(artifact['source_training_ids'])
    assert (ix >= 0).all()
    clinical = clinical.iloc[ix]
    a = {'clinical': clinical[['age', 'size', 'grade', 'node_positive', 'er_positive']].to_numpy(),
         'id': clinical.sample_id.to_numpy(), 't': clinical.os_days.to_numpy(), 'e': clinical.os_event.to_numpy(),
         'ranks': np.load(RUN / 'SCANB_ranks_transport.npy', mmap_mode='r')[ix]}
    lock = {'frozen_utc': datetime.now(timezone.utc).isoformat(), 'source_selection': selected,
            'source_freeze': freeze, 'source_selection_sha256': digest(selection),
            'source_freeze_sha256': digest(freeze_path), 'target_outcomes_used': False,
            'source_model_sha256': digest(work / 'transport_models.joblib'),
            'transport_universe_n': len(artifact['gene_symbols']), 'gmt_sha256': digest(mp.GMT)}
    dump(work / 'modern_external_model_lock.json', lock)
    models = {name: mp.fit_program(a, record['parameters'], artifact['gene_symbols']) for name, record in selected.items()}
    joblib.dump(models, work / 'modern_transport_models.joblib', compress=3)
    dump(work / 'modern_fit_complete_pending_evaluation.json', {'completed_utc': datetime.now(timezone.utc).isoformat(),
            'model_sha256': digest(work / 'modern_transport_models.joblib'), 'external_outcomes_evaluated': False})
    print('Source-only Hallmark XGBoost transport models saved.', flush=True)


def evaluate_saved_models():
    import joblib
    import program_ranking_v5 as pr
    import modern_program_v5 as mp
    from survival_experiments import baseline_predict
    work = RUN / 'locked_transport'
    pr.WORK = work
    artifact = joblib.load(work / 'transport_models.joblib')
    models, baselines = artifact['models'], artifact['baselines']
    modern = joblib.load(work / 'modern_transport_models.joblib')
    frozen = artifact['freeze']
    source_selection = frozen['source_selection']
    assert frozen == json.loads((work / 'external_model_lock.json').read_text(encoding='utf-8'))
    a = {'id': artifact['source_training_ids']}
    cols = ['age', 'size', 'grade', 'node_positive', 'er_positive']
    scan_clin = pd.read_csv(V1 / 'scanb_clinical_clean.csv')
    holdout_ids = pd.read_csv(work / 'source_holdout_ids.csv').sample_id.to_numpy()
    ix = pd.Index(scan_clin.sample_id).get_indexer(holdout_ids)
    assert (ix >= 0).all()
    s = scan_clin.iloc[ix]
    bscan = {'id': s.sample_id.to_numpy(), 'clinical': s[cols].to_numpy(),
             't': s.os_days.to_numpy(), 'e': s.os_event.to_numpy().astype(int),
             'ranks': np.load(RUN / 'SCANB_ranks_transport.npy', mmap_mode='r')[ix]}
    m = pd.read_csv(V1 / 'metabric_clinical_os.csv')
    valid = m.os_eligible.to_numpy()
    meta = {'id': m.patient_id.to_numpy()[valid], 'clinical': m[cols].to_numpy()[valid],
            't': m.os_days.to_numpy()[valid], 'e': m.os_event.to_numpy()[valid].astype(int),
            'ranks': np.load(RUN / 'METABRIC_ranks_transport.npy', mmap_mode='r')[valid]}
    clin = pd.read_csv(RUN / 'GSE42568_clinical_eligible.csv')
    external = {'id': clin.sample_id.to_numpy(), 'clinical': clin[cols].to_numpy(),
                't': clin.os_days.to_numpy(), 'e': clin.os_event.to_numpy().astype(int),
                'ranks': np.load(RUN / 'GSE42568_ranks_transport.npy')}
    target_predictions = {name: pr.predict(model, external) for name, model in models.items()}
    target_predictions.update({name: baseline_predict(model, external) for name, model in baselines.items()})
    target_predictions.update({name: mp.predict_program(model, external) for name, model in modern.items()})
    dump(work / 'evaluation_started.json', {'started_utc': datetime.now(timezone.utc).isoformat(),
                                         'model_file_sha256': digest(work / 'transport_models.joblib'),
                                         'model_lock_sha256': digest(work / 'external_model_lock.json'),
                                         'modern_model_sha256': digest(work / 'modern_transport_models.joblib'),
                                         'modern_lock_sha256': digest(work / 'modern_external_model_lock.json')})
    records = []
    predictions = {}
    for label, b in [('GSE42568_external', external), ('SCANB_holdout_bridge', bscan), ('METABRIC_bridge', meta)]:
        ps = {name: pr.predict(model, b) for name, model in models.items()}
        ps.update({name: baseline_predict(model, b) for name, model in baselines.items()})
        ps.update({name: mp.predict_program(model, b) for name, model in modern.items()})
        predictions[label] = ps
        pr.frame_predictions(b, ps).to_csv(work / f'predictions_{label}.csv', index=False)
        for name, pred in ps.items():
            assert np.isfinite(pred).all() and (pred >= 0).all() and (pred <= 1).all()
            assert np.all(np.diff(pred, axis=1) >= -1e-12)
            records.extend({'cohort': label, 'model': name, **row} for row in pr.evaluate(b, pred))
    pd.DataFrame(records).to_csv(work / 'metrics_all_horizons.csv', index=False)
    replay = max(float(np.max(np.abs(predictions['GSE42568_external'][name] - p))) for name, p in target_predictions.items())
    assert replay < 1e-12
    # Paired target bootstrap: patient resampling, same sample across methods.
    rng = np.random.default_rng(20261016)
    boot = []
    for i in range(2000):
        ix = rng.integers(len(clin), size=len(clin))
        t, e = external['t'][ix], external['e'][ix]
        if np.sum((e == 1) & (t <= 1095)) < 2 or np.sum(t > 1095) < 2:
            continue
        km = pr.km_censor(t, e)
        row = {'bootstrap': i}
        for name, p in predictions['GSE42568_external'].items():
            row[name] = pr.metric(t, e, p[ix, 2], km, 1095)['auc_ipcw']
        boot.append(row)
    bf = pd.DataFrame(boot)
    bf.to_csv(work / 'GSE42568_bootstrap_auc.csv', index=False)
    point = pd.DataFrame(records).query("cohort == 'GSE42568_external' and year == 3").set_index('model')
    intervals = []
    for name in predictions['GSE42568_external']:
        values = bf[name].dropna()
        row = {'model': name, 'auc': point.loc[name, 'auc_ipcw'], 'ci_low': values.quantile(.025), 'ci_high': values.quantile(.975), 'valid_bootstraps': len(values)}
        for base in ['clinical_RSF', 'genes30']:
            diff = (bf[name] - bf[base]).dropna()
            row.update({f'delta_vs_{base}': point.loc[name, 'auc_ipcw'] - point.loc[base, 'auc_ipcw'],
                        f'delta_vs_{base}_ci_low': diff.quantile(.025), f'delta_vs_{base}_ci_high': diff.quantile(.975)})
        intervals.append(row)
    pd.DataFrame(intervals).to_csv(work / 'GSE42568_auc_uncertainty.csv', index=False)
    for name, model in models.items():
        pd.DataFrame({'feature': model['feature_names'], 'coefficient': model['coefficients'][1:]}).to_csv(work / f'{name}_coefficients.csv', index=False)
    checks = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'source_train_count': len(a['id']),
              'source_train_ids_disjoint_holdout': True, 'source_train_ids_disjoint_external': True,
              'cross_cohort_identity_linkage_available': False,
              'id_check_scope': 'String accession IDs only, not biological identity matching across independently anonymized cohorts.',
              'repeat_pairs_training_only': len(artifact['repeat_training_ids']), 'external_preprocess_fitted': False,
              'external_outcomes_used_for_training': False, 'all_probability_range_and_monotonicity_checks': True,
              'prediction_replay_max_error': replay, 'bootstrap_replicates': len(bf),
              'source_selection_time_utc': source_selection['timestamp_utc'], 'external_freeze_time_utc': frozen['frozen_utc'],
              'target_masked_during_model_fit': True, 'fixed_models_all_reported': list(target_predictions),
              'interpretation': 'Platform-specific source-only refits; not direct validation of the original 18,150-gene fitted model.'}
    dump(work / 'verification.json', checks)
    print(pd.DataFrame(records).query('year == 3')[['cohort', 'model', 'auc_ipcw', 'brier_ipcw']].to_string(index=False), flush=True)


def build_report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    work = RUN / 'locked_transport'
    provenance = {
        'checked_date': '2026-10-16',
        'GSE42568': {'country': 'Ireland', 'collection_center': "Department of Pathology, St Vincent's University Hospital, Dublin",
            'collection_period': '1993-1997', 'primary_publication': 'https://doi.org/10.1093/carcin/bgt208',
            'evidence_section': 'Materials and methods / In-house breast cancer gene expression profiling / Patient selection',
            'full_text_accessed': 'https://scispace.com/pdf/correlating-transcriptional-networks-to-breast-cancer-qqtpwl5k27.pdf',
            'geo_source': SOURCES['GSE42568_geo'],
            'local_primary_record': 'GSE42568_series_header.txt; GEO contact country Ireland and Dublin City University.'},
        'SCANB_GSE96058': {'country': 'Sweden', 'collection_context': 'Population-based multicenter Sweden Cancerome Analysis Network Breast initiative, centralized laboratory in Lund.',
            'primary_publication': 'https://ascopubs.org/doi/10.1200/PO.17.00135',
            'supporting_primary_publication': 'https://aacrjournals.org/clincancerres/article-abstract/25/21/6368/82001',
            'context_note': 'GSE96058 is the previously prepared Swedish study, not the separate 2010-2020 SCAN-B release.'},
        'METABRIC': {'countries': ['United Kingdom', 'Canada'],
            'primary_publication': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC3440846/',
            'evidence_section': 'A breast cancer population genomic resource',
            'evidence_summary': 'Primary article describes collection of fresh-frozen primary tumors from UK and Canadian tumor banks.'},
        'independence_assessment': 'Different reported research recruitment sources and sample accession namespaces support independent cohort provenance. No common recruitment center identified.',
        'identifier_comparison': {'GSE42568_vs_SCANB_string_intersection': 0, 'GSE42568_vs_METABRIC_string_intersection': 0},
        'cross_database_patient_identity_linkage': 'Unavailable; no shared patient identifier or genomic identity comparison was used. String-ID disjointness is not a cross-cohort biological de-duplication result.',
    }
    dump(RUN / 'cohort_provenance.json', provenance)
    ci = pd.read_csv(work / 'GSE42568_auc_uncertainty.csv')
    order = ['genes30', 'hallmark', 'hybrid', 'hallmark_context', 'clinical_RSF', 'fusion_RSF', 'XGB_AFT_Hallmark', 'XGB_COX_Hallmark']
    labels = ['CA-RRL: 30 genes', 'Hallmark + ranking', 'Genes + Hallmark', 'Hallmark + age/ER', 'Clinical RSF', 'Clinical + gene RSF', 'Hallmark XGB-AFT', 'Hallmark XGB-Cox']
    ci = ci.set_index('model').loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for i, row in ci.iterrows():
        color = '#176B70' if i == 0 else '#BF6F24' if i == 1 else '#6B7280'
        ax.errorbar(row.auc, i, xerr=[[row.auc - row.ci_low], [row.ci_high - row.auc]],
                    fmt='o', capsize=3, color=color, elinewidth=1.4, markersize=5)
        ax.text(.99, i, f'{row.auc:.3f}', va='center', ha='right', color=color, fontsize=10)
    ax.set_yticks(range(len(order)), labels)
    ax.invert_yaxis()
    ax.set_xlim(.50, 1.00)
    ax.set_xlabel('3-year OS IPCW AUC (paired bootstrap 95% intervals)')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color='#E7E9EC', linewidth=.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(work / 'GSE42568_auc.png', dpi=300)
    fig.savefig(work / 'GSE42568_auc.svg')
    plt.close(fig)
    audit = json.loads((RUN / 'preparation_audit.json').read_text(encoding='utf-8'))
    metrics = pd.read_csv(work / 'metrics_all_horizons.csv').query('year == 3')
    table = metrics.pivot(index='model', columns='cohort', values='auc_ipcw').reindex(order)
    table['GSE42568 95% CI'] = [f'{r.ci_low:.4f}–{r.ci_high:.4f}' for _, r in ci.iterrows()]
    ci_show = ci[['model', 'delta_vs_clinical_RSF', 'delta_vs_clinical_RSF_ci_low', 'delta_vs_clinical_RSF_ci_high']]
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
    <title>GSE42568 独立队列准备与验证</title><style>
    body{font-family:Arial,"Microsoft YaHei",sans-serif;color:#24313f;background:#f4f6f8;margin:0;line-height:1.75}
    main{max-width:1120px;margin:32px auto;background:white;padding:32px 40px;border-radius:12px;box-sizing:border-box}
    h1{font-size:28px;line-height:1.4}h2{font-size:20px;margin-top:30px;color:#176B70}p{max-width:1000px}
    .table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border-bottom:1px solid #e2e7ec;text-align:left;white-space:nowrap}th{background:#edf3f4}
    img{max-width:100%;height:auto}.lead{font-size:18px;background:#edf5f4;padding:18px;border-left:4px solid #176B70}
    a{color:#176B70}@media(max-width:600px){main{margin:0;padding:20px 16px;border-radius:0}h1{font-size:24px}}
    </style><main><h1>新独立队列 GSE42568：准备与固定模型验证</h1><p>2026-10-16 · 模型在原 SCAN-B 开发集训练 · 外部病例未参与选参</p>
    <p class="lead">可移植 CA-RRL 在 GSE42568 的 3 年 OS AUC 为 <b>0.8195</b>，Hallmark 排序扩展为 <b>0.8174</b>，临床 RSF 为 <b>0.7727</b>。两个分子残差模型在这个此前未使用的队列保持约 0.82 的区分能力；当前样本下，相对临床基线的配对差值区间仍跨越 0。</p>
    <h2>病例与临床字段</h2><p>从 NCBI GEO 下载官方 GSE42568 Series Matrix 和 GPL570 注释。矩阵包含 121 份样本，其中 17 份正常乳腺组织按 tissue 字段排除，104 份原发乳腺癌均有有效 OS 时间和事件。全随访死亡 35 例，3 年内死亡 20 例，3 年前删失 6 例。年龄、肿瘤直径、分级、淋巴结状态均无缺失，ER 状态缺失 3 例。肿瘤直径从 cm 转为 mm；缺失 ER 保留给源训练的预处理器。</p>
    <p>原研究病例来自爱尔兰 St Vincent 医院 1993–1997 年。SCAN-B 来自瑞典，METABRIC 的主要采集来源为英国和加拿大；原始研究资料支持不同研究来源，未发现共同招募中心。不同编号空间的样本 ID 字符串交集为 0；缺乏统一患者身份跨库核验，不能将 ID 不重叠表述为已经查实患者生物身份无重叠。<a href="cohort_provenance.json">来源核查依据</a></p>
    <h2>分子输入与平台迁移</h2><p>GPL570 的 54,675 个探针中，42,904 个只对应唯一 Entrez 基因。按 Entrez ID 与源基因集连接后，36,006 个探针覆盖原 18,150 个基因中的 16,317 个。每个基因取其全部合格探针的 log2 GC-RMA 信号均值，不使用生存结局、方差或效果挑选探针。对这 16,317 个基因在每个患者内重新取平均秩并除以 16,317；SCAN-B、METABRIC 和技术重复样本全部使用相同基因集合及顺序。</p>
    <p>1,833 个未测源基因没有被补成零或中间秩。原主模型选中的 30 个基因全部可测，但由于秩参照集合发生变化，本次仍从原 2,618 个 SCAN-B 开发患者重新拟合固定配置。这里验证的是平台适配后的算法，不是原 18,150 基因模型文件直接部署的性能。50 个 Hallmark 程序均可计算，最小程序有 31 个可测基因，最低覆盖率为 87%。</p>
    <h2>参数冻结与结果</h2><p>临床 RSF、融合 RSF、四个分子表示家族及两个 Hallmark XGBoost 参数在本队列效果计算前冻结。技术重复约束仅使用 110 对开发患者的重复样本。目标结局未进入筛基因、选参、缩放、探针聚合或校准。源选择中最佳交互家族也完整报告，避免只展示外部表现较好的候选。下表两原队列是相同 16,317 基因集合的桥接结果。</p>'''
    html += '<div class="table">' + table.to_html(float_format=lambda x: f'{x:.4f}', border=0) + '</div>'
    html += '<p><img src="locked_transport/GSE42568_auc.png" alt="GSE42568 三年 OS AUC及2000次患者配对bootstrap区间"></p>'
    html += '<h2>相对临床基线的配对差值</h2><p>2,000 次患者重抽样，每次所有模型使用同一批患者，并重新估计该重抽样的删失分布。区间为百分位区间，未进行多重比较调整。CA-RRL 相对临床基线的点估计增量为 0.0468；样本量和 3 年事件数限制了差值精度。Hallmark 相对 CA-RRL 的增量为 −0.0020，95% 区间为 −0.0565 至 0.0507。</p>'
    html += '<div class="table">' + ci_show.to_html(index=False, float_format=lambda x: f'{x:.4f}', border=0) + '</div>'
    html += '''<h2>备用队列与数据记录</h2><p>同时取得 GSE20685 官方矩阵头部的 327 例元数据：83 个全程死亡、31 个 3 年内死亡。该队列在 GEO 未提供连续肿瘤直径、组织学分级或 ER 临床状态，T 分期没有被伪装成肿瘤直径，分子亚型没有被当作临床 ER。其表达下载因连接中断而不完整，已以 .partial.gz 明示保存，不参与本轮模型验证。完整的 GSE42568 与 GPL570 原始文件均通过 gzip 完整性读取并记录 SHA-256。</p>
    <p>其他检索候选包括 GSE7390（GPL96，基因覆盖较少）和 GSE199633（637 例、GPL15048 定制芯片，需另行处理注释与字段覆盖），未因预期 AUC 高低挑选本次队列。</p>
    <h2>核查与复现</h2><p>训练病例与原留出/新外部队列标识无交集，四个残差优化均收敛，所有预测在 0–1 范围且随时间不下降，模型读取重放差小于 3×10⁻¹⁷。外部预测前已保存参数、病例名单、基因集合和模型哈希。原始数据文件保留，清洗表与对齐表达矩阵分别保存。</p>
    <p><a href="external_preparation_protocol.json">准备方案与来源</a> · <a href="preparation_audit.json">数据审计</a> · <a href="locked_transport/external_model_lock.json">主模型冻结记录</a> · <a href="locked_transport/modern_external_model_lock.json">XGBoost 冻结记录</a> · <a href="locked_transport/verification.json">运行核查</a> · <a href="locked_transport/metrics_all_horizons.csv">所有时间点指标</a> · <a href="locked_transport/GSE42568_auc_uncertainty.csv">配对区间</a></p>
    <p>官方来源：<a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE42568">GSE42568</a>；<a href="https://doi.org/10.1093/carcin/bgt208">Clarke 等原始研究</a>；<a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL570">GPL570</a>；<a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE20685">GSE20685</a>；<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC3094326/">Kao 等原始研究</a>；<a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE199633">GSE199633</a>。</p></main></html>'''
    (RUN / '独立队列准备与验证.html').write_text(html, encoding='utf-8')
    dump(RUN / 'external_artifact_manifest.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': digest(Path(__file__)),
        'dependency_script_sha256': {name: digest(ROOT / 'scripts' / name) for name in
            ['program_ranking_v5.py', 'modern_program_v5.py', 'modern_baseline_v5.py', 'survival_experiments.py']},
        'files': {str(p.relative_to(RUN)): {'bytes': p.stat().st_size, 'sha256': digest(p)}
        for p in RUN.rglob('*') if p.is_file() and p.name != 'external_artifact_manifest.json'}})
    print('External cohort report and artifact manifest saved.', flush=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'transport', 'modern', 'evaluate', 'report'], default='prepare')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'transport':
        transport_experiment(evaluate_external=args.evaluate)
    elif args.stage == 'modern':
        fit_modern_transport()
    elif args.stage == 'evaluate':
        evaluate_saved_models()
    else:
        build_report()
