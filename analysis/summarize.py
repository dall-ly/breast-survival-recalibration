"""Read-only reconstruction of paired allocation evidence; no model fitting.

All allocation percentiles are empirical distributions, not patient confidence
intervals. 'Average' averages losses for three separately evaluated predictors;
it does not average their prediction probabilities or define an ensemble.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd

SOURCE=Path(__file__).resolve().parents[1]
OUT=SOURCE/'results'
OUT.mkdir(parents=True,exist_ok=True)
MAIN={'SCANB_holdout':100,'METABRIC_external':100,'GSE42568_external':50}
COHORTS=list(MAIN)
MODELS=['Clinical','Gene','Pathway']
PRIMARY=['unchanged','intercept_ML','shrink_intercept','intercept_slope_ML']
SELECTORS=['OOB_min','OOB_simple']
CANDIDATES=['unchanged','shrink_intercept','intercept_ML']
LABEL={'unchanged':'None','intercept_ML':'Ordinary','shrink_intercept':'Penalized (kappa=5)',
       'intercept_slope_ML':'Intercept and slope','OOB_min':'OOB minimum','OOB_simple':'OOB preference'}
paths={'fixed_results':SOURCE/'sensitivity/output/split_results.csv',
       'selection_results':SOURCE/'selection/selection_results/evaluation_results.csv',
       'selection_decisions':SOURCE/'selection/selection_results/selection_decisions.csv'}
hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()}
fixed=pd.read_csv(paths['fixed_results'],float_precision='round_trip')
oob=pd.read_csv(paths['selection_results'],float_precision='round_trip')
decisions=pd.read_csv(paths['selection_decisions'],float_precision='round_trip')
KEY=['evaluation','cohort','repeat','budget','model','method']
assert not fixed.duplicated(KEY).any() and len(fixed)==126000
assert not oob.duplicated(KEY).any() and len(oob)==27000
assert not decisions.duplicated(['cohort','repeat','budget','model']).any() and len(decisions)==2700
assert set(fixed.model)==set(MODELS) and set(oob.model)==set(MODELS)
assert np.isfinite(fixed.brier).all() and np.isfinite(oob.brier).all()

def add_average(frame):
    keys=['evaluation','cohort','repeat','budget','method']
    assert frame.groupby(keys).model.nunique().eq(3).all()
    avg=frame.groupby(keys,as_index=False).agg(brier=('brier','mean'),n_test=('n_test','first'))
    avg['model']='Average'
    return pd.concat([frame[KEY+['n_test','brier']],avg],ignore_index=True)

def paired_summary(frame, pairs):
    index=['evaluation','cohort','repeat','budget','model']
    wide=frame.pivot(index=index,columns='method',values='brier')
    rows=[]
    for method,reference in pairs:
        chosen=wide[[method,reference]].copy() if method!=reference else wide[[method]].copy()
        delta=wide[method]-wide[reference]
        assert not delta.isna().any()
        vals=pd.DataFrame({'brier':wide[method],'reference_brier':wide[reference],'delta':delta}).reset_index()
        for keys,g in vals.groupby(['evaluation','cohort','budget','model'],sort=False):
            x=g.delta.to_numpy()
            count=len(x)
            assert count in [280,300]
            k=int(math.ceil(.10*count))
            q05,q50,q90,q95=np.quantile(x,[.05,.5,.9,.95],method='linear')
            rows.append(dict(zip(['evaluation','cohort','budget','model'],keys))|{
                'method':method,'reference':reference,'n_allocations':count,
                'mean_brier':float(g.brier.mean()),'mean_reference_brier':float(g.reference_brier.mean()),
                'mean_delta':float(x.mean()),'q05_delta':float(q05),'median_delta':float(q50),
                'q90_delta':float(q90),'q95_delta':float(q95),'harm_fraction':float(np.mean(x>0)),
                'tie_fraction':float(np.mean(x==0)),'improved_fraction':float(np.mean(x<0)),
                'worst10_count':k,'worst10_mean_delta':float(np.sort(x)[-k:].mean())})
    result=pd.DataFrame(rows)
    for col in ['mean_delta','q05_delta','median_delta','q90_delta','q95_delta','worst10_mean_delta']:
        result[col+'_x1e3']=1000*result[col]
    result['harm_percent']=100*result.harm_fraction
    return result

fixed_aug=add_average(fixed)
oob_aug=add_average(oob)
methods=list(fixed.method.unique())
fixed_pairs=[(m,'unchanged') for m in methods]+[('shrink_intercept','intercept_ML'),('intercept_slope_ML','intercept_ML'),('intercept_slope_nonnegative_ML','intercept_slope_ML')]
oob_pairs=[(m,'unchanged') for m in list(oob.method.unique())]+[(s,r) for s in SELECTORS for r in ['shrink_intercept','intercept_ML']]
full=paired_summary(fixed_aug,fixed_pairs)
full.to_csv(OUT/'all_fixed_paired_summaries.csv',index=False)
oob_full=paired_summary(oob_aug,oob_pairs)
oob_full.to_csv(OUT/'all_oob_paired_summaries.csv',index=False)

def main_rows(frame):
    return frame[frame.budget.eq(frame.cohort.map(MAIN))].copy()

def display_columns(frame):
    frame=frame.copy()
    frame['mean_brier_display']=frame.mean_brier.map(lambda x:f'{x:.5f}')
    frame['mean_delta_display']=frame.mean_delta_x1e3.map(lambda x:f'{x:+.3f}' if x!=0 else '0.000')
    frame['allocation_interval_display']=[f'[{a:+.3f}, {b:+.3f}]' for a,b in zip(frame.q05_delta_x1e3,frame.q95_delta_x1e3)]
    frame['harm_percent_display']=frame.harm_percent.map(lambda x:f'{x:.1f}')
    frame['worst10_mean_display']=frame.worst10_mean_delta_x1e3.map(lambda x:f'{x:+.3f}' if x!=0 else '0.000')
    frame['method_label']=frame.method.map(LABEL)
    return frame

table2=display_columns(main_rows(full[(full.reference=='unchanged') & full.method.isin(PRIMARY)]))
table2.to_csv(OUT/'table2_main_budgets_all_predictors_both_designs.csv',index=False)
table2_avg=table2[(table2.model=='Average')&(table2.evaluation=='remaining')].copy()
table2_avg.to_csv(OUT/'table2_main_remaining_average.csv',index=False)
table2_pred=table2[(table2.model!='Average')&(table2.evaluation=='remaining')].copy()
table2_pred.to_csv(OUT/'table2_main_remaining_per_predictor.csv',index=False)

# A common-evaluation figure can plot the Average rows across every budget.
figure_data=full[(full.evaluation=='common')&(full.reference=='unchanged')&full.method.isin(PRIMARY)&(full.model=='Average')].copy()
figure_data.to_csv(OUT/'figure_common_budget_average.csv',index=False)
full[(full.evaluation=='common')&(full.reference=='unchanged')&full.method.isin(PRIMARY)&(full.model!='Average')].to_csv(OUT/'figure_common_budget_per_predictor.csv',index=False)
figure_alloc=fixed_aug[(fixed_aug.evaluation=='common')&(fixed_aug.model=='Average')&fixed_aug.method.isin(PRIMARY)].copy()
figure_base=figure_alloc[figure_alloc.method=='unchanged'][['evaluation','cohort','repeat','budget','model','brier']].rename(columns={'brier':'unchanged_brier'})
figure_alloc=figure_alloc.merge(figure_base,on=['evaluation','cohort','repeat','budget','model'],validate='many_to_one')
figure_alloc['delta']=figure_alloc.brier-figure_alloc.unchanged_brier
figure_alloc['delta_x1e3']=1000*figure_alloc.delta
figure_alloc['harm']=figure_alloc.delta>0
figure_alloc.to_csv(OUT/'figure_common_budget_average_paired_allocations.csv',index=False)

# Method-selection frequencies use predictor-specific decisions, not averaged
# probabilities. Aggregate rows pool 900 decisions (300 x three predictors).
freq_rows=[]
for selector in SELECTORS:
    for cohort in COHORTS:
        for model in MODELS+['Average']:
            g=decisions[decisions.cohort.eq(cohort)]
            if model!='Average':g=g[g.model.eq(model)]
            rec={'cohort':cohort,'budget':MAIN[cohort],'model':model,'method':selector,'n_decisions':len(g)}
            for candidate in CANDIDATES:
                rec[candidate+'_count']=int(g[selector].eq(candidate).sum())
                rec[candidate+'_percent']=100*float(g[selector].eq(candidate).mean())
            assert sum(rec[c+'_count'] for c in CANDIDATES)==len(g)
            freq_rows.append(rec)
freq=pd.DataFrame(freq_rows)
freq.to_csv(OUT/'selection_frequencies_complete.csv',index=False)
table3=display_columns(oob_full[(oob_full.reference=='unchanged')&oob_full.method.isin(SELECTORS)]).merge(freq,on=['cohort','budget','model','method'],validate='many_to_one')
table3.to_csv(OUT/'table3_selectors_complete.csv',index=False)
table3_avg=table3[(table3.model=='Average')&(table3.evaluation=='remaining')].copy()
table3_avg.to_csv(OUT/'table3_main_remaining_average.csv',index=False)

after280=paired_summary(fixed_aug[fixed_aug.repeat>=20],fixed_pairs)
after280.to_csv(OUT/'fixed_after_first20_paired_summaries.csv',index=False)
after280_main=display_columns(main_rows(after280[(after280.reference=='unchanged')&after280.method.isin(PRIMARY)]))
after280_main.to_csv(OUT/'pilot_exclusion_main_budgets.csv',index=False)
comparison=full.merge(after280,on=['evaluation','cohort','budget','model','method','reference'],suffixes=('_300','_280'),validate='one_to_one')
comparison.to_csv(OUT/'pilot_exclusion_full_comparison.csv',index=False)
pilot_summary=[]
for design in ['remaining','common']:
    selected=comparison[(comparison.evaluation==design)&(comparison.model!='Average')&
                        (comparison.reference=='unchanged')&comparison.method.isin(['intercept_ML','shrink_intercept'])]
    a=after280[(after280.evaluation==design)&(after280.reference=='unchanged')&(after280.model!='Average')]
    idx=['cohort','budget','model']
    wt=a.pivot(index=idx,columns='method',values='worst10_mean_delta')
    hf=a.pivot(index=idx,columns='method',values='harm_fraction')
    pilot_summary.append({'evaluation':design,'n_ordinary_or_penalized_vs_unchanged_cells':len(selected),
        'mean_delta_sign_changes_300_to_280':int((selected.mean_delta_300*selected.mean_delta_280<0).sum()),
        'n_predictor_budget_cells':len(wt),
        'k5_lower_worst10_than_ordinary_after280':int((wt.shrink_intercept<wt.intercept_ML).sum()),
        'k5_lower_harm_than_ordinary_after280':int((hf.shrink_intercept<hf.intercept_ML).sum())})
pd.DataFrame(pilot_summary).to_csv(OUT/'pilot_exclusion_diagnostic_summary.csv',index=False)

# Each design has 30 predictor x budget cells (3 x (4+4+2)).
comparison_keys=['evaluation','cohort','budget','model']
vsnone=full[full.reference=='unchanged']
shrink=vsnone[vsnone.method=='shrink_intercept'].set_index(comparison_keys)
ordinary=vsnone[vsnone.method=='intercept_ML'].set_index(comparison_keys)
tail=shrink[['mean_delta','q95_delta','harm_fraction','worst10_mean_delta']].join(ordinary[['mean_delta','q95_delta','harm_fraction','worst10_mean_delta']],lsuffix='_k5',rsuffix='_ordinary').reset_index()
for metric in ['mean_delta','q95_delta','harm_fraction','worst10_mean_delta']:
    tail[metric+'_k5_minus_ordinary']=tail[metric+'_k5']-tail[metric+'_ordinary']
tail.to_csv(OUT/'shrinkage_tail_comparison_all_cells.csv',index=False)
tail_counts=[]
for evaluation,g in tail[tail.model!='Average'].groupby('evaluation'):
    assert len(g)==30
    rec={'evaluation':evaluation,'n_predictor_budget_cells':len(g)}
    for metric in ['mean_delta','q95_delta','harm_fraction','worst10_mean_delta']:
        delta=g[metric+'_k5_minus_ordinary']
        rec[metric+'_lower_count']=int((delta < -1e-12).sum())
        rec[metric+'_higher_count']=int((delta > 1e-12).sum())
        rec[metric+'_tie_count']=int((abs(delta)<=1e-12).sum())
    tail_counts.append(rec)
tail_counts=pd.DataFrame(tail_counts)
tail_counts.to_csv(OUT/'shrinkage_tail_comparison_counts.csv',index=False)
tail_exceptions=tail[(tail.model!='Average')&(tail.worst10_mean_delta_k5_minus_ordinary>=-1e-12)]
tail_exceptions.to_csv(OUT/'shrinkage_tail_exceptions.csv',index=False)

# Independent, standard-library recomputation of the table II allocation groups.
# Use explicit sums, order statistics with linear interpolation, and strict signs.
independent_errors=[]
for r in table2_avg.to_dict('records'):
    g=fixed[(fixed.evaluation==r['evaluation'])&(fixed.cohort==r['cohort'])&(fixed.budget==r['budget'])]
    rows=g.to_dict('records')
    buckets={}
    for x in rows:buckets.setdefault((x['repeat'],x['method']),[]).append(x['brier'])
    scores=[sum(buckets[(i,r['method'])])/3 for i in range(300)]
    base=[sum(buckets[(i,'unchanged')])/3 for i in range(300)]
    ds=sorted(a-b for a,b in zip(scores,base))
    def quantile(q):
        z=(len(ds)-1)*q;lo=math.floor(z);hi=math.ceil(z)
        return ds[lo]*(hi-z)+ds[hi]*(z-lo) if lo!=hi else ds[lo]
    expected={'mean_brier':sum(scores)/300,'mean_delta':sum(ds)/300,
              'q05_delta':quantile(.05),'q95_delta':quantile(.95),
              'harm_fraction':sum(x>0 for x in ds)/300,'worst10_mean_delta':sum(ds[-30:])/30}
    independent_errors.extend(abs(expected[k]-r[k]) for k in expected)
assert max(independent_errors)<1e-14

# Selection decisions independently checked from archived score summaries.
decision_errors=0
for row in decisions.to_dict('records'):
    best=min(range(3),key=lambda j:(row[f'median_{j}'],j))
    choice=best
    for j in range(best):
        if row[f'lo_{j}_{best}']<=0<=row[f'hi_{j}_{best}']:choice=j;break
    decision_errors+=int(row['OOB_min']!=CANDIDATES[best] or row['OOB_simple']!=CANDIDATES[choice])
assert decision_errors==0
fixed_main=fixed[(fixed.budget==fixed.cohort.map(MAIN))&fixed.method.isin(CANDIDATES)]
replay=fixed_main.merge(oob[oob.method.isin(CANDIDATES)],on=KEY,suffixes=('_fixed','_oob'),validate='one_to_one')
assert len(replay)==16200
max_fixed_replay=float(abs(replay.brier_fixed-replay.brier_oob).max())
assert max_fixed_replay<1e-14

definitions={
    'delta':'Brier(method) minus Brier(reference) within the same cohort, predictor, budget and allocation; positive is worse.',
    'harm':'Proportion of 300 allocations with signed paired delta > 0; exact ties are retained and are not harmful. This is probability-prediction error, not a clinical-harm endpoint.',
    'interval':'Empirical 5th and 95th percentiles of 300 paired allocation differences, using linear interpolation; an allocation distribution, not a patient/population confidence interval.',
    'worst10':'Mean of the largest 30 signed paired allocation differences among 300 allocations. Negative differences remain negative; no truncation at zero. After excluding the first 20, average the largest 28 of 280.',
    'average':'Compute each predictor-specific Brier score separately; average the three scores equally within each allocation before forming allocation summaries. This is not an ensemble prediction model.',
    'units':'Mean Brier is unscaled. Delta means, allocation interval and worst-decile mean are displayed in units of 10^-3 (raw delta multiplied by 1000). Harm and selection frequencies are percentages.',
    'remaining':'Primary evaluation uses all patients outside the calibration sample: main n=555,1878,54.',
    'common':'Sensitivity/figure uses patients after each cohort largest calibration prefix: n=455,1778,54; held fixed across budgets within a repeat, varies between repeats.',
    'table3':'Method-selection percentages pool 900 predictor-specific decisions per cohort (300 allocations x 3 predictors). They do not describe one ensemble-level decision.',
    'pilot':'Exclusion of the original 20 allocations is a numerical diagnostic. The remaining 280 allocations reuse the same patients and do not remove historical/pilot-based study selection.'}
payload={'definitions':definitions,'table2_main_remaining_average':table2_avg.to_dict('records'),
         'table2_main_remaining_per_predictor':table2_pred.to_dict('records'),
         'table3_main_remaining_average':table3_avg.to_dict('records'),
         'pilot_exclusion_main_remaining_average':after280_main[(after280_main.model=='Average')&(after280_main.evaluation=='remaining')].to_dict('records'),
         'pilot_exclusion_diagnostic_summary':pilot_summary,
         'shrinkage_tail_counts':tail_counts.to_dict('records'),'shrinkage_tail_exceptions':tail_exceptions.to_dict('records')}
(OUT/'table_payload.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')

report={'input_hashes':hashes,'input_rows':{k:len(v) for k,v in [('fixed',fixed),('oob',oob),('decisions',decisions)]},
        'table2_independent_formula_max_error':max(independent_errors),'selection_decision_disagreements':decision_errors,
        'fixed_oob_replay_max_error':max_fixed_replay,'all_input_hashes_unchanged':all(hashlib.sha256(paths[k].read_bytes()).hexdigest()==v for k,v in hashes.items()),
        'no_fitting':True,'no_source_writes':True}
(OUT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
lines=['READ-ONLY PAIRED ALLOCATION AUDIT','',*list(definitions.values()),'','MAIN TABLE II (remaining; Average):',
       table2_avg[['cohort','method','mean_brier_display','mean_delta_display','allocation_interval_display','harm_percent_display','worst10_mean_display']].to_string(index=False),
       '','MAIN TABLE III:',table3_avg[['cohort','method','mean_brier_display','unchanged_percent','shrink_intercept_percent','intercept_ML_percent']].to_string(index=False),
       '','TAIL COUNTS:',tail_counts.to_string(index=False),'','TAIL EXCEPTIONS:',tail_exceptions.to_string(index=False),
       '','PILOT EXCLUSION DIAGNOSTIC:',json.dumps(pilot_summary,indent=2),
       '','VERIFICATION:',json.dumps(report,indent=2)]
(OUT/'numeric_summary.txt').write_text('\n'.join(lines),encoding='utf-8')
print('Verified paired summaries, 30-setting comparisons, pilot exclusion and OOB choices.')
