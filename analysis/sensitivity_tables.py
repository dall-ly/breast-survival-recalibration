from pathlib import Path
import json, hashlib, math
import numpy as np
import pandas as pd

W = Path(__file__).resolve().parent
OUT = W.parent / 'results'
OUT.mkdir(parents=True, exist_ok=True)
ROOT = W.parent
ORIG = ROOT
PILOT = ROOT / 'sensitivity/inputs/archived_results/primary_results.csv'
SRC = ORIG / 'sensitivity/output/split_results.csv'
COHORTS = ['SCANB_holdout', 'METABRIC_external', 'GSE42568_external']
LABELS = dict(zip(COHORTS, ['SCAN-B', 'METABRIC', 'GSE42568']))
MAIN = dict(zip(COHORTS, [100, 100, 50]))
MODELS = ['Clinical', 'Gene', 'Pathway']
METHODS = ['unchanged','intercept_ML','shrink_k2p5','shrink_intercept','shrink_k10','intercept_slope_ML','intercept_slope_nonnegative_ML']
KEY = ['evaluation','cohort','budget','model']
METRICS = ['mean_brier','worsening_fraction','q95_delta','upper_decile_mean_delta']
raw = pd.read_csv(SRC)
wide = raw.pivot(index=KEY+['repeat'], columns='method', values='brier').reset_index()
avg = wide.groupby(['evaluation','cohort','budget','repeat'])[METHODS].mean().reset_index()
avg['model'] = 'Average'
wide = pd.concat([wide, avg], ignore_index=True)

def summaries(data, label):
    result = []
    for key, g in data.groupby(KEY, sort=False):
        assert len(g) in (300,280)
        baseline = g['unchanged'].to_numpy()
        for method in METHODS:
            values = g[method].to_numpy()
            delta = values-baseline
            count = math.ceil(len(delta)*0.1)
            result.append(dict(zip(KEY,key)) | {'sample':label,'method':method,
                'n_allocations':len(g),'mean_brier':values.mean(),'mean_delta':delta.mean(),
                'worsening_fraction':(delta>0).mean(),'q95_delta':np.quantile(delta,.95),
                'upper_decile_mean_delta':np.sort(delta)[-count:].mean()})
    return pd.DataFrame(result)

summ = pd.concat([summaries(wide,'all300'),summaries(wide[wide['repeat']>=20],'postpilot280')],ignore_index=True)
grid = summ[summ.model.isin(MODELS)]
ordinary = grid[grid.method.eq('intercept_ML')].drop(columns='method')
paired = grid[grid.method.isin(['shrink_k2p5','shrink_intercept','shrink_k10'])].merge(ordinary,on=KEY+['sample'],suffixes=('_penalty','_ordinary'),validate='many_to_one')
for metric in METRICS:
    paired[metric+'_difference'] = paired[metric+'_penalty']-paired[metric+'_ordinary']
    paired[metric+'_lower'] = paired[metric+'_difference']<0
paired.to_csv(OUT/'all_grid_kappa_comparisons.csv',index=False)

def counts(groupcols):
    result=[]
    for key,g in paired.groupby(groupcols,sort=False):
        if not isinstance(key,tuple): key=(key,)
        r=dict(zip(groupcols,key))|{'settings':len(g),'budgets':','.join(str(x) for x in sorted(g.budget.unique()))}
        for metric in METRICS:
            r[metric+'_lower_count']=int(g[metric+'_lower'].sum())
        result.append(r)
    return pd.DataFrame(result)

cs=counts(['sample','evaluation','method','cohort'])
cp=counts(['sample','evaluation','method','cohort','model'])
cs.to_csv(OUT/'grid_counts_by_cohort.csv',index=False)
cp.to_csv(OUT/'grid_counts_by_predictor.csv',index=False)
main=summ[(summ.model=='Average') & (summ.budget==summ.cohort.map(MAIN))].copy()
main.to_csv(OUT/'sensitivity_main_brier.csv',index=False)

def get(co,method,evaluation='remaining',sample='all300',metric='mean_brier'):
    g=main[(main.cohort==co)&(main.method==method)&(main.evaluation==evaluation)&(main['sample']==sample)]
    assert len(g)==1
    return float(g.iloc[0][metric])

panel_a=[]; panel_b=[]
for co in COHORTS:
    c=cs[(cs.cohort==co)&(cs.method=='shrink_intercept')&(cs.evaluation=='remaining')&(cs['sample']=='all300')].iloc[0]
    panel_a.append({'Cohort':LABELS[co],'Settings':int(c.settings),'Lower mean Brier':int(c.mean_brier_lower_count),
        'Lower worsening fraction':int(c.worsening_fraction_lower_count),'Lower upper-decile mean':int(c.upper_decile_mean_delta_lower_count)})
    panel_b.append({'Cohort':LABELS[co],'Main calibration n':MAIN[co],
        'Mean Brier kappa 2.5':get(co,'shrink_k2p5'),'Mean Brier kappa 10':get(co,'shrink_k10'),
        'Mean Brier nonnegative slope':get(co,'intercept_slope_nonnegative_ML'),
        'Postpilot280 kappa5-minus-ordinary mean Brier x1000':1000*(get(co,'shrink_intercept',sample='postpilot280')-get(co,'intercept_ML',sample='postpilot280'))})
pd.DataFrame(panel_a).to_csv(OUT/'proposed_table_panel_a.csv',index=False)
pd.DataFrame(panel_b).to_csv(OUT/'proposed_table_panel_b.csv',index=False)

diagnostics=[]
for co in COHORTS:
    diagnostics.append({'cohort':co,'main_budget':MAIN[co],
        'all300_remaining_k5_minus_ordinary':get(co,'shrink_intercept')-get(co,'intercept_ML'),
        'postpilot280_remaining_k5_minus_ordinary':get(co,'shrink_intercept',sample='postpilot280')-get(co,'intercept_ML',sample='postpilot280'),
        'all300_common_k5_minus_ordinary':get(co,'shrink_intercept','common')-get(co,'intercept_ML','common'),
        'postpilot280_common_k5_minus_ordinary':get(co,'shrink_intercept','common','postpilot280')-get(co,'intercept_ML','common','postpilot280'),
        'nonnegative_minus_unconstrained_slope':get(co,'intercept_slope_nonnegative_ML')-get(co,'intercept_slope_ML'),
        'nonnegative_minus_ordinary':get(co,'intercept_slope_nonnegative_ML')-get(co,'intercept_ML')})
pd.DataFrame(diagnostics).to_csv(OUT/'main_paired_sensitivity_differences.csv',index=False)

pilot = pd.read_csv(PILOT)
pilot = pilot[pilot['repeat'] < 20]
pw=pilot.pivot(index=['cohort','budget','model','repeat'],columns='method',values='brier').reset_index()
prows=[]
checks=[('METABRIC_external',100,'unchanged','<=',-.001,1),('SCANB_holdout',100,'unchanged','<=',.001,2),
        ('METABRIC_external',100,'intercept_ML','<',0.,3),('SCANB_holdout',100,'intercept_ML','<',0.,3),('GSE42568_external',50,'intercept_ML','<',0.,3)]
for co,budget,ref,op,threshold,condition in checks:
    g=pw[(pw.cohort==co)&(pw.budget==budget)]
    value=float((g.shrink_intercept-g[ref]).mean())
    prows.append({'condition':condition,'cohort':co,'calibration_n':budget,'comparison':'kappa5 minus '+ref,'operator':op,'threshold':threshold,
        'observed_mean_brier_difference':value,'passes_setting':bool(value<=threshold if op=='<=' else value<threshold),
        'condition_combination':'all required' if condition<3 else 'at least 2 of these 3 settings'})
pd.DataFrame(prows).to_csv(OUT/'pilot_continuation_conditions.csv',index=False)
gate=json.loads((ORIG/'benchmark/pilot_gate.json').read_text(encoding='utf-8'))
expected=[gate['MET100_vs_unchanged'],gate['SCAN100_vs_unchanged']]+[gate['vs_intercept_ML'][co] for co in ['METABRIC_external','SCANB_holdout','GSE42568_external']]
pilot_error=max(abs(p['observed_mean_brier_difference']-v) for p,v in zip(prows,expected))
assert pilot_error<1e-14

# Reconcile source-cache recomputation with independently prepared earlier summaries.
cached=pd.read_csv(ROOT/'results/all_fixed_paired_summaries.csv')
cached=cached[cached.reference=='unchanged']
cmp=summ[summ['sample']=='all300'].merge(cached,on=KEY+['method'],suffixes=('_new','_old'),validate='one_to_one')
errors={}
for newcol,oldcol in [('mean_brier','mean_brier'),('mean_delta','mean_delta'),('worsening_fraction','harm_fraction'),('q95_delta','q95_delta'),('upper_decile_mean_delta','worst10_mean_delta')]:
    a=newcol+'_new' if newcol in cached.columns else newcol
    b=oldcol+'_old' if oldcol in summ.columns else oldcol
    errors[newcol]=float((cmp[a]-cmp[b]).abs().max())
assert max(errors.values())<1e-14


print('Verified sensitivity tables and the original pilot continuation conditions.')
