"""Bounded review-stage sensitivity: common evaluation patients and three fixed penalties."""
from pathlib import Path
import sys, json, hashlib, time, shutil
import numpy as np
import pandas as pd
from scipy.optimize import brentq

B=Path(__file__).resolve().parent
SOURCE=B/'inputs'
OUT=B/'output';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(SOURCE))
from calibration import weights, objective

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    started=time.time()
    proto=json.loads((B/'protocol.json').read_text(encoding='utf-8'))
    files=[B/'protocol.json',SOURCE/'fixed_predictions.csv',SOURCE/'calibration.py',SOURCE/'archived_results/primary_results.csv',SOURCE/'archived_results/sensitivity_results.csv']
    hashes={str(p):sha(p) for p in files}
    data=pd.read_csv(files[1],float_precision='round_trip')
    old=pd.concat([pd.read_csv(files[3],float_precision='round_trip'),pd.read_csv(files[4],float_precision='round_trip')],ignore_index=True)
    keys=['cohort','repeat','budget','model','method']
    old=old.set_index(keys,verify_integrity=True)
    records=[];fits=[];members=[];max_replay=0.;max_kappa5=0.;max_score=0.
    for ci,co in enumerate(proto['cohorts']):
        d=data[data.cohort==co].sort_values('patient_id').reset_index(drop=True)
        t=d.os_days.to_numpy();e=d.os_event.to_numpy();ids=d.patient_id.to_numpy();budgets=proto['budgets'][co]
        for repeat in range(proto['repetitions']):
            order=np.random.default_rng(proto['seed']+100000*ci+repeat).permutation(len(d))
            common=order[max(budgets):]
            common_y,common_w=weights(t[common],e[common])
            members.append({'cohort':co,'repeat':repeat,'n_test':len(common),'id_hash':hashlib.sha256('\n'.join(ids[common]).encode()).hexdigest()})
            for budget in budgets:
                train=order[:budget];remaining=order[budget:]
                assert not np.intersect1d(train,common).size
                y,w=weights(t[train],e[train]);active=w>0;y,w=y[active],w[active]
                ry,rw=weights(t[remaining],e[remaining])
                for model in proto['models']:
                    p=d['p_'+model].to_numpy();calx=np.log(-np.log1p(-np.clip(p[train][active],1e-10,1-1e-10)))
                    pars={m:(float(old.loc[(co,repeat,budget,model,m),'intercept']),float(old.loc[(co,repeat,budget,model,m),'slope'])) for m in proto['existing_methods']}
                    for kappa in proto['penalty_strengths']:
                        def score(a):return objective([a,1.],calx,y,w)[1][0]*w.sum()+kappa*a
                        a=float(brentq(score,-30.,10.,xtol=1e-11));grad=abs(score(a));max_score=max(max_score,grad)
                        if kappa==5:max_kappa5=max(max_kappa5,abs(a-pars['shrink_intercept'][0]))
                        else:pars['shrink_k'+str(kappa).replace('.','p')]=(a,1.)
                        fits.append({'cohort':co,'repeat':repeat,'budget':budget,'model':model,'kappa':kappa,'intercept':a,'score_residual':grad,'events':int(y.sum()),'weighted_events':float(w@y)})
                    for design,ix,yy,ww in [('remaining',remaining,ry,rw),('common',common,common_y,common_w)]:
                        x=np.log(-np.log1p(-np.clip(p[ix],1e-10,1-1e-10)))
                        for method,(a,b) in pars.items():
                            with np.errstate(over='ignore',under='ignore'):
                                q=p[ix] if method=='unchanged' else -np.expm1(-np.exp(a+b*x))
                            assert np.isfinite(q).all() and np.isfinite(ww).all()
                            bs=float(ww@((yy-q)**2)/len(ix))
                            if design=='remaining' and method in proto['existing_methods']:
                                max_replay=max(max_replay,abs(bs-float(old.loc[(co,repeat,budget,model,method),'brier'])))
                            records.append({'cohort':co,'repeat':repeat,'budget':budget,'model':model,'method':method,'evaluation':design,'n_test':len(ix),'calibration_events':int(y.sum()),'brier':bs})
        print(co,'finished',flush=True)
    assert max_replay<1e-10,max_replay
    assert max_kappa5<1e-8,max_kappa5
    r=pd.DataFrame(records);f=pd.DataFrame(fits)
    assert len(r)==126000 and len(f)==27000
    r.to_csv(OUT/'split_results.csv',index=False);f.to_csv(OUT/'calibration_fits.csv',index=False)
    pd.DataFrame(members).to_csv(OUT/'common_evaluation_membership_hashes.csv',index=False)
    r.groupby(['evaluation','cohort','budget','model','method']).brier.mean().unstack().reset_index().to_csv(OUT/'model_means.csv',index=False)
    r.groupby(['evaluation','cohort','budget','method']).brier.mean().unstack().reset_index().to_csv(OUT/'aggregate_means.csv',index=False)
    comparisons=[];long=[]
    permodel=r.set_index(['evaluation','cohort','budget','model','repeat','method']).brier.unstack()
    aggregate=r.groupby(['evaluation','cohort','budget','repeat','method']).brier.mean().unstack().reset_index();aggregate['model']='Average'
    both=pd.concat([permodel.reset_index(),aggregate],ignore_index=True)
    for (ev,co,b,model),group in both.groupby(['evaluation','cohort','budget','model']):
        for method in ['intercept_ML','shrink_k2p5','shrink_intercept','shrink_k10']:
            for ref in ['unchanged','intercept_ML']:
                if method==ref:continue
                delta=group[method]-group[ref]
                comparisons.append({'evaluation':ev,'cohort':co,'budget':b,'model':model,'method':method,'reference':ref,'mean_delta':delta.mean(),'relative_reduction_pct':-100*delta.mean()/group[ref].mean(),'fraction_better':float((delta<0).mean()),'median':delta.median(),'q025':delta.quantile(.025),'q90':delta.quantile(.9),'q975':delta.quantile(.975),'upper_decile_mean':delta.sort_values().iloc[-30:].mean(),'mcse_fixed_cohort':delta.std(ddof=1)/np.sqrt(300)})
    c=pd.DataFrame(comparisons);c.to_csv(OUT/'paired_comparisons.csv',index=False)
    crossings=[]
    for (ev,co,model),g in c[(c.method=='intercept_ML')&(c.reference=='unchanged')].groupby(['evaluation','cohort','model']):
        good=g[g.mean_delta<0]
        crossings.append({'evaluation':ev,'cohort':co,'model':model,'first_tested_budget_with_lower_mean_error':int(good.budget.min()) if len(good) else None,'budgets_with_lower_mean_error':';'.join(str(int(x)) for x in good.sort_values('budget').budget)})
    pd.DataFrame(crossings).to_csv(OUT/'budget_crossings.csv',index=False)
    v=c[(c.model=='Average')&(c.method!='intercept_ML')&(((c.cohort!='GSE42568_external')&(c.budget==100))|((c.cohort=='GSE42568_external')&(c.budget==50)))]
    v.to_csv(OUT/'main_sensitivity_comparisons.csv',index=False)
    common_original=r[r.evaluation=='common'].pivot_table(index=['cohort','repeat','model'],columns=['budget','method'],values='brier')
    for co in proto['cohorts']:
        z=common_original.loc[co].xs('unchanged',axis=1,level='method').dropna(axis=1)
        assert np.max(np.ptp(z.to_numpy(),axis=1))<1e-15
    assert hashes=={str(p):sha(p) for p in files}
    report={'passed':True,'scope':'Post hoc fixed-grid sensitivity; no selection or base-model training','rows':len(r),'calibration_fits':len(f),'new_penalty_fits':18000,'kappa5_verification_fits':9000,'max_archived_brier_replay_error':max_replay,'max_kappa5_intercept_error':max_kappa5,'max_root_residual':max_score,'common_unchanged_identical_across_budgets':True,'input_hashes':hashes,'seconds':time.time()-started}
    (OUT/'numerical_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='input_hashes'},indent=2),flush=True)
    expected=pd.read_csv(B/'expected_brier.csv',float_precision='round_trip')
    actual=r[['evaluation','cohort','repeat','budget','model','method','brier']]
    merge=expected.merge(actual,on=['evaluation','cohort','repeat','budget','model','method'],suffixes=('_expected','_new'),validate='one_to_one')
    error=float(abs(merge.brier_expected-merge.brier_new).max())
    assert len(merge)==126000 and error<1e-10
    (OUT/'reproduction_check.json').write_text(json.dumps({'passed':True,'rows':len(merge),'max_brier_error':error}),encoding='utf-8')
    print('Portable reproduction passed:',len(merge),'evaluations; max discrepancy',error,flush=True)
if __name__=='__main__':main()
