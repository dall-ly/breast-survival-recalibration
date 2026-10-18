"""Reproduce all patient-budget allocations; no source-model training or tuning."""
from pathlib import Path
import argparse, json, hashlib, time
import numpy as np
import pandas as pd
from calibration import fit_repaired_calibrators, weights, km_risk

ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repeats',type=int,default=300)
    ap.add_argument('--output',type=Path,default=ROOT/'output')
    args=ap.parse_args()
    if not 1<=args.repeats<=300:raise ValueError('Use 1 to 300 prespecified repeats.')
    proto=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    data=pd.read_csv(ROOT/'fixed_predictions.csv',float_precision='round_trip')
    records=[];members=[];start=time.time()
    for ci,cohort in enumerate(proto['cohorts']):
        d=data[data.cohort==cohort].sort_values('patient_id').reset_index(drop=True)
        t=d.os_days.to_numpy();e=d.os_event.to_numpy();n=len(d)
        budgets=proto['gse_budget_patients'] if cohort.startswith('GSE') else proto['budget_patients']
        for rep in range(args.repeats):
            order=np.random.default_rng(proto['seed']+ci*100000+rep).permutation(n)
            for budget in budgets:
                train,test=order[:budget],order[budget:]
                assert not np.intersect1d(train,test).size
                members.extend([cohort,rep,budget,str(d.patient_id.iloc[i])] for i in train)
                yy,ww=weights(t[test],e[test])
                observed=km_risk(t[test],e[test],np.ones(len(test),int))
                for model in ['Clinical','Gene','Pathway']:
                    p=d['p_'+model].to_numpy()
                    pars=fit_repaired_calibrators(p[train],t[train],e[train])
                    x=np.log(-np.log1p(-np.clip(p[test],1e-10,1-1e-10)))
                    for method,(a,b,fallback,boundary) in pars.items():
                        with np.errstate(over='ignore',under='ignore'):
                            pred=p[test] if method=='unchanged' else -np.expm1(-np.exp(a+b*x))
                        assert np.isfinite(pred).all()
                        records.append(dict(cohort=cohort,repeat=rep,budget=budget,model=model,method=method,
                            n_test=len(test),calibration_events=int(((t[train]<=1095)&(e[train]==1)).sum()),
                            brier=float(ww@((yy-pred)**2)/len(test)),mean_risk=float(pred.mean()),
                            observed_risk=observed,abs_mean_error=abs(float(pred.mean())-observed),
                            intercept=a,slope=b,fallback=fallback,boundary=boundary))
        print(cohort,':',args.repeats,'repetitions completed',flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    r=pd.DataFrame(records);r.to_csv(args.output/'split_results.csv',index=False)
    pd.DataFrame(members,columns=['cohort','repeat','budget','patient_id']).to_csv(args.output/'calibration_memberships.csv',index=False)
    r.groupby(['cohort','budget','model','method']).brier.agg(['mean','std']).reset_index().to_csv(args.output/'summary.csv',index=False)
    expected=pd.read_csv(ROOT/'expected_split_brier.csv')
    expected=expected[expected['repeat']<args.repeats]
    joined=expected.merge(r,on=['cohort','repeat','budget','model','method'],suffixes=('_expected','_reproduced'),validate='one_to_one')
    assert len(joined)==len(r)==len(expected)
    joined['absolute_error']=abs(joined.brier_expected-joined.brier_reproduced)
    errors=joined.groupby('method').absolute_error.max().to_dict()
    # Different finite-MLE solvers agree to numerical tolerance; major methods to 1e-10.
    for method,error in errors.items():
        assert error<(1e-7 if method.startswith('intercept_slope') else 1e-10),(method,error)
    report={'passed':True,'repeats':args.repeats,'rows_compared':len(joined),'max_brier_error_by_method':errors,
            'seconds':time.time()-start,'fixed_predictions_sha256':hashlib.sha256((ROOT/'fixed_predictions.csv').read_bytes()).hexdigest(),
            'scope':'Conditional updating of supplied fixed scores; does not reproduce or independently validate base-model development.'}
    (args.output/'reproduction_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
