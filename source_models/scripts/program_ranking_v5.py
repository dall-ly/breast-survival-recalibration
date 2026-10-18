"""Clinical-offset survival learning with biological programs and IPCW ranking.

Selection uses only original SCAN-B development patients. All feature fitting,
screening, censoring weights and clinical cross-fitting are training-fold local.
Hallmark definitions are external biological knowledge, not outcome-derived.
"""
from survival_experiments import (load_data, sub, ROOT, SEED, outcome, strata_of,
    baseline_fit, cumulative_hazard, TIMES, evaluate, km_censor, labels, metric)
import argparse, hashlib, itertools, json, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

WORK = ROOT / 'runs/gse96058_os_v5_expansion/program_ranking'
WORK.mkdir(parents=True, exist_ok=True)
GMT = WORK.parent / 'resources/MSigDB_Hallmark_2020.gmt'
ANCHOR = {'genes': 0, 'leaf': 30, 'features': 'sqrt'}
REPRESENTATIONS = ['genes30', 'hallmark', 'hybrid', 'hallmark_context']

def save(name, obj):
    (WORK/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
        default=lambda x: x.tolist() if isinstance(x,np.ndarray) else x.item()), encoding='utf-8')

def gene_sets(symbols):
    lookup = {str(s): i for i,s in enumerate(symbols)}
    sets, coverage = {}, []
    for line in GMT.read_text(encoding='utf-8').splitlines():
        parts=line.split('\t'); name=parts[0]; genes=list(dict.fromkeys(parts[2:]))
        keep=[lookup[g] for g in genes if g in lookup]
        assert len(keep)>=10, (name,len(keep))
        sets[name]=np.array(keep,int)
        coverage.append({'program':name,'source_genes':len(genes),'observed_genes':len(keep),
                         'coverage':len(keep)/len(genes)})
    assert len(sets)==50
    pd.DataFrame(coverage).to_csv(WORK/'program_gene_coverage.csv',index=False)
    return sets

def raw_features(b, pre):
    rep=pre['representation']; features=[]; names=[]
    if rep in ('genes30','hybrid'):
        features.append(b['ranks'][:,pre['gene_indices']].astype(float))
        names.extend(pre['gene_names'])
    if rep!='genes30':
        programs=np.column_stack([np.mean(b['ranks'][:,idx],axis=1,dtype=float)
                                  for idx in pre['gene_sets'].values()])
        features.append(programs); names.extend(pre['gene_sets'])
        if rep=='hallmark_context':
            clinical=np.where(np.isfinite(b['clinical']),b['clinical'],pre['clinical_medians'])
            age=np.clip((clinical[:,0]-60)/15,-3,3); er=clinical[:,4]-.5
            centered=programs-pre['program_mean']
            features.extend([centered*age[:,None],centered*er[:,None]])
            names.extend([f'{n} x age' for n in pre['gene_sets']])
            names.extend([f'{n} x ER' for n in pre['gene_sets']])
    return np.column_stack(features),names

def prepare(a,b,repeats,sets,symbols):
    y=outcome(a); ht=np.empty(len(a['t'])); h3=np.empty(len(a['t']))
    for tr,va in StratifiedKFold(3,shuffle=True,random_state=SEED+13).split(ht,strata_of(a)):
        model=baseline_fit(sub(a,tr),'rsf',ANCHOR)
        ht[va]=cumulative_hazard(model,sub(a,va),y['time'][va],individual=True)
        h3[va]=cumulative_hazard(model,sub(a,va),np.array([1095.])).ravel()
    ht=np.maximum(ht,1e-9); h3=np.maximum(h3,1e-9)
    anchor=baseline_fit(a,'rsf',ANCHOR)
    residual=y['event'].astype(float)-ht; residual-=residual.mean()
    sd=a['ranks'].std(axis=0)
    score=np.abs(residual@a['ranks'])/np.maximum(sd,1e-4)
    order=np.argsort(-score,kind='stable'); order=order[sd[order]>1e-3][:30]
    pairs=np.array([i for i,p in enumerate(a['id']) if p in repeats],int)
    repeated={'clinical':a['clinical'][pairs], 'ranks':np.stack([repeats[a['id'][i]] for i in pairs])}
    target,w=labels(a['t'],a['e'],km_censor(a['t'],a['e']))
    cases=np.where((target==1)&(w>0))[0]; controls=np.where((target==0)&(w>0))[0]
    reps={}
    program_mean=np.array([np.mean(a['ranks'][:,idx],dtype=float) for idx in sets.values()])
    for rep in REPRESENTATIONS:
        pre={'representation':rep,'gene_indices':order,'gene_names':[symbols[i] for i in order],
             'gene_sets':sets,'clinical_medians':np.nanmedian(a['clinical'],axis=0),'program_mean':program_mean}
        raw,names=raw_features(a,pre); scaler=StandardScaler().fit(raw); pre['scaler']=scaler
        x=scaler.transform(raw); delta=x[pairs]-scaler.transform(raw_features(repeated,pre)[0])
        reps[rep]={'x':x,'v':scaler.transform(raw_features(b,pre)[0]),
                   'Q':delta.T@delta/len(delta),'preprocessor':pre,'feature_names':names}
    return {'representations':reps,'event':y['event'].astype(float),'ht':ht,'h3':h3,
            'cases':cases,'controls':controls,'wc':w[cases]/w[cases].sum(),
            'wn':w[controls]/w[controls].sum(),'anchor':anchor,
            'eval_hazard':cumulative_hazard(anchor,b),'validation':{k:v for k,v in b.items() if k!='ranks'},
            'training_ids':a['id'],'repeat_ids':a['id'][pairs]}

def objective(theta,x,event,ht,h3,penalty,cases,controls,wc,wn,ranking_weight):
    eta=theta[0]+x@theta[1:]; hazard=ht*np.exp(eta)
    loss=np.mean(hazard-event*eta)+.5*theta[1:]@penalty@theta[1:]+.5e-5*theta[0]**2
    error=(hazard-event)/len(event)
    grad=np.r_[error.sum()+1e-5*theta[0],x.T@error+penalty@theta[1:]]
    if ranking_weight:
        score=np.log(h3)+eta
        difference=score[cases,None]-score[controls][None,:]
        weights=wc[:,None]*wn[None,:]
        loss+=ranking_weight*np.sum(weights*np.logaddexp(0,-difference))
        pairgrad=-ranking_weight*weights*expit(-difference)
        scoregrad=np.zeros(len(event))
        scoregrad[cases]=pairgrad.sum(axis=1); scoregrad[controls]=-pairgrad.sum(axis=0)
        grad[1:]+=x.T@scoregrad
    return float(loss),grad

def grid():
    return [{'representation':rep,'alpha':alpha,'ranking_weight':lam,'gamma':.3}
            for rep,alpha,lam in itertools.product(REPRESENTATIONS,[.03,.3,3.],[0.,.3,1.])]

def fit(prep,params):
    r=prep['representations'][params['representation']]; x=r['x']; k=x.shape[1]
    penalty=params['alpha']*np.eye(k)+params['gamma']*r['Q']
    args=(x,prep['event'],prep['ht'],prep['h3'],penalty,prep['cases'],prep['controls'],
          prep['wc'],prep['wn'],params['ranking_weight'])
    opt=minimize(objective,np.zeros(k+1),args=args,jac=True,method='L-BFGS-B',
                 options={'maxiter':700,'gtol':1e-7,'ftol':1e-11})
    if not opt.success: raise RuntimeError(f'{params}: {opt.message}; gradient {np.max(np.abs(opt.jac))}')
    p=-np.expm1(-prep['eval_hazard']*np.exp(opt.x[0]+r['v']@opt.x[1:])[:,None])
    model={'preprocessor':r['preprocessor'],'anchor':prep['anchor'],'coefficients':opt.x,
           'parameters':params,'feature_names':r['feature_names'],'training_ids':prep['training_ids']}
    return p,model,{'success':bool(opt.success),'iterations':opt.nit,'max_gradient':float(np.abs(opt.jac).max())}

def predict(model,b):
    x=model['preprocessor']['scaler'].transform(raw_features(b,model['preprocessor'])[0])
    theta=model['coefficients']; h=cumulative_hazard(model['anchor'],b)
    return -np.expm1(-h*np.exp(theta[0]+x@theta[1:])[:,None])

def selection(cache,prefix):
    rows=[]; start=time.time()
    for num,params in enumerate(grid()):
        met=[]
        for prep in cache:
            p,_,_=fit(prep,params); met.append(evaluate(prep['validation'],p)[2])
        row={**params,'cv_auc':float(np.mean([m['auc_ipcw'] for m in met])),
             'cv_brier':float(np.mean([m['brier_ipcw'] for m in met])),
             **{f'fold_{i+1}_auc':m['auc_ipcw'] for i,m in enumerate(met)}}
        rows.append(row)
        pd.DataFrame(rows).to_csv(WORK/f'{prefix}_candidates.csv',index=False)
        if (num+1)%9==0: print(prefix,num+1,'/',len(grid()),'seconds',round(time.time()-start),flush=True)
    frame=pd.DataFrame(rows); selected={}
    for rep,df in frame.groupby('representation',sort=False):
        best=df.sort_values(['cv_auc','cv_brier'],ascending=[False,True]).iloc[0]
        selected[rep]={k:best[k] for k in grid()[0]}
    best=frame.sort_values(['cv_auc','cv_brier'],ascending=[False,True]).iloc[0]
    selected['selected_family']=str(best.representation)
    selected['selected_cv_auc']=float(best.cv_auc)
    save(f'{prefix}_selected.json',selected)
    print(prefix,'selected',selected,flush=True)
    return selected

def frame_predictions(b,ps):
    return pd.concat([pd.DataFrame({'patient_id':b['id'],'year':int(t/365),'os_days':b['t'],
        'os_event':b['e'],**{n:p[:,j] for n,p in ps.items()}}) for j,t in enumerate(TIMES)],ignore_index=True)

def gradient_check():
    rng=np.random.default_rng(SEED); n,k=20,5; x=rng.normal(size=(n,k)); theta=rng.normal(0,.15,k+1)
    args=(x,rng.integers(2,size=n),rng.uniform(.01,.5,n),rng.uniform(.01,.4,n),np.eye(k)*.2,
          np.arange(5),np.arange(5,20),np.ones(5)/5,np.ones(15)/15,.3)
    err=np.max(np.abs(approx_derivative(lambda z:objective(z,*args)[0],theta).ravel()-objective(theta,*args)[1]))
    assert err<1e-6; save('gradient_check.json',{'max_error':float(err)})

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--nested',action='store_true'); args=parser.parse_args()
    start=time.time(); gradient_check()
    scan,meta,dev,test,strata,repeats,genes=load_data(); symbols=genes.Hugo_Symbol.tolist(); sets=gene_sets(symbols)
    a=sub(scan,dev)
    if not args.nested:
        save('protocol.json',{'created_utc':datetime.now(timezone.utc).isoformat(), 'endpoint':'All-cause OS at 1095 days',
             'representation':'Within-patient mean shared-gene ranks in 50 Hallmark sets; training scaling; optional age/ER interactions',
             'objective':'Working survival likelihood + IPCW case-control logistic ranking + ridge + true technical-repeat consistency',
             'ranking_weights':'Training KM; case/control weights each normalized to sum one',
             'grid':grid(),'selection':'Mean 3-fold development AUC then Brier; one selected family before evaluation',
             'reference':'Fixed v3 clinical RSF recipe; model families exploratory on previously inspected holdouts',
             'gmt_source':'https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=MSigDB_Hallmark_2020',
             'gmt_sha256':hashlib.sha256(GMT.read_bytes()).hexdigest(),'gene_universe':len(symbols)})
        cache=[]; memberships=[]
        for fold,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=SEED).split(a['t'],strata_of(a)),1):
            cache.append(prepare(sub(a,tr),sub(a,va),repeats,sets,symbols))
            memberships.extend({'patient_id':pid,'fold':fold,'role':role} for role,ix in [('train',tr),('validation',va)] for pid in a['id'][ix])
            print('Prepared development fold',fold,flush=True)
        pd.DataFrame(memberships).to_csv(WORK/'fold_membership.csv',index=False)
        selected=selection(cache,'development')
        save('selection_frozen.json',{'timestamp_utc':datetime.now(timezone.utc).isoformat(),'selected':selected})
        oof=[]
        for fold,prep in enumerate(cache,1):
            ps={rep:fit(prep,selected[rep])[0] for rep in REPRESENTATIONS}
            f=frame_predictions(prep['validation'],ps); f['fold']=fold; oof.append(f)
        pd.concat(oof).to_csv(WORK/'development_oof_predictions.csv',index=False)
        del cache
        both={k:np.concatenate([sub(scan,test)[k],meta[k]]) for k in a}
        prep=prepare(a,both,repeats,sets,symbols); models={}; ps={}; statuses={}
        for rep in REPRESENTATIONS:
            ps[rep],models[rep],statuses[rep]=fit(prep,selected[rep])
        winner=selected['selected_family']; base=selected[winner]
        for name,change in [('selected_without_ranking',{'ranking_weight':0.}),('selected_without_repeats',{'gamma':0.})]:
            ps[name],models[name],statuses[name]=fit(prep,{**base,**change})
        ps['selected_extension']=ps[winner]
        models['selected_family']=winner; models['shared_genes']=symbols
        joblib.dump(models,WORK/'models.joblib',compress=3)
        save('fit_status.json',statuses); save('training_repeat_ids.json',prep['repeat_ids'])
        records=[]
        for label,b,sl in [('SCANB_holdout',sub(scan,test),slice(0,len(test))),('METABRIC_external',meta,slice(len(test),None))]:
            frame_predictions(b,{k:v[sl] for k,v in ps.items()}).to_csv(WORK/f'predictions_{label}.csv',index=False)
            for name,p in ps.items(): records.extend({'cohort':label,'model':name,**m} for m in evaluate(b,p[sl]))
        pd.DataFrame(records).to_csv(WORK/'metrics_all_horizons.csv',index=False)
        replay=max(float(np.max(np.abs(predict(models[rep],both)-ps[rep]))) for rep in REPRESENTATIONS)
        assert replay<1e-12
        for rep in REPRESENTATIONS:
            f=models[rep]; pd.DataFrame({'feature':f['feature_names'],'coefficient':f['coefficients'][1:]}).to_csv(WORK/f'{rep}_coefficients.csv',index=False)
        save('execution.json',{'completed':True,'seconds':time.time()-start,'prediction_replay_max_error':replay})
        print(pd.DataFrame(records).query('year==3')[['cohort','model','auc_ipcw','brier_ipcw']].to_string(index=False),flush=True)
    else:
        outer=[]; records=[]
        for fold,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=SEED+101).split(a['t'],strata_of(a)),1):
            train,valid=sub(a,tr),sub(a,va); cache=[]
            for it,iv in StratifiedKFold(3,shuffle=True,random_state=SEED+fold).split(train['t'],strata_of(train)):
                cache.append(prepare(sub(train,it),sub(train,iv),repeats,sets,symbols))
            sel=selection(cache,f'nested_fold{fold}'); del cache
            prep=prepare(train,valid,repeats,sets,symbols)
            ps={rep:fit(prep,sel[rep])[0] for rep in REPRESENTATIONS}; ps['selected_extension']=ps[sel['selected_family']]
            f=frame_predictions(valid,ps); f['outer_fold']=fold; outer.append(f)
            for name,p in ps.items(): records.extend({'fold':fold,'model':name,**m} for m in evaluate(valid,p))
            pd.concat(outer).to_csv(WORK/'nested_predictions.csv',index=False)
            pd.DataFrame(records).to_csv(WORK/'nested_fold_metrics.csv',index=False)
            print('Nested fold finished',fold,flush=True)
        pooled=[]
        for year,b in pd.concat(outer).groupby('year'):
            for name in ps: pooled.append({'year':year,'model':name,**metric(b.os_days.to_numpy(),b.os_event.to_numpy(),b[name].to_numpy(),km_censor(b.os_days.to_numpy(),b.os_event.to_numpy()),365*year)})
        pd.DataFrame(pooled).to_csv(WORK/'nested_pooled_metrics.csv',index=False)
        save('nested_execution.json',{'completed':True,'seconds':time.time()-start})

if __name__=='__main__': main()
