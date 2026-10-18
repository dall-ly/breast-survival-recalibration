"""Survival CA-RRL: survival likelihood, clinical offset and replicate penalties.

All model choices use SCAN-B development patients. Endpoint: all-cause OS.
Administrative training truncation at 6 years; evaluation at years 1..5.
"""
from experiment_clinical_anchor import load_data,sub,ROOT,BASE,SEED,km_censor,labels,metric
import os,json,time,hashlib,platform,itertools,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis,CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest,GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from threadpoolctl import threadpool_limits
import joblib
threadpool_limits(2)

RUN=Path(os.environ.get('BREAST_SURVIVAL_RUN',ROOT/'runs/gse96058_os_v3_survival'))
RUN.mkdir(parents=True,exist_ok=True)
TIMES=np.array([365.,730.,1095.,1460.,1825.])
ADMIN=2190.

def write_json(name,x):
    (RUN/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda v:v.item()),encoding='utf-8')

def outcome(a):
    return Surv.from_arrays((a['e']==1)&(a['t']<=ADMIN),np.minimum(a['t'],ADMIN))

def strata_of(a):
    return np.where((a['t']<=1095)&(a['e']==1),1,np.where(a['t']>=1095,0,2))

def null_cumhaz(t,e):
    times=np.unique(t[e==1]); increments=np.array([np.sum((t==u)&(e==1))/np.sum(t>=u) for u in times])
    h=np.cumsum(increments); ix=np.searchsorted(times,t,side='right')-1
    return np.where(ix<0,0,h[np.maximum(ix,0)])

def feature_order(a):
    y=outcome(a); residual=y['event'].astype(float)-null_cumhaz(y['time'],y['event'])
    z=a['ranks']; std=z.std(axis=0); score=np.abs(residual@z)/np.maximum(std,1e-4)
    order=np.argsort(-score,kind='stable');return order[std[order]>1e-3][:200]

def features_fit(a,k):
    im=SimpleImputer(strategy='median',add_indicator=True); cs=StandardScaler()
    c=cs.fit_transform(im.fit_transform(a['clinical']))
    order=feature_order(a)[:k] if k else np.array([],int)
    gs=StandardScaler().fit(a['ranks'][:,order]) if k else None
    x=np.column_stack([c,gs.transform(a['ranks'][:,order])]) if k else c
    return x,{'imputer':im,'clinical_scaler':cs,'gene_scaler':gs,'gene_indices':order}

def features_transform(fit,b):
    c=fit['clinical_scaler'].transform(fit['imputer'].transform(b['clinical']))
    order=fit['gene_indices'];return np.column_stack([c,fit['gene_scaler'].transform(b['ranks'][:,order])]) if len(order) else c

def baseline_fit(a,kind,params):
    k=int(params.get('genes',0));x,pre=features_fit(a,k); y=outcome(a)
    if kind=='cox': model=CoxPHSurvivalAnalysis(alpha=params['alpha'],ties='breslow',n_iter=200,tol=1e-8)
    elif kind=='coxnet':model=CoxnetSurvivalAnalysis(alphas=np.geomspace(.3,params['alpha'],20),l1_ratio=params['l1_ratio'],fit_baseline_model=True,max_iter=100000,tol=1e-7)
    elif kind=='rsf':model=RandomSurvivalForest(n_estimators=160,min_samples_leaf=params['leaf'],max_features=params['features'],n_jobs=2,random_state=SEED)
    elif kind=='gb':model=GradientBoostingSurvivalAnalysis(n_estimators=params['trees'],max_depth=params['depth'],min_samples_leaf=15,learning_rate=.03,random_state=SEED)
    model.fit(x,y);return {'preprocessor':pre,'model':model,'kind':kind,'parameters':params}

def cumulative_hazard(fit,b,times=TIMES,individual=False):
    x=features_transform(fit['preprocessor'],b)
    fs=fit['model'].predict_cumulative_hazard_function(x)
    values=[]
    for i,f in enumerate(fs):
        points=np.array([times[i]]) if individual else times
        ix=np.searchsorted(f.x,points,side='right')-1
        values.append(np.where(ix<0,0.,f.a*f.y[np.maximum(ix,0)]+f.b))
    return np.asarray(values).ravel() if individual else np.asarray(values)

def baseline_predict(fit,b,times=TIMES):return -np.expm1(-cumulative_hazard(fit,b,times))

def evaluate(b,p):
    km=km_censor(b['t'],b['e']); rows=[]
    for j,tau in enumerate(TIMES):
        rows.append({'year':int(tau/365),**metric(b['t'],b['e'],p[:,j],km,tau)})
    return rows

def baseline_grid():
    grid={}
    grid['Cox_clinical']=('cox',[{'genes':0,'alpha':alpha} for alpha in [.01,.1,1.,10.]])
    grid['CoxEN_fusion']=('coxnet',[{'genes':k,'alpha':alpha,'l1_ratio':ratio} for k,alpha,ratio in itertools.product([20,60],[.003,.03],[.1,.5])])
    for suffix,k in [('clinical',0),('fusion',20)]:
        grid['RSF_'+suffix]=('rsf',[{'genes':k,'leaf':leaf,'features':feat} for leaf,feat in itertools.product([10,30],['sqrt',1.])])
        grid['SurvGB_'+suffix]=('gb',[{'genes':k,'trees':trees,'depth':depth} for trees,depth in itertools.product([100,250],[1,2])])
    return grid

def select_baselines(scan,dev):
    rows=pd.read_csv(RUN/'baseline_development_cv.csv').to_dict('records') if (RUN/'baseline_development_cv.csv').exists() else []
    selected=json.loads((RUN/'baseline_selected.json').read_text()) if (RUN/'baseline_selected.json').exists() else {}
    strata=strata_of(sub(scan,dev))
    folds=list(StratifiedKFold(3,shuffle=True,random_state=SEED).split(dev,strata))
    for name,(kind,grid) in baseline_grid().items():
        if name in selected:continue
        scores=[]
        for index,params in enumerate(grid):
            aucs=[];briers=[]
            for tr,va in folds:
                a,b=sub(scan,dev[tr]),sub(scan,dev[va]);fit=baseline_fit(a,kind,params)
                met=evaluate(b,baseline_predict(fit,b))[2];aucs.append(met['auc_ipcw']);briers.append(met['brier_ipcw'])
            row={'model':name,'candidate':index,**params,'cv_auc':np.mean(aucs),'cv_brier':np.mean(briers),**{f'fold{i+1}':v for i,v in enumerate(aucs)}}
            rows.append(row);scores.append((np.mean(aucs),-np.mean(briers),params))
        best=max(scores,key=lambda x:(x[0],x[1])); selected[name]={'kind':kind,'parameters':best[2],'cv_auc':best[0]}
        pd.DataFrame(rows).to_csv(RUN/'baseline_development_cv.csv',index=False);write_json('baseline_selected.json',selected)
        print('Selected',name,selected[name],flush=True)
    return selected

def prepare_survival_residual(a,b,repeats,anchor_params,anchor_kind='gb'):
    oof=np.zeros(len(a['t']));y=outcome(a)
    for tr,va in StratifiedKFold(3,shuffle=True,random_state=SEED+13).split(oof,strata_of(a)):
        fit=baseline_fit(sub(a,tr),anchor_kind,anchor_params)
        oof[va]=cumulative_hazard(fit,sub(a,va),y['time'][va],individual=True)
    oof=np.maximum(oof,1e-9)
    anchor=baseline_fit(a,anchor_kind,anchor_params);haz=cumulative_hazard(anchor,b)
    residual=y['event'].astype(float)-oof;residual-=residual.mean()
    std=a['ranks'].std(axis=0);scores=np.abs(residual@a['ranks'])/np.maximum(std,1e-4)
    order=np.argsort(-scores,kind='stable');order=order[std[order]>1e-3][:60]
    scaler=StandardScaler().fit(a['ranks'][:,order]);x=scaler.transform(a['ranks'][:,order]);v=scaler.transform(b['ranks'][:,order])
    paired=np.array([i for i,patient in enumerate(a['id']) if patient in repeats])
    rx=np.stack([repeats[a['id'][i]][order] for i in paired])
    delta=x[paired]-scaler.transform(rx)
    noise=.5*np.mean(delta**2,axis=0)
    return {'x':x,'v':v,'delta':delta,'noise':noise,'events':y['event'].astype(float),'oof_hazard':oof,
            'eval_hazard':haz,'anchor':anchor,'gene_indices':order,'scaler':scaler,'validation':b,'training_ids':a['id'],
            'training_replicate_ids':a['id'][paired]}

def survival_objective(theta,x,event,H,penalty):
    eta=theta[0]+x@theta[1:];risk=H*np.exp(eta)
    loss=np.mean(risk-event*eta)+.5*theta[1:]@penalty@theta[1:]+.5e-5*theta[0]**2
    err=(risk-event)/len(event)
    grad=np.r_[err.sum()+1e-5*theta[0],x.T@err+penalty@theta[1:]]
    return loss,grad

def residual_fit(prep,params):
    k=int(params['genes']);alpha=params['alpha'];gamma=params['gamma'];rho=params['reliability']
    delta=prep['delta'][:,:k]
    penalty=alpha*np.diag(1+rho*prep['noise'][:k])+gamma*(delta.T@delta)/len(delta)
    result=minimize(survival_objective,np.zeros(k+1),args=(prep['x'][:,:k],prep['events'],prep['oof_hazard'],penalty),jac=True,method='L-BFGS-B',options={'maxiter':500,'gtol':1e-7,'ftol':1e-11})
    multiplier=np.exp(result.x[0]+prep['v'][:,:k]@result.x[1:])
    prediction=-np.expm1(-prep['eval_hazard']*multiplier[:,None])
    return prediction,result.x,{'optimizer_success':bool(result.success),'gradient_max':float(np.abs(result.jac).max())}

def residual_grid():
    return [{'genes':k,'alpha':alpha,'gamma':gamma,'reliability':rho}
            for k,alpha,gamma,rho in itertools.product([10,30],[.003,.03,.3],[0.,.3,3.],[0.,10.])]

def select_residual(cache):
    rows=[]
    for params in residual_grid():
        aucs=[];briers=[]
        for prep in cache:
            p,_,_=residual_fit(prep,params);b=prep['validation'];met=evaluate(b,p)[2]
            aucs.append(met['auc_ipcw']);briers.append(met['brier_ipcw'])
        rows.append({**params,'cv_auc':np.mean(aucs),'cv_brier':np.mean(briers),**{f'fold{i+1}':v for i,v in enumerate(aucs)}})
    df=pd.DataFrame(rows);selected={}
    for name,condition in [('Surv_residual',(df.gamma==0)&(df.reliability==0)),('Surv_consistency',(df.gamma>0)&(df.reliability==0)),('Surv_reliability',(df.gamma>0)&(df.reliability>0))]:
        row=df[condition].sort_values(['cv_auc','cv_brier'],ascending=[False,True]).iloc[0]
        selected[name]={'parameters':{k:float(row[k]) for k in ['genes','alpha','gamma','reliability']},'cv_auc':row.cv_auc}
    return selected,df

def main():
    start=time.time();scan,meta,dev,test,strata,repeats,genes=load_data()
    write_json('protocol.json',{'endpoint':'All-cause OS','training_administrative_censor_days':ADMIN,'evaluation_years':[1,2,3,4,5],
               'primary_selection':'3-year development CV IPCW AUC, Brier tie-break','split':'Existing SCANB 2618 development / 655 heldout; METABRIC 1978 external',
               'gene_screening':'Training survival score for baseline; cross-fitted clinical martingale residual for CA-RRL',
               'survival_likelihood':'H_clin(T)*exp(intercept+z beta) - event*(intercept+z beta)',
               'penalties':'ridge + technical-replicate prediction consistency + gene-specific replicate noise weights',
               'replicate_pairs':'Training primary patients only','baseline_grids':baseline_grid(),'residual_grid':residual_grid(),
               'evaluation_censoring':'Evaluation cohort marginal KM as in earlier rounds; outcomes used only for scoring',
               'development_history':'This is a continued research experiment with previously evaluated holdout/external cohorts'})
    selected=select_baselines(scan,dev)
    anchor_name=max([n for n in selected if n.endswith('_clinical')],key=lambda n:selected[n]['cv_auc'])
    anchor_params=selected[anchor_name]['parameters'];anchor_kind=selected[anchor_name]['kind'];cache=[]
    write_json('residual_anchor_choice.json',{'name':anchor_name,**selected[anchor_name]})
    for fold,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=SEED).split(dev,strata[dev]),1):
        prep=prepare_survival_residual(sub(scan,dev[tr]),sub(scan,dev[va]),repeats,anchor_params,anchor_kind);cache.append(prep)
        print('Survival residual cross-fitting ready',fold,flush=True)
    residual_selected,cv=select_residual(cache);cv.to_csv(RUN/'residual_development_cv.csv',index=False)
    write_json('residual_selected.json',residual_selected)
    print('Residual candidates frozen',json.dumps(residual_selected),flush=True)
    # Save development fold predictions for subsequent calibration/risk thresholds.
    devpred=[]
    for fold,prep in enumerate(cache,1):
        b=prep['validation']
        ps={anchor_name:-np.expm1(-prep['eval_hazard'])}
        for name,chosen in residual_selected.items():ps[name]=residual_fit(prep,chosen['parameters'])[0]
        for j,year in enumerate((TIMES/365).astype(int)):
            devpred.append(pd.DataFrame({'patient_id':b['id'],'fold':fold,'year':year,'os_days':b['t'],'os_event':b['e'],**{n:p[:,j] for n,p in ps.items()}}))
    pd.concat(devpred,ignore_index=True).to_csv(RUN/'development_oof_predictions.csv',index=False)
    del cache
    train=sub(scan,dev);evals={'SCANB_holdout':sub(scan,test),'METABRIC_external':meta}
    both={k:np.concatenate([evals['SCANB_holdout'][k],meta[k]]) for k in train};slices={'SCANB_holdout':slice(0,len(test)),'METABRIC_external':slice(len(test),None)}
    predictions={cohort:{} for cohort in evals};allmetrics=[];fits={}
    for name,chosen in selected.items():
        fit=baseline_fit(train,chosen['kind'],chosen['parameters']);fits[name]=fit
        p=baseline_predict(fit,both)
        for cohort,b in evals.items():
            predictions[cohort][name]=p[slices[cohort]]
            allmetrics.extend({'cohort':cohort,'model':name,**row} for row in evaluate(b,p[slices[cohort]]))
        print('Evaluated baseline',name,flush=True)
    prep=prepare_survival_residual(train,both,repeats,anchor_params,anchor_kind)
    fitted={'anchor':prep['anchor'],'gene_indices':prep['gene_indices'],'scaler':prep['scaler'],'shared_genes':genes.Hugo_Symbol.to_list(),'evaluation_times':TIMES,'variants':{}}
    variants=dict(residual_selected)
    variants['Surv_consistency_removed']={'parameters':{**residual_selected['Surv_consistency']['parameters'],'gamma':0.}}
    variants['Surv_reliability_removed']={'parameters':{**residual_selected['Surv_reliability']['parameters'],'reliability':0.}}
    for name,chosen in variants.items():
        p,theta,info=residual_fit(prep,chosen['parameters']);fitted['variants'][name]={'parameters':chosen['parameters'],'coefficients':theta}
        write_json(name+'_fit.json',info)
        g=genes.iloc[prep['gene_indices'][:len(theta)-1]].copy();g['coefficient']=theta[1:];g['replicate_noise_variance']=prep['noise'][:len(theta)-1];g.to_csv(RUN/(name+'_genes.csv'),index=False)
        for cohort,b in evals.items():
            predictions[cohort][name]=p[slices[cohort]]
            allmetrics.extend({'cohort':cohort,'model':name,**row} for row in evaluate(b,p[slices[cohort]]))
    for cohort,b in evals.items():
        rows=[]
        for j,year in enumerate((TIMES/365).astype(int)):
            rows.append(pd.DataFrame({'patient_id':b['id'],'year':year,'os_days':b['t'],'os_event':b['e'],**{n:p[:,j] for n,p in predictions[cohort].items()}}))
        pd.concat(rows,ignore_index=True).to_csv(RUN/f'predictions_{cohort}.csv',index=False)
    pd.DataFrame(allmetrics).to_csv(RUN/'metrics_all_horizons.csv',index=False)
    joblib.dump(fitted,RUN/'survival_carrl_models.joblib');joblib.dump(fits,RUN/'survival_baselines.joblib')
    write_json('training_replicate_ids.json',prep['training_replicate_ids'].tolist())
    write_json('execution.json',{'seconds':time.time()-start,'python':platform.python_version(),'completed':True})
    print(pd.DataFrame(allmetrics).query('year == 3')[['cohort','model','auc_ipcw','brier_ipcw']].to_string(index=False),flush=True)

if __name__=='__main__':main()
