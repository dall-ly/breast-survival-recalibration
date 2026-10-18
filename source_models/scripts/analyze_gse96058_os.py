"""Reproducible exploratory OS experiment; source files are never modified."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, hashlib, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits
threadpool_limits(2)

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/gse96058_os_v1'
OUT.mkdir(parents=True,exist_ok=True)
SRC=Path(os.environ.get('SCANB_PROCESSED_DIR', str(ROOT/'data/SCAN_B_dataset/processed')))
OLD=ROOT/'runs/v2/processed'
TAU=1095
SEED=20261015

def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item()),encoding='utf-8')

def km_censor(t,e):
    u=np.unique(t); risk=np.array([(t>=v).sum() for v in u]); d=np.array([((t==v)&(e==0)).sum() for v in u])
    return u,np.cumprod(1-d/risk)

def g_at(km,x,left=False):
    u,g=km; ix=np.searchsorted(u,x,side='left' if left else 'right')-1
    return np.where(ix<0,1.,g[np.maximum(ix,0)])

def labels(t,e,km,tau=TAU):
    y=((t<=tau)&(e==1)).astype(int)
    known=(y==1)|(t>tau)|((t==tau)&(e==0))
    prob=np.where(y==1,g_at(km,t,left=True),g_at(km,np.full(len(t),tau)))
    if np.any(prob[known]<=0): raise ValueError('Censoring positivity failure')
    w=np.where(known,1/np.maximum(prob,1e-12),0)
    return y,w

def metric(t,e,p,km,tau=TAU):
    y,w=labels(t,e,km,tau); ix=w>0
    return {'auc_ipcw':roc_auc_score(y[ix],p[ix],sample_weight=w[ix]),
            'brier_ipcw':float(np.sum(w*(y-p)**2)/len(t)),
            'average_precision_ipcw':average_precision_score(y[ix],p[ix],sample_weight=w[ix]),
            'mean_predicted_risk':float(np.mean(p)), 'n':len(t),'events_by_horizon':int(y.sum()),'early_censored':int((~ix).sum())}

def clinical_scan(s):
    return pd.DataFrame({'age':s['age at diagnosis'],'size':s['tumor size'],
       'grade':s['nhg'].map({'G1':1,'G2':2,'G3':3}),
       'node_positive':s['lymph node status'].map({'NodeNegative':0,'NodePositive':1}),
       'er_positive':s['er status']}).astype(float)

def fit_predict(a,b,mode,c=1.,k=30,imp='median',tau=TAU):
    # a/b dictionaries: clinical, ranks, times, events; every fitted operation uses a only.
    im=SimpleImputer(strategy=imp,add_indicator=True)
    ca=im.fit_transform(a['clinical']); cb=im.transform(b['clinical'])
    sc=StandardScaler(); ca=sc.fit_transform(ca); cb=sc.transform(cb)
    km=km_censor(a['t'],a['e']); y,w=labels(a['t'],a['e'],km,tau); ok=w>0
    chosen=np.array([],int)
    if mode in ('genes','fusion','context'):
        xa=a['ranks']; xb=b['ranks']; ww=w/w.sum(); ym=np.sum(ww*y)
        mu=ww@xa; var=ww@((xa-mu)**2)
        score=np.abs((ww*(y-ym))@xa)/np.sqrt(np.maximum(var,1e-12))
        chosen=np.argsort(-score,kind='stable')[:k]
        gs=StandardScaler(); ga=gs.fit_transform(xa[:,chosen]); gb=gs.transform(xb[:,chosen])
        if mode=='genes': ca,cb=ga,gb
        else:
            # Predefined contextual interactions: age and ER; no outcome-derived gating.
            if mode=='context':
                ga=np.column_stack([ga,ga*ca[:,[0]],ga*ca[:,[4]]])
                gb=np.column_stack([gb,gb*cb[:,[0]],gb*cb[:,[4]]])
            ca=np.column_stack([ca,ga]); cb=np.column_stack([cb,gb])
    if mode=='clinical_boost':
        model=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=7,l2_regularization=c,learning_rate=.05,random_state=SEED)
    else:
        model=LogisticRegression(C=c,max_iter=2000,solver='lbfgs')
    model.fit(ca[ok],y[ok],sample_weight=w[ok])
    return model.predict_proba(cb)[:,1],chosen

def sub(d,ix): return {k:v[ix] for k,v in d.items()}

def main():
    s0=pd.read_csv(SRC/'sample_metadata_expression_order.csv')
    prim=~s0.is_technical_replicate
    s=s0.loc[prim].reset_index(drop=True)
    sg=pd.read_csv(SRC/'gene_ids.csv')
    fm=pd.read_csv(OLD/'metabric_expression_feature_map.csv')
    fm=fm[fm.mapping_unambiguous_within_metabric & fm.Hugo_Symbol.notna()]
    shared=fm.merge(sg,left_on='Hugo_Symbol',right_on='gene_id',validate='one_to_one').sort_values('Hugo_Symbol').reset_index(drop=True)
    shared.to_csv(OUT/'shared_gene_map.csv',index=False)
    sx=np.load(SRC/'expression_log2_fpkm_float64.npy',mmap_mode='r')
    mx=np.load(OLD/'metabric_expression_samples_by_features.npy',mmap_mode='r')
    se=np.array(sx[shared.gene_index_0based.to_numpy(),:][:,prim.to_numpy()].T)
    me=np.array(mx[:,shared.array_column_0based.to_numpy()])
    # Missing expression retained in raw aligned matrices; rank proxy uses each patient's
    # observed-gene median solely to place rare missing values at mid rank (no cohort fit).
    def ranks(x):
        z=x.copy(); ii,jj=np.where(~np.isfinite(z)); z[ii,jj]=np.nanmedian(z,axis=1)[ii]
        return (rankdata(z,axis=1,method='average')/z.shape[1]).astype('float32')
    sr,mr=ranks(se),ranks(me)
    np.save(OUT/'scanb_shared_expression.npy',se); np.save(OUT/'metabric_shared_expression.npy',me)
    np.save(OUT/'scanb_shared_ranks.npy',sr); np.save(OUT/'metabric_shared_ranks.npy',mr)
    ids=pd.read_csv(OLD/'metabric_expression_sample_order.csv').patient_id
    raw=pd.read_csv(Path(os.environ.get('METABRIC_RECURRENCE_FILE', str(ROOT/'data/METABRIC_dataset/raw/recurrence_2019/Supplementary_Table_6.txt'))),sep='\t').set_index('METABRIC.ID').loc[ids]
    mc=pd.read_csv(OLD/'metabric_clinical_features.csv').set_index('patient_id').loc[ids]
    mclin=pd.DataFrame({'age':mc.age_years,'size':mc.tumor_size_mm,'grade':mc.grade,'node_positive':mc.node_positive,'er_positive':mc.er_status.map({'positive':1,'negative':0})})
    scan={'clinical':clinical_scan(s).to_numpy(),'ranks':sr,'t':s['overall survival days'].to_numpy(float),'e':s['overall survival event'].to_numpy(int)}
    valid=raw.Death.isin([0,1]) & raw['T'].notna() & (raw['T']>0)
    meta={'clinical':mclin.to_numpy(float)[valid],'ranks':mr[valid],'t':raw['T'].to_numpy(float)[valid],'e':raw.Death.to_numpy(float)[valid].astype(int)}
    assert len(s)==3273 and s.sample_id.is_unique and len(s0)==3409
    assert set(scan['e'])=={0,1} and np.isfinite(se).all()
    assert set(s.primary_sample_id)==set(s.sample_id)
    for d in (scan,meta): assert np.isfinite(d['t']).all() and (d['t']>0).all()
    support=[]
    for name,d in [('SCANB',scan),('METABRIC',meta)]:
        for yr in [2,3,4,5]:
            y,w=labels(d['t'],d['e'],km_censor(d['t'],d['e']),365*yr)
            support.append({'cohort':name,'years':yr,'n':len(y),'deaths':int(y.sum()),'known_survivors':int(((w>0)&(y==0)).sum()),'early_censored':int((w==0).sum())})
    pd.DataFrame(support).to_csv(OUT/'followup_support.csv',index=False)
    # Version-specific split; not a reconstruction of the unavailable 2010–2020 IDs.
    strata=np.where((scan['t']<=TAU)&(scan['e']==1),1,np.where(scan['t']>=TAU,0,2))
    dev,test=train_test_split(np.arange(len(s)),test_size=.2,stratify=strata,random_state=SEED)
    membership=pd.DataFrame({'sample_id':s.sample_id,'role':'development'})
    membership.loc[test,'role']='heldout'; membership.to_csv(OUT/'scanb_membership.csv',index=False)
    cl=clinical_scan(s); cl.insert(0,'sample_id',s.sample_id); cl['os_days']=scan['t']; cl['os_event']=scan['e']; cl.to_csv(OUT/'scanb_clinical_clean.csv',index=False)
    mclean=mclin.copy(); mclean['os_days']=raw['T']; mclean['os_event']=raw.Death; mclean['os_eligible']=valid; mclean.to_csv(OUT/'metabric_clinical_os.csv')
    s0.loc[~prim,['sample_id','primary_sample_id','platform_id']].to_csv(OUT/'technical_replicates_excluded.csv',index=False)
    raw.loc[~valid,['T','Death']].to_csv(OUT/'metabric_os_exclusions.csv')
    pd.concat([clinical_scan(s).isna().sum().rename('SCANB'),mclin.isna().sum().rename('METABRIC')],axis=1).to_csv(OUT/'clinical_missingness.csv')
    dump('audit.json',{'scanb_primary':len(s),'technical_replicates':int((~prim).sum()),'metabric_os_eligible':int(valid.sum()),'shared_unambiguous_symbols':len(shared),'scanb_expression_missing':int(np.isnan(se).sum()),'metabric_expression_missing':int(np.isnan(me).sum()),'primary_horizon_days':TAU,'seed':SEED,'metabric_population':'All molecular cohort patients with valid OS; NOT prior DRFI strict cohort of 1436','checks_passed':True})
    dump('protocol.json',{'endpoint':'All-cause death by 1095 days; 1825-day supplementary','split':'80/20 SCANB, seed 20261015; 3-fold development CV; METABRIC external','selection':'Mean development CV IPCW AUC only','censoring':'Marginal Kaplan-Meier; training-fold G for fitting, evaluation-cohort G for metrics; conditional independent censoring not modeled','features':['age','size','grade','node_positive','ER','within-sample shared-gene ranks'],'missing_clinical':'mean versus median plus missing indicators, selected inside development','context_hypothesis':'Gene effects vary with age and ER; penalized rank-gene × clinical interactions','limitations':['Exploratory single split','No claim of novel algorithm priority','No treatment-benefit inference','External testing is exploratory and now exposed','Platform harmonization by ranks is a candidate, not a guarantee']})
    print('Data prepared',len(shared),'shared genes',flush=True)
    cv=StratifiedKFold(3,shuffle=True,random_state=SEED)
    folds=list(cv.split(dev,strata[dev])); allcv=[]; selected={}; preds={}; results=[]
    for mode in ['clinical','clinical_boost','genes','fusion','context']:
        candidates=[]
        for im in ['median','mean']:
            for c in ([1.,10.] if mode=='clinical_boost' else [.01,.1,1.]):
                for k in ([20,60] if mode in ['genes','fusion','context'] else [0]):
                    vals=[]
                    for tr,va in folds:
                        a,b=sub(scan,dev[tr]),sub(scan,dev[va])
                        p,_=fit_predict(a,b,mode,c,k,im)
                        vals.append(metric(b['t'],b['e'],p,km_censor(b['t'],b['e']))['auc_ipcw'])
                    row={'model':mode,'C_or_l2':c,'genes':k,'imputation':im,'cv_auc_mean':float(np.mean(vals)),'cv_auc_sd':float(np.std(vals)),**{f'fold{i+1}':v for i,v in enumerate(vals)}}
                    allcv.append(row); candidates.append(row)
        best=max(candidates,key=lambda x:x['cv_auc_mean']); selected[mode]=best
        print('Selected',mode,best,flush=True)
        a=sub(scan,dev)
        for cohort,b in [('SCANB_holdout',sub(scan,test)),('METABRIC_external',meta)]:
            p,ch=fit_predict(a,b,mode,best['C_or_l2'],best['genes'],best['imputation'])
            preds[(cohort,mode)]=p
            results.append({'cohort':cohort,'model':mode,**metric(b['t'],b['e'],p,km_censor(b['t'],b['e']))})
        if len(ch): shared.iloc[ch].to_csv(OUT/f'selected_genes_{mode}.csv',index=False)
        pd.DataFrame(allcv).to_csv(OUT/'development_cv.csv',index=False)
    dump('selected_parameters.json',selected)
    pd.DataFrame(results).to_csv(OUT/'metrics_3year.csv',index=False)
    # Paired bootstrap conditions on the fitted models; evaluation G reestimated per resample.
    rng=np.random.default_rng(SEED); boot=[]
    for cohort,b,patientids in [('SCANB_holdout',sub(scan,test),s.sample_id.iloc[test].to_numpy()),('METABRIC_external',meta,ids.to_numpy()[valid.to_numpy()])]:
        pp=pd.DataFrame({'patient_id':patientids,'os_days':b['t'],'os_event':b['e'],**{m:preds[(cohort,m)] for m in selected}})
        pp.to_csv(OUT/f'predictions_{cohort}.csv',index=False)
        values={m:[] for m in selected}
        for rep in range(500):
            ix=rng.integers(len(b['t']),size=len(b['t'])); tt,ee=b['t'][ix],b['e'][ix]; km=km_censor(tt,ee)
            for m in selected: values[m].append(metric(tt,ee,preds[(cohort,m)][ix],km)['auc_ipcw'])
        for m,v in values.items():
            lo,hi=np.quantile(v,[.025,.975]); diff=np.array(v)-values['clinical']; dl,dh=np.quantile(diff,[.025,.975])
            boot.append({'cohort':cohort,'model':m,'auc_low':lo,'auc_high':hi,'delta_clinical_low':dl,'delta_clinical_high':dh,'bootstrap_replicates':500})
    pd.DataFrame(boot).to_csv(OUT/'bootstrap_3year.csv',index=False)
    five=[]
    for mode,best in selected.items():
        for cohort,b in [('SCANB_holdout',sub(scan,test)),('METABRIC_external',meta)]:
            p,_=fit_predict(sub(scan,dev),b,mode,best['C_or_l2'],best['genes'],best['imputation'],tau=1825)
            five.append({'cohort':cohort,'model':mode,**metric(b['t'],b['e'],p,km_censor(b['t'],b['e']),tau=1825)})
    pd.DataFrame(five).to_csv(OUT/'metrics_5year_supplementary.csv',index=False)
    manifest=[]
    for f in [SRC/'sample_metadata_expression_order.csv',SRC/'gene_ids.csv',SRC/'expression_log2_fpkm_float64.npy',OLD/'metabric_expression_samples_by_features.npy',Path(__file__)]:
        h=hashlib.sha256()
        with f.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
        manifest.append({'path':str(f),'bytes':f.stat().st_size,'sha256':h.hexdigest()})
    dump('source_manifest.json',manifest)
    print(pd.DataFrame(results).to_string(index=False),flush=True)
    print('COMPLETE',OUT,flush=True)

if __name__=='__main__':main()
