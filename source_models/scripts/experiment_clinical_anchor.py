"""Clinical anchor + cross-fitted molecular residual + replicate consistency.

Development selects all parameters. Clinical predictions are cross-fitted before
residual gene selection. Only replicate pairs whose primary patient is in the
training fold contribute to the consistency penalty.
"""
from analyze_gse96058_os import *
from scipy.special import expit, logit
from scipy.optimize import minimize
import joblib

BASE=OUT
RUN=ROOT/'runs/gse96058_os_v2_anchor'
RUN.mkdir(parents=True,exist_ok=True)

def save_json(name,obj):
    (RUN/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item()),encoding='utf-8')

def load_data():
    s=pd.read_csv(BASE/'scanb_clinical_clean.csv'); m=pd.read_csv(BASE/'metabric_clinical_os.csv')
    cols=['age','size','grade','node_positive','er_positive']
    scan={'clinical':s[cols].to_numpy(),'ranks':np.load(BASE/'scanb_shared_ranks.npy'),
          't':s.os_days.to_numpy(),'e':s.os_event.to_numpy(),'id':s.sample_id.to_numpy()}
    valid=m.os_eligible.to_numpy()
    meta={'clinical':m[cols].to_numpy()[valid],'ranks':np.load(BASE/'metabric_shared_ranks.npy')[valid],
          't':m.os_days.to_numpy()[valid],'e':m.os_event.to_numpy()[valid].astype(int),'id':m.patient_id.to_numpy()[valid]}
    strata=np.where((scan['t']<=TAU)&(scan['e']==1),1,np.where(scan['t']>=TAU,0,2))
    dev,test=train_test_split(np.arange(len(s)),test_size=.2,stratify=strata,random_state=SEED)
    source_metadata=pd.read_csv(SRC/'sample_metadata_expression_order.csv')
    pair_table=pd.read_csv(SRC/'technical_replicate_pairs.csv')
    genes=pd.read_csv(BASE/'shared_gene_map.csv')
    source=np.load(SRC/'expression_log2_fpkm_float64.npy',mmap_mode='r')
    indices=source_metadata.set_index('sample_id').index.get_indexer(pair_table.replicate_sample_id)
    expression=source[np.ix_(genes.gene_index_0based.to_numpy(),indices)].T
    rr=(rankdata(expression,axis=1,method='average')/len(genes)).astype('float32')
    repeats=dict(zip(pair_table.primary_sample_id,rr))
    return scan,meta,dev,test,strata,repeats,genes

def anchor_fit(a):
    im=SimpleImputer(strategy='median',add_indicator=True)
    scaler=StandardScaler()
    x=scaler.fit_transform(im.fit_transform(a['clinical']))
    y,w=labels(a['t'],a['e'],km_censor(a['t'],a['e']))
    model=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=7,l2_regularization=10.,learning_rate=.05,random_state=SEED)
    model.fit(x[w>0],y[w>0],sample_weight=w[w>0])
    return im,scaler,model

def anchor_predict(fit,b):
    return fit[2].predict_proba(fit[1].transform(fit[0].transform(b['clinical'])))[:,1]

def prepare_fold(a,b,repeats):
    strata=np.where((a['t']<=TAU)&(a['e']==1),1,np.where(a['t']>=TAU,0,2))
    poof=np.zeros(len(a['t']))
    for tr,va in StratifiedKFold(3,shuffle=True,random_state=SEED+13).split(poof,strata):
        poof[va]=anchor_predict(anchor_fit(sub(a,tr)),sub(a,va))
    anchor=anchor_fit(a); pb=anchor_predict(anchor,b)
    y,w=labels(a['t'],a['e'],km_censor(a['t'],a['e']))
    z=a['ranks']; wn=w/w.sum(); mu=wn@z
    var=wn@((z-mu)**2); std=np.sqrt(np.maximum(var,1e-10))
    residual=wn*(y-poof); residual-=wn*residual.sum()
    score=(residual@z)/std
    order=np.argsort(-np.abs(score),kind='stable')
    # Discard practically constant features using only this training fold.
    order=order[std[order]>1e-3][:60]
    scaler=StandardScaler().fit(z[:,order])
    x=scaler.transform(z[:,order]); v=scaler.transform(b['ranks'][:,order])
    ii=np.array([i for i,patient in enumerate(a['id']) if patient in repeats],dtype=int)
    if len(ii):
        xr=np.stack([repeats[a['id'][i]][order] for i in ii])
        delta=x[ii]-scaler.transform(xr)
    else: delta=np.empty((0,len(order)))
    return {'x':x,'v':v,'delta':delta,'y':y,'w':w,'offset':logit(np.clip(poof,1e-7,1-1e-7)),
            'test_offset':logit(np.clip(pb,1e-7,1-1e-7)),'order':order,'scaler':scaler,
            'anchor':anchor,'validation':b,'replicate_patients':a['id'][ii]}

def objective(theta,x,y,w,offset,alpha,gram):
    eta=offset+x@theta
    nll=np.sum(w*(np.logaddexp(0,eta)-y*eta))/len(y)
    reg=.5*alpha*(theta@theta)+.5*theta@gram@theta
    gradient=x.T@(w*(expit(eta)-y))/len(y)+alpha*theta+gram@theta
    return nll+reg,gradient

def fit_residual(prepared,k,alpha,consistency,cap):
    x=prepared['x'][:,:k]; delta=prepared['delta'][:,:k]
    gram=consistency*(delta.T@delta)/max(1,len(delta))
    result=minimize(objective,np.zeros(k),args=(x,prepared['y'],prepared['w'],prepared['offset'],alpha,gram),
                    method='L-BFGS-B',jac=True,options={'maxiter':600,'ftol':1e-11,'gtol':1e-7})
    change=prepared['v'][:,:k]@result.x
    if cap: change=cap*np.tanh(change/cap)
    p=expit(prepared['test_offset']+change)
    return p,result.x,{'optimizer_success':bool(result.success),'gradient_max':float(np.abs(result.jac).max()),
                       'replicate_logit_gap_rms':float(np.sqrt(np.mean((delta@result.x)**2))) if len(delta) else None}

def main():
    scan,meta,dev,test,strata,repeats,genes=load_data()
    protocol={'method':'Clinical-anchored replicate-consistent residual learning (CA-RRL)',
              'horizon_days':TAU,'clinical_anchor':'IPCW histogram gradient boosting, fixed preceding development-selected parameters',
              'molecular_residual':'Train-fold OOF clinical residual screening; ridge logistic offset model',
              'objective':'mean IPCW logistic loss(offset + X beta) + alpha/2 ||beta||² + gamma/2 mean((Xprimary-Xrepeat) beta)²',
              'crossfit':'3 internal folds for anchor, nested within each of 3 development selection folds',
              'parameters':{'genes':[10,20,40],'alpha':[.003,.01,.03,.1],'gamma':[0.,.1,1.],'logit_cap':[0.,.5,1.]},
              'selection':'Mean development CV IPCW AUC; no external outcomes used',
              'split':'Same patient identities and CV fold order as v1',
              'replicate_use':'Only train-patient replicate pairs; heldout and external pairs never used',
              'clinical_only_and_previous_methods':'Retained as performance references'}
    save_json('protocol.json',protocol)
    cache=[]; fold_ids=[]
    for fold,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=SEED).split(dev,strata[dev]),1):
        a,b=sub(scan,dev[tr]),sub(scan,dev[va]); prepared=prepare_fold(a,b,repeats); cache.append(prepared)
        fold_ids.extend({'sample_id':patient,'fold':fold,'role':role} for role,data in [('train',a),('validation',b)] for patient in data['id'])
        print(f'Prepared development fold {fold}: {len(a["t"])} training, {len(prepared["replicate_patients"])} training replicate pairs',flush=True)
    pd.DataFrame(fold_ids).to_csv(RUN/'development_fold_membership.csv',index=False)
    rows=[]
    for k in [10,20,40]:
        for alpha in [.003,.01,.03,.1]:
            for gamma in [0.,.1,1.]:
                for cap in [0.,.5,1.]:
                    aucs=[]; briers=[]
                    for prepared in cache:
                        p,theta,info=fit_residual(prepared,k,alpha,gamma,cap)
                        b=prepared['validation']; met=metric(b['t'],b['e'],p,km_censor(b['t'],b['e']))
                        aucs.append(met['auc_ipcw']); briers.append(met['brier_ipcw'])
                    rows.append({'genes':k,'alpha':alpha,'gamma':gamma,'logit_cap':cap,'cv_auc_mean':np.mean(aucs),
                                 'cv_brier_mean':np.mean(briers),**{f'fold{i+1}_auc':v for i,v in enumerate(aucs)}})
        print('Finished residual search for',k,'genes',flush=True)
    cv=pd.DataFrame(rows); cv.to_csv(RUN/'development_cv.csv',index=False)
    selections={}
    for name,subset in [('anchor_residual',cv[cv.gamma==0]),('anchor_consistency',cv[cv.gamma>0])]:
        selections[name]=subset.sort_values(['cv_auc_mean','cv_brier_mean'],ascending=[False,True]).iloc[0].to_dict()
    save_json('selected_parameters.json',selections)
    print('Selected using development only:',json.dumps(selections),flush=True)
    # Models frozen above; evaluate both prespecified variants and export the fitted model.
    a=sub(scan,dev); all_eval={k:np.concatenate([sub(scan,test)[k],meta[k]],axis=0) for k in scan}
    prepared=prepare_fold(a,all_eval,repeats)
    metrics=[]; ntest=len(test)
    clinicalp=expit(prepared['test_offset'])
    for cohort,data,sl in [('SCANB_holdout',sub(scan,test),slice(0,ntest)),('METABRIC_external',meta,slice(ntest,None))]:
        metrics.append({'cohort':cohort,'model':'clinical_anchor',**metric(data['t'],data['e'],clinicalp[sl],km_censor(data['t'],data['e']))})
    full_predictions={name:pd.DataFrame({'patient_id':data['id'],'os_days':data['t'],'os_event':data['e'],'clinical_anchor':clinicalp[sl]})
                      for name,data,sl in [('SCANB_holdout',sub(scan,test),slice(0,ntest)),('METABRIC_external',meta,slice(ntest,None))]}
    fitted={'clinical_anchor':prepared['anchor'],'shared_gene_symbols':genes.Hugo_Symbol.to_list(),'scaler':prepared['scaler'],
            'gene_indices':prepared['order'],'variants':{}}
    evaluation_variants=dict(selections)
    evaluation_variants['consistency_removed_same_parameters']={**selections['anchor_consistency'],'gamma':0.}
    for name,params in evaluation_variants.items():
        p,theta,info=fit_residual(prepared,int(params['genes']),params['alpha'],params['gamma'],params['logit_cap'])
        fitted['variants'][name]={'parameters':params,'coefficients':theta}
        selected_genes=genes.iloc[prepared['order'][:len(theta)]].copy(); selected_genes['residual_coefficient']=theta
        selected_genes.to_csv(RUN/f'{name}_genes.csv',index=False)
        save_json(f'{name}_fit.json',info)
        for cohort,data,sl in [('SCANB_holdout',sub(scan,test),slice(0,ntest)),('METABRIC_external',meta,slice(ntest,None))]:
            full_predictions[cohort][name]=p[sl]
            metrics.append({'cohort':cohort,'model':name,**metric(data['t'],data['e'],p[sl],km_censor(data['t'],data['e']))})
    for name,df in full_predictions.items(): df.to_csv(RUN/f'predictions_{name}.csv',index=False)
    pd.DataFrame(metrics).to_csv(RUN/'metrics_3year.csv',index=False)
    joblib.dump(fitted,RUN/'fitted_models.joblib')
    save_json('training_replicate_ids.json',prepared['replicate_patients'].tolist())
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)

if __name__=='__main__':main()
