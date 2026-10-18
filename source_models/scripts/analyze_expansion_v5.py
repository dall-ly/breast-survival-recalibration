"""Patient-matched aggregation of all v5 branches with paired bootstrap."""
from survival_experiments import ROOT, SEED, labels, metric, km_censor
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN=ROOT/'runs/gse96058_os_v5_expansion'
MODELS={
 'Surv_consistency':'Original CA-RRL', 'RSF_clinical':'Clinical RSF (5)', 'RSF_fusion':'Gene fusion RSF',
 'fixed3y':'Fixed-horizon CA-RRL',
 'program_hallmark':'Program CA-RRL (5)', 'program_hallmark_context':'Program interactions (5)',
 'node_hallmark':'Program CA-RRL (6)', 'node_hallmark_context':'Program interactions (6)',
 'clinical_node_burden_clinical':'Clinical RSF (6)', 'clinical_node_burden_residual':'Gene CA-RRL (6)',
 'XGB_COX_fusion':'XGB Cox + genes','XGB_AFT_fusion':'XGB AFT + genes',
 'XGB_COX_Hallmark':'XGB Cox + programs','XGB_AFT_Hallmark':'XGB AFT + programs',
 'blend_hallmark_blend':'Program/horizon mixture (5)',
}

def km_fast(t,e):
    u,inv,cnt=np.unique(t,return_inverse=True,return_counts=True)
    cens=np.bincount(inv,weights=e==0,minlength=len(u));risk=np.cumsum(cnt[::-1])[::-1]
    return u,np.cumprod(1-cens/risk)

def attach(base,other,columns,prefix=''):
    other=other.set_index('patient_id').loc[base.index]
    assert other.index.is_unique
    assert np.array_equal(base.os_days.to_numpy(),other.os_days.to_numpy())
    assert np.array_equal(base.os_event.to_numpy(),other.os_event.to_numpy())
    for col in columns:
        name=prefix+col;assert name not in base
        base[name]=other[col].to_numpy()

def data(cohort):
    base=pd.read_csv(ROOT/'runs/gse96058_os_v3_survival'/f'predictions_{cohort}.csv').query('year==3').set_index('patient_id')
    base=base[['os_days','os_event','Surv_consistency','RSF_clinical','RSF_fusion']]
    old=pd.read_csv(ROOT/'runs/gse96058_os_v2_anchor'/f'predictions_{cohort}.csv').rename(columns={'anchor_consistency':'fixed3y'})
    attach(base,old,['fixed3y'])
    specs=[('program_ranking','program_', ['genes30','hallmark','hybrid','hallmark_context','selected_extension']),
           ('program_node','node_',['genes30','hallmark','hybrid','hallmark_context','selected_extension']),
           ('program_blend','blend_',['genes30_blend','hallmark_blend','hybrid_blend','hallmark_context_blend','selected_blend']),
           ('modern_baseline','',['XGB_AFT_clinical','XGB_AFT_fusion','XGB_COX_clinical','XGB_COX_fusion']),
           ('modern_baseline/programs','',['XGB_AFT_Hallmark','XGB_COX_Hallmark']),
           ('program_node_ablation','node_ablation_',['program_without_ranking','program_without_repeats','program_ridge_only'])]
    for folder,prefix,cols in specs:
        f=pd.read_csv(RUN/folder/f'predictions_{cohort}.csv').query('year==3')
        attach(base,f,cols,prefix)
    f=pd.read_csv(RUN/'clinical_expansion/evaluation_predictions.csv')
    for branch in ['base','node_burden','node_receptors']:
        ff=f[(f.cohort==cohort)&(f.branch==branch)&(f.year==3)]
        attach(base,ff,['clinical','residual']+(['selected_gate'] if branch=='base' else []),'clinical_'+branch+'_')
    assert base.notna().all().all()
    return base

def main():
    rng=np.random.default_rng(SEED+510); rows=[]; differences=[]; combined=[];checks=[]
    for cohort in ['SCANB_holdout','METABRIC_external']:
        d=data(cohort);names=[n for n in d if n not in ['os_days','os_event']]
        t,e=d.os_days.to_numpy(),d.os_event.to_numpy();p=d[names].to_numpy()
        assert np.all((p>=0)&(p<=1))
        legacy=km_censor(t,e);fast=km_fast(t,e)
        assert np.array_equal(legacy[0],fast[0]) and np.max(np.abs(legacy[1]-fast[1]))<1e-14
        boot=np.empty((1000,len(names)))
        for i in range(len(boot)):
            ix=rng.integers(len(d),size=len(d));yy,ww=labels(t[ix],e[ix],km_fast(t[ix],e[ix]));ok=ww>0
            for j in range(len(names)):boot[i,j]=roc_auc_score(yy[ok],p[ix,j][ok],sample_weight=ww[ok])
        np.save(RUN/f'bootstrap_auc_{cohort}.npy',boot)
        mets={n:metric(t,e,p[:,j],fast) for j,n in enumerate(names)}
        for j,name in enumerate(names):
            lo,hi=np.quantile(boot[:,j],[.025,.975]);m=mets[name]
            rows.append({'cohort':cohort,'model':name,'label':MODELS.get(name,name),**m,'auc_lower':lo,'auc_upper':hi})
            for ref in ['Surv_consistency','RSF_clinical','clinical_node_burden_clinical','clinical_node_burden_residual','program_hallmark']:
                diff=boot[:,j]-boot[:,names.index(ref)];lo,hi=np.quantile(diff,[.025,.975])
                differences.append({'cohort':cohort,'model':name,'reference':ref,'difference':m['auc_ipcw']-mets[ref]['auc_ipcw'],'lower':lo,'upper':hi})
        checks.extend([{'cohort':cohort,'check':'patient IDs/outcomes identical across all model files','passed':True,'n':len(d)},
                       {'cohort':cohort,'check':'fast KM agrees with original convention','passed':True}])
        d['cohort']=cohort;combined.append(d.reset_index())
        print('Bootstrap completed',cohort,len(names),'models',flush=True)
    c=pd.DataFrame(rows);c.to_csv(RUN/'comparison_3year.csv',index=False)
    pd.DataFrame(differences).to_csv(RUN/'paired_auc_differences.csv',index=False)
    pd.concat(combined).to_csv(RUN/'combined_3year_predictions.csv',index=False)
    nested={}
    for folder,prefix in [('program_ranking','program_'),('program_node','node_'),('program_blend','blend_'),('modern_baseline',''),('modern_baseline/programs','')]:
        f=pd.read_csv(RUN/folder/'nested_pooled_metrics.csv').query('year==3')
        nested.update({prefix+r.model:r.auc_ipcw for r in f.itertuples()})
    old=pd.read_csv(ROOT/'runs/gse96058_os_v3_survival/nested_pooled_metrics.csv').query('year==3')
    nested.update({r.model:r.auc_ipcw for r in old.itertuples()})
    table=[]
    for name in MODELS:
        f=c[c.model==name].set_index('cohort');s=f.loc['SCANB_holdout'];m=f.loc['METABRIC_external']
        table.append({'model':name,'label':MODELS[name],'nested_auc':nested.get(name,np.nan),
                      'SCANB_auc':s.auc_ipcw,'METABRIC_auc':m.auc_ipcw,'METABRIC_brier':m.brier_ipcw})
    pd.DataFrame(table).to_csv(RUN/'paper_main_table.csv',index=False)
    (RUN/'aggregation_audit.json').write_text(json.dumps({'checks':checks,'passed':True,'bootstrap_seed':SEED+510,'replicates':1000,
        'model_order':names,'interpretation':'Patient-paired fixed-model uncertainty; not corrected for repeated research or multiple comparisons'},indent=2),encoding='utf-8')
    plot(c)
    print(pd.DataFrame(table).to_string(index=False),flush=True)

def plot(c):
    selected=['RSF_clinical','clinical_node_burden_clinical','RSF_fusion','XGB_COX_fusion','XGB_AFT_Hallmark','Surv_consistency','clinical_node_burden_residual','program_hallmark','node_hallmark']
    fig,axes=plt.subplots(1,2,figsize=(11.5,5.3),sharey=True)
    for ax,cohort,title in zip(axes,['SCANB_holdout','METABRIC_external'],['SCAN-B research holdout','METABRIC external cohort']):
        f=c[c.cohort==cohort].set_index('model')
        for i,n in enumerate(selected):
            r=f.loc[n];color='#0d8278' if n=='node_hallmark' else '#60788c'
            ax.errorbar(r.auc_ipcw,i,xerr=[[r.auc_ipcw-r.auc_lower],[r.auc_upper-r.auc_ipcw]],fmt='o',color=color,capsize=3)
            ax.text(.99,i,f'{r.auc_ipcw:.3f}',transform=ax.get_yaxis_transform(),va='center',ha='right',fontsize=9)
        ax.set(xlim=(.52,.94),ylim=(len(selected)-.5,-.5),title=title,xlabel='Three-year IPCW AUC',yticks=range(len(selected)),yticklabels=[MODELS[n] for n in selected]);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(RUN/'auc_comparison.png',dpi=300);fig.savefig(RUN/'auc_comparison.svg');plt.close(fig)

if __name__=='__main__':main()
