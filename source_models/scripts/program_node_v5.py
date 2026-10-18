"""Same program/ranking learner with one additional clinical node-burden feature.

Usage: python program_node_v5.py [--nested]
The underlying program_ranking_v5 module is not modified. Its fixed anchor
recipe and 36-candidate grid are preserved, with local folds and parameters.
"""
import numpy as np
import pandas as pd
import program_ranking_v5 as pg

_load=pg.load_data
_save=pg.save
pg.WORK=pg.ROOT/'runs/gse96058_os_v5_expansion/program_node'
pg.WORK.mkdir(parents=True,exist_ok=True)
pg.ANCHOR={'genes':0,'leaf':30,'features':'sqrt'}
FEATURES=['age','size','grade','node_positive','er_positive','node_ge4']

def load_node_data():
    scan,meta,dev,test,strata,repeats,genes=_load()
    source=pg.WORK.parent/'clinical_expansion'
    for cohort,data in [('SCANB',scan),('METABRIC',meta)]:
        table=pd.read_csv(source/f'{cohort}_node_burden_features.csv').set_index('patient_id')
        assert table.index.is_unique
        clinical=table.loc[data['id'],FEATURES].to_numpy()
        assert np.allclose(clinical[:,:5],data['clinical'],equal_nan=True)
        assert np.isin(clinical[np.isfinite(clinical[:,5]),5],[0.,1.]).all()
        data['clinical']=clinical
    return scan,meta,dev,test,strata,repeats,genes

def save(name,obj):
    if name=='protocol.json':
        obj={**obj,'clinical_features':FEATURES,
             'additional_variable':'Node count >=4 from harmonized original metadata; no new receptor proxies.',
             'clinical_anchor_fixed':pg.ANCHOR,
             'feature_extension_design':'Same algorithm/grid/folds as program_ranking_v5; one extra clinical measurement.',
             'external_requirements':'Requires node burden >=4. Cohorts offering only node positive/negative cannot evaluate this branch without missing-variable transport.'}
    _save(name,obj)

pg.load_data=load_node_data
pg.save=save

if __name__=='__main__':
    pg.main()
