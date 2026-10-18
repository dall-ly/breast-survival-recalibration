"""IPCW recalibration, including explicit non-finite MLE handling."""
import numpy as np
from scipy.optimize import minimize, brentq
TAU=1095.

def weights(t,e):
    # Independent product-limit implementation, with the frozen tie convention.
    times,counts=np.unique(t,return_counts=True)
    risk=len(t)-np.r_[0,np.cumsum(counts[:-1])]
    cens_times,cens_count=np.unique(t[e==0],return_counts=True)
    c=np.zeros(len(times));c[np.searchsorted(times,cens_times)]=cens_count
    g=np.cumprod((risk-c)/risk)
    left=np.r_[1.,g[:-1]]
    case=(t<=TAU)&(e==1);control=t>TAU
    control|=(t==TAU)&(e==0)
    horizon=np.searchsorted(times,TAU,side='right')-1
    gt=g[horizon] if horizon>=0 else 1.
    w=np.zeros(len(t));w[case]=1/left[np.searchsorted(times,t[case])];w[control]=1/gt
    return case.astype(float),w

def objective(theta, z, y, w):
    eta=theta[0]+theta[1]*z
    if np.max(eta)>700:return np.inf,np.zeros(2)
    with np.errstate(over='ignore',under='ignore',divide='ignore',invalid='ignore'):
        h=np.exp(eta);event=y==1
        values=h.copy();derivative=h.copy()
        values[event]=-np.log(-np.expm1(-h[event]))
        derivative[event]=-h[event]*np.exp(-h[event])/(-np.expm1(-h[event]))
    if not np.isfinite(values).all() or not np.isfinite(derivative).all():return np.inf,np.zeros(2)
    v=float(w@values/w.sum());g=np.array([w@derivative,w@(derivative*z)])/w.sum()
    return v,g

def fit(p,t,e,intercept_reference):
    y,w=weights(t,e);active=w>0;y,w=y[active],w[active]
    x=np.log(-np.log1p(-np.clip(p[active],1e-10,1-1e-10)))
    if y.sum()==0:return -np.inf,1.,True,True,False,0.,'no_events'
    if y.sum()==len(y):return np.inf,1.,True,True,False,0.,'all_events'
    # Complete or quasi-complete positive separation has no finite MLE.
    if np.ptp(x)>1e-10 and x[y==1].min()>=x[y==0].max():
        return intercept_reference,1.,True,False,False,0.,'positive_separation_fallback'
    mean,sd=x.mean(),x.std()
    if sd<=1e-10:return intercept_reference,1.,True,False,False,0.,'constant_scores'
    z=(x-mean)/sd
    rate=float(w@y/w.sum())
    const_a=float(np.log(-np.log1p(-rate)))
    val0,g0=objective([const_a,0.],z,y,w)
    if g0[1]>=-1e-10:
        return const_a,0.,False,False,True,float(abs(g0[0])),'analytic_slope_boundary'
    fun=lambda theta:objective(theta,z,y,w)
    start=np.array([intercept_reference+mean,sd])
    res=minimize(fun,start,jac=True,method='L-BFGS-B',bounds=[(None,None),(0.,None)],options={'maxiter':1000,'gtol':1e-10,'ftol':1e-14,'maxls':50})
    def stationarity(theta):
        val,g=fun(theta)
        projected=np.array([g[0],min(0.,g[1]) if theta[1]<=1e-9 else g[1]])
        return val,float(np.max(abs(projected)))
    val,grad=stationarity(res.x)
    solver='L-BFGS-B'
    if not np.isfinite(val) or grad>1e-6:
        res=minimize(fun,start,jac=True,method='SLSQP',bounds=[(None,None),(0.,None)],options={'maxiter':1000,'ftol':1e-13})
        val,grad=stationarity(res.x);solver='SLSQP_recovery'
    if not np.isfinite(val) or grad>1e-6:
        return intercept_reference,1.,True,False,False,grad,'intercept_fallback'
    slope=float(res.x[1]/sd);a=float(res.x[0]-slope*mean)
    return a,slope,False,False,bool(res.x[1]<=1e-9),grad,solver

def fit_repaired_calibrators(p,t,e):
    """Portable API: method -> (intercept, slope, fallback, event_boundary).

    Uses only the supplied calibration patients. Includes the original fixed
    kappa=5 method and the audit-added monotone comparator. All two-parameter
    non-finite MLEs invoke the declared intercept-only fallback. It does not
    access project files, test labels, or select any tuning parameter.
    """
    p,t,e=np.asarray(p,float),np.asarray(t,float),np.asarray(e,float)
    yy,ww=weights(t,e);active=ww>0
    if not active.any():raise ValueError('No outcomes observable at the horizon in this calibration sample.')
    x=np.log(-np.log1p(-np.clip(p[active],1e-10,1-1e-10)))
    y,w=yy[active],ww[active]
    out={'unchanged':(0.,1.,False,False)}
    def score(a,penalty):
        return objective([a,1.],x,y,w)[1][0]*w.sum()+penalty*a
    if y.sum()==0:ml=(-np.inf,1.,False,True)
    elif y.sum()==len(y):ml=(np.inf,1.,False,True)
    else:ml=(float(brentq(lambda a:score(a,0.),-30.,10.,xtol=1e-11)),1.,False,False)
    out['intercept_ML']=ml
    shrink=float(brentq(lambda a:score(a,5.),-30.,10.,xtol=1e-11))
    out['shrink_intercept']=(shrink,1.,False,False)
    out['intercept_slope_ML']=(ml[0],1.,True,ml[3])
    separated=False
    if not ml[3] and np.ptp(x)>1e-10:
        separated=(x[y==1].min()>=x[y==0].max()) or (x[y==1].max()<=x[y==0].min())
    if not ml[3] and not separated and x.std()>1e-10:
        center,scale=x.mean(),x.std();z=(x-center)/scale
        fun=lambda theta:objective(theta,z,y,w)
        start=np.array([ml[0]+center,scale])
        res=minimize(fun,start,jac=True,method='L-BFGS-B',options={'maxiter':1000,'gtol':1e-10,'ftol':1e-14,'maxls':50})
        val,grad=fun(res.x)
        if not np.isfinite(val) or np.max(abs(grad))>1e-6:
            res=minimize(fun,start,jac=True,method='BFGS',options={'maxiter':1000,'gtol':1e-10})
            val,grad=fun(res.x)
        if np.isfinite(val) and np.max(abs(grad))<=1e-6:
            slope=float(res.x[1]/scale)
            out['intercept_slope_ML']=(float(res.x[0]-slope*center),slope,False,False)
    a,b,fallback,boundary,*_=fit(p,t,e,ml[0])
    out['intercept_slope_nonnegative_ML']=(a,b,fallback,boundary)
    return out

def km_risk(t, e, selected):
    s = 1.
    for tm in np.unique(t[(e == 1) & (t <= TAU) & (selected > 0)]):
        n = int(selected[t >= tm].sum())
        d = int(selected[(t == tm) & (e == 1)].sum())
        s *= 1. - d / n
    return 1. - s
