from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf,pacf,adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera, shapiro

INPUT = Path(__file__).with_name('ibmclose.csv')
OUT = Path(__file__).with_name('hw3_ibm_outputs')
OUT.mkdir(parents=True,exist_ok=True)
raw = pd.read_csv(INPUT)['x'].to_numpy(float)
n=len(raw)
series = {'log': np.log(raw), 'sqrt': np.sqrt(raw), 'square':raw**2}
scale={'log':100.,'sqrt':10.,'square':1e-4}

fig,axs=plt.subplots(3,1,figsize=(12,9))
axs[0].plot(np.arange(1,n+1),raw,linewidth=1.1)
axs[0].set(title=f'IBM closing prices (n={n})',xlabel='Time',ylabel='Price')
axs[1].stem(np.arange(0,31),acf(raw,nlags=30,fft=True),linefmt='C0-',markerfmt='C0o',basefmt='k-')
axs[1].axhline(1.96/np.sqrt(n),color='gray',ls='--');axs[1].axhline(-1.96/np.sqrt(n),color='gray',ls='--')
axs[1].set(title='ACF of IBM closing prices',xlabel='Lag')
axs[2].stem(np.arange(0,31),pacf(raw,nlags=30,method='ywm'),linefmt='C0-',markerfmt='C0o',basefmt='k-')
axs[2].axhline(1.96/np.sqrt(n),color='gray',ls='--');axs[2].axhline(-1.96/np.sqrt(n),color='gray',ls='--')
axs[2].set(title='PACF of IBM closing prices',xlabel='Lag')
fig.tight_layout();fig.savefig(OUT/'01_original_acf_pacf.png',dpi=170);plt.close(fig)

fig,axs=plt.subplots(3,1,figsize=(12,9),sharex=True)
diags=[]
for ax,(name,v) in zip(axs,series.items()):
    d=np.diff(v)*scale[name]
    blocks=np.array_split(d,6)
    sds=np.array([np.std(b,ddof=1) for b in blocks])
    diags.append({'transform':name,'std_cv_across_6_blocks':np.std(sds,ddof=1)/np.mean(sds),
                  'adf_p':adfuller(d,autolag='AIC')[1], 'scale_for_fitting':scale[name],
                  'diff_std_on_original_scale':np.std(np.diff(v),ddof=1)})
    ax.plot(np.arange(2,n+1),d,lw=0.85)
    ax.set_ylabel(name+' diff, scaled');ax.grid(alpha=.2)
axs[0].set_title('First differences after the three required transformations')
axs[-1].set_xlabel('Time')
fig.tight_layout();fig.savefig(OUT/'02_transformed_differences.png',dpi=170);plt.close(fig)
pd.DataFrame(diags).to_csv(OUT/'transformation_diagnostics.csv',index=False)

rows=[];fits={}
for name,v in series.items():
    d=np.diff(v)*scale[name]
    for p in range(4):
        for q in range(4):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    result=ARIMA(d,order=(p,0,q),trend='c',enforce_stationarity=True,
                                 enforce_invertibility=True).fit(method_kwargs={'maxiter':300})
                k=len(result.params)
                converged=result.mle_retvals.get('converged',True)
                rows.append({'transform':name,'p':p,'q':q,'aic':result.aic,'bic':result.bic,'aicc':result.aicc,
                             'llf':result.llf,'k':k,'converged':converged})
                fits[(name,p,q,'with_mean')]=result
            except Exception as e:
                print('FAILED',name,p,q,str(e)[:120])
orders=pd.DataFrame(rows)
orders.to_csv(OUT/'model_grid.csv',index=False)
winners=[]
for name in series:
    dd=orders[(orders['transform']==name)&orders['converged']]
    for criterion in ['aic','bic','aicc']:
        win=dd.loc[dd[criterion].idxmin()].to_dict()
        win['criterion']=criterion
        winners.append(win)
win_df=pd.DataFrame(winners)
win_df.to_csv(OUT/'selected_orders.csv',index=False)

rows=[]
for name in series:
    best=win_df[(win_df['transform']==name)&(win_df['criterion']=='aic')].iloc[0]
    p,q=int(best.p),int(best.q)
    fit1=fits[(name,p,q,'with_mean')]
    d=np.diff(series[name])*scale[name]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit0=ARIMA(d,order=(p,0,q),trend='n',enforce_stationarity=True,
                   enforce_invertibility=True).fit(method_kwargs={'maxiter':300})
    fits[(name,p,q,'no_mean')]=fit0
    for label,fit in [('with_mean',fit1),('no_mean',fit0)]:
        mu=float(fit.params[0]) if label=='with_mean' else 0.
        ci=fit.conf_int()[0] if label=='with_mean' else [np.nan,np.nan]
        rows.append({'transform':name,'p':p,'q':q,'specification':label,'aic':fit.aic,'bic':fit.bic,
                     'aicc':fit.aicc,'mean_on_scaled_diff':mu, 'mean_CI_lower':ci[0], 'mean_CI_upper':ci[1]})
meancomp=pd.DataFrame(rows)
meancomp.to_csv(OUT/'mean_comparison.csv',index=False)

diag_rows=[]
for name in series:
    best=win_df[(win_df['transform']==name)&(win_df.criterion=='aic')].iloc[0]
    p,q=int(best.p),int(best.q)
    # For residual diagnostics, use the mean specification favored by AICc.
    cc=meancomp[meancomp['transform']==name]
    label=cc.loc[cc.aicc.idxmin(),'specification']
    fit=fits[(name,p,q,label)]
    resid=np.asarray(fit.resid)
    std_resid=resid / np.sqrt(float(fit.params[-1]))
    lj=acorr_ljungbox(std_resid,lags=[10,20],model_df=p+q,return_df=True)
    l10=lj.iloc[0]['lb_pvalue'];l20=lj.iloc[1]['lb_pvalue']
    jb=jarque_bera(std_resid)
    diag_rows.append({'transform':name,'p':p,'q':q,'mean_spec':label,'lag10_LB_p':l10,'lag20_LB_p':l20,
                      'JB_p':jb.pvalue, 'standardized_residual_sd':np.std(std_resid,ddof=1),
                      'abs_standardized_resid_gt_3':int(np.sum(np.abs(std_resid)>3))})
    from scipy import stats
    fig,axs=plt.subplots(3,1,figsize=(11,9))
    axs[0].plot(std_resid,lw=.9);axs[0].axhline(0,color='k',ls='--')
    axs[0].set(title=f'{name}: ARMA({p},{q}), {label}, standardized residuals',ylabel='Standardized residual')
    axs[1].stem(np.arange(0,31),acf(std_resid,nlags=30,fft=True),linefmt='C0-',markerfmt='C0o',basefmt='k-')
    axs[1].axhline(1.96/np.sqrt(len(resid)),color='gray',ls='--');axs[1].axhline(-1.96/np.sqrt(len(resid)),color='gray',ls='--')
    axs[1].set(title='Residual ACF',xlabel='Lag')
    stats.probplot(std_resid,dist='norm',plot=axs[2]);axs[2].set_title('Residual normal QQ plot')
    fig.tight_layout();fig.savefig(OUT/f'03_{name}_diagnostics.png',dpi=170);plt.close(fig)
pd.DataFrame(diag_rows).to_csv(OUT/'model_diagnostics.csv',index=False)

with open(OUT/'results_summary.txt','w',encoding='utf8') as f:
    def pr(x):
        print(x);print(x,file=f)
    pr('IBM Homework Problem 4 | n='+str(n))
    pr('Level ADF test p = %.6g (null: unit root)'%adfuller(raw,autolag='AIC')[1])
    pr('\nTransformation first-difference stability (CV of block SDs):\n'+pd.DataFrame(diags).round(5).to_string(index=False))
    pr('\nARMA grid p,q=0,...,3 with mean (criteria are within-scale only):')
    pr(win_df[['transform','criterion','p','q','aic','bic','aicc']].round(5).to_string(index=False))
    pr('\nMean vs no mean for AIC-selected order:')
    pr(meancomp.round(6).to_string(index=False))
    pr('\nResidual diagnostics (AICc-preferred mean specification):')
    pr(pd.DataFrame(diag_rows).round(6).to_string(index=False))
    for name in series:
        best=win_df[(win_df['transform']==name)&(win_df.criterion=='aic')].iloc[0];p,q=int(best.p),int(best.q)
        for label in ['with_mean','no_mean']:
            fit=fits[(name,p,q,label)]
            pr(f'\n{name} ARMA({p},{q}) {label}: params {dict(zip(fit.param_names, fit.params.round(8)))}')
    pr('\nNOTE: Gaussian likelihood levels/AIC are not comparable across different transformations; use variance stability to pick transform and criteria to compare model orders within a transform.')
print('All outputs saved at',OUT)
