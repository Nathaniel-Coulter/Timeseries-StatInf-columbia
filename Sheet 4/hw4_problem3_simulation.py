from __future__ import annotations
import argparse
from pathlib import Path
import warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from statsmodels.tsa.arima.estimators.innovations import innovations_mle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_data(kind: str, n: int, seed: int):
    rng=np.random.default_rng(seed)
    e=rng.standard_normal(n+300)
    if kind=='MA2':
        z=lfilter([1.0,0.8,-0.1],[1.0],e)
    elif kind=='AR3':
        z=lfilter([1.0],[1.0,-0.8,0.2,0.1],e)
    else:
        raise ValueError(kind)
    return z[300:]


def fit_orders(item):
    kind,n,seed=item
    data=make_data(kind,n,seed)
    maxorder=4 if kind=='MA2' else 5
    ics={'AIC': [],'BIC': [],'AICc': []}
    failed=0
    for order in range(maxorder+1):
        p,q=(0,order) if kind=='MA2' else (order,0)
        k=order+2  # intercept/mean + AR or MA coefficients + sigma^2
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                params,extra=innovations_mle(data,order=(p,0,q),demean=True,
                                  minimize_kwargs={'options':{'maxiter':100}})
            minus_ll=float(extra.minimize_results.fun)
            if not np.isfinite(minus_ll):
                raise ArithmeticError('Nonfinite likelihood')
            aic=2*minus_ll+2*k
            bic=2*minus_ll+k*np.log(n)
            aicc=aic+2*k*(k+1)/(n-k-1)
        except Exception:
            aic=bic=aicc=np.inf
            failed+=1
        for name,v in [('AIC',aic),('BIC',bic),('AICc',aicc)]:
            ics[name].append(v)
    result={name:int(np.argmin(values)) for name,values in ics.items()}
    return (kind, result, failed)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--n',type=int,default=200,help='observations per simulated series')
    parser.add_argument('--reps',type=int,default=1000,help='datasets per generating model')
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--seed',type=int,default=5263)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'hw3_simulation_outputs')
    args=parser.parse_args()
    if args.n<30 or args.n<=7:
        raise ValueError('n must be at least 30')
    args.output.mkdir(parents=True,exist_ok=True)

    # Independent reproducible seeds for each series and both experiments
    seeds=np.random.SeedSequence(args.seed).generate_state(2*args.reps)
    items=[(kind,args.n,int(seeds[offset+i]))
           for offset,kind in [(0,'MA2'),(args.reps,'AR3')]
           for i in range(args.reps)]
    counts={kind:{ic:np.zeros(5 if kind=='MA2' else 6,dtype=int)
                  for ic in ['AIC','BIC','AICc']} for kind in ['MA2','AR3']}
    failures=0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for idx,(kind,chosen,failed) in enumerate(pool.map(fit_orders,items,chunksize=5),start=1):
            failures+=failed
            for criterion,order in chosen.items():
                counts[kind][criterion][order]+=1
            if idx % 250==0:
                print(f'Processed {idx}/{2*args.reps} data sets',flush=True)

    for kind in ['MA2','AR3']:
        rows=[]
        for order in range(len(counts[kind]['AIC'])):
            rows.append({'order':order,
                         **{criterion:counts[kind][criterion][order]/args.reps
                            for criterion in ['AIC','BIC','AICc']}})
        df=pd.DataFrame(rows)
        df.to_csv(args.output/f'{kind}_selection_probabilities.csv',index=False)
        print(f'\n{kind} selection probabilities (n={args.n}, simulations={args.reps}):')
        print(df.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
        ax=df.set_index('order').plot(kind='bar',figsize=(9,4),rot=0,title=f'{kind}: AIC, BIC, AICc selection probabilities')
        ax.set(ylabel='Probability',xlabel='Selected '+('q' if kind=='MA2' else 'p'))
        plt.tight_layout();plt.savefig(args.output/f'{kind}_selection.png',dpi=170);plt.close()
    print(f'\nTotal candidate-fit failures: {failures} (out of {args.reps*11} attempts)')
    print('Outputs:',args.output.resolve())

if __name__=='__main__':
    main()
