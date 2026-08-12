import numpy as np
RNG=np.random.default_rng(11)

def mse(T,K,plo,phi,tau,kind,trials=80):
    """Time-averaged MSE of a windowed estimate of a jumping Bernoulli rate."""
    out=[]
    L=T//K
    for _ in range(trials):
        seg=np.repeat(RNG.choice([plo,phi],size=K),L)[:T]
        x=(RNG.random(T)<seg).astype(float)
        if kind=="rect":
            w=np.ones(int(tau))
        else:                                   # exponential kernel, same L1 mass
            k=np.arange(0,int(10*tau)); w=np.exp(-k/tau)
        w=w/w.sum()
        est=np.convolve(x,w)[:T]
        burn=min(int(10*tau),T//4)
        out.append(np.mean((est[burn:]-seg[burn:])**2))
    return np.mean(out)

def oracle(T,K,plo,phi,trials=80):
    L=T//K; out=[]
    for _ in range(trials):
        seg=RNG.choice([plo,phi],size=K)
        x=(RNG.random((K,L))<seg[:,None]).astype(float)
        out.append(np.mean((x.mean(1)-seg)**2))
    return np.mean(out)

print("K=8, plo=0.02, phi=0.08   (rect / exp kernels)")
print(f"{'T':>8} {'tau*':>7} {'MSE_win':>11} {'MSE_reset':>11} {'ratio':>8}")
rows=[]
for T in [4000,8000,16000,32000,64000]:
    taus=np.unique(np.round(np.logspace(0.7,np.log10(T/8),22)).astype(int))
    vals=[mse(T,8,0.02,0.08,t,"rect") for t in taus]
    i=int(np.argmin(vals)); mo=oracle(T,8,0.02,0.08)
    rows.append((T,taus[i],vals[i],mo))
    print(f"{T:8d} {taus[i]:7d} {vals[i]:11.3e} {mo:11.3e} {vals[i]/mo:8.2f}")

R=np.array(rows,float)
for name,col,pred in [("tau*",1,0.5),("MSE_win",2,-0.5),("MSE_reset",3,-1.0),
                      ("ratio",None,0.5)]:
    y=R[:,col] if col else R[:,2]/R[:,3]
    s=np.polyfit(np.log(R[:,0]),np.log(y),1)[0]
    print(f"  d log {name:9s}/d log T = {s:+.3f}   (predicted {pred:+.3f})")
