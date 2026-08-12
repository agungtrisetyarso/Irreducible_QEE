import numpy as np
RNG=np.random.default_rng(5)
# w: nonincreasing probability weights.  Exact functionals of the estimator:
#   tau_var = 1/sum w^2                (effective sample size; Var = v/tau_var)
#   tau_bias = sum_s T(s)^2, T(s)=sum_{u>s} w(u)   (post-jump squared-bias mass)
def funcs(w):
    w=w/w.sum()
    T=np.cumsum(w[::-1])[::-1]          # T[s] = sum_{u>=s} w
    T=np.append(T[1:],0.0)              # shift: tail strictly beyond s
    return 1/np.sum(w**2), np.sum(T**2)

print("canonical families:  tau_var   tau_bias   ratio")
for N in [2,4,8,32,128,512]:
    tv,tb=funcs(np.ones(N)); print(f"  rect N={N:4d}   {tv:9.2f} {tb:9.2f} {tb/tv:7.4f}")
for r in [0.5,0.9,0.99,0.999]:
    u=np.arange(0,int(40/(1-r))); tv,tb=funcs((1-r)*r**u)
    print(f"  exp r={r:<6}  {tv:9.2f} {tb:9.2f} {tb/tv:7.4f}")

# search for the worst case: minimise tau_bias/tau_var over nonincreasing w
best={}
for trial in range(300000):
    L=RNG.integers(2,60)
    w=np.sort(RNG.random(L))[::-1]**RNG.uniform(0.2,6)
    tv,tb=funcs(w)
    b=int(np.floor(np.log2(max(tv,1.0001))))
    if b not in best or tb/tv<best[b][0]: best[b]=(tb/tv,tv,tb)
print("\nworst observed ratio, binned by tau_var:")
for b in sorted(best):
    r,tv,tb=best[b]; print(f"  tau_var in [2^{b},2^{b+1}):  min tau_bias/tau_var = {r:.4f}  (tau_var={tv:.2f})")

# Test the conjectured sharp inequality  tau_bias >= (tau_var-1)^2/(4 tau_var)
worst=1e9; arg=None
for trial in range(600000):
    L=RNG.integers(2,80)
    style=RNG.integers(0,3)
    if style==0: w=np.sort(RNG.random(L))[::-1]**RNG.uniform(0.1,8)
    elif style==1: w=np.sort(RNG.exponential(1,L))[::-1]
    else:
        k=RNG.integers(1,L); w=np.concatenate([np.full(k,RNG.uniform(1,50)),np.ones(L-k)])
    tv,tb=funcs(w)
    if tv<=1.0001: continue
    ratio=tb/((tv-1)**2/(4*tv))
    if ratio<worst: worst,arg=ratio,(tv,tb)
print(f"\nmin over 600k kernels of tau_bias / [(tau_var-1)^2/(4 tau_var)] = {worst:.4f}")
print(f"  attained near tau_var={arg[0]:.3f}, tau_bias={arg[1]:.4f}")
