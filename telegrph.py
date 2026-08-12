import numpy as np
RNG=np.random.default_rng(17)

def beta_exact(w,L):
    """beta(w) = E[(sum_u w(u) chi_u)^2], chi_u = 1{odd # switches in last u rounds},
    telegraph drift with per-round switch probability 1/L."""
    w=np.asarray(w,float); w=w/w.sum(); n=len(w); rho=1-2.0/L
    u=np.arange(n); P=(1-rho**u)/2.0                      # E[chi_u]
    du=np.abs(u[:,None]-u[None,:])
    lo=np.minimum(u[:,None],u[None,:])
    M=((1-rho**lo)/2.0)*((1+rho**du)/2.0)                 # E[chi_u chi_u']
    return float(w@M@w), float(w@P)

def tau_var(w):
    w=np.asarray(w,float); w=w/w.sum(); return 1/np.sum(w**2)

L=2000
print(f"telegraph drift, L={L}     tau_var    beta     beta*L/tau_var   bound 1/(64e)={1/(64*np.e):.5f}")
for N in [10,50,200,800]:
    w=np.ones(N); b,_=beta_exact(w,L); tv=tau_var(w)
    print(f"  rect N={N:4d}        {tv:8.1f} {b:9.5f}   {b*L/tv:8.4f}")
for r in [0.9,0.99,0.995]:
    u=np.arange(int(30/(1-r))); w=(1-r)*r**u; b,_=beta_exact(w,L); tv=tau_var(w)
    print(f"  exp r={r:<6}       {tv:8.1f} {b:9.5f}   {b*L/tv:8.4f}")
# the adversarial comb that breaks the periodic-drift lemma is harmless here:
w=np.zeros(4*L)
for j in range(2): w[2*j*L:(2*j+1)*L]=1.0
b,_=beta_exact(w,L); print(f"  comb (period 2L)   {tau_var(w):8.1f} {b:9.5f}   {b*L/tau_var(w):8.4f}")

# stress test the claimed bound beta >= min(tau_var,L)/(64 e L)
worst=1e9
for _ in range(4000):
    n=RNG.integers(2,1500); style=RNG.integers(0,3)
    if style==0: w=RNG.random(n)**RNG.uniform(0.2,6)
    elif style==1: w=np.sort(RNG.random(n))[::-1]
    else:
        w=np.zeros(n); k=RNG.integers(1,n); w[RNG.choice(n,k,replace=False)]=RNG.random(k)
    if w.sum()<=0: continue
    b,_=beta_exact(w,L); tv=tau_var(w)
    worst=min(worst, b/(min(tv,L)/(64*np.e*L)))
print(f"\nmin over 4000 kernels of beta / [min(tau_var,L)/(64 e L)] = {worst:.2f}   (claim: >= 1)")

print("\ncorrected bound:  beta >= (min(tau_var,L) - 4)_+ / (64 e L)")
worst=1e9; arg=None
for _ in range(6000):
    n=RNG.integers(2,1500); style=RNG.integers(0,4)
    if style==0: w=RNG.random(n)**RNG.uniform(0.2,6)
    elif style==1: w=np.sort(RNG.random(n))[::-1]
    elif style==2:
        w=np.zeros(n); k=RNG.integers(1,n); w[RNG.choice(n,k,replace=False)]=RNG.random(k)
    else:
        w=np.zeros(n); w[0]=RNG.uniform(0,1); w[RNG.integers(1,n)]=1-w[0]
    if w.sum()<=0: continue
    b,_=beta_exact(w,L); tv=tau_var(w)
    denom=max(min(tv,L)-4,0)/(64*np.e*L)
    if denom<=0: continue
    if b/denom<worst: worst,arg=b/denom,(tv,b)
print(f"min over 6000 kernels of beta / bound = {worst:.2f}   (claim: >= 1)")
print(f"  attained at tau_var={arg[0]:.1f}, beta={arg[1]:.2e}")
