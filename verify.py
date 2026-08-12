import numpy as np
RNG=np.random.default_rng(23)

def beta_exact(w,L):
    w=np.asarray(w,float); w=w/w.sum(); n=len(w); rho=1-2.0/L
    u=np.arange(n); du=np.abs(u[:,None]-u[None,:]); lo=np.minimum(u[:,None],u[None,:])
    M=((1-rho**lo)/2.0)*((1+rho**du)/2.0)
    return float(w@M@w)
def tau_var(w):
    w=np.asarray(w,float); w=w/w.sum(); return 1/np.sum(w**2)

# --- does v4's Step 2 claim "sum_u w chi_u = T(U)" hold?  It ignores lags u>M,
# whose chi is NOT determined by the conditioning event.  Concretely: a kernel
# whose mass sits just inside and just beyond m2 makes T(U)-T(M) ~ 0.
L=2000
w=np.zeros(4*L); w[0:5]=0.75/5; w[300:305]=0.2499/5; w[3000:3005]=0.0001/5
Tl=np.cumsum(w[::-1])[::-1]; Tl=np.append(Tl[1:],0.0)
m1=int(np.argmax(Tl<=0.75)); m2=int(np.argmax(Tl<=0.25))
print(f"v4 quartile lags: m1={m1}, m2={m2};  T(m1)={Tl[m1]:.4f}, T(m2)={Tl[m2]:.6f}")
print(f"  for U just below m2, T(U)-T(M) = {Tl[m2-1]-Tl[m2]:.6f}  <<  1/4 claimed by v4")

# --- corrected bound: beta >= (min(tau_var,L)-32)_+/(256 e L)
def bound(w,L): return max(min(tau_var(w),L)-32,0)/(256*np.e*L)
print("\ncanonical families:      tau_var     beta      beta/bound")
for N in [10,50,200,800]:
    w=np.ones(N); print(f"  rect N={N:4d}   {tau_var(w):9.1f} {beta_exact(w,L):9.5f}  {beta_exact(w,L)/max(bound(w,L),1e-30):9.1f}")
for r in [0.9,0.99,0.995]:
    u=np.arange(int(30/(1-r))); w=(1-r)*r**u
    print(f"  exp r={r:<6}  {tau_var(w):9.1f} {beta_exact(w,L):9.5f}  {beta_exact(w,L)/max(bound(w,L),1e-30):9.1f}")

worst=1e9; arg=None
for _ in range(6000):
    n=RNG.integers(2,1500); s=RNG.integers(0,4)
    if s==0: w=RNG.random(n)**RNG.uniform(0.2,6)
    elif s==1: w=np.sort(RNG.random(n))[::-1]
    elif s==2:
        w=np.zeros(n); k=RNG.integers(1,n); w[RNG.choice(n,k,replace=False)]=RNG.random(k)
    else:
        w=np.zeros(n); q=RNG.integers(1,4)
        for _q in range(q):
            a=RNG.integers(0,n); w[a:a+RNG.integers(1,20)]=RNG.random()
    if w.sum()<=0: continue
    b=bound(w,L)
    if b<=0: continue
    r=beta_exact(w,L)/b
    if r<worst: worst,arg=r,(tau_var(w),beta_exact(w,L))
print(f"\nmin over 6000 kernels of beta/bound = {worst:.2f}  (claim >=1) at tau_var={arg[0]:.1f}")
