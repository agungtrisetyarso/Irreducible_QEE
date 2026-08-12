import numpy as np
RNG=np.random.default_rng(3)
# Global (non-local) decoder: reset to nearest codeword each step. Should scale.
def lifetime(n,p,trials=4000,max_steps=2_000_000):
    e=np.zeros((trials,n),np.int8); alive=np.ones(trials,bool)
    tf=np.full(trials,max_steps,np.int64)
    for t in range(1,max_steps+1):
        a=np.flatnonzero(alive)
        if a.size==0: break
        ea=e[a]^(RNG.random((a.size,n))<p)
        w=ea.sum(1); bad=w>n/2
        d=a[bad]
        if d.size: tf[d]=t; alive[d]=False
        ea[~bad]=0
        e[a]=ea
    return tf[tf<max_steps].mean() if (tf<max_steps).any() else max_steps
for n in [5,7,9,11]:
    print(n, f"{lifetime(n,0.10):12.0f}")
