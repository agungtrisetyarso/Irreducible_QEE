import numpy as np
RNG = np.random.default_rng(7)

def lifetime(n,p,q,tau,rule,theta=0.5,trials=400,max_steps=200_000):
    lam = 0.0 if tau<=1 else 1.0-1.0/tau
    e=np.zeros((trials,n),dtype=np.int8); M=np.zeros((trials,n))
    alive=np.ones(trials,bool); tf=np.full(trials,max_steps,np.int64); half=n/2.0
    for t in range(1,max_steps+1):
        a=np.flatnonzero(alive)
        if a.size==0: break
        ea,Ma=e[a],M[a]
        ea^=(RNG.random(ea.shape)<p)
        s=ea^np.roll(ea,-1,axis=1)
        if q>0: s=s^(RNG.random(s.shape)<q)
        f=s.astype(np.int8)+np.roll(s,1,axis=1).astype(np.int8)
        if rule=="symmetric":
            Ma*=lam; Ma+=(f-1); flip=Ma>theta; Ma[flip]=0.0
        elif rule=="diffusive":              # coin-flip tie-break at f==1
            flip=(f==2)|((f==1)&(RNG.random(f.shape)<0.5))
        e[a],M[a]=ea^flip,Ma
        d=a[(ea^flip).sum(axis=1)>=half]
        if d.size: tf[d]=t; alive[d]=False
    return tf.mean()

ns=[7,11,15,19,23]
print("control: diffusive tie-break, memoryless, q=0, p=0.02")
print("  n   T_fail")
for n in ns: print(f"{n:3d} {lifetime(n,0.02,0.0,1,'diffusive'):9.0f}")
