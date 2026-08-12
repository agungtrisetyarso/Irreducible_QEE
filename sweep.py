"""
Go/no-go experiment: does the sub-threshold failure exponent alpha depend on the
memory horizon tau, and only when the syndrome is unreliable (q > 0)?

Model
-----
1D repetition code, ring of n bits, one encoded bit.
Per step:
  1. data errors: e[i] ^= Bern(p)
  2. true syndrome on links: s[i] = e[i] ^ e[i+1]
  3. measured syndrome: stilde = s ^ Bern(q)
  4. local rule at bit i, evidence from its two incident links:
         f_i = stilde[i-1] + stilde[i]  in {0,1,2}
         M_i <- lam * M_i + (f_i - 1)          [ +1 / 0 / -1 ]
         if M_i > theta:  e[i] ^= 1 ; M_i <- 0
     lam = 1 - 1/tau, so tau = 1 gives lam = 0 (MEMORYLESS: flip iff both
     incident links fire) and tau > 1 is a leaky integrator in H_1.

Everything is a fixed nonnegative kernel + monotone readout: inside the class.
Nothing depends on the cell's own state, so GKL-type rules are not represented.

Failure: weight(e) >= n/2, i.e. the residual has drifted to the other codeword.
Memory lifetime T_fail; sub-threshold scaling log T_fail ~ alpha * n * log(1/p),
so alpha is read off the slope of log T_fail vs n (up to the fixed log(1/p)).
"""
import numpy as np

RNG = np.random.default_rng(20260812)


def lifetime(n, p, q, tau, theta=0.5, trials=400, max_steps=200_000):
    """Mean steps to logical failure, vectorised over trials."""
    lam = 0.0 if tau <= 1 else 1.0 - 1.0 / tau
    e = np.zeros((trials, n), dtype=np.int8)
    M = np.zeros((trials, n), dtype=np.float64)
    alive = np.ones(trials, dtype=bool)
    tfail = np.full(trials, max_steps, dtype=np.int64)
    half = n / 2.0

    for t in range(1, max_steps + 1):
        a = np.flatnonzero(alive)
        if a.size == 0:
            break
        ea, Ma = e[a], M[a]

        ea ^= (RNG.random(ea.shape) < p)                      # data errors
        s = ea ^ np.roll(ea, -1, axis=1)                       # link i joins i,i+1
        if q > 0:
            s = s ^ (RNG.random(s.shape) < q)                  # readout errors
        f = s.astype(np.int8) + np.roll(s, 1, axis=1).astype(np.int8)  # links i-1, i

        Ma *= lam
        Ma += (f - 1)
        flip = Ma > theta
        ea ^= flip
        Ma[flip] = 0.0

        e[a], M[a] = ea, Ma
        dead = a[ea.sum(axis=1) >= half]
        if dead.size:
            tfail[dead] = t
            alive[dead] = False

    return tfail.mean(), (tfail == max_steps).mean()


def exponent(ns, logT):
    """Slope of log T_fail vs n."""
    A = np.vstack([ns, np.ones_like(ns)]).T
    slope, _ = np.linalg.lstsq(A, logT, rcond=None)[0]
    return slope


if __name__ == "__main__":
    p = 0.02
    ns = np.array([7, 11, 15, 19, 23], dtype=float)
    taus = [1, 2, 4, 8, 16, 32]

    for q in (0.0, 0.02):
        print(f"\n=== q = {q}  (p = {p}) ===")
        print(f"{'tau':>5} {'slope d(logT)/dn':>18}   " +
              "  ".join(f"n={int(n)}" for n in ns))
        for tau in taus:
            row, cens = [], []
            for n in ns:
                mt, c = lifetime(int(n), p, q, tau)
                row.append(mt)
                cens.append(c)
            sl = exponent(ns, np.log(row))
            flag = " *censored*" if max(cens) > 0.02 else ""
            print(f"{tau:>5} {sl:>18.4f}   " +
                  "  ".join(f"{v:9.0f}" for v in row) + flag)
