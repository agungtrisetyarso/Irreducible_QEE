#!/usr/bin/env python3
"""
Reproduces every new numerical result in v2 of

  "Calibration-limited quantum error correction: linear syndrome memory,
   a universal curvature at the Nishimori line, and an irreducible qubit
   overhead"

Sections produced:
  A  gauge invariance and matched exponent            -> Sec. VI E, checks 1
  B  universal curvature 1/8                          -> Table IV, Fig. 3(a)
  C  projection form with inhomogeneous rates         -> Sec. VI E, check 3
  D  tilted importance sampling of P_fail             -> Sec. VI F
  E  end-to-end drift -> window -> decay rate         -> Table V, Fig. 3(b)
  F  device-parameter overhead table                  -> Table III

Runtime: ~8 minutes single core.  Only numpy/scipy/matplotlib required.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.optimize import minimize_scalar

# ---------------------------------------------------------------- utilities

def wgt(p):
    """Matched decoder weight w = ln[(1-p)/p] = 2J."""
    return np.log((1.0 - p) / p)


def exponent(pvec, what):
    """Exact Cramer exponent per mechanism, Eq. (18) of the paper:

        kappa = (1/d) sup_{s>0} [ (s/2) sum_e what_e
                                  - sum_e ln(1 - p_e + p_e e^{s what_e}) ]

    for the two-coset failure event  sum_e what_e X_e > (1/2) sum_e what_e
    with X_e ~ Bern(p_e) independent.  The objective is strictly concave in s,
    so the 1-D bounded minimisation is exact to machine precision.
    """
    pvec = np.atleast_1d(np.asarray(pvec, float))
    what = np.asarray(what, float)
    d = what.size
    if pvec.size == 1:
        pvec = np.full(d, pvec[0])
    half = 0.5 * what.sum()

    def negg(s):
        z = s * what
        m = np.maximum(z, 0.0)                      # log-sum-exp stabilisation
        val = m + np.log((1 - pvec) * np.exp(-m) + pvec * np.exp(z - m))
        return -(s * half - val.sum()) / d

    return -minimize_scalar(negg, bounds=(1e-9, 80.0), method="bounded",
                            options={"xatol": 1e-13}).fun


def kappa0(p):
    """Matched (Bhattacharyya) exponent, Eq. (19)."""
    return -np.log(2.0 * np.sqrt(p * (1.0 - p)))


def perp_norm2(delta, w):
    """||delta_perp||^2, the component of delta orthogonal to w."""
    return float(delta @ delta - (delta @ w) ** 2 / (w @ w))


# ---------------------------------------------------------- A: invariances

def part_A(rng):
    print("=" * 72)
    print("A.  matched exponent, s* = 1, and exact gauge invariance")
    print("=" * 72)
    for p in (0.001, 0.01, 0.05, 0.1):
        w = np.full(40, wgt(p))
        print(f"   p={p:<7} kappa={exponent(p, w):.10f}   "
              f"-ln(2 sqrt(p(1-p)))={kappa0(p):.10f}")

    p, d = 0.02, 40
    w = np.full(d, wgt(p))
    base = exponent(p, w)
    print("   multiplicative rescaling  what -> c what:")
    for c in (0.3, 0.7, 1.0, 1.5, 4.0):
        print(f"      c={c:<5} kappa-kappa0 = {exponent(p, c * w) - base:+.3e}")
    print("   uniform additive shift    what -> what + const:")
    for c in (-0.5, -0.1, 0.1, 0.5):
        print(f"      shift={c:+.2f}  kappa-kappa0 = {exponent(p, w + c) - base:+.3e}")
    print()


# ------------------------------------------------- B: the universal 1/8

def part_B(rng):
    print("=" * 72)
    print("B.  universal curvature  (kappa0 - kappa) / Var_e(delta)   [-> 0.125]")
    print("=" * 72)
    d = 256
    sds = (0.02, 0.05, 0.1, 0.2, 0.4)
    table = {}
    print("   p        " + "".join(f"sd={s:<9}" for s in sds))
    for p in (0.005, 0.02, 0.10):
        row = []
        for sd in sds:
            L, V = [], []
            for _ in range(40):
                dl = rng.normal(0, sd, d)
                dl -= dl.mean()
                V.append(dl.var())
                L.append(kappa0(p) - exponent(p, wgt(p) + dl))
            row.append(np.mean(L) / np.mean(V))
        table[p] = row
        print(f"   {p:<8} " + "".join(f"{r:<12.5f}" for r in row))
    print()
    return table


def panel_a_data(rng):
    """Data for Fig. 3(a): loss vs Var over several decades."""
    d = 128
    out = {}
    for p in (0.005, 0.02, 0.10):
        Vs, Ls = [], []
        for sd in (0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5):
            vv, ll = [], []
            for _ in range(25):
                dl = rng.normal(0, sd, d)
                dl -= dl.mean()
                vv.append(dl.var())
                ll.append(kappa0(p) - exponent(p, wgt(p) + dl))
            Vs.append(np.mean(vv)); Ls.append(np.mean(ll))
        out[p] = (np.array(Vs), np.array(Ls))
    return out


# ------------------------------- C: projection form, inhomogeneous rates

def part_C(rng):
    print("=" * 72)
    print("C.  projection form with p_e spanning two decades")
    print("    loss  vs  ||delta_perp||^2 / 8d")
    print("=" * 72)
    d = 200
    for trial in range(4):
        pv = 10 ** rng.uniform(-3, -1, d)
        w = wgt(pv)
        k0 = exponent(pv, w)
        for sd in (0.02, 0.05, 0.15):
            dl = rng.normal(0, sd, d)
            loss = k0 - exponent(pv, w + dl)
            pred = perp_norm2(dl, w) / (8 * d)
            naive = dl.var() / 8                      # wrong predictor
            print(f"   trial {trial}  sd={sd:<5}  loss={loss:.5e}  "
                  f"pred={pred:.5e}  ratio={loss/pred:.4f}   "
                  f"(Var/8 would give ratio {loss/naive:.3f})")
    print()


# --------------------------------- D: importance-sampled failure probability

def _tilted_pfail(p, what, nshot, rng, chunk=500_000):
    d = what.size
    half = 0.5 * what.sum()

    def negG(s):
        z = s * what
        m = np.maximum(z, 0.0)
        return -(s * half - np.sum(m + np.log((1 - p) * np.exp(-m)
                                              + p * np.exp(z - m))))
    s = minimize_scalar(negG, bounds=(1e-9, 60), method="bounded",
                        options={"xatol": 1e-13}).x
    z = s * what
    q = p * np.exp(z) / (1 - p + p * np.exp(z))       # tilted Bernoulli rates
    lr1 = np.log(p) - np.log(q)
    lr0 = np.log(1 - p) - np.log(1 - q)
    tot, done = 0.0, 0
    while done < nshot:
        n = min(chunk, nshot - done)
        X = rng.random((n, d)) < q
        acc = (X @ what) > half
        tot += np.where(acc, np.exp(X @ lr1 + (~X) @ lr0), 0.0).sum()
        done += n
    return tot / done


def part_D(rng):
    print("=" * 72)
    print("D.  direct failure probability by exponential tilting  (p = 0.1)")
    print("    slope of ln[P(what)/P(w)] vs d  should equal the exponent loss")
    print("=" * 72)
    p = 0.10
    w0 = wgt(p)
    ds = [20, 40, 60, 80]
    patterns = {d: [rng.normal(0, 1, d) for _ in range(24)] for d in ds}
    for V in (0.04, 0.09, 0.16):
        diffs, kex = [], []
        for d in ds:
            lm = -np.log(binom.sf(d // 2, d, p))       # exact matched tail
            acc = []
            for z0 in patterns[d]:
                dl = z0 - z0.mean()
                dl *= np.sqrt(V / dl.var())
                acc.append(-np.log(_tilted_pfail(p, w0 + dl, 1_000_000, rng)))
                kex.append(exponent(p, w0 + dl))
            diffs.append(lm - np.mean(acc))
        slope = np.polyfit(np.array(ds, float), np.array(diffs), 1)[0]
        print(f"   Var={V:<6} MC slope={slope:.5f}   "
              f"exact Cramer loss={kappa0(p)-np.mean(kex):.5f}   V/8={V/8:.5f}")
    print("   (the V=0.04 case is not resolvable: the lattice/non-lattice")
    print("    prefactor difference dominates over the accessible range of d)")
    print()


# ------------------------------------------------------------- E: end to end

def part_E(rng):
    print("=" * 72)
    print("E.  end to end: telegraph drift -> sliding window -> decay rate")
    print("=" * 72)
    d, T, L = 49, 60_000, 2000
    pbar, Delta = 0.05, 0.04
    v = (pbar - Delta / 2) * (1 - (pbar - Delta / 2))

    # independent telegraph sign per mechanism
    flip = rng.random((T, d)) < 1.0 / L
    eps = np.empty((T, d))
    s = np.where(rng.random(d) < 0.5, 1.0, -1.0)
    for t in range(T):
        s = np.where(flip[t], -s, s)
        eps[t] = s
    ptrue = pbar + 0.5 * Delta * eps

    x = (rng.random((T, d)) < ptrue).astype(float)
    cx = np.vstack([np.zeros(d), np.cumsum(x, axis=0)])

    def jeffreys(cnt, n):
        return (cnt + 0.5) / (n + 1.0)

    def win_est(N, t):
        lo = max(0, t - N + 1)
        return jeffreys(cx[t + 1] - cx[lo], t + 1 - lo)

    # oracle change-point smoother: average over the whole dwell interval
    starts = np.zeros((T, d), int)
    ends = np.zeros((T, d), int)
    for e in range(d):
        bnd = np.concatenate(([0], np.flatnonzero(flip[:, e]), [T]))
        for i in range(len(bnd) - 1):
            starts[bnd[i]:bnd[i + 1], e] = bnd[i]
            ends[bnd[i]:bnd[i + 1], e] = bnd[i + 1]

    def cp_est(t):
        a, b = starts[t], ends[t]
        return jeffreys(cx[b, np.arange(d)] - cx[a, np.arange(d)], b - a)

    tsamp = rng.choice(np.arange(4000, T), size=250, replace=False)
    k0 = np.mean([exponent(ptrue[t], wgt(ptrue[t])) for t in tsamp])
    print(f"   oracle-matched kappa0 = {k0:.5f}")
    print(f"   rectangular optimum sqrt(3 v L / Delta^2) = "
          f"{np.sqrt(3*v*L/Delta**2):.0f}")
    print(f"   MSE floor Delta sqrt(v/L) = {Delta*np.sqrt(v/L):.3e}")
    print()
    print("      N    ||d_perp||^2/d   kappa     dk/k0     pred/8")
    rows = []
    for N in (25, 50, 100, 200, 300, 500, 800, 1200, 2000, 4000):
        ks, vs = [], []
        for t in tsamp:
            pe = ptrue[t]
            ph = np.clip(win_est(N, t), 1e-4, 0.45)
            w = wgt(pe)
            dl = wgt(ph) - w
            vs.append(perp_norm2(dl, w) / d)
            ks.append(exponent(pe, wgt(ph)))
        kk, VV = np.mean(ks), np.mean(vs)
        rows.append((N, VV, kk, k0 - kk))
        print(f"   {N:6d}   {VV:.5f}       {kk:.5f}   "
              f"{(k0-kk)/k0*100:5.2f}%   {VV/8:.5f}")

    ks, vs = [], []
    for t in tsamp:
        pe = ptrue[t]
        ph = np.clip(cp_est(t), 1e-4, 0.45)
        w = wgt(pe)
        dl = wgt(ph) - w
        vs.append(perp_norm2(dl, w) / d)
        ks.append(exponent(pe, wgt(ph)))
    kcp, Vcp = np.mean(ks), np.mean(vs)
    print(f"   change-pt  {Vcp:.5f}       {kcp:.5f}   "
          f"{(k0-kcp)/k0*100:5.2f}%   {Vcp/8:.5f}")

    best = min(rows, key=lambda r: r[3])
    print()
    print(f"   best window N={best[0]}: loss {best[3]/k0*100:.2f}% of kappa0"
          f"  ->  {2*best[3]/k0*100:.2f}% physical-qubit overhead (2-D code)")
    print(f"   loss ratio window/change-point = {best[3]/(k0-kcp):.2f}"
          f"   (predicted Delta sqrt(L/v) = {Delta*np.sqrt(L/v):.2f})")
    print()
    return np.array(rows, float), (k0, kcp, Vcp), v, L, Delta


# ------------------------------------------------ F: device overhead table

def part_F():
    print("=" * 72)
    print("F.  device-parameter overhead, Eq. (30), at a = Delta/pbar = 0.3")
    print("=" * 72)
    a = 0.3
    print("   pbar     L        N_err   kappa0   LTI ovh    CP ovh   ratio")
    for pb, L in ((1e-3, 1e6), (1e-4, 1e6), (1e-5, 1e6),
                  (1e-3, 1e4), (1e-4, 1e4)):
        Ne = pb * L
        dk = (2 / np.sqrt(3)) * a / ((1 - pb) ** 1.5 * np.sqrt(Ne)) / 8
        dkcp = 1.0 / (Ne * (1 - pb)) / 8
        k0 = kappa0(pb)
        print(f"   {pb:<8.0e} {L:<8.0e} {Ne:<7.4g} {k0:<8.3f} "
              f"{2*dk/k0*100:7.3f}%  {2*dkcp/k0*100:7.4f}%  {dk/dkcp:6.1f}")
    print("   (LTI uses the achievable constant of Eq. (23); the rigorous")
    print("    floor is a factor 15 smaller.  Ratio = a sqrt(N_err).)")
    print()


# ---------------------------------------------------------------- figure

def make_figure(panel_a, rows, cp, v, L, Delta, fname="f3.png"):
    k0, kcp, Vcp = cp
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.7))

    marks = {0.005: "o", 0.02: "s", 0.10: "^"}
    for p, m in marks.items():
        Vs, Ls = panel_a[p]
        ax[0].loglog(Vs, Ls, m, ms=5, label=f"$p={p}$")
    xx = np.logspace(-5, -0.4, 50)
    ax[0].loglog(xx, xx / 8, "k--", lw=1,
                 label=r"$\frac{1}{8}\,\mathrm{Var}_e(\delta)$")
    ax[0].set_xlabel(r"$\mathrm{Var}_e(\delta_e)$")
    ax[0].set_ylabel(r"$\kappa_0-\kappa(\hat w)$")
    ax[0].set_title("(a) universal curvature at the Nishimori line", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False)
    ax[0].grid(alpha=.25, which="both")

    N, V, kk, loss = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    ax[1].loglog(N, loss / k0 * 100, "o-", ms=5,
                 label=r"measured $\Delta\kappa/\kappa_0$")
    ax[1].loglog(N, (V / 8) / k0 * 100, "x--", ms=5,
                 label=r"$\|\delta_\perp\|^2/8d\,/\,\kappa_0$")
    ax[1].axhline((k0 - kcp) / k0 * 100, color="g", ls=":", lw=1.4,
                  label="change-point calibration")
    ax[1].axvline(np.sqrt(3 * v * L / Delta ** 2), color="gray", ls="-.", lw=1,
                  label=r"$\sqrt{3vL/\Delta^2}$")
    ax[1].set_xlabel(r"window length $N=\tau_{\rm var}$")
    ax[1].set_ylabel("decay-rate loss " + r"$\Delta\kappa/\kappa_0$" + "  (%)")
    ax[1].set_title("(b) calibration overhead, $d=49$ ring code", fontsize=9)
    ax[1].legend(fontsize=7, frameon=False)
    ax[1].grid(alpha=.25, which="both")

    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    print(f"wrote {fname}")


if __name__ == "__main__":
    part_A(np.random.default_rng(7))
    part_B(np.random.default_rng(7))
    part_C(np.random.default_rng(17))
    part_D(np.random.default_rng(808))
    rows, cp, v, L, Delta = part_E(np.random.default_rng(20260812))
    part_F()
    make_figure(panel_a_data(np.random.default_rng(3)),
                rows, cp, v, L, Delta)
