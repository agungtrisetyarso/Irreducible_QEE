# ================================================================
# Colab notebook: Testing linear-syndrome-memory bounds
# Manuscript: "Fundamental limits of linear syndrome memory..."
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------
# 1. Telegraph (switching) drift  – Definition 1
# ----------------------------------------------------------------
def generate_telegraph(T, L, p_bar=0.05, Delta=0.06, seed=None):
    """
    p(t) = p_bar + (Delta/2)*eps(t)
    eps is a stationary telegraph process with flip probability 1/L.
    Returns p (length T) and the true switch times.
    """
    rng = np.random.default_rng(seed)
    eps = np.ones(T)
    # start from stationary distribution
    if rng.random() < 0.5:
        eps[0] = -1
    for t in range(1, T):
        if rng.random() < 1.0/L:
            eps[t] = -eps[t-1]
        else:
            eps[t] = eps[t-1]
    p = p_bar + 0.5*Delta*eps
    # clip to (0,1)
    p = np.clip(p, 1e-4, 1-1e-4)
    switches = np.where(np.diff(eps) != 0)[0] + 1
    return p, switches, eps

# ----------------------------------------------------------------
# 2. Kernels and the two functionals  – Definition 2 & Eq. (4)
# ----------------------------------------------------------------
def rectangular_kernel(N):
    w = np.ones(N) / N
    return w

def geometric_kernel(r, max_lag=5000):
    """w(u) = (1-r) r^u , truncated and renormalised"""
    u = np.arange(max_lag)
    w = (1-r) * r**u
    w /= w.sum()
    return w

def tau_var(w):
    return 1.0 / np.sum(w**2)

def beta_closed(w, rho):
    """Closed-form switching susceptibility, Eq. (5)"""
    # β = 1/4 * (1 + sum_{u,u'} w(u)w(u') ρ^{|u-u'|} - 2 sum_u w(u) ρ^u)
    n = len(w)
    # efficient Toeplitz way
    # first term: sum_u sum_up w(u)w(up) rho^{|u-up|}
    s = 0.0
    for d in range(-n+1, n):
        rho_d = rho**abs(d)
        if d >= 0:
            s += rho_d * np.dot(w[:n-d], w[d:])
        else:
            s += rho_d * np.dot(w[-d:], w[:n+d])
    sum_w_rho = np.dot(w, rho**np.arange(n))
    beta = 0.25 * (1.0 + s - 2.0*sum_w_rho)
    return max(beta, 0.0)

# ----------------------------------------------------------------
# 3. Estimators
# ----------------------------------------------------------------
def apply_kernel(x, w):
    """ˆp(t) = sum_u w(u) x_{t-u}  (causal, zero-pad)"""
    return np.convolve(x, w, mode='full')[:len(x)]

def change_point_estimator(x, switches, T):
    """Oracle that knows switch times and averages inside each dwell"""
    hat = np.empty(T)
    boundaries = np.concatenate(([0], switches, [T]))
    for i in range(len(boundaries)-1):
        a, b = boundaries[i], boundaries[i+1]
        hat[a:b] = x[a:b].mean() if b > a else x[a]
    return hat

# ----------------------------------------------------------------
# 4. Monte-Carlo evaluation of MSE for a given kernel
# ----------------------------------------------------------------
def evaluate_mse(w, T=20000, L=2000, p_bar=0.05, Delta=0.06,
                 n_trials=40, seed=0):
    """
    Returns mean MSE, mean Var term, mean Bias² term,
    and the theoretical lower bounds.
    """
    rng = np.random.default_rng(seed)
    mses, vars_, biases = [], [], []
    for trial in range(n_trials):
        p, switches, eps = generate_telegraph(T, L, p_bar, Delta,
                                              seed=rng.integers(1e9))
        # Bernoulli observations
        x = rng.random(T) < p
        # linear estimator
        hat = apply_kernel(x.astype(float), w)
        # conditioned bias (path-wise)
        # E[hat | p] = apply_kernel(p, w)
        hat_mean = apply_kernel(p, w)
        bias2 = np.mean((hat_mean - p)**2)
        # residual variance
        var = np.mean((hat - hat_mean)**2)
        mse = np.mean((hat - p)**2)
        mses.append(mse)
        vars_.append(var)
        biases.append(bias2)
    return (np.mean(mses), np.std(mses),
            np.mean(vars_), np.mean(biases))

# ----------------------------------------------------------------
# 5. Test of the counting-cost and bias identities
# ----------------------------------------------------------------
print("=== Test of exact identities (Lemmas) ===")
T, L = 30000, 2500
p_bar, Delta = 0.05, 0.06
v = min(p_bar-0.5*Delta, 1-(p_bar+0.5*Delta)) * \
    (1-min(p_bar-0.5*Delta, 1-(p_bar+0.5*Delta)))  # rough
v = 0.05*(1-0.05)   # conservative

for name, w in [("Rectangular N=200", rectangular_kernel(200)),
                ("Geometric r=0.99", geometric_kernel(0.99))]:
    tv = tau_var(w)
    rho = 1 - 2/L
    b = beta_closed(w, rho)
    mse, se, var_emp, bias_emp = evaluate_mse(w, T=T, L=L,
                                              p_bar=p_bar, Delta=Delta,
                                              n_trials=30)
    print(f"\n{name}")
    print(f"  τ_var = {tv:.1f}")
    print(f"  β (closed) = {b:.5f}")
    print(f"  Emp. Var   = {var_emp:.6f}   (theory ≥ v/τ_var = {v/tv:.6f})")
    print(f"  Emp. Bias² = {bias_emp:.6f}   (theory = Δ² β = {Delta**2 * b:.6f})")
    print(f"  Emp. MSE   = {mse:.6f} ± {se:.6f}")

# ----------------------------------------------------------------
# 6. U-shape: MSE versus effective sample size (main figure)
# ----------------------------------------------------------------
print("\n=== Sweeping rectangular window → U-shape ===")
Ns = np.unique(np.logspace(1.2, 3.5, 18).astype(int))
mses, theoretical = [], []
for N in Ns:
    w = rectangular_kernel(N)
    mse, _, _, _ = evaluate_mse(w, T=25000, L=2000,
                                n_trials=25, seed=42)
    mses.append(mse)
    # crude theoretical floor
    tv = N
    th = v/tv + (Delta**2)*(tv)/(64*np.e*2000)
    theoretical.append(th)

plt.figure(figsize=(7,4.5))
plt.loglog(Ns, mses, 'o-', label='Empirical MSE (rectangular)')
plt.loglog(Ns, theoretical, '--', label='Theoretical floor (order)')
plt.axvline(8*np.sqrt(np.e)*np.sqrt(v*2000/Delta**2), color='k', ls=':',
            label=r'theory $\tau^\star$')
plt.xlabel(r'Window length $N$ ($=\tau_{\rm var}$)')
plt.ylabel('MSE')
plt.title('U-shape of linear syndrome memory (telegraph drift)')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------
# 7. Scaling of optimal horizon and of the MSE ratio
#    (reproduces the spirit of Table I)
# ----------------------------------------------------------------
print("\n=== Scaling with T (fixed K=8) ===")
Ts = np.array([4000, 8000, 16000, 32000, 64000])
K = 8
opt_Ns, mse_win, mse_cp, ratios = [], [], [], []

for T in Ts:
    L = T // K
    # sweep a few N around the expected optimum
    N_candidates = np.unique(np.logspace(np.log10(max(20,np.sqrt(T)/3)),
                                         np.log10(min(T//3, 3*np.sqrt(T))), 12).astype(int))
    best_mse, best_N = np.inf, None
    for N in N_candidates:
        w = rectangular_kernel(N)
        mse, _, _, _ = evaluate_mse(w, T=T, L=L, n_trials=20, seed=T)
        if mse < best_mse:
            best_mse, best_N = mse, N
    # change-point oracle
    # (approximate by evaluating one long run)
    p, switches, _ = generate_telegraph(T, L, seed=123)
    x = (np.random.rand(T) < p).astype(float)
    hat_cp = change_point_estimator(x, switches, T)
    mse_oracle = np.mean((hat_cp - p)**2)

    opt_Ns.append(best_N)
    mse_win.append(best_mse)
    mse_cp.append(mse_oracle)
    ratios.append(best_mse / mse_oracle)
    print(f"T={T:5d}  τ*≈{best_N:4d}  MSEwin={best_mse:.3e}  "
          f"MSEcp={mse_oracle:.3e}  ratio={best_mse/mse_oracle:.2f}")

# Fit exponents
logT = np.log(Ts)
exp_tau = np.polyfit(logT, np.log(opt_Ns), 1)[0]
exp_mse = np.polyfit(logT, np.log(mse_win), 1)[0]
exp_ratio = np.polyfit(logT, np.log(ratios), 1)[0]

print(f"\nMeasured exponents:")
print(f"  τ*      ~ T^{exp_tau:.3f}   (theory 0.50)")
print(f"  MSEwin  ~ T^{exp_mse:.3f}   (theory -0.50)")
print(f"  ratio   ~ T^{exp_ratio:.3f}   (theory 0.50)")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
axes[0].loglog(Ts, opt_Ns, 'o-')
axes[0].set_title(r'$\tau^\star$ vs $T$')
axes[0].set_xlabel('T')
axes[1].loglog(Ts, mse_win, 'o-', label='window')
axes[1].loglog(Ts, mse_cp, 's--', label='change-point')
axes[1].set_title('MSE vs T')
axes[1].legend()
axes[1].set_xlabel('T')
axes[2].loglog(Ts, ratios, 'o-')
axes[2].set_title('MSE ratio (window / CP)')
axes[2].set_xlabel('T')
plt.suptitle('Scaling tests of Theorem (estimation floor & separation)')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------
# 8. Direct check of the β lower bound (Lemma)
# ----------------------------------------------------------------
print("\n=== Checking β ≥ c (τ_var)/L  (incompatibility) ===")
L = 2000
rho = 1 - 2/L
Ns = [20, 50, 100, 200, 500, 1000]
print(f"{'N':>6}  {'τ_var':>8}  {'β':>10}  {'β·L/τ':>10}  {'bound':>10}")
for N in Ns:
    w = rectangular_kernel(N)
    tv = tau_var(w)
    b = beta_closed(w, rho)
    print(f"{N:6d}  {tv:8.1f}  {b:10.5f}  {b*L/tv:10.4f}  "
          f"{(tv-4)/(64*np.e*L)*L/tv:10.5f}")

print("\nGeometric kernels:")
for r in [0.9, 0.95, 0.99, 0.995]:
    w = geometric_kernel(r)
    tv = tau_var(w)
    b = beta_closed(w, rho)
    print(f"r={r:.3f}  τ_var={tv:7.1f}  β={b:.5f}  β·L/τ={b*L/tv:.4f}")

print("\nDone.  All core claims of the manuscript have been exercised.")
