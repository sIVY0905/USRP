"""V8: redraw the diagnostic figure with horizon as the primary axis.

Every model is pushed through the same rolling-origin recursion and the same
uniform random numbers (common random numbers), so differences between models
cannot come from the simulation.

Per horizon h in {1,2,4,6,8} (= 15,30,60,90,120 min) we compute
  - RMSE from the analytic conditional mean (no Monte Carlo noise)
  - CRPS from the predictive ensemble
  - 95% coverage and PIT uniformity
  - paired moving-block-bootstrap CI for CRPS differences
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, t as student_t, kstest
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.simplefilter("ignore")
rng_master = np.random.default_rng(20260812)

y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")
center = float(np.load("center_value.npy"))
full = np.concatenate([y_train, y_test])
s, n, ntr = 96, len(y_test), len(y_train)
H = 8
HOR = [1, 2, 4, 6, 8]
M = 1000

res_s = SARIMAX(y_train, order=(2, 0, 2), seasonal_order=(1, 0, 0, 96), trend="n").fit(disp=False, maxiter=500)
res_a = SARIMAX(y_train, order=(2, 0, 2), trend="n").fit(disp=False, maxiter=500)
pm_, pa_ = dict(zip(res_s.param_names, res_s.params)), dict(zip(res_a.param_names, res_a.params))

MODELS = {
    "Gaussian SARIMA": dict(phi1=1.297, phi2=-0.408, theta1=0.819, theta2=0.134,
                            Phi1=0.180, sigma=6.041, dist="norm", color="#1f77b4"),
    "Student-t SARIMA": dict(phi1=1.306, phi2=-0.409, theta1=0.779, theta2=0.111,
                             Phi1=0.186, sigma=5.058, nu=6.777, dist="t", color="#b22222"),
    "SARIMA MLE (exact)": dict(phi1=pm_["ar.L1"], phi2=pm_["ar.L2"], theta1=pm_["ma.L1"],
                               theta2=pm_["ma.L2"], Phi1=pm_["ar.S.L96"],
                               sigma=float(np.sqrt(pm_["sigma2"])), dist="norm", color="#7f7f7f"),
    "ARMA(2,2) no seasonal": dict(phi1=pa_["ar.L1"], phi2=pa_["ar.L2"], theta1=pa_["ma.L1"],
                                  theta2=pa_["ma.L2"], Phi1=0.0,
                                  sigma=float(np.sqrt(pa_["sigma2"])), dist="norm", color="#2ca02c"),
}

# common random numbers: one uniform block reused by every model
U = rng_master.random((M, H, n))


def warm_residuals(p):
    e = np.zeros(ntr)
    for t in range(s + 2, ntr):
        e[t] = (y_train[t] - p["phi1"] * y_train[t - 1] - p["phi2"] * y_train[t - 2]
                - p["Phi1"] * y_train[t - s] + p["phi1"] * p["Phi1"] * y_train[t - s - 1]
                + p["phi2"] * p["Phi1"] * y_train[t - s - 2]
                - p["theta1"] * e[t - 1] - p["theta2"] * e[t - 2])
    return e


def run_model(p):
    """Rolling origin. Returns conditional means and predictive draws per horizon."""
    inv = (lambda u: p["sigma"] * norm.ppf(u)) if p["dist"] == "norm" \
        else (lambda u: p["sigma"] * student_t.ppf(u, df=p["nu"]))
    E = np.concatenate([warm_residuals(p), np.zeros(n)])
    Y = full.copy()
    mus = np.full((H, n), np.nan)
    draws = np.full((H, n, M), np.nan)
    for k in range(n):
        t0 = ntr + k
        y1 = np.full(M, Y[t0 - 1])
        y2 = np.full(M, Y[t0 - 2])
        e1 = np.full(M, E[t0 - 1])
        e2 = np.full(M, E[t0 - 2])
        m1, m2 = Y[t0 - 1], Y[t0 - 2]
        c1, c2 = E[t0 - 1], E[t0 - 2]
        for j in range(H):
            t = t0 + j
            if t >= len(full):
                break
            seas = (p["Phi1"] * Y[t - s] - p["phi1"] * p["Phi1"] * Y[t - s - 1]
                    - p["phi2"] * p["Phi1"] * Y[t - s - 2])
            mu_vec = p["phi1"] * y1 + p["phi2"] * y2 + seas + p["theta1"] * e1 + p["theta2"] * e2
            mu_sc = p["phi1"] * m1 + p["phi2"] * m2 + seas + p["theta1"] * c1 + p["theta2"] * c2
            eps = inv(U[:, j, k])
            ynew = mu_vec + eps
            draws[j, t - ntr] = ynew
            mus[j, t - ntr] = mu_sc
            y2, y1 = y1, ynew
            e2, e1 = e1, eps
            m2, m1 = m1, mu_sc
            c2, c1 = c1, 0.0
        # reveal the observation, advance the filter
        t = t0
        E[t] = (full[t] - p["phi1"] * Y[t - 1] - p["phi2"] * Y[t - 2] - p["Phi1"] * Y[t - s]
                + p["phi1"] * p["Phi1"] * Y[t - s - 1] + p["phi2"] * p["Phi1"] * Y[t - s - 2]
                - p["theta1"] * E[t - 1] - p["theta2"] * E[t - 2])
    return mus, draws


def crps_ens(obs, dr):
    a = np.mean(np.abs(dr - obs[:, None]), axis=1)
    sd = np.sort(dr, axis=1)
    m = sd.shape[1]
    w = (2 * np.arange(1, m + 1) - m - 1)[None, :]
    return a - np.sum(w * sd, axis=1) / m ** 2


def block_boot_ci(d, B=4000, L=12, seed=3):
    rg = np.random.default_rng(seed)
    N = len(d)
    nb = int(np.ceil(N / L))
    starts = rg.integers(0, N - L + 1, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B, -1)[:, :N]
    return np.quantile(d[idx].mean(axis=1), [0.025, 0.975])


out = {name: run_model(p) for name, p in MODELS.items()}

rows, per_target_crps = [], {}
for name, p in MODELS.items():
    mus, draws = out[name]
    for h in HOR:
        v, dr = mus[h - 1], draws[h - 1]
        m = ~np.isnan(v)
        obs = y_test[m]
        c = crps_ens(obs, dr[m])
        per_target_crps[(name, h)] = c
        lo = np.quantile(dr[m], 0.025, axis=1)
        hi = np.quantile(dr[m], 0.975, axis=1)
        pit = np.mean(dr[m] < obs[:, None], axis=1)
        rows.append(dict(model=name, horizon_min=h * 15,
                         RMSE=float(np.sqrt(np.mean((obs - v[m]) ** 2))),
                         CRPS=float(c.mean()),
                         cov95=float(np.mean((obs >= lo) & (obs <= hi))),
                         width95=float(np.mean(hi - lo)),
                         PIT_KS_p=float(kstest(pit, "uniform").pvalue),
                         R2=float(1 - np.mean((obs - v[m]) ** 2) / np.var(y_test, ddof=1))))
tab = pd.DataFrame(rows)
tab.to_csv("multi_horizon_full_metrics.csv", index=False, encoding="utf-8-sig")
print(tab.pivot(index="horizon_min", columns="model", values="CRPS").to_string(float_format=lambda v: f"{v:.3f}"))
print()
print(tab.pivot(index="horizon_min", columns="model", values="RMSE").to_string(float_format=lambda v: f"{v:.2f}"))

REF = "Gaussian SARIMA"
ci_rows = []
for name in MODELS:
    if name == REF:
        continue
    for h in HOR:
        d = per_target_crps[(name, h)] - per_target_crps[(REF, h)]
        lo, hi = block_boot_ci(d)
        ci_rows.append(dict(model=name, horizon_min=h * 15, mean=float(d.mean()),
                            lo=float(lo), hi=float(hi), sig=bool(lo > 0 or hi < 0)))
ci = pd.DataFrame(ci_rows)
ci.to_csv("crps_difference_ci.csv", index=False, encoding="utf-8-sig")
print()
print("Paired CRPS difference vs Gaussian SARIMA (moving block bootstrap, L=12, 95% CI)")
print(ci.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(14, 9.5))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.3)
xh = np.array(HOR) * 15

ax = fig.add_subplot(gs[0, 0])
for name, p in MODELS.items():
    d = tab[tab.model == name].sort_values("horizon_min")
    ax.plot(d.horizon_min, d.CRPS, "o-", color=p["color"], lw=1.7, ms=5, label=name)
ax.set(title="A. Predictive skill (CRPS) vs horizon", xlabel="forecast horizon (min)",
       ylabel="CRPS (mg/dL, lower better)")
ax.set_xticks(xh)
ax.legend(fontsize=7.5)
ax.grid(alpha=0.25)

ax = fig.add_subplot(gs[0, 1])
for name, p in MODELS.items():
    if name == REF:
        continue
    d = ci[ci.model == name].sort_values("horizon_min")
    ax.errorbar(d.horizon_min + (list(MODELS).index(name) - 2) * 2.2, d["mean"],
                yerr=[d["mean"] - d.lo, d.hi - d["mean"]], fmt="o", color=p["color"],
                capsize=3, ms=5, lw=1.4, label=name)
ax.axhline(0, color="black", lw=1.2)
ax.set(title=f"B. Paired ΔCRPS vs {REF}\n(95% moving-block bootstrap CI)",
       xlabel="forecast horizon (min)", ylabel="ΔCRPS (mg/dL)")
ax.set_xticks(xh)
ax.legend(fontsize=7.5)
ax.grid(alpha=0.25)
ax.text(0.02, 0.03, "CI crossing 0 = models not distinguishable", transform=ax.transAxes,
        fontsize=7.5, style="italic", color="#555555")

ax = fig.add_subplot(gs[0, 2])
for name, p in MODELS.items():
    d = tab[tab.model == name].sort_values("horizon_min")
    ax.plot(d.horizon_min, d.cov95, "o-", color=p["color"], lw=1.7, ms=5)
ax.axhline(0.95, color="black", ls="--", lw=1.2)
ax.fill_between([xh[0] - 5, xh[-1] + 5], 0.95 - 2 * np.sqrt(.95 * .05 / n),
                0.95 + 2 * np.sqrt(.95 * .05 / n), color="black", alpha=0.08)
ax.set(title="C. 95% interval coverage vs horizon", xlabel="forecast horizon (min)",
       ylabel="empirical coverage", xlim=(xh[0] - 5, xh[-1] + 5), ylim=(0.87, 1.0))
ax.text(0.03, 0.06, "grey band = ±2 binomial SE around 0.95", transform=ax.transAxes,
        fontsize=7.5, style="italic", color="#555555")
ax.set_xticks(xh)
ax.grid(alpha=0.25)

ax = fig.add_subplot(gs[1, 0])
for name, p in MODELS.items():
    mus_, draws_ = out[name]
    ps = []
    for h in HOR:
        v, dr = mus_[h - 1], draws_[h - 1]
        mm = ~np.isnan(v)
        pit_all = np.mean(dr[mm] < y_test[mm][:, None], axis=1)
        ps.append(kstest(pit_all[::h], "uniform").pvalue)
    ax.plot(xh, ps, "o-", color=p["color"], lw=1.7, ms=5)
ax.axhline(0.05, color="red", ls="--", lw=1.2)
ax.set(title="D. Distributional calibration (PIT KS p)\nnon-overlapping targets only",
       xlabel="forecast horizon (min)", ylabel="KS p-value", ylim=(0, 1))
ax.set_xticks(xh)
ax.grid(alpha=0.25)
for x, nn in zip(xh, [n // h for h in HOR]):
    ax.annotate(f"n={nn}", (x, 0.96), fontsize=7, ha="center", color="#777777")
ax.text(0.03, 0.10, "below red line = predictive shape rejected;\nn shrinks with horizon, so power drops",
        transform=ax.transAxes, fontsize=7.5, style="italic", color="#555555")

ax = fig.add_subplot(gs[1, 1])
gs_ = tab[tab.model == "SARIMA MLE (exact)"].sort_values("horizon_min").RMSE.to_numpy()
ar_ = tab[tab.model == "ARMA(2,2) no seasonal"].sort_values("horizon_min").RMSE.to_numpy()
ax.bar(np.arange(len(HOR)), ar_ - gs_, color=["#2ca02c" if v > 0 else "#d62728" for v in ar_ - gs_])
ax.axhline(0, color="black", lw=1.2)
ax.set(title="E. Value of the seasonal AR(1)[96] term\nRMSE(no seasonal) − RMSE(seasonal)",
       xlabel="forecast horizon (min)", ylabel="ΔRMSE (mg/dL)")
ax.set_xticks(np.arange(len(HOR)), [str(v) for v in xh])
ax.grid(alpha=0.25, axis="y")
for i, v in enumerate(ar_ - gs_):
    ax.annotate(f"{v:+.2f}", (i, v), textcoords="offset points",
                xytext=(0, 6 if v > 0 else -12), fontsize=7.5, ha="center")
ax.text(0.03, 0.9, "> 0: seasonal term helps\n< 0: seasonal term hurts",
        transform=ax.transAxes, fontsize=7.5, style="italic", color="#555555", va="top")

ax = fig.add_subplot(gs[1, 2])
d = tab[tab.model == "Gaussian SARIMA"].sort_values("horizon_min")
ax.plot(d.horizon_min, d.R2, "o-", color="#1f77b4", lw=1.9, ms=6)
ax.axhline(0, color="black", ls="--", lw=1.2)
ax.fill_between(d.horizon_min, 0, d.R2, color="#1f77b4", alpha=0.12)
ax.set(title="F. Remaining predictable variance vs horizon",
       xlabel="forecast horizon (min)", ylabel=r"$R^2$ against holdout variance", ylim=(-0.05, 1.02))
ax.set_xticks(xh)
ax.grid(alpha=0.25)
for x, y in zip(d.horizon_min, d.R2):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=7.5, ha="center")

fig.suptitle("One-step to two-hour predictive comparison, 192-point holdout, "
             "fixed parameters, common random numbers", fontsize=13.5, fontweight="bold")
fig.savefig("replacement_figure_v2_horizon.png", dpi=200, bbox_inches="tight")
print("\nSaved replacement_figure_v2_horizon.png, multi_horizon_full_metrics.csv, crps_difference_ci.csv")

# ---- validity check for panel D: KS on overlapping h-step PITs is not a valid test
print()
print("PIT KS p on NON-OVERLAPPING subsamples (every h-th target), Gaussian SARIMA:")
mus, draws = out["Gaussian SARIMA"]
for h in HOR:
    v, dr = mus[h - 1], draws[h - 1]
    m = ~np.isnan(v)
    obs, drm = y_test[m], dr[m]
    pit_all = np.mean(drm < obs[:, None], axis=1)
    sub = pit_all[::h]
    print(f"  h={h*15:>3}min  overlapping n={len(pit_all):3d} p={kstest(pit_all,'uniform').pvalue:.2e}"
          f"   |  non-overlapping n={len(sub):3d} p={kstest(sub,'uniform').pvalue:.3f}"
          f"   |  mean PIT={sub.mean():.3f} (target 0.5)")
