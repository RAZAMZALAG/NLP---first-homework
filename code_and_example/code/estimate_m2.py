"""Calibrated Model 2 accuracy estimate for the report (the |estimated - actual| score).

Traces a learning curve (repeated CV at increasing train sizes), fits a power law
acc(n) = A - B*n^(-C), and extrapolates to the full 250-sentence model. Linear extrapolation
overestimates because the curve is concave; the power law captures the diminishing returns.
Saves a figure to guide/m2_learning_curve.png. Run from code_and_example:
    .\\.venv\\Scripts\\python.exe code\\estimate_m2.py
"""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from eval_m2 import cfg_thr_subset, repeated_cv, TRAIN
from main import MODEL2_CONFIGS
from preprocessing import induce_clusters

CONFIG, LAM, K, REPEATS, SEED = "G2", 1.0, 5, 8, 0

lines = [ln for ln in open(TRAIN) if ln.strip()]
thr, subset = cfg_thr_subset(CONFIG)
# G2's f_clust needs the unsupervised cluster map (train1+train2, no tags) in every fold.
clusters = None
cspec = MODEL2_CONFIGS[CONFIG].get("clusters")
if cspec:
    spec = dict(cspec)
    clusters = induce_clusters(spec.pop("corpus", [TRAIN]), **spec)
rng = np.random.default_rng(SEED)
order = rng.permutation(len(lines))

xs, ys, es = [], [], []
for frac in (0.5, 0.7, 0.85, 1.0):
    nsub = max(K + 1, int(len(lines) * frac))
    sub = [lines[i] for i in order[:nsub]]
    a = repeated_cv(sub, thr, subset, LAM, K, REPEATS, SEED, clusters)
    xs.append(nsub * (K - 1) // K)      # ~ train sentences per fold
    ys.append(a.mean()); es.append(1.96 * a.std(ddof=1) / np.sqrt(len(a)))
xs, ys, es = np.array(xs), np.array(ys), np.array(es)


def f(n, A, B, C):
    return A - B * np.power(n, -C)


popt, _ = curve_fit(f, xs, ys, p0=[0.95, 1.0, 0.5], maxfev=20000)
est_250 = f(250, *popt)

print(f"config {CONFIG} lam={LAM}")
for n, y, e in zip(xs, ys, es):
    print(f"  ~{n:3d} sents -> {y*100:.2f}% (+/-{e*100:.2f})")
print(f"power-law fit: A={popt[0]:.4f} B={popt[1]:.4f} C={popt[2]:.4f}")
print(f"=> full-250 estimate = {est_250*100:.2f}%   "
      f"(plateau A = {popt[0]*100:.2f}%)")

# The report places this figure at 3.48 in wide, so effective pt = source pt * 3.48/figwidth.
# figsize 4.6 in with 13/14 pt fonts -> 9.8-10.6 pt printed, clearing the report's 9 pt minimum.
plt.rcParams.update({"font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14,
                     "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13})
plt.figure(figsize=(4.6, 3.02))
plt.errorbar(xs, ys * 100, yerr=es * 100, fmt="o", capsize=3, label="repeated 5-fold CV")
xx = np.linspace(xs.min(), 255, 200)
plt.plot(xx, f(xx, *popt) * 100, "-", label=r"power-law fit $A-Bn^{-C}$")
plt.scatter([250], [est_250 * 100], c="red", zorder=5,
            label=f"250-sent estimate = {est_250*100:.2f}%")
plt.axvline(250, ls="--", c="gray", lw=0.8)
plt.xlabel("# training sentences"); plt.ylabel("CV accuracy (%)")
plt.title(f"Model 2 learning curve ({CONFIG}, $\\lambda$={LAM})", fontsize=13)
plt.legend(loc="lower right", fontsize=12.5); plt.tight_layout()
plt.savefig("../guide/m2_learning_curve.png", dpi=200)
print("saved figure -> guide/m2_learning_curve.png")
