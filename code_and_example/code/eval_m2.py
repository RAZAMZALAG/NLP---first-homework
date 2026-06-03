"""Model 2 evaluation + search harness: repeated k-fold cross-validation on train2.wtag.

train2 has no held-out test set, so CV is our only honest accuracy signal -- and the report is
graded on |estimated - actual|, so a stable, well-calibrated estimate matters.

Modes:
  * single config  : repeated k-fold CV for one config/threshold, mean +/- 95% CI, optional
                     learning curve (extrapolates fold accuracy to the full-250 model).
  * grid (--configs): CV-rank several configs x lambdas, print a leaderboard, and optionally
                     finalize the winner (train on all 250 -> weights_2.pkl + comp_m2_<sid>.wtag).

Run on the VM:
  uv run python code/eval_m2.py --config A --lam 1.0 --repeats 5 --curve
  uv run python code/eval_m2.py --configs A B C D E F --lams 0.5 1.0 2.0 --finalize --sid 000000000
"""
import argparse
import os
import pickle
import shutil
import numpy as np
from preprocessing import preprocess_train, FEATURE_CLASSES
from optimization import get_optimal_vector
from inference import tag_all_test, compute_accuracy
from main import MODEL2_CONFIGS, MODEL2_FEATURE_SUBSET

TRAIN = "data/train2.wtag"
COMP = "data/comp2.words"
WDIR = "cv_tmp"          # throwaway; gitignored, never submitted


def cfg_thr_subset(name):
    cfg = MODEL2_CONFIGS[name]
    subset = [c for c in FEATURE_CLASSES if c not in cfg.get("drop", [])]
    return cfg["thr"], subset


def full_count(threshold, subset):
    """Feature count of the model trained on ALL 250 sentences -- the number the cap applies to."""
    _, f2i = preprocess_train(TRAIN, threshold, subset)
    return f2i.n_total_features


def _train_eval(train_lines, val_lines, threshold, subset, lam, tag):
    """Materialize a train/val split, train, tag val, return accuracy."""
    os.makedirs(WDIR, exist_ok=True)
    ftr, fval = f"{WDIR}/{tag}_tr.wtag", f"{WDIR}/{tag}_val.wtag"
    with open(ftr, "w") as f:  f.writelines(train_lines)
    with open(fval, "w") as f: f.writelines(val_lines)
    stats, f2i = preprocess_train(ftr, threshold, subset)
    wpath = f"{WDIR}/{tag}_w.pkl"
    get_optimal_vector(statistics=stats, feature2id=f2i, weights_path=wpath, lam=lam)
    with open(wpath, "rb") as f:
        params, _ = pickle.load(f)
    pred = f"{WDIR}/{tag}_pred.wtag"
    tag_all_test(fval, params[0], f2i, pred, tagged=True)
    return compute_accuracy(pred, fval)


def kfold(lines, threshold, subset, lam, k, seed):
    """One k-fold pass over sentence indices. Returns list of k fold accuracies."""
    idx = np.arange(len(lines))
    np.random.default_rng(seed).shuffle(idx)
    folds = np.array_split(idx, k)
    accs = []
    for i, val_i in enumerate(folds):
        tr_i = np.concatenate([folds[j] for j in range(k) if j != i])
        accs.append(_train_eval([lines[j] for j in tr_i], [lines[j] for j in val_i],
                                threshold, subset, lam, f"s{seed}f{i}"))
    return accs


def repeated_cv(lines, threshold, subset, lam, k, repeats, seed):
    """Repeated k-fold CV; returns the flat array of all fold accuracies."""
    accs = []
    for r in range(repeats):
        accs += kfold(lines, threshold, subset, lam, k, seed + r)
    return np.array(accs)


def ci95(a):
    return 1.96 * a.std(ddof=1) / np.sqrt(len(a))


def finalize(name, lam, sid):
    """Train the chosen config on ALL 250 sentences, save weights_2.pkl, tag comp2."""
    threshold, subset = cfg_thr_subset(name)
    os.makedirs("trained_models", exist_ok=True)
    stats, f2i = preprocess_train(TRAIN, threshold, subset, verbose=True)
    get_optimal_vector(statistics=stats, feature2id=f2i,
                       weights_path="trained_models/weights_2.pkl", lam=lam)
    with open("trained_models/weights_2.pkl", "rb") as f:
        params, f2i = pickle.load(f)
    out = f"comp_m2_{sid}.wtag"
    tag_all_test(COMP, params[0], f2i, out, tagged=False)
    print(f"finalized config={name} lam={lam}: {f2i.n_total_features} features "
          f"-> trained_models/weights_2.pkl + {out}")


def main():
    ap = argparse.ArgumentParser(description="Repeated k-fold CV + grid search for Model 2.")
    ap.add_argument("--config", default=None, help="Single named MODEL2_CONFIGS preset.")
    ap.add_argument("--configs", nargs="+", default=None, help="Grid of presets to CV-rank.")
    ap.add_argument("--lams", type=float, nargs="+", default=[1.0], help="Lambdas for the grid.")
    ap.add_argument("--threshold", type=int, default=1, help="Scalar threshold (single mode, no config).")
    ap.add_argument("--subset", action="store_true", help="Use MODEL2_FEATURE_SUBSET (single scalar mode).")
    ap.add_argument("--lam", type=float, default=1.0, help="Lambda (single mode).")
    ap.add_argument("--k", type=int, default=5, help="Folds.")
    ap.add_argument("--repeats", type=int, default=5, help="Repeated CV passes (different seeds).")
    ap.add_argument("--seed", type=int, default=0, help="Base seed.")
    ap.add_argument("--curve", action="store_true", help="Learning curve (single mode).")
    ap.add_argument("--finalize", action="store_true", help="Train+save the grid winner.")
    ap.add_argument("--sid", default="000000000", help="Student ID for the comp file name.")
    args = ap.parse_args()

    with open(TRAIN) as f:
        lines = [ln for ln in f if ln.strip()]

    try:
        if args.configs:                         # ---- GRID MODE ----
            results = []
            for name in args.configs:
                threshold, subset = cfg_thr_subset(name)
                n = full_count(threshold, subset)
                if n > 500:
                    print(f"  {name}: {n} features OVER 500 -> SKIP")
                    continue
                for lam in args.lams:
                    a = repeated_cv(lines, threshold, subset, lam, args.k, args.repeats, args.seed)
                    results.append((a.mean(), ci95(a), a.std(ddof=1), name, lam, n))
                    print(f"  {name} lam={lam}: CV {a.mean()*100:.2f}% +/-{ci95(a)*100:.2f}  "
                          f"(std {a.std(ddof=1)*100:.2f}, {n} feat)")
            if not results:
                print("No configs under the cap.")
                return
            results.sort(reverse=True)
            print("\n=== MODEL 2 CV LEADERBOARD (train2, repeated k-fold) ===")
            print(f"{'CV%':>7} {'+/-':>5} {'cfg':>4} {'lam':>5} {'feat':>5}")
            for m, ci, sd, name, lam, n in results:
                print(f"{m*100:7.2f} {ci*100:5.2f} {name:>4} {lam:>5} {n:>5}")
            best = results[0]
            print(f"\nBEST: config={best[3]} lam={best[4]}  CV={best[0]*100:.2f}% +/-{best[1]*100:.2f}")
            if args.finalize:
                finalize(best[3], best[4], args.sid)

        else:                                    # ---- SINGLE MODE ----
            if args.config:
                threshold, subset = cfg_thr_subset(args.config)
                label = f"config={args.config}"
            else:
                threshold = args.threshold
                subset = MODEL2_FEATURE_SUBSET if args.subset else None
                label = f"thr={args.threshold}{' subset' if args.subset else ''}"
            n = full_count(threshold, subset)
            flag = "OK <=500" if n <= 500 else "OVER 500 -> GRADE 0"
            print(f"[{label} lam={args.lam}]  full-250 features = {n}  [{flag}]")
            if n > 500:
                return
            a = repeated_cv(lines, threshold, subset, args.lam, args.k, args.repeats, args.seed)
            print(f"  CV mean = {a.mean()*100:.2f}% +/- {ci95(a)*100:.2f} (95% CI, "
                  f"n={len(a)} folds)  std={a.std(ddof=1)*100:.2f}")

            if args.curve:
                print("\n  learning curve (CV mean vs ~#train sentences/fold):")
                rng = np.random.default_rng(args.seed)
                order = rng.permutation(len(lines))
                curve = []
                for frac in (0.5, 0.7, 0.85, 1.0):
                    nsub = max(args.k + 1, int(len(lines) * frac))
                    sub = [lines[i] for i in order[:nsub]]
                    m = float(repeated_cv(sub, threshold, subset, args.lam, args.k, 2, args.seed).mean())
                    curve.append((nsub * (args.k - 1) // args.k, m))
                    print(f"    ~{curve[-1][0]:3d} sents -> {m*100:.2f}%")
                (x0, y0), (x1, y1) = curve[-2], curve[-1]
                slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0.0
                est = y1 + slope * (len(lines) - x1)
                print(f"  => extrapolated full-250 estimate ~ {est*100:.2f}%")
    finally:
        shutil.rmtree(WDIR, ignore_errors=True)


if __name__ == "__main__":
    main()
