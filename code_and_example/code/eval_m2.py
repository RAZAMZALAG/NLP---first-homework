"""Model 2 evaluation harness: repeated k-fold cross-validation on train2.wtag.

train2 has no held-out test set, so CV is our only honest accuracy signal -- and the report is
graded on |estimated - actual|, so a stable, well-calibrated estimate matters. This script:
  * runs repeated k-fold CV for a given feature config + lambda,
  * reports mean / std / 95% CI over all folds,
  * reports the FULL-250 model's feature count (the number the 500-cap applies to),
  * optionally traces a learning curve to extrapolate fold accuracy up to the full-250 model
    (fold models train on fewer sentences, so plain CV slightly UNDER-estimates the final model).

Run on the VM:
  uv run python code/eval_m2.py --threshold 10 --subset --lam 1.0
  uv run python code/eval_m2.py --config A --lam 0.5 --repeats 5 --curve
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
WDIR = "cv_tmp"          # throwaway; gitignored, never submitted


def _resolve(args):
    """Return (threshold, feature_subset, label) from --config or --threshold/--subset."""
    if args.config:
        cfg = MODEL2_CONFIGS[args.config]
        subset = [c for c in FEATURE_CLASSES if c not in cfg.get("drop", [])]
        return cfg["thr"], subset, f"config={args.config}"
    subset = MODEL2_FEATURE_SUBSET if args.subset else None
    return args.threshold, subset, f"thr={args.threshold}{' subset' if args.subset else ''}"


def _train_eval(train_lines, val_lines, threshold, subset, lam, tag):
    """Materialize a train/val split, train, tag val, return (accuracy, n_features)."""
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
    return compute_accuracy(pred, fval), f2i.n_total_features


def kfold(lines, threshold, subset, lam, k, seed):
    """One k-fold pass over sentence indices. Returns list of k fold accuracies."""
    idx = np.arange(len(lines))
    np.random.default_rng(seed).shuffle(idx)
    folds = np.array_split(idx, k)
    accs = []
    for i, val_i in enumerate(folds):
        tr_i = np.concatenate([folds[j] for j in range(k) if j != i])
        acc, _ = _train_eval([lines[j] for j in tr_i], [lines[j] for j in val_i],
                             threshold, subset, lam, f"s{seed}f{i}")
        accs.append(acc)
    return accs


def main():
    ap = argparse.ArgumentParser(description="Repeated k-fold CV for Model 2.")
    ap.add_argument("--config", default=None, help="Named MODEL2_CONFIGS preset.")
    ap.add_argument("--threshold", type=int, default=1, help="Scalar threshold if no --config.")
    ap.add_argument("--subset", action="store_true", help="Use MODEL2_FEATURE_SUBSET.")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=5, help="Folds.")
    ap.add_argument("--repeats", type=int, default=5, help="Repeated CV passes (different seeds).")
    ap.add_argument("--seed", type=int, default=0, help="Base seed.")
    ap.add_argument("--curve", action="store_true", help="Also trace a learning curve.")
    args = ap.parse_args()

    with open(TRAIN) as f:
        lines = [ln for ln in f if ln.strip()]
    threshold, subset, label = _resolve(args)

    # Full-250 feature count -- this is the number the 500-cap is measured against.
    full_stats, full_f2i = preprocess_train(TRAIN, threshold, subset)
    n_full = full_f2i.n_total_features
    cap_flag = "OK <=500" if n_full <= 500 else "OVER 500 -> GRADE 0"
    print(f"[{label} lam={args.lam}]  full-250 features = {n_full}  [{cap_flag}]")
    if n_full > 500:
        print("  refusing CV: config exceeds the cap, redesign thresholds first.")
        shutil.rmtree(WDIR, ignore_errors=True)
        return

    try:
        all_accs = []
        for r in range(args.repeats):
            accs = kfold(lines, threshold, subset, args.lam, args.k, args.seed + r)
            all_accs += accs
            print(f"  repeat {r}: " + " ".join(f"{a*100:.2f}" for a in accs)
                  + f"   (mean {np.mean(accs)*100:.2f})")
        a = np.array(all_accs)
        ci = 1.96 * a.std(ddof=1) / np.sqrt(len(a))
        print(f"\n  CV mean = {a.mean()*100:.2f}%  +/- {ci*100:.2f} (95% CI, n={len(a)} folds)"
              f"  std={a.std(ddof=1)*100:.2f}")

        if args.curve:
            print("\n  learning curve (CV mean acc vs #train sentences):")
            base = lines
            rng = np.random.default_rng(args.seed)
            order = rng.permutation(len(base))
            curve = []
            for frac in (0.5, 0.7, 0.85, 1.0):
                n = max(args.k + 1, int(len(base) * frac))
                sub = [base[i] for i in order[:n]]
                accs = kfold(sub, threshold, subset, args.lam, args.k, args.seed)
                m = float(np.mean(accs))
                curve.append((n * (args.k - 1) // args.k, m))   # ~train size per fold
                print(f"    ~{curve[-1][0]:3d} train sents -> {m*100:.2f}%")
            # Linear extrapolation of the last two points to the full-250 model's train size.
            (x0, y0), (x1, y1) = curve[-2], curve[-1]
            full_train = len(base)
            slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0.0
            est = y1 + slope * (full_train - x1)
            print(f"  => extrapolated full-250 estimate ~ {est*100:.2f}%  "
                  f"(plain CV mean was {a.mean()*100:.2f}%)")
    finally:
        shutil.rmtree(WDIR, ignore_errors=True)


if __name__ == "__main__":
    main()
