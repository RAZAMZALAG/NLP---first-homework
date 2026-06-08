"""Experiment (allowed: use Model 1 data to train Model 2): keep config G's feature SET selected
from train2 (the biomedical target domain, <=500 params), but estimate the WEIGHTS on
train1+train2 so the domain-general families (suffix, tag-context, shape) get far more evidence.
Evaluated by repeated k-fold CV on train2 (held-out biomedical) vs the train2-only baseline.
Reproducible from the provided data only. Run from code_and_example with .venv.
"""
import os, pickle, shutil
import numpy as np
from preprocessing import preprocess_train, FeatureStatistics, FEATURE_CLASSES
from optimization import get_optimal_vector
from inference import tag_all_test, compute_accuracy

AUX, T2, WDIR = "data/train1.wtag", "data/train2.wtag", "cv_tmp"
G = {"drop": ["f100", "f102", "f106", "f107", "f_prev_shape", "f_next_shape"],
     "thr": {"f101": 15, "f103": 20, "f104": 20, "f105": 1, "f_shape": 7, "f_lower": 30}}
SUB = [c for c in FEATURE_CLASSES if c not in G["drop"]]


def _stats(lines, tag):
    os.makedirs(WDIR, exist_ok=True)
    p = f"{WDIR}/{tag}.wtag"
    with open(p, "w") as f: f.writelines(lines)
    fs = FeatureStatistics(); fs.get_word_tag_pair_count(p)
    return fs


def train_eval(feat_lines, weight_lines, val_lines, lam, tag):
    """Feature set from feat_lines; weights fit on weight_lines; accuracy on val_lines."""
    os.makedirs(WDIR, exist_ok=True)
    fp = f"{WDIR}/{tag}_f.wtag"
    with open(fp, "w") as f: f.writelines(feat_lines)
    _, f2i = preprocess_train(fp, G["thr"], SUB)          # feature_to_idx fixed from feat_lines
    if weight_lines is not feat_lines:
        f2i.feature_statistics = _stats(weight_lines, f"{tag}_w")  # histories from weight corpus
        f2i.calc_represent_input_with_features()                   # rebuild matrices, same features
    wp = f"{WDIR}/{tag}.pkl"
    get_optimal_vector(statistics=f2i.feature_statistics, feature2id=f2i, weights_path=wp, lam=lam)
    params, _ = pickle.load(open(wp, "rb"))
    vp = f"{WDIR}/{tag}_v.wtag"
    with open(vp, "w") as f: f.writelines(val_lines)
    pred = f"{WDIR}/{tag}_p.wtag"
    tag_all_test(vp, params[0], f2i, pred, tagged=True)
    return compute_accuracy(pred, vp), f2i.n_total_features


def cv(use_aux, lam=1.0, k=5, repeats=3, seed=0):
    aux = [l for l in open(AUX) if l.strip()] if use_aux else []
    t2 = [l for l in open(T2) if l.strip()]
    accs, nf = [], 0
    for r in range(repeats):
        idx = np.arange(len(t2)); np.random.default_rng(seed + r).shuffle(idx)
        folds = np.array_split(idx, k)
        for i in range(k):
            val = [t2[j] for j in folds[i]]
            tr2 = [t2[j] for ff in range(k) if ff != i for j in folds[ff]]
            weights = (aux + tr2) if use_aux else tr2
            acc, nf = train_eval(tr2, weights, val, lam, f"s{seed+r}f{i}")
            accs.append(acc)
    a = np.array(accs)
    return a.mean(), 1.96 * a.std(ddof=1) / np.sqrt(len(a)), nf


if __name__ == "__main__":
    try:
        m, ci, nf = cv(False)
        print(f"baseline  (weights: train2 only)        : {m*100:.2f}% +/-{ci*100:.2f}  ({nf} feat)")
        m, ci, nf = cv(True)
        print(f"augmented (weights: train1+train2)       : {m*100:.2f}% +/-{ci*100:.2f}  ({nf} feat)")
    finally:
        shutil.rmtree(WDIR, ignore_errors=True)
