import argparse
import os
import pickle
import shutil
import numpy as np
from preprocessing import preprocess_train, FEATURE_CLASSES
from optimization import get_optimal_vector
from inference import tag_all_test, compute_accuracy

# Per-model defaults. Model 2 drops expensive classes to stay under the 500-feature cap.
MODEL2_FEATURE_SUBSET = ["f100", "f101", "f102", "f103", "f104", "f105",
                        "f_cap", "f_num", "f_shape", "f_all_upper", "f_first_upper", "f_is_number"]

# Model 1 feature-budget configs. The assignment REQUIRES every family f100-f107,
# so no family is dropped -- we only raise per-family thresholds to fit < 10,000 params.
# Classes absent from "thr" default to threshold 1 (kept fully). All three keep the
# full f100-f107 set + required cap/num + orthographic extras; they differ in where the
# budget goes (suffix/morphology vs. lexical-context prev/next-word vs. balanced).
# "drop" stays [] for Model 1; it exists only for completeness.
MODEL1_CONFIGS = {
    # --- Rounds 1-2 configs A-H, RETROFITTED (Round 5) with the proven case-backoff
    #     lever. Originally tuned before the f_lower / f_prev_shape / f_next_shape
    #     families existed, so they left those at thr 1 and blew past 10k (auto-skipped).
    #     Each now carries L's winning fixed block (f_lower:5, prev/next shape:2) and
    #     pays for it by raising f100/f101; the f102-f107 split still encodes each
    #     config's original CHARACTER, so this re-tests those shapes against L (95.93%).
    #     Run `tune.py --dry_run --configs A B C D E F G H` first to verify all land <10k.
    # A: balanced -- every family meaningfully represented (L-like baseline).
    "A": {"drop": [], "thr": {"f100": 30, "f101": 30, "f102": 150, "f103": 20,
                              "f104": 7, "f106": 25, "f107": 25, "f_shape": 2,
                              "f_lower": 5, "f_prev_shape": 2, "f_next_shape": 2}},
    # B: lexical-context leaning -- more prev/next-WORD (low f106/f107), pay via suffix.
    "B": {"drop": [], "thr": {"f100": 30, "f101": 40, "f102": 200, "f103": 20,
                              "f104": 7, "f106": 15, "f107": 15, "f_shape": 2,
                              "f_lower": 5, "f_prev_shape": 2, "f_next_shape": 2}},
    # C: morphology-max -- pour budget into suffix (low f101), context families thin.
    "C": {"drop": [], "thr": {"f100": 35, "f101": 20, "f102": 300, "f103": 25,
                              "f104": 8, "f106": 40, "f107": 40, "f_shape": 3,
                              "f_lower": 5, "f_prev_shape": 2, "f_next_shape": 2}},
    # D: starve prefix (high f102), reinvest in suffix + prev/next-word context.
    "D": {"drop": [], "thr": {"f100": 30, "f101": 20, "f102": 300, "f103": 20,
                              "f104": 7, "f106": 18, "f107": 18, "f_shape": 2,
                              "f_lower": 5, "f_prev_shape": 2, "f_next_shape": 2}},
    # E: rare words + max suffix, near-minimal prefix; moderate context.
    "E": {"drop": [], "thr": {"f100": 20, "f101": 18, "f102": 400, "f103": 25,
                              "f104": 8, "f106": 30, "f107": 30, "f_shape": 2,
                              "f_lower": 5, "f_prev_shape": 2, "f_next_shape": 2}},
    # F: more rare WORDS (low f100), trim suffix/trigram to pay for it.
    "F": {"drop": [], "thr": {"f100": 15, "f101": 30, "f102": 250, "f103": 25,
                              "f104": 8, "f106": 30, "f107": 25, "f_shape": 2,
                              "f_lower": 5, "f_prev_shape": 2, "f_next_shape": 2}},
    # G: more prev/next-WORD context (low f106/f107), prefix near-minimal.
    "G": {"drop": [], "thr": {"f100": 35, "f101": 35, "f102": 300, "f103": 25,
                              "f104": 7, "f106": 12, "f107": 12, "f_shape": 2,
                              "f_lower": 5, "f_prev_shape": 2, "f_next_shape": 2}},
    # H: more tag structure (low trigram f103 + bigram f104), pay via suffix/context.
    "H": {"drop": [], "thr": {"f100": 35, "f101": 40, "f102": 300, "f103": 12,
                              "f104": 3, "f106": 30, "f107": 25, "f_shape": 2,
                              "f_lower": 5, "f_prev_shape": 2, "f_next_shape": 2}},
    # --- Round 3: add generalizing/OOV families (f_lower case backoff, prev/next-word shape).
    #     New families need EXPLICIT thresholds here; absent => thr 1 => budget blows past 10k.
    #     All three pay for the new families by trimming F's winning shape.
    # I: lean on case backoff (f_lower thr8=1899). Pay via f100/f101 (redundant w/ f_lower). ~9869.
    "I": {"drop": [], "thr": {"f100": 15, "f101": 20, "f102": 200, "f103": 20,
                              "f104": 7, "f106": 25, "f107": 20, "f_shape": 2,
                              "f_lower": 8, "f_prev_shape": 2, "f_next_shape": 2}},
    # J: lean on context-shape (prev/next thr1=805/806); f_lower thin (thr15=992). ~9980.
    "J": {"drop": [], "thr": {"f100": 10, "f101": 20, "f102": 200, "f103": 20,
                              "f104": 7, "f106": 25, "f107": 20, "f_shape": 2,
                              "f_lower": 15, "f_prev_shape": 1, "f_next_shape": 1}},
    # K: balanced new families (f_lower thr12=1270, shapes thr2). ~9927.
    "K": {"drop": [], "thr": {"f100": 10, "f101": 20, "f102": 200, "f103": 20,
                              "f104": 7, "f106": 20, "f107": 20, "f_shape": 2,
                              "f_lower": 12, "f_prev_shape": 2, "f_next_shape": 2}},
    # --- Round 4: push case-backoff harder (I won R3). Cut f100/f101 (redundant w/ f_lower).
    # L: f_lower thr5=3003. f100 thr30=455, f101 thr30=1686. ~9814.
    "L": {"drop": [], "thr": {"f100": 30, "f101": 30, "f102": 200, "f103": 20,
                              "f104": 7, "f106": 25, "f107": 20, "f_shape": 2,
                              "f_lower": 5, "f_prev_shape": 2, "f_next_shape": 2}},
    # M: f_lower thr3=4749 (max). Everything else to floor.
    "M": {"drop": [], "thr": {"f100": 50, "f101": 50, "f102": 300, "f103": 30,
                              "f104": 10, "f106": 30, "f107": 30, "f_shape": 3,
                              "f_lower": 3, "f_prev_shape": 3, "f_next_shape": 3}},
}

# Model 2 (small, train2.wtag, <=500 params). 250 biomedical sentences with heavy OOV =>
# exact-word features overfit; budget goes to generalizing families (suffix, shape,
# tag-context, cap/num). Same per-family-threshold mechanism as Model 1. Filled after
# measuring train2 feature counts: `uv run python code/measure_features.py 2`.
MODEL2_CONFIGS = {
}


def _train_and_save(train_path: str, threshold: int, lam: float, weights_path: str,
                    feature_subset=None):
    """Train weights on train_path and pickle them to weights_path."""
    statistics, feature2id = preprocess_train(train_path, threshold, feature_subset)
    get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)
    with open(weights_path, "rb") as f:
        optimal_params, feature2id = pickle.load(f)
    return optimal_params[0], feature2id


def _kfold_indices(n: int, k: int, seed: int = 42):
    """Yield (train_idx, val_idx) for k folds. numpy-only (sklearn forbidden)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    for i in range(k):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, val


def cross_validate(train_path: str, k: int, threshold, lam: float, feature_subset):
    """k-fold CV for Model 2 evaluation (report requirement).

    All fold artifacts go to a throwaway `cv_tmp/` dir that is deleted afterwards,
    so they never leak into `trained_models/` (which gets zipped into the submission).
    Returns (mean_acc, accs_list).
    """
    cv_dir = "cv_tmp"
    os.makedirs(cv_dir, exist_ok=True)
    try:
        with open(train_path) as f:
            lines = f.readlines()
        accs = []
        for fold, (tr_idx, val_idx) in enumerate(_kfold_indices(len(lines), k)):
            # Materialize fold files (preprocess_train reads from disk).
            fold_train = os.path.join(cv_dir, f"fold{fold}_train.wtag")
            fold_val = os.path.join(cv_dir, f"fold{fold}_val.wtag")
            with open(fold_train, "w") as f: f.writelines(lines[i] for i in tr_idx)
            with open(fold_val, "w") as f:   f.writelines(lines[i] for i in val_idx)

            fold_w = os.path.join(cv_dir, f"fold{fold}_weights.pkl")
            fold_pred = os.path.join(cv_dir, f"fold{fold}_pred.wtag")
            weights, feature2id = _train_and_save(fold_train, threshold, lam, fold_w, feature_subset)
            # tagged=True so read_test parses gold tags for accuracy reporting downstream.
            tag_all_test(fold_val, weights, feature2id, fold_pred, tagged=True)
            acc = compute_accuracy(fold_pred, fold_val)
            print(f"[fold {fold}] acc={acc*100:.2f}%")
            accs.append(acc)
        mean = float(np.mean(accs))
        print(f"[CV mean] {mean*100:.2f}%  (folds: {[f'{a*100:.2f}' for a in accs]})")
        return mean, accs
    finally:
        shutil.rmtree(cv_dir, ignore_errors=True)  # keep trained_models/ clean for submission


def main():
    parser = argparse.ArgumentParser(description="Train MEMM and tag competition file.")
    parser.add_argument("--sid", help="student ID")
    parser.add_argument("--model_number", type=int, choices=[1, 2], default=1)
    parser.add_argument("--threshold", type=int, default=1)
    parser.add_argument("--lam", type=float, default=1.0)
    # Optional add-ons: evaluate on test1.wtag (Model 1) or k-fold CV (Model 2).
    parser.add_argument("--eval_test", action="store_true",
                        help="Also tag data/test<N>.wtag and report accuracy.")
    parser.add_argument("--cv", type=int, default=0,
                        help="If >0, run k-fold CV on train file before final training.")
    parser.add_argument("--config", choices=sorted(MODEL1_CONFIGS), default=None,
                        help="Model 1 feature-budget preset (per-family thresholds). Overrides --threshold.")
    args = parser.parse_args()

    sid = f"{args.sid}"
    model_number = args.model_number
    threshold = args.threshold
    lam = args.lam
    trained_models_dir = "trained_models"

    if not os.path.exists(trained_models_dir):
        os.makedirs(trained_models_dir)

    train_path = f"data/train{model_number}.wtag"
    comp_path = f"data/comp{model_number}.words"
    test_path = f"data/test{model_number}.wtag"
    weights_path = f"{trained_models_dir}/weights_{model_number}.pkl"
    predictions_path = f"comp_m{model_number}_{sid}.wtag"

    # Model 2 uses the reduced subset to satisfy the 500-feature cap.
    feature_subset = MODEL2_FEATURE_SUBSET if model_number == 2 else None

    # Model 1 preset: per-family thresholds + dropped classes, engineered to fit < 10k params.
    if args.config:
        cfg = MODEL1_CONFIGS[args.config]
        feature_subset = [c for c in FEATURE_CLASSES if c not in cfg["drop"]]
        threshold = cfg["thr"]
        print(f"[config {args.config}] drop={cfg['drop']} thr={cfg['thr']}")

    # Optional CV pass before fitting the production model.
    if args.cv > 0:
        cross_validate(train_path, args.cv, threshold, lam, feature_subset)

    # Final training on the full train set, save weights for graders to reproduce.
    pre_trained_weights, feature2id = _train_and_save(
        train_path, threshold, lam, weights_path, feature_subset)

    # Optional: report accuracy on the official held-out test set (only Model 1 has one).
    if args.eval_test and os.path.exists(test_path):
        test_pred = f"{trained_models_dir}/test{model_number}_pred.wtag"
        tag_all_test(test_path, pre_trained_weights, feature2id, test_pred, tagged=True)
        acc = compute_accuracy(test_pred, test_path)
        print(f"[test{model_number}.wtag] accuracy = {acc*100:.2f}%")

    # Mandatory: tag the competition file (no gold tags) -> comp_mN_<sid>.wtag.
    tag_all_test(comp_path, pre_trained_weights, feature2id, predictions_path, tagged=False)


if __name__ == "__main__":
    main()
