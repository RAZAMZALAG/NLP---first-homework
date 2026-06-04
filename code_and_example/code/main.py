"""Train, save weights, tag the competition file. Added to the provided driver:
- MODEL1_CONFIGS / MODEL2_CONFIGS: per-family-threshold presets fitting the caps; the submitted
  models are config L (Model 1, lam 0.3) and config G (Model 2, lam 1.0).
- --model_number N defaults to that model's tuned winner so the command reproduces the exact
  submitted weights; --config selects another preset, an explicit --threshold keeps scalar mode.
- cross_validate: k-fold CV for Model 2 (no held-out test set).
"""
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
# tag-context, cap/num). Same per-family-threshold mechanism as Model 1. Thresholds chosen
# from the train2 feature-count table (`measure_features.py 2`) to land each ~476-497 (<500).
# All drop f106/f107 (prev/next WORD: ~2800 sparse features, overfit on 250 sentences).
# Cheap tag-keyed flags (f_cap/f_num/f_is_number/f_hyphen/uppers, ~38 total) kept at thr 1.
MODEL2_CONFIGS = {
    # A: pure generalizer -- suffix + tag-context + shape + case/shape backoff. No exact word/prefix.
    "A": {"drop": ["f100", "f102", "f106", "f107", "f_prev_shape"],
          "thr": {"f101": 20, "f103": 20, "f104": 20, "f105": 1, "f_shape": 5,
                  "f_lower": 20, "f_next_shape": 20}},
    # B: add prefix (biomedical anti-/intra-), no lexical backoff.
    "B": {"drop": ["f100", "f106", "f107", "f_lower", "f_prev_shape", "f_next_shape"],
          "thr": {"f101": 30, "f102": 20, "f103": 20, "f104": 20, "f105": 1, "f_shape": 10}},
    # C: keep frequent exact words (function words) via f100, plus morphology + structure.
    "C": {"drop": ["f102", "f106", "f107", "f_lower", "f_prev_shape", "f_next_shape"],
          "thr": {"f100": 15, "f101": 20, "f103": 20, "f104": 15, "f105": 1, "f_shape": 10}},
    # D: structure-only -- suffix + tag trigram/bigram/unigram, no orthography at all.
    "D": {"drop": ["f100", "f102", "f106", "f107", "f_cap", "f_num", "f_shape", "f_all_upper",
                   "f_first_upper", "f_is_number", "f_hyphen", "f_lower", "f_prev_shape", "f_next_shape"],
          "thr": {"f101": 20, "f103": 7, "f104": 10, "f105": 1}},
    # E: orthography-heavy -- shape + prev/next-word shape for OOV context.
    "E": {"drop": ["f100", "f102", "f106", "f107"],
          "thr": {"f101": 30, "f103": 20, "f104": 20, "f105": 1, "f_shape": 3,
                  "f_lower": 20, "f_prev_shape": 15, "f_next_shape": 15}},
    # F: balanced -- suffix + light prefix + thin case backoff.
    "F": {"drop": ["f100", "f106", "f107", "f_prev_shape", "f_next_shape"],
          "thr": {"f101": 20, "f102": 50, "f103": 20, "f104": 20, "f105": 1, "f_shape": 10,
                  "f_lower": 30}},
    # --- Round 2: refinements around winner A (92.52%). A uses 484/500, 16 free. ---
    # G: max suffix -- pour budget into f101 (thr15=242), drop next-shape to pay.
    "G": {"drop": ["f100", "f102", "f106", "f107", "f_prev_shape", "f_next_shape"],
          "thr": {"f101": 15, "f103": 20, "f104": 20, "f105": 1, "f_shape": 7, "f_lower": 30}},
    # H: A + finer shape granularity (f_shape thr3=69).
    "H": {"drop": ["f100", "f102", "f106", "f107", "f_prev_shape"],
          "thr": {"f101": 20, "f103": 20, "f104": 20, "f105": 1, "f_shape": 3,
                  "f_lower": 20, "f_next_shape": 20}},
    # I: more case backoff (f_lower thr10=76), drop next-shape to pay.
    "I": {"drop": ["f100", "f102", "f106", "f107", "f_prev_shape", "f_next_shape"],
          "thr": {"f101": 20, "f103": 20, "f104": 20, "f105": 1, "f_shape": 5, "f_lower": 10}},
    # J: more tag-context (f103/f104 thr15), no lexical backoff -- structure vs backoff test.
    "J": {"drop": ["f100", "f102", "f106", "f107", "f_lower", "f_prev_shape", "f_next_shape"],
          "thr": {"f101": 20, "f103": 15, "f104": 15, "f105": 1, "f_shape": 7}},
    # K: A + prev-shape too (use the headroom for full left+right shape context).
    "K": {"drop": ["f100", "f102", "f106", "f107"],
          "thr": {"f101": 20, "f103": 20, "f104": 20, "f105": 1, "f_shape": 10,
                  "f_lower": 30, "f_prev_shape": 20, "f_next_shape": 20}},
    # --- Round 3: combine the two winning levers from R2 -- max suffix (G) + more backoff (I). ---
    # N: max suffix (f101 thr15) + mid backoff (f_lower thr15), keep some shape; trim f104 to pay.
    "N": {"drop": ["f100", "f102", "f106", "f107", "f_prev_shape", "f_next_shape"],
          "thr": {"f101": 15, "f103": 20, "f104": 30, "f105": 1, "f_shape": 10, "f_lower": 15}},
    # P: max suffix + max backoff, no shape (pure lexical morphology + case backoff + tag-context).
    "P": {"drop": ["f100", "f102", "f106", "f107", "f_shape", "f_prev_shape", "f_next_shape"],
          "thr": {"f101": 15, "f103": 20, "f104": 20, "f105": 1, "f_lower": 10}},
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
    parser.add_argument("--threshold", type=int, default=None,
                        help="Scalar feature threshold. Omit to use the model's tuned config.")
    parser.add_argument("--lam", type=float, default=None,
                        help="L2 lambda. Omit to use the tuned config's lambda.")
    # Optional add-ons: evaluate on test1.wtag (Model 1) or k-fold CV (Model 2).
    parser.add_argument("--eval_test", action="store_true",
                        help="Also tag data/test<N>.wtag and report accuracy.")
    parser.add_argument("--cv", type=int, default=0,
                        help="If >0, run k-fold CV on train file before final training.")
    parser.add_argument("--config", default=None,
                        choices=sorted(set(MODEL1_CONFIGS) | set(MODEL2_CONFIGS)),
                        help="Feature-budget preset; defaults to the model's tuned winner (M1=L, M2=G).")
    args = parser.parse_args()

    sid = f"{args.sid}"
    model_number = args.model_number
    trained_models_dir = "trained_models"

    if not os.path.exists(trained_models_dir):
        os.makedirs(trained_models_dir)

    train_path = f"data/train{model_number}.wtag"
    comp_path = f"data/comp{model_number}.words"
    test_path = f"data/test{model_number}.wtag"
    weights_path = f"{trained_models_dir}/weights_{model_number}.pkl"
    predictions_path = f"comp_m{model_number}_{sid}.wtag"

    # Tuned winners from our search -> bare `main.py --model_number N` reproduces the submitted
    # models exactly (M1: config L lam 0.3 = 95.93% / 9814 feat; M2: config G lam 1.0 / 491 feat).
    DEFAULT_CONFIG = {1: ("L", 0.3), 2: ("G", 1.0)}
    configs = MODEL1_CONFIGS if model_number == 1 else MODEL2_CONFIGS

    if args.threshold is not None:
        # Explicit scalar threshold: original behaviour (Model 2 keeps its reduced subset).
        threshold = args.threshold
        feature_subset = MODEL2_FEATURE_SUBSET if model_number == 2 else None
        lam = args.lam if args.lam is not None else 1.0
        print(f"[model {model_number} scalar threshold={threshold} lam={lam}]")
    else:
        cfgname = args.config or DEFAULT_CONFIG[model_number][0]
        if cfgname not in configs:
            parser.error(f"--config {cfgname} is not a Model {model_number} config "
                         f"(choose from {sorted(configs)})")
        cfg = configs[cfgname]
        feature_subset = [c for c in FEATURE_CLASSES if c not in cfg["drop"]]
        threshold = cfg["thr"]
        lam = args.lam if args.lam is not None else DEFAULT_CONFIG[model_number][1]
        print(f"[model {model_number} config {cfgname} lam={lam}] thr={cfg['thr']}")

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
