import argparse
import os
import pickle
import numpy as np
from preprocessing import preprocess_train
from optimization import get_optimal_vector
from inference import tag_all_test, compute_accuracy

# Per-model defaults. Model 2 drops expensive classes to stay under the 500-feature cap.
MODEL2_FEATURE_SUBSET = ["f100", "f101", "f102", "f103", "f104", "f105",
                        "f_cap", "f_num", "f_shape", "f_all_upper", "f_first_upper", "f_is_number"]


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


def cross_validate(train_path: str, k: int, threshold: int, lam: float,
                   feature_subset, trained_models_dir: str):
    """k-fold CV for Model 2 evaluation (report requirement).

    Writes per-fold split files to trained_models_dir, trains + scores each fold,
    returns (mean_acc, accs_list).
    """
    with open(train_path) as f:
        lines = f.readlines()
    accs = []
    for fold, (tr_idx, val_idx) in enumerate(_kfold_indices(len(lines), k)):
        # Materialize fold files (preprocess_train reads from disk).
        fold_train = os.path.join(trained_models_dir, f"fold{fold}_train.wtag")
        fold_val = os.path.join(trained_models_dir, f"fold{fold}_val.wtag")
        with open(fold_train, "w") as f: f.writelines(lines[i] for i in tr_idx)
        with open(fold_val, "w") as f:   f.writelines(lines[i] for i in val_idx)

        fold_w = os.path.join(trained_models_dir, f"fold{fold}_weights.pkl")
        fold_pred = os.path.join(trained_models_dir, f"fold{fold}_pred.wtag")
        weights, feature2id = _train_and_save(fold_train, threshold, lam, fold_w, feature_subset)
        # tagged=True so read_test parses gold tags for accuracy reporting downstream.
        tag_all_test(fold_val, weights, feature2id, fold_pred, tagged=True)
        acc = compute_accuracy(fold_pred, fold_val)
        print(f"[fold {fold}] acc={acc*100:.2f}%")
        accs.append(acc)
    mean = float(np.mean(accs))
    print(f"[CV mean] {mean*100:.2f}%  (folds: {[f'{a*100:.2f}' for a in accs]})")
    return mean, accs


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

    # Optional CV pass before fitting the production model.
    if args.cv > 0:
        cross_validate(train_path, args.cv, threshold, lam, feature_subset, trained_models_dir)

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
