import argparse
import os
import pickle
from preprocessing import preprocess_train
from optimization import get_optimal_vector
from inference import tag_all_test


def main():
    parser = argparse.ArgumentParser(description="Generate competition tagged files from trained models.")
    parser.add_argument("--sid", help="student ID")
    parser.add_argument("--model_number", help="Which model to use for tagging (1 or 2)", type=int, choices=[1, 2], default=1)
    parser.add_argument("--threshold", help="Threshold for feature selection", type=int, default=1)
    parser.add_argument("--lam", help="L2 regularization parameter", type=float, default=1)

    args = parser.parse_args()
    
    sid = f"{args.sid}"
    model_number = args.model_number
    threshold = args.threshold
    lam = args.lam
    trained_models_dir = "trained_models"
    
    if not os.path.exists(trained_models_dir):
        os.makedirs(trained_models_dir)

    train_path = f"data/train{model_number}.wtag"
    test_path = f"data/comp{model_number}.words"

    weights_path = f'{trained_models_dir}/weights_{model_number}.pkl'
    predictions_path = f'comp_m{model_number}_{sid}.wtag'

    statistics, feature2id = preprocess_train(train_path, threshold)
    get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]

    print(pre_trained_weights)
    tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path, tagged=False)


if __name__ == '__main__':
    main()
