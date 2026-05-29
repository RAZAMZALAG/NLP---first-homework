import pickle
import argparse
from inference import tag_all_test


def load_model(weights_path):
    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    return optimal_params[0], feature2id


def main():
    parser = argparse.ArgumentParser(description="Generate competition tagged files from trained models.")
    parser.add_argument("--sid", help="student ID")
    parser.add_argument("--model_number", help="Which model to use for tagging (1 or 2)", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()

    sid = f"{args.sid}"
    model_number = args.model_number

    print(f"Loading model {model_number}...")
    weights, feature2id = load_model(f"trained_models/weights_{model_number}.pkl")
    out_path = f"comp_m{model_number}_{sid}.wtag"
    print(f"Tagging comp{model_number}.words -> {out_path}")
    tag_all_test(f"data/comp{model_number}.words", weights, feature2id, out_path)

    print("Done.")


if __name__ == '__main__':
    main()
