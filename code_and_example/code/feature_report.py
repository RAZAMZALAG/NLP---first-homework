import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import preprocess_train

TRAIN1 = "data/train1.wtag"
TRAIN2 = "data/train2.wtag"
THRESHOLD = 1

FEATURE_DESCRIPTIONS = {
    "f100": "word+tag pair                (c_word, c_tag)",
    "f101": "word suffix len-3 + tag      (suffix3, c_tag)",
    "f102": "word prefix len-3 + tag      (prefix3, c_tag)",
    "f103": "trigram tags                 (pp_tag, p_tag, c_tag)",
    "f104": "bigram tags                  (p_tag, c_tag)",
    "f105": "unigram tag                  (c_tag)",
    "f106": "previous word + tag          (p_word, c_tag)",
    "f107": "next word + tag              (n_word, c_tag)",
}


def report(train_path: str, model_num: int):
    print(f"\n{'='*65}")
    print(f"  MODEL {model_num} — {train_path}")
    print(f"{'='*65}")

    stats, f2id = preprocess_train(train_path, THRESHOLD)

    implemented = list(stats.feature_rep_dict.keys())
    print(f"\n  {'Feature':<8}  {'Description':<48}  {'Gold Fires':>10}  {'Unique':>8}")
    print(f"  {'-'*8}  {'-'*48}  {'-'*10}  {'-'*8}")

    for feat_class in implemented:
        counts = stats.feature_rep_dict[feat_class]
        total_gold = sum(counts.values())
        unique_feats = len(counts)
        desc = FEATURE_DESCRIPTIONS.get(feat_class, "(no description)")
        print(f"  {feat_class:<8}  {desc:<48}  {total_gold:>10,}  {unique_feats:>8,}")

    print(f"\n  Total implemented feature classes : {len(implemented)}")
    print(f"  Total features after threshold={THRESHOLD}: {f2id.n_total_features:,}")
    print(f"  Total tags seen                  : {len(stats.tags)}")
    print(f"  Total training histories         : {len(stats.histories):,}")
    print()


if __name__ == "__main__":
    for i, path in [(1, TRAIN1), (2, TRAIN2)]:
        if os.path.exists(path):
            report(path, i)
        else:
            print(f"\n[skip] {path} not found")
