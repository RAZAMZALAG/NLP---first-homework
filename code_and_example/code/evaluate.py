import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import preprocess_train, read_test
from optimization import get_optimal_vector
from inference import memm_viterbi
import pickle
from collections import defaultdict

TRAIN_PATH = "data/train1.wtag"
TEST_PATH  = "data/test1.wtag"
WEIGHTS_PATH = "trained_models/weights_eval1.pkl"
THRESHOLD = 10
LAM = 1.0

os.makedirs("trained_models", exist_ok=True)

print("=== Training ===")
statistics, feature2id = preprocess_train(TRAIN_PATH, THRESHOLD)
get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=WEIGHTS_PATH, lam=LAM)

with open(WEIGHTS_PATH, "rb") as f:
    optimal_params, feature2id = pickle.load(f)
weights = optimal_params[0]

print("\n=== Evaluating on test1.wtag ===")
sentences = read_test(TEST_PATH, tagged=True)

total = 0
correct = 0
tag_correct = defaultdict(int)
tag_total   = defaultdict(int)
errors = defaultdict(int)

for words, gold_tags in sentences:
    pred_tags = memm_viterbi(words, weights, feature2id)[1:]
    real_words = words[2:-1]
    real_gold  = gold_tags[2:-1]

    for w, g, p in zip(real_words, real_gold, pred_tags):
        total += 1
        tag_total[g] += 1
        if g == p:
            correct += 1
            tag_correct[g] += 1
        else:
            errors[(g, p)] += 1

acc = correct / total * 100
print(f"\nOverall accuracy : {correct:,} / {total:,}  ({acc:.2f}%)")

print(f"\n{'Tag':<10}  {'Correct':>8}  {'Total':>8}  {'Acc%':>7}")
print(f"{'-'*10}  {'-'*8}  {'-'*8}  {'-'*7}")
for tag in sorted(tag_total, key=lambda t: -tag_total[t])[:20]:
    t_acc = tag_correct[tag] / tag_total[tag] * 100
    print(f"{tag:<10}  {tag_correct[tag]:>8,}  {tag_total[tag]:>8,}  {t_acc:>6.1f}%")

print(f"\nTop 10 confusion pairs (gold -> pred):")
print(f"{'Gold':<10}  {'Pred':<10}  {'Count':>8}")
print(f"{'-'*10}  {'-'*10}  {'-'*8}")
for (g, p), cnt in sorted(errors.items(), key=lambda x: -x[1])[:10]:
    print(f"{g:<10}  {p:<10}  {cnt:>8,}")
