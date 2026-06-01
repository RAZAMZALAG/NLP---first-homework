import numpy as np
from typing import List
from preprocessing import read_test, represent_input_with_features, Feature2id
from tqdm import tqdm

BEAM = 50  # max (prev_tag, cur_tag) states kept per position


def memm_viterbi(sentence: List[str], pre_trained_weights: np.ndarray, feature2id: Feature2id) -> List[str]:
    """
    Runs trigram MEMM Viterbi (log-space, with beam) over a single sentence.

    @param sentence: padded word list ["*", "*", w1, ..., wN, "~"] from read_test.
    @param pre_trained_weights: weight vector w of shape (n_features,).
    @param feature2id: the Feature2id object (feature_to_idx, feature_statistics.tags).
    @return: N+1 tags; the caller drops index 0, keeping tags for w1..wN.
    """
    feat_to_idx = feature2id.feature_to_idx
    all_tags = [t for t in feature2id.feature_statistics.tags if t not in ("*", "~")]

    # Candidate-tag pruning: tags each word was seen with in training (from f100).
    word_tags = {}
    for word, tag in feat_to_idx["f100"]:
        word_tags.setdefault(word, set()).add(tag)

    def cands(idx: int) -> List[str]:
        if idx < 2:
            return ["*"]
        seen = word_tags.get(sentence[idx])
        return sorted(seen) if seen else all_tags

    n = len(sentence) - 2          # index of the last real word
    pred = ["*"] * n               # pred[k-1] holds the tag of position k
    if n < 2:
        return pred

    pi = {("*", "*"): 0.0}         # (tag[k-2], tag[k-1]) -> best log-prob
    bp = {}                        # bp[k][(u, v)] = best tag[k-2]

    for k in range(2, n + 1):
        c_word, p_word, pp_word, n_word = sentence[k], sentence[k - 1], sentence[k - 2], sentence[k + 1]
        Sv = cands(k)
        new_pi, bp_k = {}, {}
        for (t, u), base in pi.items():
            scores = np.array([
                pre_trained_weights[idx].sum() if idx else 0.0
                for idx in (represent_input_with_features(
                    (c_word, v, p_word, u, pp_word, t, n_word), feat_to_idx) for v in Sv)
            ])
            log_q = scores - (scores.max() + np.log(np.exp(scores - scores.max()).sum()))
            for v, lq in zip(Sv, log_q):
                val, key = base + lq, (u, v)
                if key not in new_pi or val > new_pi[key]:
                    new_pi[key], bp_k[key] = val, t
        if len(new_pi) > BEAM:
            new_pi = dict(sorted(new_pi.items(), key=lambda kv: kv[1], reverse=True)[:BEAM])
        pi, bp[k] = new_pi, bp_k

    u, v = max(pi, key=pi.get)
    pred[n - 1] = v
    pred[n - 2] = u
    for k in range(n, 2, -1):      # recover tag[k-2] back to the start
        t = bp[k][(u, v)]
        pred[k - 3] = t
        u, v = t, u
    return pred


def tag_all_test(test_path: str, pre_trained_weights: np.ndarray, feature2id: Feature2id, predictions_path: str, tagged: bool = False) -> None:
    """
    Tags all sentences in test_path using memm_viterbi and writes results to predictions_path.
    @param tagged: set to True if test_path is a .wtag file (word_TAG format), False for plain .words files
    """
    test = read_test(test_path, tagged=tagged)

    output_file = open(predictions_path, "w")

    for sen in tqdm(test, total=len(test)):
        sentence = sen[0]
        # [1:] because memm_viterbi should return len(sentence) - 2 tags (N+1),
        # where the first tag corresponds to the second * padding token (and gets discarded).
        pred = memm_viterbi(sentence, pre_trained_weights, feature2id)[1:]
        sentence = sentence[2:]
        for i in range(len(pred)):
            if i > 0:
                output_file.write(" ")
            output_file.write(f"{sentence[i]}_{pred[i]}")
        output_file.write("\n")
    output_file.close()
