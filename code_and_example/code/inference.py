"""Decoding. We implemented the provided memm_viterbi stub and helpers:
- Trigram Viterbi in log-space with a beam (top-BEAM states/position) for tractable decoding.
- OOV candidate fall-back (cands): tags seen with the word, else its shape, else its suffix,
  else all tags -- so unseen words still get a sensible guess.
- compute_accuracy: word-level accuracy, used by our CV harness.
"""
import numpy as np
from typing import List
from preprocessing import read_test, represent_input_with_features, Feature2id, get_word_shape
from tqdm import tqdm

BEAM = 50  # max (prev_tag, cur_tag) states kept per position


def _build_word_tags(feature2id: Feature2id, all_tags: List[str]) -> dict:
    """Build word -> candidate-tag-set map for Viterbi pruning.

    Prefer the explicit `word_tags_dict` collected in FeatureStatistics
    (every (word, tag) seen, regardless of feature-threshold pruning).
    Fall back to scanning surviving f100 keys for older pickles.
    """
    stats = feature2id.feature_statistics
    wt = getattr(stats, "word_tags_dict", None)
    if wt:
        return {w: set(tags) for w, tags in wt.items()}
    # Backwards-compatible fallback: derive from f100 keys that passed threshold.
    out = {}
    for word, tag in feature2id.feature_to_idx.get("f100", {}):
        out.setdefault(word, set()).add(tag)
    return out


def _build_shape_tags(feature2id: Feature2id) -> dict:
    """shape -> tag-set, derived from surviving f_shape keys. OOV fallback signal."""
    out = {}
    for key in feature2id.feature_to_idx.get("f_shape", {}):
        shape, tag = key
        out.setdefault(shape, set()).add(tag)
    return out


def _build_suffix_tags(feature2id: Feature2id) -> dict:
    """suffix -> tag-set, derived from surviving f101 keys. Last-resort OOV signal."""
    out = {}
    for key in feature2id.feature_to_idx.get("f101", {}):
        suffix, tag = key
        out.setdefault(suffix, set()).add(tag)
    return out


def memm_viterbi(sentence: List[str], pre_trained_weights: np.ndarray, feature2id: Feature2id) -> List[str]:
    """
    Runs trigram MEMM Viterbi (log-space, with beam) over a single sentence.

    @param sentence: padded word list ["*", "*", w1, ..., wN, "~"] from read_test.
    @param pre_trained_weights: weight vector w of shape (n_features,).
    @param feature2id: the Feature2id object (feature_to_idx, feature_statistics.tags).
    @return: N+1 tags; the caller drops index 0, keeping tags for w1..wN.
    """
    feat_to_idx = feature2id.feature_to_idx
    clusters = getattr(feature2id.feature_statistics, "clusters", None)  # word-cluster map if any
    all_tags = [t for t in feature2id.feature_statistics.tags if t not in ("*", "~")]

    # Candidate-tag sources, in priority order: seen-with-word > seen-with-shape > seen-with-suffix.
    word_tags = _build_word_tags(feature2id, all_tags)
    shape_tags = _build_shape_tags(feature2id)
    suffix_tags = _build_suffix_tags(feature2id)

    def cands(idx: int) -> List[str]:
        """OOV fallback chain: word -> shape -> longest matching suffix -> all tags."""
        if idx < 2:
            return ["*"]
        w = sentence[idx]
        seen = word_tags.get(w)
        if seen:
            return sorted(seen)
        # Shape fallback: e.g. unseen proper noun "Smith" -> shape "Xx" -> NNP-like tags.
        sh = get_word_shape(w)
        seen = shape_tags.get(sh)
        if seen:
            return sorted(seen)
        # Suffix fallback: try longest -> shortest; "-ing" -> VBG, "-ed" -> VBD/VBN, etc.
        for k in range(min(4, len(w)), 0, -1):
            seen = suffix_tags.get(w[-k:])
            if seen:
                return sorted(seen)
        return all_tags

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
            # Score every candidate v under context (t, u) and softmax-normalize over Sv.
            scores = np.array([
                pre_trained_weights[idx].sum() if idx else 0.0
                for idx in (represent_input_with_features(
                    (c_word, v, p_word, u, pp_word, t, n_word), feat_to_idx, clusters) for v in Sv)
            ])
            log_q = scores - (scores.max() + np.log(np.exp(scores - scores.max()).sum()))
            for v, lq in zip(Sv, log_q):
                val, key = base + lq, (u, v)
                if key not in new_pi or val > new_pi[key]:
                    new_pi[key], bp_k[key] = val, t
        # Beam prune: keep top-BEAM states by score to cap runtime on long sentences.
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


def compute_accuracy(predictions_path: str, gold_path: str) -> float:
    """Word-level accuracy: compare predicted .wtag to gold .wtag, same sentence order."""
    correct = total = 0
    with open(predictions_path) as pf, open(gold_path) as gf:
        for pl, gl in zip(pf, gf):
            for pt, gt in zip(pl.split(), gl.split()):
                # rsplit so words containing '_' don't break the split.
                pw, ptag = pt.rsplit("_", 1)
                gw, gtag = gt.rsplit("_", 1)
                total += 1
                if ptag == gtag:
                    correct += 1
    return correct / total if total else 0.0
