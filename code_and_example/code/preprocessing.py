"""Feature extraction and indexing. Added to the provided f100 scaffold:
- iter_features: Ratnaparkhi f101-f107 + required capital/number features, plus orthographic and
  OOV back-off families (word shape, f_lower, prev/next-word shape) that key on coarser signals
  than the exact word to generalize to unseen words. Single definition shared by train/inference.
- Feature2id._thr: per-family thresholds (threshold may be a {class: min_count} dict) to fit the
  parameter caps by family; optional feature_subset drops whole families.
"""
from scipy import sparse
from collections import OrderedDict, defaultdict
import numpy as np
from typing import List, Dict, Tuple, Iterator

WORD = 0
TAG = 1

# History tuple: (c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word)
History = Tuple[str, str, str, str, str, str, str]

# Ratnaparkhi (1996) f100-f107 plus HW-required capital/number features
# and orthographic extras (shape, all-upper, first-upper-mid-sentence, numeric).
FEATURE_CLASSES = [
    "f100", "f101", "f102", "f103", "f104", "f105", "f106", "f107",
    "f_cap", "f_num",          # HW-required: word-has-uppercase / word-has-digit
    "f_shape",                 # word shape pattern (e.g. "Xx", "dd-dd", "Xx-Xx")
    "f_all_upper",             # whole word in uppercase (acronyms: NASA, IBM)
    "f_first_upper",           # capital mid-sentence (proper-noun signal, not sentence start)
    "f_is_number",             # whole word is numeric (covers floats: 3.14, -2)
    "f_hyphen",                # word contains an internal hyphen (JJ/CD: "year-earlier")
    # Generalizing / OOV families: back off from exact lexical word to a coarser key,
    # so evidence transfers to words unseen (or seen only in another case) in training.
    "f_lower",                 # lowercased current word + tag (case backoff: "The"~"the")
    "f_prev_shape",            # shape of previous word + tag (context that fires on OOV neighbors)
    "f_next_shape",            # shape of next word + tag
]


def get_word_shape(word: str) -> str:
    """Collapse word into orthographic shape pattern.

    Runs are compressed: "Apple"->"Xx", "HELLO"->"X", "Wi-Fi"->"Xx-Xx",
    "3D"->"dX", "123"->"d". Strong OOV signal for proper nouns / numbers.
    """
    if not word:
        return ""
    shape = []
    for ch in word:
        if ch.isupper():   shape.append("X")
        elif ch.islower(): shape.append("x")
        elif ch.isdigit(): shape.append("d")
        else:              shape.append(ch)
    # collapse consecutive duplicates so "Apple"=Xxxxx -> "Xx"
    out, prev = [], None
    for c in shape:
        if c != prev:
            out.append(c)
            prev = c
    return "".join(out)


def is_number(word: str) -> bool:
    """True iff whole word parses as a number (3.14, -2, 1e5). Tighter than has-digit."""
    try:
        float(word)
        return True
    except ValueError:
        return False


def iter_features(history: History) -> Iterator[Tuple[str, tuple]]:
    """Yields (feature_class, feature_key) for every feature that fires on `history`.

    Single source of truth: used both when counting features (training) and
    when representing a history as active indices (matrix build + inference).
    """
    c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word = history

    yield "f100", (c_word, c_tag)                      # word + tag (Ratnaparkhi base)
    for k in range(1, 5):                              # affixes of length 1..4
        if len(c_word) >= k:
            yield "f101", (c_word[-k:], c_tag)         # suffix + tag
            yield "f102", (c_word[:k], c_tag)          # prefix + tag
    yield "f103", (pp_tag, p_tag, c_tag)               # tag trigram
    yield "f104", (p_tag, c_tag)                       # tag bigram
    yield "f105", (c_tag,)                             # tag unigram
    yield "f106", (p_word, c_tag)                      # previous word + tag
    yield "f107", (n_word, c_tag)                      # next word + tag

    # HW-required: capital/number flags keyed on tag only (low-cost, generalize across OOV).
    if any(ch.isupper() for ch in c_word):
        yield "f_cap", (c_tag,)
    if any(ch.isdigit() for ch in c_word):
        yield "f_num", (c_tag,)

    # Orthographic extras (keyed on tag, NOT on full word, to keep param count down).
    yield "f_shape", (get_word_shape(c_word), c_tag)   # shape pattern + tag
    if c_word.isupper() and len(c_word) > 1:
        yield "f_all_upper", (c_tag,)                  # acronym signal
    # Distinguish mid-sentence capital (proper noun) from sentence-initial capital.
    if c_word[:1].isupper() and p_word != "*":
        yield "f_first_upper", (c_tag,)
    if is_number(c_word):
        yield "f_is_number", (c_tag,)                  # whole-word numeric (floats too)
    if "-" in c_word[1:-1]:
        yield "f_hyphen", (c_tag,)                     # internal hyphen (compound adj/number)

    # Generalizing backoff families (keyed coarser than exact word -> transfer to OOV).
    yield "f_lower", (c_word.lower(), c_tag)           # case-insensitive word backoff
    yield "f_prev_shape", (get_word_shape(p_word), c_tag)  # prev-word shape (fires even if p_word OOV)
    yield "f_next_shape", (get_word_shape(n_word), c_tag)  # next-word shape


class FeatureStatistics:
    def __init__(self):
        self.n_total_features = 0
        self.feature_rep_dict = {fc: defaultdict(int) for fc in FEATURE_CLASSES}  # class -> (feature -> count)
        self.tags = {"~"}
        self.tags_counts = defaultdict(int)
        self.words_count = defaultdict(int)
        self.histories = []
        # Maps word -> set of tags seen with it in training. Used by Viterbi to
        # prune the candidate tag set per position (huge speedup, small acc cost).
        self.word_tags_dict = defaultdict(set)

    def get_word_tag_pair_count(self, file_path: str) -> None:
        """
        Reads a tagged file, updates feature counts, tag/word counts, and histories list.
        Each history is: (c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word)
        """
        with open(file_path) as f:
            for line in f:
                pairs = line.rstrip("\n").split()
                sentence = [("*", "*"), ("*", "*")] + [tuple(p.split("_")) for p in pairs] + [("~", "~")]

                for i in range(2, len(sentence) - 1):
                    c, p, pp, n = sentence[i], sentence[i - 1], sentence[i - 2], sentence[i + 1]
                    self.tags.add(c[1])
                    self.tags_counts[c[1]] += 1
                    self.words_count[c[0]] += 1
                    self.word_tags_dict[c[0]].add(c[1])  # record observed (word, tag) for Viterbi pruning
                    history = (c[0], c[1], p[0], p[1], pp[0], pp[1], n[0])
                    self.histories.append(history)
                    for feat_class, key in iter_features(history):
                        self.feature_rep_dict[feat_class][key] += 1


class Feature2id:
    def __init__(self, feature_statistics: FeatureStatistics, threshold, feature_subset: List[str] = None):
        """
        @param feature_statistics: the feature statistics object
        @param threshold: minimum appearances for a feature to be kept. Either an int
                          (same threshold for every class) or a dict {feat_class: int}
                          for per-family control (classes absent from the dict default to 1).
        @param feature_subset: optional list of feature classes to keep. None = all classes.
                               Used for Model 2 (500-param cap) to drop expensive classes.
        """
        self.feature_statistics = feature_statistics
        self.threshold = threshold
        # active_features = classes actually used; everything else is dropped at index assignment.
        self.active_features = list(feature_subset) if feature_subset else list(FEATURE_CLASSES)
        self.n_total_features = 0
        self.feature_to_idx = {fc: OrderedDict() for fc in self.active_features}
        self.histories_features = OrderedDict()
        self.small_matrix = sparse.csr_matrix
        self.big_matrix = sparse.csr_matrix

    def _thr(self, feat_class: str) -> int:
        """Per-family threshold: dict lookup (default 1) or scalar for all classes."""
        if isinstance(self.threshold, dict):
            return self.threshold.get(feat_class, 1)
        return self.threshold

    def get_features_idx(self) -> None:
        """Assigns an index to each feature meeting its (possibly per-family) threshold."""
        # Only iterate over active_features so dropped classes get zero indices.
        for feat_class in self.active_features:
            thr = self._thr(feat_class)
            for feat, count in self.feature_statistics.feature_rep_dict[feat_class].items():
                if count >= thr:
                    self.feature_to_idx[feat_class][feat] = self.n_total_features
                    self.n_total_features += 1

    def calc_represent_input_with_features(self) -> None:
        """Builds small_matrix (true-tag histories) and big_matrix (all-tag histories) as sparse bool matrices."""
        tags = self.feature_statistics.tags
        histories = self.feature_statistics.histories
        n_hist = len(histories)
        n_tags = len(tags)

        small_rows, small_cols = [], []
        big_rows, big_cols = [], []

        for small_r, hist in enumerate(histories):
            cols = represent_input_with_features(hist, self.feature_to_idx)
            small_rows += [small_r] * len(cols)
            small_cols += cols

            for big_r_offset, y_tag in enumerate(tags):
                demi_hist = (hist[0], y_tag, hist[2], hist[3], hist[4], hist[5], hist[6])
                cols = represent_input_with_features(demi_hist, self.feature_to_idx)
                self.histories_features[demi_hist] = cols
                big_r = small_r * n_tags + big_r_offset
                big_rows += [big_r] * len(cols)
                big_cols += cols

        ones = np.ones(len(small_rows))
        self.small_matrix = sparse.csr_matrix(
            (ones, (small_rows, small_cols)), shape=(n_hist, self.n_total_features), dtype=bool
        )
        ones = np.ones(len(big_rows))
        self.big_matrix = sparse.csr_matrix(
            (ones, (big_rows, big_cols)), shape=(n_hist * n_tags, self.n_total_features), dtype=bool
        )


def represent_input_with_features(history: History, dict_of_dicts: Dict[str, Dict[Tuple, int]]) -> List[int]:
    """
    Returns the list of active feature indices for a given history.
    Filters via dict_of_dicts so dropped classes (not in subset) contribute nothing.
    @param history: (c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word)
    @param dict_of_dicts: maps feature class name -> {feature_key -> index}
    """
    features = []
    for feat_class, key in iter_features(history):
        class_map = dict_of_dicts.get(feat_class)
        if class_map is None:
            continue                      # class not in active subset (Model 2 path)
        idx = class_map.get(key)
        if idx is not None:
            features.append(idx)
    return features


def preprocess_train(train_path: str, threshold, feature_subset: List[str] = None,
                     verbose: bool = False) -> Tuple[FeatureStatistics, Feature2id]:
    """Build statistics + Feature2id. `threshold` is an int or per-family dict;
    `feature_subset` restricts which classes survive (Model 2). `verbose` prints the
    per-family feature breakdown (off by default so CV/grid runs stay quiet)."""
    statistics = FeatureStatistics()
    statistics.get_word_tag_pair_count(train_path)

    feature2id = Feature2id(statistics, threshold, feature_subset)
    feature2id.get_features_idx()
    feature2id.calc_represent_input_with_features()

    if verbose:
        print(f"{feature2id.n_total_features} features:")
        for feat_class, idx in feature2id.feature_to_idx.items():
            print(f"  {feat_class} {len(idx)}")
    return statistics, feature2id


def read_test(file_path: str, tagged: bool = True) -> List[Tuple[List[str], List[str]]]:
    """
    Reads a test/validation file.
    @param tagged: if True, expects `word_TAG` tokens; if False, expects plain words
    @return: list of (words, tags) tuples, each padded with ["*","*"] start and ["~"] end tokens
    """
    sentences = []
    with open(file_path) as f:
        for line in f:
            words, tags = ["*", "*"], ["*", "*"]
            for token in line.rstrip("\n").split():
                w, t = token.split("_") if tagged else (token, "")
                words.append(w)
                tags.append(t)
            words.append("~")
            tags.append("~")
            sentences.append((words, tags))
    return sentences
