from scipy import sparse
from collections import OrderedDict, defaultdict
import numpy as np
from typing import List, Dict, Tuple


WORD = 0
TAG = 1

# History tuple: (c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word)
History = Tuple[str, str, str, str, str, str, str]


class FeatureStatistics:
    def __init__(self):
        self.n_total_features = 0
        self.feature_rep_dict = {"f100": defaultdict(int)}  # feature class -> (feature -> count)
        self.tags = {"~"}
        self.tags_counts = defaultdict(int)
        self.words_count = defaultdict(int)
        self.histories = []

    def get_word_tag_pair_count(self, file_path: str) -> None:
        """
        Reads a tagged file, updates feature counts, tag/word counts, and histories list.
        Each history is: (c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word)
        """
        with open(file_path) as f:
            for line in f:
                pairs = line.rstrip("\n").split()
                sentence = [("*", "*"), ("*", "*")] + [tuple(p.split("_")) for p in pairs] + [("~", "~")]

                for word, tag in sentence[2:-1]:
                    self.tags.add(tag)
                    self.tags_counts[tag] += 1
                    self.words_count[word] += 1
                    self.feature_rep_dict["f100"][(word, tag)] += 1

                for i in range(2, len(sentence) - 1):
                    c, p, pp, n = sentence[i], sentence[i-1], sentence[i-2], sentence[i+1]
                    self.histories.append((c[0], c[1], p[0], p[1], pp[0], pp[1], n[0]))


class Feature2id:
    def __init__(self, feature_statistics: FeatureStatistics, threshold: int):
        """
        @param feature_statistics: the feature statistics object
        @param threshold: minimum number of appearances for a feature to be included
        """
        self.feature_statistics = feature_statistics
        self.threshold = threshold
        self.n_total_features = 0
        self.feature_to_idx = {"f100": OrderedDict()}
        self.histories_features = OrderedDict()
        self.small_matrix = sparse.csr_matrix
        self.big_matrix = sparse.csr_matrix

    def get_features_idx(self) -> None:
        """Assigns an index to each feature that appears at least `threshold` times."""
        for feat_class, counts in self.feature_statistics.feature_rep_dict.items():
            if feat_class not in self.feature_to_idx:
                continue
            for feat, count in counts.items():
                if count >= self.threshold:
                    self.feature_to_idx[feat_class][feat] = self.n_total_features
                    self.n_total_features += 1
        print(f"you have {self.n_total_features} features!")

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
    @param history: (c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word)
    @param dict_of_dicts: maps feature class name -> {feature_key -> index}
    """
    c_word, c_tag = history[0], history[1]
    features = []

    # f100: (word, tag) pair
    if (c_word, c_tag) in dict_of_dicts["f100"]:
        features.append(dict_of_dicts["f100"][(c_word, c_tag)])

    return features


def preprocess_train(train_path: str, threshold: int) -> Tuple[FeatureStatistics, Feature2id]:
    statistics = FeatureStatistics()
    statistics.get_word_tag_pair_count(train_path)

    feature2id = Feature2id(statistics, threshold)
    feature2id.get_features_idx()
    feature2id.calc_represent_input_with_features()
    print(feature2id.n_total_features)

    for feat_class, idx in feature2id.feature_to_idx.items():
        print(feat_class, len(idx))
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
