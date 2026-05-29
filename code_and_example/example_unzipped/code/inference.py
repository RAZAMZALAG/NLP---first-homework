import numpy as np
from typing import List
from preprocessing import read_test, represent_input_with_features, Feature2id
from tqdm import tqdm


def memm_viterbi(sentence: List[str], pre_trained_weights: np.ndarray, feature2id: Feature2id) -> List[str]:
    """
    Runs MEMM Viterbi over a single sentence and returns predicted tags.

    @param sentence: a padded word list of the form ["*", "*", w1, w2, ..., wN, "~"],
                     as produced by read_test. Indices 0-1 are start padding, index -1 is end padding.
    @param pre_trained_weights: the weight vector w of shape (n_features,)
    @param feature2id: the Feature2id object (provides feature_to_idx, feature_statistics.tags, etc.)
    @return: a list of N+1 predicted tags — one for each position from index 1 to N+1 inclusive
             (i.e. the second "*" padding token through the last real word).
             The caller discards result[0], so only the tags for w1..wN are used.

    You may implement Beam Search to improve runtime.
    Use represent_input_with_features to compute active feature indices for a history,
    and compute q(tag | history) as softmax over dot products with pre_trained_weights.
    """
    # TODO: implement this function
    return ["NN"] * (len(sentence) - 2)  # dummy: N+1 tags; replace with actual Viterbi output


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
