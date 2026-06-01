# `code_and_example/` — Project Code Walkthrough

This document explains every file in the example project we were given, marks which files we need to change, and walks through how the whole pipeline works end-to-end. Read this before writing any code — once you understand the flow, the implementation work becomes mechanical.

---

## 1. Folder Layout

```
code_and_example/
├── HW1_000000000.zip           # Example submission zip (dummy contents)
├── init.sh                      # Environment setup script
├── submission_check.py          # Validates your final submission zip
└── code/                        # The skeleton we develop
    ├── main.py                  # Entry point: train -> save weights -> tag comp file
    ├── preprocessing.py         # Read data, count features, build feature matrices
    ├── optimization.py          # L-BFGS-B optimizer (objective + gradient)
    ├── inference.py             # Viterbi decoder (we implement)
    └── generate_comp_tagged.py  # Standalone re-tagging from saved weights
```

The example `HW1_000000000.zip` contains exactly the same `code/` (no implementation), placeholder competition files where every tag is `NN`, a 9-byte fake PDF, and dummy `weights_1.pkl` / `weights_2.pkl`. The example exists only to show the **expected zip structure** — not a working solution.

---

## 2. Pipeline Overview (How the Project Works)

```
                       ┌───────────────────┐
                       │  train1.wtag      │
                       │  (tagged corpus)  │
                       └────────┬──────────┘
                                │
                                ▼
       ┌─────────────────────────────────────────────────┐
       │ preprocessing.py                                 │
       │   FeatureStatistics  -> counts every feature     │
       │   Feature2id         -> assigns indices,         │
       │                         builds sparse matrices   │
       └────────────────────┬────────────────────────────┘
                            │ (statistics, feature2id)
                            ▼
       ┌─────────────────────────────────────────────────┐
       │ optimization.py                                  │
       │   calc_objective_per_iter  -> log-likelihood     │
       │                              + gradient          │
       │   fmin_l_bfgs_b            -> optimizes weights  │
       │   saves (weights, feature2id) to weights_N.pkl   │
       └────────────────────┬────────────────────────────┘
                            │ pre_trained_weights
                            ▼
       ┌─────────────────────────────────────────────────┐
       │ inference.py                                     │
       │   memm_viterbi  -> for each sentence,            │
       │                    find argmax tag sequence      │
       │                    using log-linear q(t | hist)  │
       └────────────────────┬────────────────────────────┘
                            │
                            ▼
                comp_mN_<sid>.wtag  (final tagged output)
```

Key idea: **MEMM = log-linear model that scores `q(tag | history)`, applied position by position via Viterbi**. Training = maximize regularized log-likelihood of the gold tag at each history in the corpus, where each history's distribution is the softmax over feature scores.

---

## 3. File-by-File Walkthrough

### 3.1 `init.sh` (do not change)

Bash script that:
- Installs the `uv` package manager (Python project manager) if missing.
- Runs `uv sync` to create a virtual environment and install dependencies declared in the project's `pyproject.toml`.

Run once per machine:

```bash
bash init.sh
```

### 3.2 `submission_check.py` (do not change)

Standalone validator. Reads your final `HW1_<sid>.zip`, verifies:

1. Exactly 5 root entries: 2 `.wtag` files, `report_<sid>.pdf`, `code/`, `trained_models/`.
2. `code/` has all 5 required py files.
3. `trained_models/weights_1.pkl` and `weights_2.pkl` are loadable, and report their feature counts: **Model 1 ≤ 10,000**, **Model 2 ≤ 500**. Exceeding the limit = grade 0.
4. Each `.wtag` file has the same number of sentences as the `.words` input, every token is `word_TAG`, and words match the input exactly.

Run from a directory containing `HW1_<sid>.zip` and the `data/` folder:

```bash
uv run submission_check.py --sid <your_id>
```

A clean run prints `All checks passed`. Anything else is a blocker.

### 3.3 `code/preprocessing.py` (**we modify heavily**)

The data-prep module. Two classes plus helpers.

#### Constants and types
```python
WORD = 0
TAG  = 1
History = Tuple[c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word]
```

A **history** is a 7-tuple: current word/tag, previous word/tag, previous-previous word/tag, next word. **No future tag** — that's the Markov assumption (the next tag isn't known when we score the current position).

#### `FeatureStatistics`
- `feature_rep_dict` — maps feature class name (e.g. `"f100"`) to a counter `{feature_key -> count}`.
- `tags` — set of all observed tags (plus the `~` end marker).
- `tags_counts`, `words_count` — for diagnostics or future features.
- `histories` — flat list of every training history.

Method `get_word_tag_pair_count(file_path)`:
1. Pads every sentence with `("*","*")` start tokens and `("~","~")` end token.
2. For each real (word, tag): adds the tag, updates counters, and increments `feature_rep_dict["f100"][(word, tag)]`.
3. Walks every position and appends the 7-tuple history to `self.histories`.

**This is where we add new feature classes.** When we add (for example) `f101` for suffix/tag pairs, we add a new entry `feature_rep_dict["f101"] = defaultdict(int)` in `__init__`, and we extend the loop in `get_word_tag_pair_count` to populate it from each (word, tag) seen.

#### `Feature2id`
- `feature_to_idx` — maps feature class -> `OrderedDict(feature_key -> integer id)`.
- `n_total_features` — running counter, also reported by `submission_check.py`.
- `histories_features` — cached active-feature indices for every (history, candidate-tag) combination, used by inference.
- `small_matrix`, `big_matrix` — sparse boolean matrices used by the optimizer.

Method `get_features_idx()`:
- Walks every feature class in `feature_rep_dict`, and for each feature whose count `>= threshold`, assigns it the next free index.
- `threshold` is the main feature-pruning knob. HW1 PDF demonstrates that `--threshold 10` is enough to satisfy the size limits at the default feature set.

Method `calc_represent_input_with_features()`:
- Builds two sparse matrices:
  - **`small_matrix`** `(n_histories, n_features)`: row `i` = 1 wherever a feature fires for history `i` with its gold tag.
  - **`big_matrix`** `(n_histories * n_tags, n_features)`: for every history, has one row per candidate tag (with that candidate substituted for the current tag).
- These two matrices are the heart of fast vectorized objective/gradient computation in `optimization.py`. Don't be intimidated — once we add features, the matrices are still built by the same loop.

#### `represent_input_with_features(history, dict_of_dicts) -> List[int]`

Given a history 7-tuple and the feature-class -> feature-key -> id maps, returns the list of feature indices that "fire" for this history.

**This is the second place we modify.** The skeleton only handles `f100`. We extend it with one block per feature class:

```python
# f101: current word ends with suffix of length <=4, paired with tag
for k in range(1, 5):
    suffix = c_word[-k:]
    key = (suffix, c_tag)
    if key in dict_of_dicts.get("f101", {}):
        features.append(dict_of_dicts["f101"][key])
# ...same pattern for f102 (prefix), f103 (trigram tags), etc.
```

#### `preprocess_train(train_path, threshold)`

Convenience wrapper: builds `FeatureStatistics`, builds `Feature2id`, calls all the right methods, and prints the final feature count.

#### `read_test(file_path, tagged)`

Reads a test or competition file into `[(words, tags), ...]` with the `*/*` start padding and `~` end padding. Used by `inference.py`.

### 3.4 `code/optimization.py` (do not change, unless we want to)

Wraps the regularized maximum-entropy training via `scipy.optimize.fmin_l_bfgs_b`. Two functions:

#### `calc_objective_per_iter(w_i, *args)`

Computes the **negative** log-likelihood and its **negative** gradient (because `fmin_l_bfgs_b` minimizes):

```
L(v) = sum_i [ v . f(x_i, y_i) ]            # linear term: scores on gold tags
     - sum_i log sum_{y'} exp(v . f(x_i, y'))  # log-normalizer over all tags
     - (lam / 2) ||v||^2                      # L2 regularization

dL/dv = empirical_counts - expected_counts - lam * v
```

Implemented as vector/matrix products against `small_matrix` and `big_matrix` — see lecture 3 slides "Calculating the Maximum Likelihood Estimates" and "L2 Regularization" for the exact derivation.

#### `get_optimal_vector(statistics, feature2id, lam, weights_path)`

1. Initializes `w_0` randomly (`N(0,1)`).
2. Calls L-BFGS-B with up to 750 iterations.
3. Pickles `(optimal_params, feature2id)` to `weights_path`. The `submission_check` and `generate_comp_tagged` both rely on this exact tuple layout.

We may want to tune `lam`, `maxiter`, or the initialization, but the math is correct as-is.

### 3.5 `code/inference.py` (**we modify — implement Viterbi**)

Two functions.

#### `memm_viterbi(sentence, pre_trained_weights, feature2id)` — STUB to implement

Input: a padded word list `["*", "*", w1, w2, ..., wN, "~"]`.

Output: a list of `N+1` predicted tags (the caller drops index 0; only positions 1..N are kept).

What we need to write: classic MEMM Viterbi (lecture 3 slide "The full Algorithm (with backpointers)"). For each position `k`, for each (prev-prev tag `t`, prev tag `u`, current tag `v`), update:

```
pi(k, u, v) = max_t pi(k-1, t, u) * q(v | t, u, w_[1:n], k)
bp(k, u, v) = argmax t
```

where `q(v | t, u, w, k) = exp(w . f(history_with_tag_v)) / sum_{v'} exp(w . f(history_with_tag_v'))`.

Practical tips:
- Use `represent_input_with_features(history_with_tag, feature2id.feature_to_idx)` to get the active feature indices, then `pre_trained_weights[indices].sum()` gives the score. Apply softmax over candidate tags.
- Work in log-space (`pi` in logs, `+` instead of `*`) to avoid underflow on long sentences.
- Cap the candidate-tag set: at position `k`, only consider tags actually seen for word `w_k` in training (plus a fallback set for unknown words). This is the standard speed hack.
- Beam search is optional but recommended for runtime on Model 1.

Current placeholder returns `["NN"] * (len(sentence) - 2)`, which is why the example wtag files have every word tagged `NN`.

#### `tag_all_test(test_path, weights, feature2id, predictions_path, tagged)`

Reads every sentence with `read_test`, calls `memm_viterbi`, strips the leading `*/*` and trailing `~`, writes `word_TAG` per line, space-separated. This is correct already — we just need to make Viterbi return real tags.

### 3.6 `code/main.py` (small tweaks at most)

CLI to train and tag in one shot. Arguments:

- `--sid` student ID, used in the output filename.
- `--model_number` 1 or 2.
- `--threshold` minimum feature-occurrence count to keep a feature (default 1, but realistically we'll set 10 for Model 1 and higher for Model 2).
- `--lam` L2 regularization strength (default 1).

Flow:
1. `preprocess_train(train_path, threshold)` -> `(statistics, feature2id)`.
2. `get_optimal_vector(...)` trains and pickles `trained_models/weights_<n>.pkl`.
3. Re-loads the pickle, extracts `optimal_params[0]` (the weight vector).
4. `tag_all_test(comp_path, weights, feature2id, "comp_mN_<sid>.wtag", tagged=False)` writes the competition output.

Note: `main.py` tags the **comp** file (`data/compN.words`), not `testN.wtag`. The test-set accuracy reporting we need for the report is something we'll add ourselves — easiest is to call `tag_all_test` on `test1.wtag` with `tagged=True`, then compare predicted vs gold tags.

### 3.7 `code/generate_comp_tagged.py` (do not change)

Standalone re-tagger. Loads `trained_models/weights_<n>.pkl`, tags `data/compN.words` -> `comp_mN_<sid>.wtag`. Used when we already have trained weights and only want to refresh the competition output. The graders may run this to reproduce our results — so keep weights deterministic enough (set a numpy seed if we tweak `optimization.py`).

---

## 4. What We Change vs. What We Leave Alone

| File | Action | What we add |
|---|---|---|
| `preprocessing.py` | **EDIT** | new feature classes f101–f107 + numeric/capital features in `FeatureStatistics.__init__`, populate them in `get_word_tag_pair_count`, and extend `represent_input_with_features` to extract their active indices. |
| `inference.py` | **EDIT** | full `memm_viterbi` implementation. |
| `main.py` | **OPTIONAL** | maybe add test-accuracy evaluation, set defaults, or log results. |
| `optimization.py` | **LEAVE** | math is correct. Only touch if we want to tune lam/maxiter. |
| `generate_comp_tagged.py` | **LEAVE** | works as-is. |
| `init.sh`, `submission_check.py` | **LEAVE** | infra only. |

---

## 5. The Mandatory Feature Set (Ratnaparkhi 1996)

From the assignment PDF + lecture 3:

| Code | Description | Key tuple |
|---|---|---|
| f100 | (word, tag) pair | `(c_word, c_tag)` — already in skeleton |
| f101 | suffix of length ≤4, with tag | `(suffix, c_tag)` |
| f102 | prefix of length ≤4, with tag | `(prefix, c_tag)` |
| f103 | tag-trigram | `(pp_tag, p_tag, c_tag)` |
| f104 | tag-bigram | `(p_tag, c_tag)` |
| f105 | tag-unigram | `(c_tag,)` |
| f106 | previous word + current tag | `(p_word, c_tag)` |
| f107 | next word + current tag | `(n_word, c_tag)` |

Plus the HW1-required extras:
- **Capital-letter feature** — fires when the current word contains an uppercase letter (or starts with one), paired with the current tag.
- **Number feature** — fires when the current word contains a digit, paired with the current tag.

Optional additions allowed (good for Model 2 with the 500-feature cap): word-shape patterns (`Xxxx`, `dd-dd`), word length buckets, hyphenation flag, etc.

---

## 6. Run Order (Reference)

```bash
# One-time
bash init.sh

# Train Model 1 (large, threshold 10 to stay under 10k features)
uv run python code/main.py --sid 123456789 --model_number 1 --threshold 10

# Train Model 2 (small, threshold tuned to stay under 500 features)
uv run python code/main.py --sid 123456789 --model_number 2 --threshold 10

# Write the report PDF as report_123456789.pdf

# Build the submission zip
zip HW1_123456789.zip -r code/ trained_models/ \
    comp_m1_123456789.wtag comp_m2_123456789.wtag report_123456789.pdf

# Verify before submitting
uv run submission_check.py --sid 123456789
```

---

## 7. Mental Model — Why This Works

- Training learns weights `v` such that on every training history `(x, y_gold)`, the model assigns higher score to `y_gold` than to other tags. The L-BFGS-B optimizer does the math: we just supply the feature counts via the sparse matrices.
- Each feature is a binary indicator (`1` if the pattern holds at this history with this tag, `0` otherwise). Adding more feature classes = giving the model more lenses through which to view a history.
- Inference is the standard trigram Viterbi, except `q(v | t, u, x, k)` comes from a softmax over our learned scores instead of from HMM counts. Same dynamic programming, different scoring function.
- Feature thresholds prune rare features — those generalize poorly anyway and would blow the parameter budget.

Once we internalize that "features fire on histories, weights score features, softmax gives `q`, Viterbi chains the q's", the rest of the project is plumbing.
