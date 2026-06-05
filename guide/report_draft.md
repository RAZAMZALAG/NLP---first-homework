# HW1 — MEMM POS Tagging — Report

**Name:** ____________   **ID:** 000000000

> Final-PDF formatting: one A4 page, Arial 10, 1.15 spacing, 2.54 cm margins, single column,
> figure centred with caption below. This .md is the content draft, not the submitted file.

## Overview

I train a trigram MEMM by L2-regularized maximum likelihood (L-BFGS). The provided code applied
a **single global count threshold** to every feature; with the strict parameter caps (Model 1
≤10,000, Model 2 ≤500) this is too blunt — it discards rare-but-informative features while keeping
many redundant ones. I replaced it with **per-family thresholds**: the threshold is a dict
`{feature_family: min_count}` (default 1), so each family is pruned independently and the budget is
steered to the families that actually help.

**Filtering to meet the size cap.** A feature `(family, key)` enters the model only if it occurs at
least its family's `min_count` times in the training data (`get_features_idx`). To choose the
thresholds I tabulated each family's surviving count over a range of `min_count` values
(`measure_features.py`) and picked the per-family values that maximize accuracy while the total
stays under the cap — yielding **9,814 ≤ 10,000** parameters for Model 1 and **491 ≤ 500** for
Model 2. The search itself used `test1` for Model 1 and cross-validation for Model 2.

## Feature families and their method

Every feature is a binary indicator on a `(key, tag)` pair; `iter_features` defines them once so
training and inference agree. **The tag is always the current tag `t` (the one being predicted)**;
the families differ only in the key. Beyond the provided `f100 = (word, t)` I implemented:

- **Lexical/morphological:** `f101 = (suffix, t)` and `f102 = (prefix, t)` for every affix of
  length 1–4 — capture morphology and fire on unseen words.
- **Tag context (vocabulary-free):** `f103 = (t-2, t-1, t)` trigram, `f104 = (t-1, t)` bigram,
  `f105 = (t)` unigram.
- **Word context:** `f106 = (previous word, t)`, `f107 = (next word, t)`.
- **Required capital/number + orthography**, each fired only when its condition holds and keyed on
  `t` alone: `f_cap` (word has an uppercase letter), `f_num` (has a digit), `f_is_number` (whole
  word is numeric, incl. floats/`1e5`), `f_all_upper` (all caps, length > 1 → acronym),
  `f_first_upper` (capital mid-sentence → proper noun), `f_hyphen` (internal hyphen).
- **Word shape** `f_shape = (shape(word), t)`, plus `f_prev_shape`/`f_next_shape` for the
  neighbours: `shape` maps each character to `X` (upper), `x` (lower), `d` (digit), else itself,
  then collapses consecutive duplicates — `Apple`→`Xx`, `HELLO`→`X`, `Wi-Fi`→`Xx-Xx`, `3.14`→`d.d`.
  A strong signal for proper nouns/numbers and unseen words.
- **Back-off** `f_lower = (lower-cased word, t)`: `"The"` shares evidence with `"the"`.

**Model 1** (config L, λ=0.3): keeps all families; `f_lower` at its sweet spot (3,003 feat) is the
decisive lever → 9,814 params. **Model 2** (config G, λ=1.0): 250 OOV-heavy biomedical sentences,
so I drop the sparse word-context families and spend the 500 budget on suffixes (`f101`, ~half),
tag context, shape and a thin back-off → 491 params.

## Training and inference

`main.py --model_number N` builds the config's features and fits the weights by L-BFGS with the
provided convex objective/gradient (linear term − log-normalizer − ½λ‖w‖²); the optimum is unique,
so training is reproducible. Weights are pickled to `trained_models/weights_N.pkl` after stripping
the training-only matrices (<1 MB). `memm_viterbi` decodes with trigram Viterbi in log-space and a
beam (top-50 states/position); per word the candidate tags follow an OOV back-off chain — tags seen
with the word, else its shape, else its longest suffix, else all — speeding decoding and handling
unseen words.

## Test, evaluation and competition

Model 1 scores **95.93 %** on the held-out `test1.wtag`. Model 2 has no test set, so I use
**repeated 5-fold cross-validation** (5 seeds × 5 folds): **92.7 % ± 0.24 (95 % CI)**. As folds
train on only ~200/250 sentences, I fit a power-law learning curve (Fig. 1) that extrapolates the
full 250-sentence model to **93.6 %**; since comp2's OOV rate (21.0 %) matches the CV held-out rate
(20.8 %), the CV estimate transfers and I **predict ≈ 93 % on `comp2`**. `generate_comp_tagged.py`
reproduces `comp_m{1,2}_<id>.wtag` exactly from the saved weights (`word_TAG`, original order).

![Model 2 learning curve](m2_learning_curve.png)

*Figure 1. Model 2 (config G) repeated-5-fold CV accuracy vs. training-set size, with a power-law
fit; the submitted model trains on all 250 sentences (red, ≈ 93.6 %).*
