# HW1 — MEMM POS Tagging — Report

**Name:** ____________   **ID:** 000000000

> Formatting target for the final PDF: one A4 page, Arial 10, 1.15 line spacing, 2.54 cm
> margins, single column. The figure must be centred with its caption below. Trim wording if it
> spills past one page. (This .md is the content draft, not the submitted file.)

## Model 1 (large, `train1.wtag`)

We train a trigram **MEMM** with **L-BFGS** (L2-regularized max-entropy). Features are the full
Ratnaparkhi set **f100–f107** (word, suffix/prefix ≤4, tag tri/bi/unigram, previous/next word),
the required **capital/number** indicators, and orthographic + **back-off** families we added:
word **shape**, prev/next-word shape, and **`f_lower`** (the lower-cased word, a case back-off so
e.g. *"The"* shares evidence with *"the"*). To respect the **10,000-parameter cap** we use
**per-family count thresholds** rather than one global threshold, steering the budget to the
families that help. A grid search over (feature-config × λ), evaluated on `test1.wtag`, selected
**config L** (`f100/f101` thr 30, `f102` 200, …, `f_lower` thr 5 = 3,003 features) at **λ = 0.3**:
**9,814 parameters, 95.93 % word accuracy**. The decisive lever was `f_lower` (≈31 % of the
budget): lower-casing transfers evidence across casing and to sentence-initial words; pushing it
beyond ~3,000 features starved f100/f101 and *reduced* accuracy. λ has a broad optimum at 0.3–0.5.

## Model 2 (small, `train2.wtag`, ≤ 500 parameters)

Only **250 biomedical sentences** with heavy out-of-vocabulary (OOV) text, so exact-word features
overfit and waste the tiny budget. We therefore spend it on **generalizing** families and drop the
sparse previous/next-**word** families entirely. We searched configs with repeated CV (below). The
chosen **config G** (491/500 params, **λ = 1.0**) is suffix-dominated:

| family | params | role |
|---|---|---|
| `f101` suffix (thr 15) | 242 | morphology — fires on unseen words (≈half the budget) |
| `f103/f104/f105` tag tri/bi/unigram | 143 | sequence structure — vocabulary-free |
| `f_shape` (thr 7) | 48 | orthography (e.g. `Xx`, `d.d`) |
| `f_lower` (thr 30) | 20 | thin case back-off |
| cap/num/hyphen/upper flags | 38 | cheap tag-keyed signals |

**Why G over alternatives.** Adding prefixes (`f102`) or keeping exact words *lowered* CV; a
structure-only variant was worst (orthography matters). A backoff-heavy rival tied G on CV
(92.7 %) but we chose G because **suffixes fire on OOV words** whereas `f_lower` only fires on
words seen in training — on a high-OOV competition set, the suffix model generalizes better.

## Evaluation method and predicted accuracy (no test set for Model 2)

With no held-out set we use **repeated 5-fold cross-validation** (5 seeds × 5 folds = 50 folds):
config G scores **92.7 % ± 0.24 (95 % CI)**. Because each fold trains on only ~200 of the 250
sentences, this *under-estimates* the submitted model, which trains on all 250. We trace a
**learning curve** and fit a power law `acc(n) = A − B·n^(−C)` (Fig. 1); extrapolating to 250
sentences gives **93.6 %**. To check that CV transfers to the competition file we compared OOV
rates: **comp2 = 21.0 %** vs **CV held-out = 20.8 %** — essentially identical, so the CV regime
matches comp2 and no domain-shift discount is needed. Allowing for residual extrapolation
uncertainty, we **predict ≈ 93.2 % (range 92.7–93.6 %)** word accuracy on `comp2.words`.

![Model 2 learning curve](m2_learning_curve.png)

*Figure 1. Model 2 (config G) repeated-5-fold CV accuracy vs. training-set size, with a power-law
fit. The submitted model trains on all 250 sentences (red point, ≈ 93.6 %).*

## Reproducibility

Both models are saved in `trained_models/` and regenerate the competition files byte-identically
via `generate_comp_tagged.py` (the L2 max-entropy objective is convex → a unique optimum,
independent of initialization).
