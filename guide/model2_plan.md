# Model 2 — Task & Plan of Work

Goal: **be the best in the class on Model 2.** This document captures the task constraints,
where points are won/lost, and the phased plan.

---

## 1. The task (from the assignment)

- **Train** on `data/train2.wtag` — only **250 tagged sentences** (small data).
- **Features:** any set allowed, including all of Model 1's.
- **Size limit:** the trained model must have **≤ 500 parameters**. Exceeding it = **grade 0**.
- **No dedicated test file.** The report must describe how we evaluate given the limited data.
- **Competition output:** run inference on `comp2.words` → `comp_m2_<sid>.wtag`
  (`word_TAG` format, sentence order preserved).

### How Model 2 is graded
| Component | Weight | Notes |
|---|---|---|
| Model 2 implementation + competition | 20 % | shared with Model-2 comp accuracy on `comp2.words` |
| Model 2 evaluation (report) | 5 % | scored on the **absolute difference `|estimated − actual|`** accuracy |
| Submission report | 10 % | 1 page; creativity, rigor, clarity, novelty |
| Code-length competition | 10 % | fewer non-blank/non-comment lines = better |

### Hard facts from `submission_check.py`
- Parameter count measured = **`feature2id.n_total_features`** (same number `tune.py` prints).
  Cap for Model 2 = **500**.
- Zip top level must contain **exactly 5 items**: `report_<sid>.pdf`, `code/`,
  `trained_models/`, `comp_m1_<sid>.wtag`, `comp_m2_<sid>.wtag`. More than 5 → "redundant files".
- Required files in `code/`: `main.py, generate_comp_tagged.py, preprocessing.py, inference.py,
  optimization.py`. Extra files are *allowed* but **count against the code-length competition**.
- Allowed packages: stdlib, `scipy`, `numpy`, `pandas`, `matplotlib`, `tqdm`.
  Forbidden: any Viterbi/MEMM/n-gram **library**.

---

## 2. Where the class will win or lose points

1. **250 sentences vs ≤500 params.** Exact-word features (`f100`) overfit and waste budget.
   Winners lean on **generalizing** features — suffix (`f101`), word shape, tag-context
   (`f103/f104/f105`), cap/num flags, and **`f_lower`** (case backoff — the Model-1 MVP).
2. **The `|estimated − actual|` score.** Most students report a single CV number, which
   *underestimates* the final model (fold models train on fewer sentences). A
   **learning-curve-corrected estimate + confidence interval** wins this 5 %.
3. **The 450 MB weights pickle.** It bloats the zip and is "files we don't need" in the literal
   sense. Must be slimmed before submission (affects both models).

---

## 3. Plan of work (phased)

### Phase 0 — Evaluation harness *(do first; everything is judged by it)*
Build a robust **repeated stratified k-fold CV** on `train2.wtag` as the single source of truth.
Choose k (likely 5), repeat over several seeds → stable **mean ± CI**. Serves as both the tuning
metric and the basis for the report's accuracy estimate. Harden the existing `cross_validate`.

### Phase 1 — Feature/threshold design under 500
Run `measure_features.py` on `train2.wtag` for per-family counts at each threshold. Define
candidate `MODEL2_CONFIGS` (per-family thresholds) that spend the budget on generalizing
families, not exact words. **Target ~480–499 params** (using the full budget; `--threshold 10`
→ 75 params wastes it).

### Phase 2 — Grid search on the VM *(free + long-running)*
Adapt `tune.py` to a Model-2 mode: selection metric = **CV accuracy** (no `test1`), enforce
≤ 500, search **(config × λ)**. Large grid + repeated CV — exactly what the free VM is for.

### Phase 3 — Lock Model 2
Train the winner on all 250 sentences → `weights_2.pkl` + `comp_m2_<sid>.wtag`. Verify ≤ 500
through the `submission_check` code path.

### Phase 4 — Calibrated accuracy estimate (the 5 % eval score)
Final CV estimate + **learning-curve extrapolation** to the full-250 model + CI. This is the
number reported.

### Phase 5 — Slim the weights pickle (both models)
Stop pickling `FeatureStatistics.histories` + sparse matrices; keep only what inference needs.
450 MB → < 1 MB, with byte-identical competition-file reproduction.

### Phase 6 — Package + validate
Build the zip (exactly 5 root items, 5 required code files), run `submission_check.py`, fix any
format issues.

### Phase 7 — Report + code-length
Write the 1-page report (Arial 10, 1.15 spacing, 2.54 cm margins): feature/param choices for
**both** models, the CV methodology, predicted Model-2 accuracy + CI, key improvements. Minimize
the submitted `code/` line count (helper scripts kept in the repo, excluded from the zip).

---

## 4. Open decisions (to confirm before/at execution)

- **Start point:** Phase 0 (eval harness) recommended.
- **CV strategy:** repeated 5-fold + learning-curve correction (best calibration for the
  `|est − actual|` score) vs plain k-fold. Can be decided after prototyping the harness on real
  data variance.
- **k value** and number of repeat seeds — set after seeing fold-to-fold variance.

---

*Related docs:* `model1_implementation.md` (Model 1 design + tuning journey, reuses the same
per-family-threshold mechanism and the `f_lower` lever this plan builds on).
