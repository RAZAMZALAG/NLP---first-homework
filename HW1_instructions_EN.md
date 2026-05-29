# Natural Language Processing — Homework 1 (English Version)

## Task Description

In this assignment you will implement a **Maximum Entropy Markov Model (MEMM)** for **Part-of-Speech (POS) tagging**, as taught in Lecture 3.

- Implementation language: **Python 3**.
- Submission: **individual** work only.
- You must implement and train **two models**:
  - **Model 1 (large)** — trained on `train1.wtag`.
  - **Model 2 (small)** — trained on `train2.wtag`.
- The accuracy metric is **word-level accuracy**.

## Grade Breakdown

| Component | % of Final Grade | Description |
|---|---|---|
| Model 1 implementation | 35% | Full implementation of Model 1, training on `train1.wtag`, and reporting accuracy on `test1.wtag`. |
| Model 1 competition | 20% | Accuracy on the competition file `comp1.words`. |
| Model 2 implementation + competition | 20% | Implementation and training of Model 2 on `train2.wtag`, and accuracy on the competition file `comp2.words`. |
| Model 2 evaluation | 5% | Analysis and evaluation of Model 2 performance, written in the report. |
| Submission report | 10% | Concise report meeting all formatting requirements. |
| Code length | 10% | Competition for the shortest overall code. |

## Provided Data (`data.zip`)

| File | Model | Description |
|---|---|---|
| `train1.wtag` | Model 1 (large) | 5,000 tagged sentences for training. Format: `word_TAG`. |
| `test1.wtag` | Model 1 (large) | 1,000 tagged sentences for testing. |
| `comp1.words` | Model 1 (large) | 1,000 **un-tagged** sentences for the competition. Format: `word word ... .` |
| `train2.wtag` | Model 2 (small) | 250 tagged sentences for training. |
| `comp2.words` | Model 2 (small) | 1,000 un-tagged sentences for the competition. |

**Format note:** Tags use the Penn Treebank scheme. Example:
```
the_D dog_N barks_V ._.
```

## Model Requirements

### Model 1 (Large)

- **Training:** on `train1.wtag`.
- **Features:** you must implement the full feature set from **Ratnaparkhi, 1996 (f100–f107)**, plus additional features that capture **numbers** and **words containing capital letters**. You may add more features.
- **Size limit:** the trained model must contain **at most 10,000 parameters**. Failing this constraint = **grade 0**.
- **Test:** run inference on `test1.wtag` and report the accuracy.

### Model 2 (Small)

- **Training:** on `train2.wtag`.
- **Features:** you may define any feature set (including those from Model 1).
- **Size limit:** the trained model must contain **at most 500 parameters**. Failing this constraint = **grade 0**.
- **Test:** there is no dedicated test file. The report must describe how you handled model evaluation given the limited training data.

## Competitions

There are several competitions in this assignment:

1. **Model 1 accuracy competition:** word-level accuracy on `comp1.words`.
2. **Model 2 accuracy competition:** word-level accuracy on `comp2.words`.
3. **Code-length competition:** scored on total number of lines. **Blank lines and comment-only lines do not count.** Readable, documented code is still required.
4. **Model 2 evaluation (in report):** analysis of Model 2's performance. The score is based on the **absolute difference** between your estimated accuracy and the actual measured accuracy.

You must run **inference** on the competition files (`comp1.words`, `comp2.words`) and write the results into new files in `.wtag` format.

- **Required output format:** for the input `the dog barks .` with predicted tags `D N V .`, produce the line:
  ```
  the_D dog_N barks_V ._.
  ```
- **Submission filenames:** `comp_m1_123456789.wtag` and `comp_m2_123456789.wtag` (replace `123456789` with your student ID).
- The order of sentences in the output file must match the original input file exactly.

## Environment and Code

- **Environment:** the project must run on the supplied machine.
- **Allowed packages:** standard Python, `scipy`, `numpy`, `pandas`, `matplotlib`, `tqdm`.
- **LBFGS:** it is recommended to use an LBFGS implementation (e.g. from `scipy`) for optimization. You must provide it with the objective function and the gradient.
- **Provided code:**
  - `preprocessing.py` — includes feature `f100`. You must implement the other required features.
  - `optimization.py` — not required to change (but may be modified).
  - `inference.py` — you must implement the `memm_viterbi` function.
  - `main.py` — trains a model, saves its weights, and saves the tagged competition files for the trained model.
  - `submission_check.py` — checks that your submission is valid. Run it from a directory that contains `HW1_<ID>.zip` and the `data` folder.
  - `generate_comp_tagged.py` — runs inference only, on the trained models, and generates the tagged competition files.
- **Forbidden packages:** any implementation of Viterbi, any implementation of MEMM, any text-processing libraries such as repetition / N-gram counters.

## Submission Requirements

### Short, concise report

As part of the Model 2 competition, you must write a short report describing your choices for the Model 2 you submit (how you chose features, parameters, and any additional improvements).

The report is evaluated based on:

- **Creativity:** in feature selection and in handling the limited data.
- **Writing and phrasing:** clarity of reporting.
- **Rigorousness:** correct use of evaluation methods.
- **Novelty:** original ideas and their implementation.

#### Format requirements

- Up to **one A4 page**.
- Standard margins — **2.54 cm** on all sides.
- **Single column** format.
- **Arial font, size 10**.
- **1.15 line spacing**.
- One blank line spacing between paragraphs.
- Every image must be **centered, without text wrap**, and must include a **caption below**.
- Text inside images (legends, titles, etc.) must be at least **9 pt**. If you must zoom above 100% to read text in an image, the image will not be accepted.
- **Filename:** `report_123456789.pdf`. Must contain concise explanations, reporting, and analysis of results.
- **Required content:** author's name and ID; performance evaluation report for Model 2; explanation of the parameter choices, features, and final improvements in **both** models.

### Submission package

- The submission is a single file: `HW1_123456789.zip`.
- **Code files:** documented, readable code.
- **Tagged competition files:** `comp_m1_123456789.wtag` and `comp_m2_123456789.wtag`.
- **Trained models:** saved in a folder named `trained_models` as `weights_1.pkl` and `weights_2.pkl`.
- **Reproducibility interface:** the code and the trained models must allow exact reproduction of the competition files.

**Important:** any mismatch in file format means a **grade of 0**.

A submission example file is provided that contains exactly the file structure that should appear in your final zip.

### Plagiarism

Code sharing between students is **strictly forbidden**. AI tools may be used, but the code is your responsibility, and you must not share prompts or AI-generated code snippets with other students.

## Run Instructions (Setup and Submission)

```
1. Run `bash init.sh`
2. Exit terminal and open a new one
3. uv run python code/main.py --sid 000000000 --model 1
3. uv run python code/main.py --sid 000000000 --model 2
4. Create the report pdf:                report_000000000.pdf
4. Zip the submission:
   zip HW1_000000000.zip -r code/ -r trained_models/ \
       comp_m1_000000000.wtag comp_m2_000000000.wtag report_000000000.pdf
5. uv run submission_check.py --sid 000000000
```

Running the defaults should give you:
```
Model 1: 3062 features
Model 2: 664 features
  ERROR: exceeds the limit of 500 features.
  Model 1 competition file: OK (1000 sentences).
  Model 2 competition file: OK (1000 sentences).

Please fix the issues above before submitting.
```

6. If you add `--threshold 10` to `main` for both models:
```
uv run python code/main.py --sid 000000000 --model 1 --threshold 10
uv run python code/main.py --sid 000000000 --model 2 --threshold 10
```

Re-running the zip and submission check should then produce:
```
Model 1: 1561 features
Model 2: 75 features
  Model 1 competition file: OK (1000 sentences).
  Model 2 competition file: OK (1000 sentences).

All checks passed — it looks like you are ready to submit!
```
