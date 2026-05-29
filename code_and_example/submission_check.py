import argparse
import os
import pickle
import shutil
import sys
import zipfile


def unzip_directory(zip_path, output_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_path)


def open_zip(sid, unzip_dir, comp_files_dir):
    errors = False
    zip_file_path = f"HW1_{sid}.zip"
    if not os.path.exists(zip_file_path):
        print(f"{zip_file_path} does not exist.")
        return True

    unzip_directory(zip_file_path, unzip_dir)
    dir_files = os.listdir(unzip_dir)

    if len(dir_files) > 5:
        print("The submission contains redundant files.")
        errors = True

    comp_files = [f"comp_m1_{sid}.wtag", f"comp_m2_{sid}.wtag"]
    req_files = [f"report_{sid}.pdf", "code", "trained_models"] + comp_files
    req_code_files = ["main.py", "generate_comp_tagged.py", "preprocessing.py", "inference.py", "optimization.py"]

    for file in req_files:
        if file not in dir_files:
            print(f"Missing required file: {file}")
            errors = True

    code_dir = os.path.join(unzip_dir, "code")
    if os.path.isdir(code_dir):
        code_dir_files = os.listdir(code_dir)
        for file in req_code_files:
            if file not in code_dir_files:
                print(f"Missing required file in code/: {file}")
                errors = True

    student_comp_dir = os.path.join(comp_files_dir, sid)
    os.makedirs(student_comp_dir, exist_ok=True)
    for file in comp_files:
        src = os.path.join(unzip_dir, file)
        if os.path.exists(src):
            shutil.copy(src, student_comp_dir)

    return errors


def check_model_size(unzip_dir):
    models_path = os.path.join(unzip_dir, "trained_models")
    sys.path.insert(0, os.path.join(unzip_dir, "code"))
    errors = False

    checks = [(1, 10_000), (2, 500)]
    for model_n, limit in checks:
        pkl_path = os.path.join(models_path, f"weights_{model_n}.pkl")
        try:
            with open(pkl_path, 'rb') as f:
                _, feature2id = pickle.load(f)
            n = feature2id.n_total_features
            print(f"Model {model_n}: {n} features")
            if n > limit:
                print(f"  ERROR: exceeds the limit of {limit:,} features.")
                errors = True
        except Exception as err:
            print(f"Model {model_n}: could not load weights_{model_n}.pkl — {err}")
            errors = True

    return errors


def validate_tagged_output(words_path, wtag_path, model_n):
    """
    Validates that a student's .wtag file matches the structure of the input .words file:
    - Same number of sentences
    - Each token is in word_TAG format
    - Words in each sentence match the input exactly
    Reports problematic sentences without revealing accuracy.
    """
    with open(words_path) as f:
        ref_sentences = [line.rstrip("\n").split() for line in f if line.strip()]
    with open(wtag_path) as f:
        pred_lines = [line.rstrip("\n") for line in f if line.strip()]

    if len(pred_lines) != len(ref_sentences):
        print(f"  ERROR (model {model_n}): expected {len(ref_sentences)} sentences, "
              f"got {len(pred_lines)}.")
        return True

    prob_sent = []
    for idx, (ref_words, pred_line) in enumerate(zip(ref_sentences, pred_lines)):
        # Normalise common formatting quirk
        if pred_line.endswith('._.') and not pred_line.endswith(' ._.'):
            pred_line = pred_line[:-3] + ' ._.'
        pred_tokens = pred_line.split()

        # Strip trailing ~ token if present
        if pred_tokens and pred_tokens[-1] == '~':
            pred_tokens = pred_tokens[:-1]

        issues = []
        if len(pred_tokens) != len(ref_words):
            issues.append(f"expected {len(ref_words)} tokens, got {len(pred_tokens)}")
        else:
            for i, (ref_w, tok) in enumerate(zip(ref_words, pred_tokens)):
                if '_' not in tok:
                    issues.append(f"token {i} '{tok}' is not in word_TAG format")
                    break
                pred_w = tok.rsplit('_', 1)[0]
                if pred_w != ref_w:
                    issues.append(f"token {i} word mismatch: expected '{ref_w}', got '{pred_w}'")
                    break

        if issues:
            prob_sent.append((idx, issues[0]))

    if prob_sent:
        print(f"  ERROR (model {model_n}): {len(prob_sent)} sentence(s) have format issues "
              f"(e.g. sentence {prob_sent[0][0]}: {prob_sent[0][1]}).")
        return True

    print(f"  Model {model_n} competition file: OK ({len(ref_sentences)} sentences).")
    return False


def check_format(sid, comp_files_dir):
    errors = False
    student_comp_dir = os.path.join(comp_files_dir, sid)
    if not os.path.isdir(student_comp_dir):
        print("Skipping competition file validation (zip extraction failed).")
        return True
    comp1_files = [x for x in os.listdir(student_comp_dir) if x.startswith('comp_m1')]
    comp2_files = [x for x in os.listdir(student_comp_dir) if x.startswith('comp_m2')]

    if len(comp1_files) != 1:
        print("Problem with comp_m1 file (missing or duplicate).")
        errors = True
    else:
        errors = validate_tagged_output(
            os.path.join("data", "comp1.words"),
            os.path.join(student_comp_dir, comp1_files[0]),
            model_n=1,
        ) or errors

    if len(comp2_files) != 1:
        print("Problem with comp_m2 file (missing or duplicate).")
        errors = True
    else:
        errors = validate_tagged_output(
            os.path.join("data", "comp2.words"),
            os.path.join(student_comp_dir, comp2_files[0]),
            model_n=2,
        ) or errors

    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Verify your HW1 submission format before submitting."
    )
    parser.add_argument("--sid", required=True, help="Your student ID (e.g. 123456789)")
    parser.add_argument("--unzip-dir", default="your_unzip_submission",
                        help="Directory to extract the zip into (default: your_unzip_submission)")
    parser.add_argument("--comp-files-dir", default="comps_files",
                        help="Directory to store extracted competition files (default: comps_files)")
    args = parser.parse_args()

    errors = open_zip(args.sid, args.unzip_dir, args.comp_files_dir)
    errors = check_model_size(args.unzip_dir) or errors
    errors = check_format(args.sid, args.comp_files_dir) or errors

    if not errors:
        print("\nAll checks passed — it looks like you are ready to submit!")
    else:
        print("\nPlease fix the issues above before submitting.")


if __name__ == '__main__':
    main()
