"""Model 1 hyperparameter search.

Grid over (feature-config x lambda), evaluate each on test1.wtag, print a
leaderboard, then finalize the winner: copy its weights to
trained_models/weights_1.pkl and tag data/comp1.words -> comp_m1_<sid>.wtag.

Efficiency: feature building (preprocess_train) is the slow step, so we do it
ONCE per config and reuse the same Feature2id across every lambda -- only the
L-BFGS training is repeated. w_0 is seeded so runs are reproducible/comparable.

Run on the VM:
    uv run python code/tune.py --sid 000000000
    uv run python code/tune.py --sid 000000000 --lams 0.5 1 2 --configs A B
"""
import argparse
import os
import pickle
import shutil
import time
import numpy as np
from preprocessing import preprocess_train, FEATURE_CLASSES, induce_clusters
from optimization import get_optimal_vector
from inference import tag_all_test, compute_accuracy
from main import MODEL1_CONFIGS

TRAIN = "data/train1.wtag"
TEST = "data/test1.wtag"
COMP = "data/comp1.words"
WDIR = "tune_tmp"          # throwaway: never goes into the submission


def _subset(cfg):
    return [c for c in FEATURE_CLASSES if c not in cfg["drop"]]


def main():
    ap = argparse.ArgumentParser(description="Model 1 (config x lambda) grid search on test1.")
    ap.add_argument("--sid", default="000000000")
    ap.add_argument("--lams", type=float, nargs="+", default=[0.1, 0.3, 0.5, 1.0, 2.0],
                    help="L2 lambda values to try.")
    ap.add_argument("--configs", nargs="+", default=sorted(MODEL1_CONFIGS),
                    help="Which MODEL1_CONFIGS presets to search.")
    ap.add_argument("--seed", type=int, default=42, help="Seed for w_0 init (fair comparison).")
    ap.add_argument("--dry_run", action="store_true",
                    help="Build features per config and print n_features only; skip all training.")
    args = ap.parse_args()

    os.makedirs(WDIR, exist_ok=True)
    os.makedirs("trained_models", exist_ok=True)
    t_start = time.time()
    results = []   # (acc, config_name, lam, n_features)

    def log(msg):
        """Timestamped, flushed print -> shows up live even when piped to a file."""
        el = time.time() - t_start
        print(f"[{time.strftime('%H:%M:%S')} +{el/60:5.1f}m] {msg}", flush=True)

    total_runs = len(args.configs) * len(args.lams)
    done = 0
    log(f"START grid: {len(args.configs)} configs x {len(args.lams)} lams = {total_runs} runs")
    log(f"configs={args.configs}  lams={args.lams}")

    for ci, cname in enumerate(args.configs, 1):
        cfg = MODEL1_CONFIGS[cname]
        log(f"=== config {cname} ({ci}/{len(args.configs)}): preprocessing... thr={cfg['thr']}")
        tp = time.time()
        clusters = induce_clusters(TRAIN, **cfg["clusters"]) if cfg.get("clusters") else None
        stats, f2i = preprocess_train(TRAIN, cfg["thr"], _subset(cfg), clusters=clusters)
        nfeat = f2i.n_total_features
        fit = "OK <10k" if nfeat < 10000 else "OVER >=10k"
        log(f"    config {cname}: built {nfeat} features in {time.time()-tp:.0f}s  [{fit}]")
        if args.dry_run:
            results.append((0.0, cname, 0.0, nfeat))   # reuse leaderboard writer below
            done += len(args.lams)
            continue
        if nfeat >= 10000:
            log(f"    !! {cname} has {nfeat} >= 10000 -- SKIPPING (would zero the grade)")
            done += len(args.lams)
            continue
        for lam in args.lams:
            done += 1
            log(f"    -> [{done}/{total_runs}] config {cname} lam={lam}: training...")
            np.random.seed(args.seed)                      # identical w_0 across lams -> fair
            wpath = f"{WDIR}/w_{cname}_{lam}.pkl"
            t0 = time.time()
            get_optimal_vector(statistics=stats, feature2id=f2i, weights_path=wpath, lam=lam)
            with open(wpath, "rb") as f:
                params, _ = pickle.load(f)
            pred = f"{WDIR}/pred_{cname}_{lam}.wtag"
            tag_all_test(TEST, params[0], f2i, pred, tagged=True)   # reuse in-memory f2i
            acc = compute_accuracy(pred, TEST)
            results.append((acc, cname, lam, nfeat))
            dt = time.time() - t0
            avg = (time.time() - t_start) / done
            eta = avg * (total_runs - done) / 60
            best_so_far = max(results)
            log(f"    DONE [{done}/{total_runs}] {cname} lam={lam}: test1={acc*100:.2f}%  "
                f"({dt:.0f}s)  best={best_so_far[0]*100:.2f}%({best_so_far[1]},lam={best_so_far[2]})  ETA~{eta:.0f}m")
            # Persist partial leaderboard after every run so a kill never loses progress.
            with open("tune_results.txt", "w") as f:
                f.write(f"# partial: {done}/{total_runs} runs done\n")
                for a, c, l, n in sorted(results, reverse=True):
                    f.write(f"{a*100:6.2f}  {c:>6}  lam={l:<5}  feat={n}\n")

    if not results:
        log("No valid configs (all >= 10000 features). Adjust thresholds.")
        return

    if args.dry_run:
        log("DRY RUN -- feature counts only (no training):")
        for _, c, _, n in sorted(results, key=lambda r: r[3]):
            log(f"    config {c}: {n} features  [{'OK <10k' if n < 10000 else 'OVER >=10k'}]")
        return

    results.sort(reverse=True)
    lines = [f"{'acc%':>6}  {'config':>6}  {'lam':<5}  feat"]
    lines += [f"{a*100:6.2f}  {c:>6}  {l:<5}  {n}" for a, c, l, n in results]
    print("\n=== LEADERBOARD (test1.wtag) ===", flush=True)
    print("\n".join(lines), flush=True)
    with open("tune_results.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    best_acc, bc, bl, _ = results[0]
    log(f"BEST: config={bc} lam={bl}  test1={best_acc*100:.2f}%")

    # Finalize without retraining: the winning pickle already holds (params, feature2id).
    log("finalizing -> trained_models/weights_1.pkl + comp file ...")
    shutil.copy(f"{WDIR}/w_{bc}_{bl}.pkl", "trained_models/weights_1.pkl")
    with open("trained_models/weights_1.pkl", "rb") as f:
        params, f2i_best = pickle.load(f)
    tag_all_test(COMP, params[0], f2i_best, f"comp_m1_{args.sid}.wtag", tagged=False)
    log(f"ALL DONE in {(time.time()-t_start)/60:.1f} min. "
        f"Best config={bc} lam={bl} ({best_acc*100:.2f}% on test1). "
        f"-> weights_1.pkl + comp_m1_{args.sid}.wtag")


if __name__ == "__main__":
    main()
