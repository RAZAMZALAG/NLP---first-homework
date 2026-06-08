"""Throwaway sizing probe: cluster-family feature counts on train1 at various thresholds, per K.
Run: uv run python code/_size_clusters.py   (delete after sizing the LC presets)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import induce_clusters, FeatureStatistics

for K in (128, 256, 512):
    cl = induce_clusters("data/train1.wtag", k=K, min_freq=5, seed=42)
    fs = FeatureStatistics(clusters=cl)
    fs.get_word_tag_pair_count("data/train1.wtag")
    print(f"\n=== K={K}: {len(cl)} words clustered ===")
    for fam in ("f_clust", "f_prev_clust", "f_next_clust"):
        d = fs.feature_rep_dict[fam]
        cnt = " ".join(f"thr{t}:{sum(1 for v in d.values() if v >= t)}"
                       for t in (3, 5, 10, 20, 30, 50, 75, 100))
        print(f"  {fam}: {cnt}")
