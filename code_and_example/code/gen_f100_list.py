# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import FeatureStatistics

THRESHOLD = 10
lines = []

for model_num in [1, 2]:
    stats = FeatureStatistics()
    stats.get_word_tag_pair_count(f"data/train{model_num}.wtag")
    counts = stats.feature_rep_dict["f100"]

    features = [(word, tag, cnt) for (word, tag), cnt in counts.items() if cnt >= THRESHOLD]
    features.sort(key=lambda x: (-x[2], x[1], x[0]))

    lines.append("=" * 55)
    lines.append(f"  MODEL {model_num} -- train{model_num}.wtag  (threshold >= {THRESHOLD})")
    lines.append(f"  Total f100 features: {len(features)}")
    lines.append("=" * 55)
    lines.append(f"{'#':>4}  {'Word':<22}  {'Tag':<8}  {'Gold Count':>10}")
    lines.append(f"{'----':>4}  {'-'*22}  {'-'*8}  {'-'*10}")
    for i, (word, tag, cnt) in enumerate(features, 1):
        lines.append(f"{i:>4}  {word:<22}  {tag:<8}  {cnt:>10}")
    lines.append("")

out_path = os.path.join(os.path.dirname(__file__), "..", "f100_features.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Written to {os.path.abspath(out_path)}")
print(f"Total lines: {len(lines)}")
