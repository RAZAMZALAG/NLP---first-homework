import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import FeatureStatistics, FEATURE_CLASSES

fs = FeatureStatistics()
fs.get_word_tag_pair_count("data/train1.wtag")
print("histories:", len(fs.histories), " tags:", len(fs.tags))
print(f'{"thr":>4} ' + " ".join(f"{fc:>6}" for fc in FEATURE_CLASSES) + f'{"TOTAL":>8}')
for thr in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]:
    c = {fc: sum(1 for v in fs.feature_rep_dict[fc].values() if v >= thr) for fc in FEATURE_CLASSES}
    print(f"{thr:>4} " + " ".join(f"{c[fc]:>6}" for fc in FEATURE_CLASSES) + f"{sum(c.values()):>8}")
