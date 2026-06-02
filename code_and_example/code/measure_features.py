import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import FeatureStatistics, FEATURE_CLASSES
from main import MODEL1_CONFIGS

fs = FeatureStatistics()
fs.get_word_tag_pair_count("data/train1.wtag")
print("histories:", len(fs.histories), " tags:", len(fs.tags))
print(f'{"thr":>4} ' + " ".join(f"{fc:>6}" for fc in FEATURE_CLASSES) + f'{"TOTAL":>8}')
for thr in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]:
    c = {fc: sum(1 for v in fs.feature_rep_dict[fc].values() if v >= thr) for fc in FEATURE_CLASSES}
    print(f"{thr:>4} " + " ".join(f"{c[fc]:>6}" for fc in FEATURE_CLASSES) + f"{sum(c.values()):>8}")

# Exact total per Model 1 config (must be < 10000).
print("\nMODEL 1 CONFIGS (cap 10000):")
for name, cfg in sorted(MODEL1_CONFIGS.items()):
    keep = [fc for fc in FEATURE_CLASSES if fc not in cfg["drop"]]
    total, per = 0, {}
    for fc in keep:
        thr = cfg["thr"].get(fc, 1)
        n = sum(1 for v in fs.feature_rep_dict[fc].values() if v >= thr)
        per[fc] = n
        total += n
    flag = "OK" if total < 10000 else "OVER!"
    print(f"  {name}: {total:>6}  [{flag}]  drop={cfg['drop']}")
    print(f"       " + "  ".join(f"{fc}:{per[fc]}" for fc in keep))
