"""Per-family feature counts at a range of thresholds, for designing configs under the
parameter cap. Usage:  uv run python code/measure_features.py [1|2]   (default 1)"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import FeatureStatistics, FEATURE_CLASSES
from main import MODEL1_CONFIGS, MODEL2_CONFIGS, MODEL2_FEATURE_SUBSET

model = sys.argv[1] if len(sys.argv) > 1 else "1"
train = f"data/train{model}.wtag"
# Sweep over ALL families so we can see every family's budget (even ones a config may drop).
families = FEATURE_CLASSES
configs = MODEL1_CONFIGS if model == "1" else MODEL2_CONFIGS
cap = 10000 if model == "1" else 500

fs = FeatureStatistics()
fs.get_word_tag_pair_count(train)
print(f"train{model}: histories={len(fs.histories)}  tags={len(fs.tags)}  cap={cap}")
print(f'{"thr":>4} ' + " ".join(f"{fc:>6}" for fc in families) + f'{"TOTAL":>8}')
for thr in [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50]:
    c = {fc: sum(1 for v in fs.feature_rep_dict[fc].values() if v >= thr) for fc in families}
    print(f"{thr:>4} " + " ".join(f"{c[fc]:>6}" for fc in families) + f"{sum(c.values()):>8}")

print(f"\nMODEL {model} CONFIGS (cap {cap}):")
for name, cfg in sorted(configs.items()):
    keep = [fc for fc in families if fc not in cfg["drop"]]
    per = {fc: sum(1 for v in fs.feature_rep_dict[fc].values() if v >= cfg["thr"].get(fc, 1)) for fc in keep}
    total = sum(per.values())
    flag = "OK" if total <= cap else "OVER!"
    print(f"  {name}: {total:>6}  [{flag}]")
    print(f"       " + "  ".join(f"{fc}:{per[fc]}" for fc in keep))
