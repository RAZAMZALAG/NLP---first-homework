"""Throwaway: re-tag test1 with a saved model at several Viterbi beam widths (no retrain).
Usage: uv run python code/beam_sweep.py <weights_path>"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(__file__))
import inference
from inference import tag_all_test, compute_accuracy

params, f2i = pickle.load(open(sys.argv[1], "rb"))
w = params[0]
os.makedirs("cv_tmp", exist_ok=True)
for beam in (30, 50, 75, 100, 200, 10000):
    inference.BEAM = beam
    tag_all_test("data/test1.wtag", w, f2i, "cv_tmp/bs.wtag", tagged=True)
    print(f"beam={beam}: {compute_accuracy('cv_tmp/bs.wtag', 'data/test1.wtag')*100:.2f}%")
