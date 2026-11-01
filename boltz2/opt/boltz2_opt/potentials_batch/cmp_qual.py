"""Are batched predictions WORSE, or just different? Per-record confidence
across batch sizes, paired (same record, so the comparison is within-record)."""
import glob, json, os, sys, numpy as np
B = sys.argv[1]
TAGS = sys.argv[2].split(",")
KEY = "complex_plddt"

def vals(tag):
    d = {}
    for p in glob.glob(f"{B}/out_{tag}/**/confidence_*.json", recursive=True):
        n = os.path.basename(p).replace("confidence_", "").replace("_model_0.json", "")
        try:
            d[n] = float(json.load(open(p))[KEY])
        except Exception:
            pass
    return d

data = {t: vals(t) for t in TAGS}
common = sorted(set.intersection(*[set(v) for v in data.values()])) if all(data.values()) else []
print(f"{KEY}, n={len(common)} records common to {TAGS}\n")
print(f"{'tag':8} {'mean':>8} {'sd':>7} {'min':>7} {'max':>7}")
ref = None
for t in TAGS:
    a = np.array([data[t][k] for k in common])
    print(f"{t:8} {a.mean():8.4f} {a.std():7.4f} {a.min():7.4f} {a.max():7.4f}")
    if ref is None: ref, reft = a, t
print()
print(f"{'paired delta vs '+reft:28} {'mean':>9} {'max|d|':>9}")
for t in TAGS[1:]:
    a = np.array([data[t][k] for k in common])
    d = a - ref
    print(f"{t:28} {d.mean():+9.4f} {np.abs(d).max():9.4f}")
