"""Compare the probe record's coordinates across runs. The probe yaml is
byte-identical in every leg; only its batch-mates (or batch size) differ."""
import glob, sys, numpy as np

BENCH = sys.argv[1] if len(sys.argv) > 1 else "/w/bench_det2"

def probe(tag):
    hits = glob.glob(f"{BENCH}/out_{tag}/**/probe*.pdb", recursive=True)
    hits = [h for h in sorted(hits) if "model_0" in h] or sorted(hits)
    if not hits:
        return None, None
    xyz = [(float(l[30:38]), float(l[38:46]), float(l[46:54]))
           for l in open(hits[0]) if l.startswith(("ATOM", "HETATM"))]
    return np.array(xyz), hits[0]

ref, pref = probe("detA1")
if ref is None:
    print("detA1 probe MISSING"); sys.exit(1)
print(f"reference: {pref}  ({len(ref)} atoms)\n")
print(f"{'comparison':34} {'max|dxyz| (A)':>14} {'rmsd (A)':>10}   meaning")
rows = [("detA2", "same-set (reproducibility floor)"),
        ("detB1", "cross-set (different batch-mates)"),
        ("detS",  "unbatched (bs=1, alone)"),
        ("detR",  "remainder group (batches of 1 and 4)")]
worst = 0.0
for tag, meaning in rows:
    x, p = probe(tag)
    if x is None:
        print(f"detA1 vs {tag:<10} MISSING"); continue
    if x.shape != ref.shape:
        print(f"detA1 vs {tag:<10} SHAPE {x.shape} vs {ref.shape}"); continue
    mx = float(np.abs(x - ref).max())
    rmsd = float(np.sqrt(((x - ref) ** 2).sum(axis=1).mean()))
    worst = max(worst, mx)
    print(f"detA1 vs {tag:<24} {mx:14.6f} {rmsd:10.6f}   {meaning}")
print(f"\nworst max|dxyz|: {worst:.6f} A")
print("VERDICT:", "IDENTICAL" if worst < 1e-3 else
      ("CLOSE" if worst < 0.5 else "DIFFERENT - investigate"))
sys.exit(0 if worst < 1e-3 else 1)
