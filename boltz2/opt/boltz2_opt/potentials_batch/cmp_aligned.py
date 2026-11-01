"""Same probe across batch sizes, compared BOTH raw and after rigid-body
superposition. Boltz emits every structure in an arbitrary global frame (the
random augmentation), so a raw coordinate diff conflates 'different structure'
with 'same structure, different orientation'. Kabsch separates them."""
import glob, sys, numpy as np
BENCH = sys.argv[1] if len(sys.argv) > 1 else "/w/bench_bsinv"
TAGS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["bs1", "bs2", "bs3", "bs4"]
REF = sys.argv[3] if len(sys.argv) > 3 else "bs4"

def probe(tag):
    h = [p for p in sorted(glob.glob(f"{BENCH}/out_{tag}/**/probe*.pdb", recursive=True))
         if "model_0" in p]
    if not h: return None
    return np.array([(float(l[30:38]), float(l[38:46]), float(l[46:54]))
                     for l in open(h[0]) if l.startswith(("ATOM", "HETATM"))])

def kabsch_rmsd(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    V, S, Wt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt
    Pr = Pc @ R
    return float(np.sqrt(((Pr - Qc) ** 2).sum(1).mean())), float(np.abs(Pr - Qc).max())

ref = probe(REF)
if ref is None:
    print(f"{REF} MISSING"); sys.exit(1)
print(f"reference {REF}, {len(ref)} atoms\n")
print(f"{'tag':6} {'raw max|d|':>12} {'raw rmsd':>10} | {'aligned rmsd':>13} {'aligned max|d|':>15}")
for t in TAGS:
    x = probe(t)
    if x is None or x.shape != ref.shape:
        print(f"{t:6} MISSING/SHAPE"); continue
    raw_mx = float(np.abs(x - ref).max())
    raw_rmsd = float(np.sqrt(((x - ref) ** 2).sum(1).mean()))
    a_rmsd, a_mx = kabsch_rmsd(x, ref)
    print(f"{t:6} {raw_mx:12.6f} {raw_rmsd:10.6f} | {a_rmsd:13.6f} {a_mx:15.6f}")
