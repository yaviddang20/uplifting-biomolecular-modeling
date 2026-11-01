"""All records, two runs, aligned RMSD per record -> mean/max. Mirrors
tools/boltz2_batching/compare.py's gate (default 1.0 A)."""
import glob, os, sys, numpy as np
BENCH = sys.argv[1]; GATE = 1.0

def recs(tag):
    out = {}
    for p in glob.glob(f"{BENCH}/out_{tag}/**/*_model_0.pdb", recursive=True):
        name = os.path.basename(p).replace("_model_0.pdb", "")
        out[name] = np.array([(float(l[30:38]), float(l[38:46]), float(l[46:54]))
                              for l in open(p) if l.startswith(("ATOM", "HETATM"))])
    return out

def kabsch(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    V, S, Wt = np.linalg.svd(Pc.T @ Qc)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(V @ Wt))])
    return float(np.sqrt((((Pc @ (V @ D @ Wt)) - Qc) ** 2).sum(1).mean()))

print(f"{'comparison':28} {'n':>3} {'mean RMSD':>10} {'max RMSD':>9}   verdict (gate {GATE} A)")
for a, b, label in (("g_b1","g_b4","guided   b1 vs b4"),
                    ("g_b1","g_b8","guided   b1 vs b8"),
                    ("u_b1","u_b4","unguided b1 vs b4"),
                    ("u_b1","u_b8","unguided b1 vs b8")):
    A, B = recs(a), recs(b)
    keys = sorted(set(A) & set(B))
    keys = [k for k in keys if A[k].shape == B[k].shape]
    if not keys:
        print(f"{label:28} {'--':>3}  no comparable records"); continue
    v = [kabsch(A[k], B[k]) for k in keys]
    mean, mx = float(np.mean(v)), float(np.max(v))
    print(f"{label:28} {len(keys):3d} {mean:10.4f} {mx:9.4f}   "
          f"{'PASS' if mx <= GATE else 'FAIL'}")
