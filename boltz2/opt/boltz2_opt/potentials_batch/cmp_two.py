import glob, os, sys, numpy as np
(bA, tA, bB, tB) = sys.argv[1:5]
def recs(b, t):
    d = {}
    for p in glob.glob(f"{b}/out_{t}/**/*_model_0.pdb", recursive=True):
        d[os.path.basename(p).replace("_model_0.pdb", "")] = np.array(
            [(float(l[30:38]), float(l[38:46]), float(l[46:54]))
             for l in open(p) if l.startswith(("ATOM", "HETATM"))])
    return d
def kab(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    V, S, Wt = np.linalg.svd(Pc.T @ Qc)
    D = np.diag([1., 1., np.sign(np.linalg.det(V @ Wt))])
    return float(np.sqrt((((Pc @ (V @ D @ Wt)) - Qc) ** 2).sum(1).mean()))
A, B = recs(bA, tA), recs(bB, tB)
k = [x for x in sorted(set(A) & set(B)) if A[x].shape == B[x].shape]
if not k:
    print(f"no comparable records ({len(A)} vs {len(B)} found)"); sys.exit(0)
v = [kab(A[x], B[x]) for x in k]
raw = [float(np.abs(A[x] - B[x]).max()) for x in k]
verdict = "PASS" if max(v) <= 1.0 else "FAIL"
print(f"{tA} vs {tB}: n={len(k)} aligned mean={np.mean(v):.4f} max={np.max(v):.4f} "
      f"| raw max={max(raw):.6f} -> {verdict}")
print("  bitwise identical" if max(raw) == 0.0 else f"  not bitwise (raw max {max(raw):.2e})")
