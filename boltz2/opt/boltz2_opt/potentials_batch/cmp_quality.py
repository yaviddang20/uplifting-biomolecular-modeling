"""Batched vs unbatched prediction quality, same per-record seeds.

Both legs run with BOLTZ_BATCH_POTENTIALS=1 so the per-record noise carrier
seeds each record identically; only the batch size differs. Records are
independent, so a correct implementation should give the SAME structure.
Reports per-record max atom deviation and RMSD, plus confidence deltas.
"""
import json, sys, glob, os
import numpy as np

A, B = sys.argv[1], sys.argv[2]


def atoms(path):
    xyz = []
    for ln in open(path):
        if ln.startswith(("ATOM", "HETATM")):
            xyz.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
    return np.array(xyz)


def find(root, name, ext):
    hits = glob.glob(f"{root}/**/{name}*{ext}", recursive=True)
    return sorted(hits)[0] if hits else None


names = ["lig_1", "lig_2", "lig_3", "lig_4", "pp_1", "pp_2", "pp_3", "pp_4"]
print(f"{'record':8} {'n_atoms':>8} {'max|dxyz|':>11} {'rmsd':>9}   confidence")
worst = 0.0
for n in names:
    pa, pb = find(A, n, ".pdb"), find(B, n, ".pdb")
    if not pa or not pb:
        print(f"{n:8} MISSING  a={bool(pa)} b={bool(pb)}")
        continue
    xa, xb = atoms(pa), atoms(pb)
    if xa.shape != xb.shape:
        print(f"{n:8} SHAPE MISMATCH {xa.shape} vs {xb.shape}")
        continue
    d = np.abs(xa - xb)
    mx = d.max()
    rmsd = float(np.sqrt((((xa - xb) ** 2).sum(axis=1)).mean()))
    worst = max(worst, mx)

    cf = ""
    ja, jb = find(A, f"confidence_{n}", ".json"), find(B, f"confidence_{n}", ".json")
    if ja and jb:
        da, db = json.load(open(ja)), json.load(open(jb))
        k = "complex_plddt" if "complex_plddt" in da else next(iter(da))
        cf = f"  {k}: {da[k]:.4f} vs {db[k]:.4f}  (d={db[k]-da[k]:+.4f})"
    print(f"{n:8} {len(xa):8d} {mx:11.6f} {rmsd:9.6f}{cf}")

print(f"\nworst max|dxyz| across records: {worst:.6f} A")
print("VERDICT:", "IDENTICAL" if worst < 1e-3 else
      ("CLOSE" if worst < 0.5 else "DIFFERENT - investigate"))
