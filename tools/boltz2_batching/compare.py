#!/usr/bin/env python3
"""Compare two Boltz-2 kit output directories record by record (each: <dir>/by_seed/<name>/s<seed>/<name>_model_0.cif + confidence json).
Prints per record: atoms, all-atom RMSD after optimal superposition (Kabsch), and the confidence_score / complex_plddt of each side.
usage: compare.py <dir_a> <dir_b> [--max-rmsd X]     exit 1 when a record is missing on one side or (with --max-rmsd) any RMSD exceeds X."""
import sys, os, json, glob
import numpy as np


def read_cif_xyz(path):
    """Cartn_x/y/z of every ATOM / HETATM row of an mmCIF atom_site loop (order as written)."""
    cols, xyz, in_loop = [], [], False
    for line in open(path):
        t = line.strip()
        if t == "loop_":
            cols, in_loop = [], True; continue
        if in_loop and t.startswith("_atom_site."):
            cols.append(t.split(".", 1)[1]); continue
        if in_loop and cols and "Cartn_x" in cols and t and not t.startswith(("_", "#", "loop_")):
            f = t.split()
            if len(f) >= len(cols):
                xyz.append([float(f[cols.index(c)]) for c in ("Cartn_x", "Cartn_y", "Cartn_z")])
            continue
        if in_loop and t.startswith("_") and not t.startswith("_atom_site."):
            if xyz: break
            cols, in_loop = [], False
        if t.startswith("#") and xyz:
            break
    return np.asarray(xyz, dtype=np.float64)


def kabsch_rmsd(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    u, s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt((((a @ r) - b) ** 2).sum(1).mean()))


def units(root):
    out = {}
    for d in sorted(glob.glob(os.path.join(root, "by_seed", "*", "s*"))):
        name = os.path.basename(os.path.dirname(d)); seed = os.path.basename(d)
        cif = os.path.join(d, f"{name}_model_0.cif"); cj = os.path.join(d, f"confidence_{name}_model_0.json")
        if os.path.isfile(cif):
            out[(name, seed)] = (cif, json.load(open(cj)) if os.path.isfile(cj) else {})
    return out


def main(argv):
    a_dir, b_dir = argv[1], argv[2]
    lim = float(argv[argv.index("--max-rmsd") + 1]) if "--max-rmsd" in argv else None
    A, B = units(a_dir), units(b_dir)
    bad = False
    print(f"{'record':<22}{'seed':<6}{'atoms':>7}{'rmsd_A':>9}{'conf_a':>9}{'conf_b':>9}{'plddt_a':>9}{'plddt_b':>9}")
    rms = []
    for key in sorted(set(A) | set(B)):
        if key not in A or key not in B:
            print(f"{key[0]:<22}{key[1]:<6} MISSING in {'A' if key not in A else 'B'}"); bad = True; continue
        xa, xb = read_cif_xyz(A[key][0]), read_cif_xyz(B[key][0])
        if xa.shape != xb.shape or len(xa) == 0:
            print(f"{key[0]:<22}{key[1]:<6} atom count differs: {xa.shape} vs {xb.shape}"); bad = True; continue
        r = kabsch_rmsd(xa, xb); rms.append(r)
        ca, cb = A[key][1], B[key][1]
        print(f"{key[0]:<22}{key[1]:<6}{len(xa):>7}{r:>9.3f}{ca.get('confidence_score', float('nan')):>9.4f}{cb.get('confidence_score', float('nan')):>9.4f}"
              f"{ca.get('complex_plddt', float('nan')):>9.4f}{cb.get('complex_plddt', float('nan')):>9.4f}")
        if lim is not None and r > lim:
            bad = True
    if rms:
        print(f"records compared: {len(rms)}   rmsd mean {np.mean(rms):.3f}  max {np.max(rms):.3f} A" + (f"   (limit {lim})" if lim is not None else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
