import glob, os, sys, numpy as np
B = sys.argv[1]
def recs(t):
    d = {}
    for p in glob.glob(f"{B}/out_{t}/**/*_model_0.pdb", recursive=True):
        d[os.path.basename(p).replace("_model_0.pdb","")] = np.array(
            [(float(l[30:38]),float(l[38:46]),float(l[46:54]))
             for l in open(p) if l.startswith(("ATOM","HETATM"))])
    return d
def kab(P,Q):
    Pc,Qc = P-P.mean(0), Q-Q.mean(0)
    V,S,Wt = np.linalg.svd(Pc.T@Qc)
    D = np.diag([1.,1.,np.sign(np.linalg.det(V@Wt))])
    return float(np.sqrt((((Pc@(V@D@Wt))-Qc)**2).sum(1).mean()))
print(f"{'comparison':26} {'n':>3} {'mean':>8} {'max':>8}   verdict")
for a,b in (("g_b1","g_b4"),("g_b1","g_b8"),("g_b4","g_b8"),
            ("u_b1","u_b4"),("u_b1","u_b8"),("u_b4","u_b8")):
    A,Bd = recs(a), recs(b)
    k=[x for x in sorted(set(A)&set(Bd)) if A[x].shape==Bd[x].shape]
    if not k: print(f"{a} vs {b}: none"); continue
    v=[kab(A[x],Bd[x]) for x in k]
    print(f"{a+' vs '+b:26} {len(k):3d} {np.mean(v):8.4f} {np.max(v):8.4f}   "
          f"{'PASS' if max(v)<=1.0 else 'FAIL'}")
