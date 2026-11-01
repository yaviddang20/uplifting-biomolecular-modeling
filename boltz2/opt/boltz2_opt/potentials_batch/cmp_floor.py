import glob, os, sys, numpy as np
B = sys.argv[1]
def recs(tag):
    d = {}
    for p in glob.glob(f"{B}/out_{tag}/**/*_model_0.pdb", recursive=True):
        d[os.path.basename(p).replace("_model_0.pdb","")] = np.array(
            [(float(l[30:38]),float(l[38:46]),float(l[46:54]))
             for l in open(p) if l.startswith(("ATOM","HETATM"))])
    return d
def kab(P,Q):
    Pc,Qc = P-P.mean(0), Q-Q.mean(0)
    V,S,Wt = np.linalg.svd(Pc.T@Qc)
    D = np.diag([1.,1.,np.sign(np.linalg.det(V@Wt))])
    return float(np.sqrt((((Pc@(V@D@Wt))-Qc)**2).sum(1).mean()))
print(f"{'comparison':46} {'n':>3} {'mean':>8} {'max':>8}")
for a,b,lab in (("A1","A2","guided floor: same config x2, DET OFF"),
                ("D1","D2","guided floor: same config x2, DET ON"),
                ("D1","E", "guided scale: fast vs exact, DET ON")):
    A,Bd = recs(a), recs(b)
    k = [x for x in sorted(set(A)&set(Bd)) if A[x].shape==Bd[x].shape]
    if not k: print(f"{lab:46} {'--':>3}   no comparable records"); continue
    v=[kab(A[x],Bd[x]) for x in k]
    print(f"{lab:46} {len(k):3d} {np.mean(v):8.4f} {np.max(v):8.4f}")
