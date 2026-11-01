import glob, json, os, sys, numpy as np
B = "/w/bench_eq2"; KEY = "complex_plddt"
def vals(tag):
    d = {}
    for p in glob.glob(f"{B}/out_{tag}/**/confidence_*.json", recursive=True):
        n = os.path.basename(p).replace("confidence_", "").replace("_model_0.json", "")
        d[n] = float(json.load(open(p))[KEY])
    return d
tags = ["g_b1","g_b4","g_b8","u_b1","u_b4","u_b8"]
D = {t: vals(t) for t in tags}
common = sorted(set.intersection(*[set(D[t]) for t in tags]))
print(f"n={len(common)} records, paired\n")
print(f"{'comparison':22} {'mean d':>9} {'sd d':>8} {'t':>7} {'|t| vs 2.36':>12}")
for a, b in (("g_b1","g_b4"),("g_b1","g_b8"),("g_b4","g_b8"),
             ("u_b1","u_b4"),("u_b1","u_b8"),("u_b4","u_b8")):
    d = np.array([D[b][k] - D[a][k] for k in common])
    sd = d.std(ddof=1)
    t = d.mean() / (sd / np.sqrt(len(d))) if sd > 0 else float("nan")
    flag = "significant" if abs(t) > 2.36 else "not sig."
    print(f"{a+' -> '+b:22} {d.mean():+9.4f} {sd:8.4f} {t:7.2f}   {flag}")
print("\n(t crit = 2.36 for 7 df, two-sided p=0.05; negative t = lower pLDDT at larger batch)")
