import glob, os, sys, numpy as np
B = sys.argv[1] if len(sys.argv) > 1 else "/w/bench_div"
for tag in ("noise", "coords"):
    a = sorted(glob.glob(f"{B}/dump_b1/{tag}_*.npy"))
    b = sorted(glob.glob(f"{B}/dump_b4/{tag}_*.npy"))
    print(f"--- {tag}: b1 has {len(a)}, b4 has {len(b)}")
    for x, y in zip(a, b):
        A, Bb = np.load(x), np.load(y)
        if A.shape != Bb.shape:
            print(f"   {os.path.basename(x)}: SHAPE {A.shape} vs {Bb.shape}")
            continue
        d = float(np.abs(A - Bb).max())
        verdict = "IDENTICAL" if d == 0 else "DIFFERS"
        print(f"   {os.path.basename(x)}: shape={A.shape} max|d|={d:.6e}  {verdict}")
