"""8 EQUAL-LENGTH variants of the Top7 fold (confidently predicted without an
MSA, pLDDT ~0.85). Equal length => identical padding at every batch size, so a
b1-vs-bN difference cannot be a padding effect."""
import os, sys
sys.path.insert(0, "/w/bz-potbatch/tools/boltz2_batching")
from make_inputs import TOP7, yaml
out = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
os.makedirs(out, exist_ok=True)
base = TOP7
for k in range(n):
    p = (k * 7) % len(base)                      # one substitution, length preserved
    s = base[:p] + ("A" if base[p] != "A" else "G") + base[p + 1:]
    assert len(s) == len(base)
    name = f"eq{k:02d}"
    open(os.path.join(out, name + ".yaml"), "w").write(yaml([("A", s)]))
    print(name, len(s))
