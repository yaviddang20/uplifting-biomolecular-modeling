"""Block-diagonal batching: B=3, UNEQUAL atom counts (so unequal pair counts),
multiplicity>1 laid out record-major. Compares against the B=1 path, which is
upstream's own code, and asks for BITWISE agreement -- the new reduction sums
exactly the columns upstream sums, in the same shape, so it should hold."""
import importlib.util as ilu, sys, torch

PATCH = sys.argv[1] if len(sys.argv) > 1 else "/kit/boltz2/opt/boltz2_opt/potentials_batch/patch.py"
spec = ilu.spec_from_file_location("_potb", PATCH)
POTB = ilu.module_from_spec(spec); spec.loader.exec_module(POTB)
print("applied to:", POTB.apply())
from boltz.model.potentials.potentials import VDWOverlapPotential

DEV = "cuda"; torch.manual_seed(1)

def feats_for(ns, cs):
    B, N = len(ns), max(ns)
    apm = torch.zeros(B, N, dtype=torch.bool, device=DEV)
    a2t = torch.zeros(B, N, N, device=DEV)
    asym = torch.zeros(B, N, dtype=torch.long, device=DEV)
    rel = torch.zeros(B, N, 128, device=DEV)
    for b, n in enumerate(ns):
        apm[b, :n] = True
        a2t[b, :n, :n] = torch.eye(n, device=DEV)
        asym[b, :n] = (torch.arange(n, device=DEV) >= n // 2).long()
        rel[b, :n, 6] = 1.0
    co = torch.zeros(B, N, 3, device=DEV)
    for b, c in enumerate(cs):
        co[b, : c.shape[0]] = c
    return {"atom_pad_mask": apm, "atom_to_token": a2t, "asym_id": asym,
            "ref_element": rel,
            "connected_chain_index": torch.zeros(B, 2, 0, dtype=torch.long, device=DEV)}, co

NS = [14, 9, 11]
CS = [torch.randn(n, 3, device=DEV) * 2.0 for n in NS]
pm = {"buffer": 0.2, "resampling_weight": 1.0, "guidance_weight": 1.0}
ok = True

# ---- reference: each record on its own, through upstream's B==1 path ----
alone_e, alone_g = [], []
for n, c in zip(NS, CS):
    f, x = feats_for([n], [c])
    alone_e.append(VDWOverlapPotential().compute(x, f, pm))
    alone_g.append(VDWOverlapPotential().compute_gradient(x, f, pm))
print("B=1 energies   :", [round(v.item(), 8) for v in alone_e])

fb, xb = feats_for(NS, CS)
want_e1 = torch.cat(alone_e)

for m in (1, 2, 3):
    xm = xb.repeat_interleave(m, 0)                     # record-major, as the sampler lays it out
    em = VDWOverlapPotential().compute(xm, fb, pm)
    want = want_e1.repeat_interleave(m)
    exact = torch.equal(em, want)
    close = torch.allclose(em, want, atol=1e-6)
    d = (em - want).abs().max().item()
    print(f"  B=3 m={m}  energy  max|d|={d:.3e}  bitwise={exact}  close={close}")
    ok = ok and close

    gm = VDWOverlapPotential().compute_gradient(xm, fb, pm)
    gd, pad, gexact = 0.0, 0.0, True
    for b in range(3):
        for s in range(m):
            got = gm[b * m + s, : NS[b]]
            ref = alone_g[b][0, : NS[b]]
            gd = max(gd, (got - ref).abs().max().item())
            gexact = gexact and torch.equal(got, ref)
            tailv = gm[b * m + s, NS[b]:]
            if tailv.numel():
                pad = max(pad, tailv.abs().max().item())
    print(f"  B=3 m={m}  grad    max|d|={gd:.3e}  bitwise={gexact}  padding_leak={pad:.3e}")
    ok = ok and gd < 1e-6 and pad < 1e-6

print("VERDICT:", "OK" if ok else "BROKEN")
sys.exit(0 if ok else 1)
