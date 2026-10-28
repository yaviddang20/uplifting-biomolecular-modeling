"""B=3, unequal sizes, multiplicity>1 (record-major, as repeat_interleave lays out)."""
import sys, torch
sys.path.insert(0, "/mnt/scratch/user/daviyang")
import pot_batch_patch; pot_batch_patch.apply()
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

alone = []
for n, c in zip(NS, CS):
    f, x = feats_for([n], [c])
    alone.append(VDWOverlapPotential().compute(x, f, pm).item())
print("B=1 separately :", [round(v, 6) for v in alone])

fb, xb = feats_for(NS, CS)
eb = VDWOverlapPotential().compute(xb, fb, pm)
print("B=3 m=1        :", [round(v, 6) for v in eb.tolist()])
ok1 = torch.allclose(eb, torch.tensor(alone, device=DEV), atol=1e-5)

for m in (2, 3):
    xm = xb.repeat_interleave(m, 0)                      # record-major
    em = VDWOverlapPotential().compute(xm, fb, pm)
    want = torch.tensor(alone, device=DEV).repeat_interleave(m)
    okm = em.shape == want.shape and torch.allclose(em, want, atol=1e-5)
    print(f"B=3 m={m}        : {[round(v,6) for v in em.tolist()]}  -> {'OK' if okm else 'BROKEN'}")
    ok1 = ok1 and okm

    gm = VDWOverlapPotential().compute_gradient(xm, fb, pm)
    gd = max((gm[b * m + s, : NS[b]] - VDWOverlapPotential().compute_gradient(
                 *(lambda fx: (fx[1], fx[0], pm))(feats_for([NS[b]], [CS[b]])))[0, : NS[b]]
              ).abs().max().item()
             for b in range(3) for s in range(m))
    padmax = max([gm[b * m + s, NS[b]:].abs().max().item()
                  for b in range(3) for s in range(m)
                  if gm[b * m + s, NS[b]:].numel()] or [0.0])
    print(f"            grad max|diff|={gd:.3e} padding={padmax:.3e}")
    ok1 = ok1 and gd < 1e-5 and padmax < 1e-5

print("VERDICT:", "OK" if ok1 else "BROKEN")
sys.exit(0 if ok1 else 1)
