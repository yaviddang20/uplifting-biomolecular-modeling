import sys, torch
sys.path.insert(0, "/mnt/scratch/user/daviyang")
if "--patch" in sys.argv:
    import pot_batch_patch; print("patched:", pot_batch_patch.apply())
from boltz.model.potentials.potentials import VDWOverlapPotential

DEV = "cuda"; torch.manual_seed(0)

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

N1, N2 = 12, 10
c1 = torch.randn(N1, 3, device=DEV) * 2.0
c2 = torch.randn(N2, 3, device=DEV) * 2.0
pm = {"buffer": 0.2, "resampling_weight": 1.0, "guidance_weight": 1.0}

f1, x1 = feats_for([N1], [c1]); e1 = VDWOverlapPotential().compute(x1, f1, pm)
f2, x2 = feats_for([N2], [c2]); e2 = VDWOverlapPotential().compute(x2, f2, pm)
fb, xb = feats_for([N1, N2], [c1, c2]); eb = VDWOverlapPotential().compute(xb, fb, pm)

want = torch.tensor([e1.item(), e2.item()], device=DEV)
print(f"B=1 separately : {want.tolist()}")
print(f"B=2 stacked    : {eb.tolist()}")
ok = eb.shape == want.shape and torch.allclose(eb, want, atol=1e-5)
print("VERDICT:", "OK - per-record energies match B=1" if ok else "BROKEN")

# --- gradients ---
g1 = VDWOverlapPotential().compute_gradient(x1, f1, pm)
g2 = VDWOverlapPotential().compute_gradient(x2, f2, pm)
gb = VDWOverlapPotential().compute_gradient(xb, fb, pm)
d0 = (gb[0, :N1] - g1[0, :N1]).abs().max().item()
d1 = (gb[1, :N2] - g2[0, :N2]).abs().max().item()
pad = gb[1, N2:].abs().max().item()
print(f"grad max|diff| record0={d0:.3e} record1={d1:.3e} padding_grad={pad:.3e}")
gok = d0 < 1e-5 and d1 < 1e-5 and pad < 1e-5
print("GRAD VERDICT:", "OK" if gok else "BROKEN")
sys.exit(0 if (ok and gok) else 1)
