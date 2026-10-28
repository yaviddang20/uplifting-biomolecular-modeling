"""Make boltz's steering potentials batch-aware (B > 1).

Upstream builds pair topology from feats[...][0] -- record 0 only -- and then
broadcasts that one pair list across the whole leading axis, so every record
after the first is scored with record 0's connectivity and reaches into its
own padding atoms. See pot_correctness.py.

This builds each record's topology from its own slice, concatenates, and
tags every pair with its record so the energy reduction only sums the pairs
that belong to the record being scored. Atom indices need no offset: feats
are already padded to a common width, so a record's indices are valid in the
shared atom axis.
"""
import torch
from boltz.model.potentials import potentials as P

_PAIR_POTENTIALS = ("VDWOverlapPotential", "ConnectionsPotential",
                    "PoseBustersPotential", "ChiralAtomPotential",
                    "StereoBondPotential", "PlanarBondPotential")


def _batched_compute_args(cls, orig):
    def compute_args(self, feats, parameters):
        B = feats["atom_pad_mask"].shape[0]
        if B == 1:
            self._pair_record, self._B = None, 1
            return orig(self, feats, parameters)

        idx, ks, los, his, recs = [], [], [], [], []
        for b in range(B):
            sub = {k: (v[b : b + 1] if torch.is_tensor(v) else v)
                   for k, v in feats.items()}
            index, args, com, ref, op = orig(self, sub, parameters)
            if com is not None or ref is not None or op is not None:
                raise NotImplementedError(
                    f"{cls.__name__}: com/ref/operator args not batched yet")
            k, lo, hi = args
            idx.append(index)
            ks.append(k)
            los.append(lo)
            his.append(hi)
            recs.append(torch.full((index.shape[1],), b, dtype=torch.long,
                                   device=index.device))

        self._pair_record = torch.cat(recs)
        self._B = B
        hi_cat = None if his[0] is None else torch.cat(his)
        return (torch.cat(idx, dim=1), (torch.cat(ks), torch.cat(los), hi_cat),
                None, None, None)

    return compute_args


_orig_compute = P.Potential.compute


def _compute(self, coords, feats, parameters):
    index, args, com, ref, op = self.compute_args(feats, parameters)
    pr = getattr(self, "_pair_record", None)
    if pr is None or com is not None or ref is not None or op is not None:
        return _orig_compute(self, coords, feats, parameters)
    if index.shape[1] == 0:
        return torch.zeros(coords.shape[:-2], device=coords.device)

    value = self.compute_variable(coords, index, ref_coords=None, ref_mask=None,
                                  compute_gradient=False)
    energy = self.compute_function(value, *args, negation_mask=None,
                                   compute_derivative=False)
    L = energy.shape[0]                      # B * samples_per_record
    m = L // self._B
    rec_of = torch.arange(L, device=energy.device) // m
    mask = pr.unsqueeze(0) == rec_of.unsqueeze(1)      # (L, P)
    return (energy * mask).sum(dim=tuple(range(1, energy.dim())))


_orig_grad = P.Potential.compute_gradient


def _compute_gradient(self, coords, feats, parameters):
    index, args, com, ref, op = self.compute_args(feats, parameters)
    pr = getattr(self, "_pair_record", None)
    if pr is None or com is not None or ref is not None or op is not None:
        return _orig_grad(self, coords, feats, parameters)
    if index.shape[1] == 0:
        return torch.zeros_like(coords)

    value, grad_value = self.compute_variable(coords, index, ref_coords=None,
                                              ref_mask=None, compute_gradient=True)
    energy, dEnergy = self.compute_function(value, *args, negation_mask=None,
                                            compute_derivative=True)
    L = dEnergy.shape[0]
    m = L // self._B
    rec_of = torch.arange(L, device=dEnergy.device) // m
    dEnergy = dEnergy * (pr.unsqueeze(0) == rec_of.unsqueeze(1))   # foreign pairs contribute nothing

    prod = dEnergy.tile(grad_value.shape[-3]).unsqueeze(-1) * grad_value.flatten(
        start_dim=-3, end_dim=-2)
    if prod.dim() > 3:
        prod = prod.sum(dim=list(range(1, prod.dim() - 2)))
    return torch.zeros_like(coords).scatter_reduce(
        -2,
        index.flatten(start_dim=0, end_dim=1).unsqueeze(-1).expand(
            (*coords.shape[:-2], -1, 3)),
        prod, "sum")


def apply():
    P.Potential.compute = _compute
    P.Potential.compute_gradient = _compute_gradient
    for name in _PAIR_POTENTIALS:
        cls = getattr(P, name, None)
        if cls is not None and "compute_args" in cls.__dict__:
            cls.compute_args = _batched_compute_args(cls, cls.__dict__["compute_args"])
    return [n for n in _PAIR_POTENTIALS if getattr(P, n, None) is not None]
