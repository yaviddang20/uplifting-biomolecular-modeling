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

        # compute_args runs on every compute()/compute_gradient() -- ~20 guidance
        # steps x ~200 denoising steps. The result depends only on feats and the
        # parameter values, both constant across a guidance loop, so memoise it;
        # rebuilding per-record topology that often is most of the batched cost.
        try:
            key = (id(feats), tuple(sorted((k, float(v)) for k, v in parameters.items()
                                           if isinstance(v, (int, float, bool)))))
        except Exception:
            key = None
        hit = getattr(self, "_args_cache", None)
        if key is not None and hit is not None and hit[0] == key:
            self._pair_record, self._B = hit[1], B
            return hit[2]

        idx, argcols, recs = [], [], []
        for b in range(B):
            sub = {k: (v[b : b + 1] if torch.is_tensor(v) else v)
                   for k, v in feats.items()}
            index, args, com, ref, op = orig(self, sub, parameters)
            if com is not None or ref is not None or op is not None:
                raise NotImplementedError(
                    f"{cls.__name__}: com/ref/operator args not batched yet")
            idx.append(index)
            argcols.append(args)
            recs.append(torch.full((index.shape[1],), b, dtype=torch.long,
                                   device=index.device))

        self._pair_record = torch.cat(recs)
        self._B = B
        # args differ in arity and any element may be None (an unbounded side of a
        # flat-bottom potential): concatenate per position, keep None as None.
        cat_args = tuple(None if col[0] is None else torch.cat(col)
                         for col in zip(*argcols))
        out = (torch.cat(idx, dim=1), cat_args, None, None, None)
        if key is not None:
            self._args_cache = (key, self._pair_record, out)
        return out

    return compute_args


def _guard(self, feats, parameters):
    """A potential this patch does not batch must NOT quietly run upstream's
    record-0 code on a B>1 batch: that is the silent corruption this whole
    change exists to remove.

    But an unbatched potential that has no work contributes nothing, so it is
    safe to skip: TemplateReferencePotential without templates, ContactPotentital
    without contact restraints and SymmetricChainCOMPotential without symmetric
    chains all return an empty index, which is exactly NISE's input shape (one
    protein, one ligand, no templates or constraints). Check every record, not
    just record 0 -- emptiness is per record. Otherwise raise, and the kit's own
    fallback re-runs the group one record per step: slow, but right.

    Returns True when the potential contributes nothing and may be skipped.
    """
    B = feats["atom_pad_mask"].shape[0]
    if B == 1 or getattr(type(self), "_potb_batched", False):
        return False

    key = id(feats)
    cached = getattr(self, "_inactive_for", None)
    if cached is not None and cached[0] == key:
        if cached[1]:
            return True
        raise NotImplementedError(cached[2])

    orig = type(self).compute_args
    empty = True
    for b in range(B):
        sub = {k: (v[b : b + 1] if torch.is_tensor(v) else v) for k, v in feats.items()}
        try:
            index = orig(self, sub, parameters)[0]
        except Exception:
            empty = False
            break
        if index is not None and index.shape[-1] != 0:
            empty = False
            break

    msg = (f"{type(self).__name__} is not batch-aware and has work (B={B}); "
           f"refusing to score records 1.. with record 0 topology")
    self._inactive_for = (key, empty, msg)
    if empty:
        return True
    raise NotImplementedError(msg)


_orig_compute = P.Potential.compute


def _compute(self, coords, feats, parameters):
    if _guard(self, feats, parameters):
        return torch.zeros(coords.shape[:-2], device=coords.device)
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
    if _guard(self, feats, parameters):
        return torch.zeros_like(coords)
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
            cls._potb_batched = True
    return [n for n in _PAIR_POTENTIALS if getattr(P, n, None) is not None]


# ------------------------------------------------------------- noise carrier ---
# Upstream's eager sample() draws its randomness from one stream over the whole
# leading axis, so with B>1 a record's trajectory would depend on its batch-mates.
# bz_sampler (the rollout lever) solves this by pre-drawing per record, but
# potentials drop that lever. Same idea here, for the eager path: one generator
# per record, every draw sliced per record. Seeds come from the batched worker
# loop, exactly as bz_sampler.BATCH["seeds"] does.
BATCH = {"seeds": None}
_ACTIVE = {"noise": None}


class _Noise:
    def __init__(self, B, device, seeds):
        self.B, self.device = B, device
        self.gens = []
        for b in range(B):
            g = torch.Generator(device=device)
            g.manual_seed(int(seeds[b]) % (2 ** 63 - 1))
            self.gens.append(g)

    def randn(self, shape, dtype=torch.float32):
        m = shape[0] // self.B
        out = torch.empty(shape, device=self.device, dtype=dtype)
        for b, g in enumerate(self.gens):
            out[b * m : (b + 1) * m] = torch.randn(
                (m, *shape[1:]), device=self.device, dtype=dtype, generator=g)
        return out

    def augment(self, n, s_trans, dtype):
        from boltz.model.modules.utils import _copysign, quaternion_to_matrix
        m = n // self.B
        Rs, trs = [], []
        for g in self.gens:
            o = torch.randn((m, 4), dtype=dtype, device=self.device, generator=g)
            s = (o * o).sum(1)
            o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
            Rs.append(quaternion_to_matrix(o))
            trs.append(torch.randn((m, 1, 3), dtype=dtype, device=self.device,
                                   generator=g) * s_trans)
        return torch.cat(Rs), torch.cat(trs)


def _begin(B, device):
    """Called at the top of each sample(); None => fall back to stock draws."""
    seeds = BATCH.get("seeds")
    if B == 1 and not seeds:
        _ACTIVE["noise"] = None
        return None
    if not seeds or len(seeds) != B:
        seeds = [int(x) for x in torch.randint(0, 2 ** 62, (B,), device=device).tolist()]
    n = _Noise(B, device, seeds)
    _ACTIVE["noise"] = n
    return n


def _augment(multiplicity, s_trans=1.0, device=None, dtype=torch.float32):
    n = _ACTIVE.get("noise")
    if n is None:
        return _ORIG_AUGMENT(multiplicity, s_trans=s_trans, device=device, dtype=dtype)
    return n.augment(multiplicity, s_trans, dtype)


_ORIG_AUGMENT = None


# ---------------------------------------------------------------- sampler ---
# AtomDiffusion.sample() sizes its steering buffers with `multiplicity`, which
# is the per-record sample count. The real leading axis is batch * multiplicity
# (atom_mask is repeat_interleave'd). At B=1 they coincide; at B>1 every one of
# these is short by a factor of B. Patched by source rewrite rather than a hand
# copy, so an upstream change to the body shows up as a failed substitution
# instead of silently stale code.
import inspect
import textwrap

_SAMPLE_SUBS = [
    ("energy_traj = torch.empty((multiplicity, 0), device=self.device)",
     "energy_traj = torch.empty((_Bsz * multiplicity, 0), device=self.device)"),
    ("resample_weights = torch.ones(multiplicity, device=self.device).reshape(",
     "resample_weights = torch.ones(_Bsz * multiplicity, device=self.device).reshape("),
    ("(multiplicity, *atom_mask.shape[1:], 3),",
     "(_Bsz * multiplicity, *atom_mask.shape[1:], 3),"),
    ("max_parallel_samples = multiplicity",
     "max_parallel_samples = _Bsz * multiplicity"),
    ("multiplicity, device=atom_coords.device, dtype=atom_coords.dtype",
     "_Bsz * multiplicity, device=atom_coords.device, dtype=atom_coords.dtype"),
    ("sample_ids = torch.arange(multiplicity).to(atom_coords_noisy.device)",
     "sample_ids = torch.arange(_Bsz * multiplicity).to(atom_coords_noisy.device)"),
    ("multiplicity % max_parallel_samples + 1",
     "(_Bsz * multiplicity) % max_parallel_samples + 1"),
    ("multiplicity=sample_ids_chunk.numel(),",
     "multiplicity=sample_ids_chunk.numel() // _Bsz,"),
    ("energy = torch.zeros(multiplicity, device=self.device)",
     "energy = torch.zeros(_Bsz * multiplicity, device=self.device)"),
    ("atom_coords = init_sigma * torch.randn(shape, device=self.device)",
     "atom_coords = init_sigma * (_nz.randn(shape) if _nz is not None else torch.randn(shape, device=self.device))"),
    ("eps = sqrt(noise_var) * torch.randn(shape, device=self.device)",
     "eps = sqrt(noise_var) * (_nz.randn(shape) if _nz is not None else torch.randn(shape, device=self.device))"),
]


def patch_sampler():
    from boltz.model.modules import diffusionv2 as D

    src = textwrap.dedent(inspect.getsource(D.AtomDiffusion.sample))
    missing = [a for a, _ in _SAMPLE_SUBS if a not in src]
    if missing:
        raise RuntimeError(f"sample() source changed; no match for: {missing}")
    for a, b in _SAMPLE_SUBS:
        src = src.replace(a, b)

    # capture the pre-expansion batch size before atom_mask is repeat_interleave'd
    anchor = "    if steering_args is not None and ("      # dedented to module level
    assert anchor in src, "sample() prologue changed"
    src = src.replace(
        anchor,
        "    _Bsz = atom_mask.shape[0]\n    _nz = _POTB_BEGIN(_Bsz, atom_mask.device)\n" + anchor,
        1)

    # share the module's LIVE globals, not a copy: a copy freezes the helpers at
    # patch time and the rewritten body then calls the unoptimised originals.
    global _ORIG_AUGMENT
    if _ORIG_AUGMENT is None:
        _ORIG_AUGMENT = D.compute_random_augmentation
    D.compute_random_augmentation = _augment     # per-record rotations / translations

    ns = D.__dict__
    ns["_POTB_BEGIN"] = _begin
    exec(compile(src, "<batched_sample>", "exec"), ns)
    D.AtomDiffusion.sample = ns.pop("sample")
    return len(_SAMPLE_SUBS)
