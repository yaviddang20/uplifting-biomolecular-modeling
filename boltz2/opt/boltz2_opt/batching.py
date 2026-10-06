"""Batched structure inference for the worker route (`fast` only): several DIFFERENT inputs per predict step, padded to one shape.

Not upstream behaviour: stock Boltz 2.2.1 predicts one record per step (inferencev2 ``batch_size=1``; ``AtomDiffusion.sample``, the writer and
the confidence module at ``diffusion_samples > 1`` are single-record bodies). ``BOLTZ_BATCH_SIZE=N`` (N > 1) opts a ``pred --mode fast`` run in.
The model's trunk is batch-capable as trained (padding masks at every site), and the row's kernels take a leading batch; what this module and
the batched worker loop (bz_worker_levf2.py) add around them:

* **Buckets** (:func:`plan`): the pass's inputs sorted by token count and cut into batches of at most N whose longest / shortest token-count
  ratio stays under ``BOLTZ_BATCH_MAX_PAD`` (1.25) — padding is paid in O(N^2)-O(N^3) pair work, and the CUDA-graph levers key their captures
  by shape. An input declaring an affinity property is a batch of one (upstream's affinity leg is single-record).
* **Features** (:class:`Records`, :func:`collate_group`): stock's PredictionDataset per record, stock's ``collate`` (``pad_to_max``) for the
  batch — and each record's own UNPADDED features beside it (``singles``).
* **Per-record randomness**: the featurizer's torch stream and the sampler's noise (bz_sampler ``BATCH["seeds"]``) are seeded per record from
  (the run's seed, the record id) — :func:`record_seed`. A record's prediction then does not depend on its batch-mates, its slot, or the batch
  size, up to floating-point rounding (padding moves kernel tile boundaries). It is NOT the stream a non-batched run draws: batched and
  non-batched results of one input differ as two seeds do.
* **Confidence per record, unpadded** (:func:`install`): the confidence module runs once per record on that record's own features and the
  cropped trunk outputs — stock's single-record statements (``assert z.shape[0] == 1`` at ``diffusion_samples > 1``; pad tokens sit at the
  origin in the contact mask otherwise).
* **Outputs per record** (:func:`record_prediction`): the prediction dict stock's writer takes for a single record — that record's samples,
  its unpadded masks, its own confidence outputs and ranking — so every file is written by stock's single-record writer body.

Refused by name, never served wrong: ``--use_potentials`` and inputs with forced contact / pocket guidance (upstream's per-step steering is
single-record) run as batches of one on the non-batched statements.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Optional, Sequence

ENV_SIZE = "BOLTZ_BATCH_SIZE"            # records per predict step; unset / 1 = upstream's one record per step (nothing of this module runs)
ENV_MAX_PAD = "BOLTZ_BATCH_MAX_PAD"      # longest / shortest token count allowed inside one batch (default 1.25)
DEFAULT_MAX_PAD = 1.25
MODES = ("fast",)                        # the rows whose levers are shown batch-capable (the sampler roll-out and the fused DiT step carry B records)

CTX: Dict[str, Any] = {"singles": None, "conf": None, "busy": False, "installed": False, "calls": 0, "records": 0}


def size(env: Optional[dict] = None) -> int:
    """The requested batch size (``BOLTZ_BATCH_SIZE``); 1 when unset. ValueError names a value that is not an integer >= 1."""
    v = (os.environ if env is None else env).get(ENV_SIZE)
    if v is None or str(v).strip() == "":
        return 1
    try:
        n = int(str(v).strip())
    except ValueError:
        raise ValueError(f"{ENV_SIZE}={v!r} is not an integer") from None
    if n < 1:
        raise ValueError(f"{ENV_SIZE}={n} is below 1")
    return n


def max_pad(env: Optional[dict] = None) -> float:
    v = (os.environ if env is None else env).get(ENV_MAX_PAD)
    if v is None or str(v).strip() == "":
        return DEFAULT_MAX_PAD
    r = float(v)
    if r < 1.0:
        raise ValueError(f"{ENV_MAX_PAD}={v!r} is below 1.0")
    return r


def record_seed(seed: int, record_id: str, what: str = "noise") -> int:
    """A 62-bit seed that is a function of (run seed, record id, stream name) only."""
    h = hashlib.sha256(f"{int(seed)}|{record_id}|{what}".encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 62) - 1)


def n_tokens_hint(record) -> int:
    """The bucket key of a record: its residue count over valid chains (a ligand chain counts its atoms upstream; exact enough to sort by)."""
    return int(sum(int(getattr(c, "num_residues", 0) or 0) for c in getattr(record, "chains", []) if getattr(c, "valid", True)))


def plan(sizes: Sequence[int], batch: int, ratio: float = DEFAULT_MAX_PAD, alone: Optional[Sequence[bool]] = None) -> List[List[int]]:
    """Index groups over `sizes`: sorted by size (ties by index), cut greedily into groups of at most `batch` whose max / min size stays
    <= `ratio`; an index flagged `alone` is its own group. Deterministic; every index appears exactly once."""
    alone = list(alone) if alone is not None else [False] * len(sizes)
    order = sorted((i for i in range(len(sizes)) if not alone[i]), key=lambda i: (sizes[i], i))
    groups: List[List[int]] = []
    cur: List[int] = []
    for i in order:
        if cur and (len(cur) >= batch or sizes[i] > max(1, sizes[cur[0]]) * ratio):
            groups.append(cur); cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    groups += [[i] for i in range(len(sizes)) if alone[i]]
    return groups


class Records:
    """A map-style dataset over the pass's records, each with its own processed directories (the worker parses one input per directory):
    ONE stock PredictionDataset whose record / directories are switched per index, so stock's own ``__getitem__`` featurizes every record.
    The process's torch / numpy / python streams are seeded per record first (featurization draws from them: the reference conformers'
    augmentation), so a record's features do not depend on which DataLoader worker or batch it lands in."""

    def __init__(self, units: List[dict], mol_dir, seed: int, override_method=None):
        self.units = units; self.mol_dir = mol_dir; self.seed = int(seed); self.override_method = override_method; self._base = None

    def __len__(self) -> int:
        return len(self.units)

    def _dataset(self, u):
        from boltz.data.module.inferencev2 import PredictionDataset
        if self._base is None:
            self._base = PredictionDataset(manifest=u["manifest"], target_dir=u["targets_dir"], msa_dir=u["msa_dir"], mol_dir=self.mol_dir,
                                           constraints_dir=u["constraints_dir"], template_dir=u["template_dir"], extra_mols_dir=u["extra_mols_dir"],
                                           override_method=self.override_method)
        b = self._base
        b.manifest = u["manifest"]; b.target_dir = u["targets_dir"]; b.msa_dir = u["msa_dir"]; b.constraints_dir = u["constraints_dir"]
        b.template_dir = u["template_dir"]; b.extra_mols_dir = u["extra_mols_dir"]
        return b

    def __getitem__(self, i: int) -> dict:
        import random
        import numpy as np
        import torch
        u = self.units[i]
        s = record_seed(self.seed, u["name"], "features")
        torch.manual_seed(s); np.random.seed(s % (2 ** 32)); random.seed(s)
        feats = self._dataset(u)[0]
        if feats["record"].id != u["name"]:                       # stock's __getitem__ answers a failed record with record 0 of its manifest: here that is the same record or a defect
            raise RuntimeError(f"featurizing {u['name']} returned record {feats['record'].id}")
        return feats


def collate_group(data: List[dict]) -> dict:
    """The DataLoader's collate: stock's padded batch, and every record's own single-record batch beside it."""
    from boltz.data.module.inferencev2 import collate
    return {"batch": collate(data), "singles": [collate([d]) for d in data]}


NO_DEVICE = ("all_coords", "all_resolved_mask", "crop_to_all_atom_map", "chain_symmetries", "amino_acids_symmetries", "ligand_symmetries", "record", "affinity_mw")


def to_device(batch: dict, device) -> dict:
    """Boltz2InferenceDataModule.transfer_batch_to_device's statements (the same key list)."""
    for key in batch:
        if key not in NO_DEVICE:
            batch[key] = batch[key].to(device)
    return batch


def _pad_cat(tensors):
    """Row-concatenate tensors whose trailing dims differ (per-record token counts): zero-padded to the largest."""
    import torch
    import torch.nn.functional as F
    if all(t.shape[1:] == tensors[0].shape[1:] for t in tensors):
        return torch.cat(tensors, dim=0)
    nd = tensors[0].dim()
    mx = [max(int(t.shape[d]) for t in tensors) for d in range(nd)]
    out = []
    for t in tensors:
        pad = []
        for d in range(nd - 1, 0, -1):
            pad += [0, mx[d] - int(t.shape[d])]
        out.append(F.pad(t, pad))
    return torch.cat(out, dim=0)


def _merge_conf(outs: List[dict]) -> dict:
    """One dict for Boltz2.forward / predict_step from the per-record confidence outputs: rows record-major, token dims zero-padded, the
    chain-pair table over the union of chain indices (zeros where a record has no such chain). The exact per-record outputs stay in CTX["conf"]."""
    import torch
    merged = {}
    for key in outs[0]:
        if key != "pair_chains_iptm":
            merged[key] = _pad_cat([o[key] for o in outs]) if torch.is_tensor(outs[0][key]) else outs[0][key]
            continue
        ref = next(iter(next(iter(outs[0][key].values())).values()))
        idx = sorted({i for o in outs for i in o[key]})
        merged[key] = {i: {j: torch.cat([(o[key][i][j] if i in o[key] and j in o[key][i] else torch.zeros_like(ref)) for o in outs], dim=0) for j in idx} for i in idx}
    return merged


def install() -> bool:
    """Wrap ConfidenceModule.forward once (the batched worker loop calls this): with a batch's ``singles`` in CTX, the module runs once per
    record on that record's own unpadded features and the cropped trunk outputs; any other call is the wrapped forward unchanged."""
    if CTX["installed"]:
        return False
    from boltz.model.modules.confidencev2 import ConfidenceModule
    inner = ConfidenceModule.forward

    def forward(self, s_inputs, s, z, x_pred, feats, pred_distogram_logits, multiplicity=1, run_sequentially=False, use_kernels=False):
        singles = CTX["singles"]
        if singles is None or CTX["busy"] or int(z.shape[0]) != len(singles):
            return inner(self, s_inputs, s, z, x_pred, feats, pred_distogram_logits, multiplicity=multiplicity, run_sequentially=run_sequentially, use_kernels=use_kernels)
        m = int(multiplicity)
        if x_pred.dim() == 4:
            x_pred = x_pred.reshape(-1, x_pred.shape[-2], 3)
        outs = []
        CTX["busy"] = True
        try:
            for b, fb in enumerate(singles):
                nb = int(fb["token_pad_mask"].shape[1]); ab = int(fb["atom_pad_mask"].shape[1])
                outs.append(inner(self, s_inputs[b:b + 1, :nb], s[b:b + 1, :nb], z[b:b + 1, :nb, :nb], x_pred[b * m:(b + 1) * m, :ab], fb,
                                  pred_distogram_logits[b:b + 1, :nb, :nb], multiplicity=m, run_sequentially=run_sequentially, use_kernels=use_kernels))
        finally:
            CTX["busy"] = False
        CTX["conf"] = outs; CTX["calls"] += 1; CTX["records"] += len(outs)
        return _merge_conf(outs)

    forward._bz_batched = True
    ConfidenceModule.forward = forward
    CTX["installed"] = True
    return True


CONF_KEYS = ("pde", "plddt", "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde")
PAE_KEYS = ("pae", "ptm", "iptm", "ligand_iptm", "protein_iptm", "pair_chains_iptm")


def record_prediction(pred: dict, b: int, single: dict, multiplicity: int, conf: Optional[dict]) -> dict:
    """Record b's prediction as stock's predict_step would return it for that record alone (boltz2.py predict_step's keys): its m samples
    cropped to its atoms, its own masks, its trunk embeddings cropped to its tokens, its own confidence outputs and — the ranking stock's
    writer sorts by — its confidence_score from predict_step's own expression."""
    import torch
    m = int(multiplicity)
    nb = int(single["token_pad_mask"].shape[1]); ab = int(single["atom_pad_mask"].shape[1])
    out = {"exception": False, "masks": single["atom_pad_mask"], "token_masks": single["token_pad_mask"],
           "coords": pred["coords"][b * m:(b + 1) * m, :ab], "s": pred["s"][b:b + 1, :nb], "z": pred["z"][b:b + 1, :nb, :nb]}
    if conf is not None:
        for k in CONF_KEYS:
            out[k] = conf[k]
        iptm, ptm = conf.get("iptm"), conf.get("ptm")
        if iptm is not None:
            out["confidence_score"] = (4 * conf["complex_plddt"] + (iptm if not torch.allclose(iptm, torch.zeros_like(iptm)) else ptm)) / 5
            if "pae" in pred:
                for k in PAE_KEYS:
                    out[k] = conf[k]
    return out


def describe() -> dict:
    return {"batch_size": size(), "max_pad": max_pad(), "confidence_calls": CTX["calls"], "confidence_records": CTX["records"], "installed": CTX["installed"]}
