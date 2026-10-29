# Batching with `--use_potentials`

`BOLTZ_BATCH_SIZE` is honoured but vacuous under `--use_potentials`: the worker
marks every guided record `alone`, so each batch holds one record. This note
covers the work that lifts that, behind `BOLTZ_BATCH_POTENTIALS=1`.

**Status: off by default.** Correct and measured on the inputs below; not yet
folded into the image, never run inside NISE, untested above 8 inputs.

## Why it was off

Three separate things had to be true, and none were.

1. **The potentials read record 0 only.** Every `compute_args` indexes
   `feats[...][0]`, then `compute` broadcasts that one pair list across the whole
   leading axis. Hand it a batch and record 1 is scored with record 0's
   connectivity, reaching into its own padding atoms. Measured before the fix:
   record 1's VDW energy came out 10.588521 against a true 7.707700, with record
   0 exactly right — so a spot-check passes and the run looks fine.
2. **The sampler sizes its steering buffers with `multiplicity`**, the per-record
   sample count, while the real leading axis is `batch * multiplicity`
   (`atom_mask` is expanded by `repeat_interleave`). Nine sites, short by a
   factor of B. These crash rather than corrupt.
3. **The multi-record carrier lived inside the `rollout` lever**, which
   potentials drop by name, so `SMOD` was None and the plan collapsed to batches
   of one however correct the potentials were.

What was *not* a problem: FK steering. Resampling is confined to rows of
`reshape(-1, num_particles)` with a row-offset index, so with the record-major
layout it cannot mix records. That was the part worth worrying about and it
needed nothing.

## What the patch does

`potentials_batch/patch.py`, applied in `bz_worker_levf2.py` before the kit
wraps `AtomDiffusion.sample`:

- Builds each record's topology from its own slice, tags every pair with its
  record, and masks the energy and `dEnergy` reductions so a record sums only
  its own pairs. Atom indices need no offset: feats are already padded to a
  common width.
- Rewrites `sample()` from source (not a hand copy, so an upstream change fails
  the substitution instead of going stale) for the nine `multiplicity` sites.
- Gives the eager sampler a per-record noise carrier: one generator per record,
  every draw sliced per record (init, per-step `eps`, and the rotation and
  translation inside `compute_random_augmentation`), seeded from the worker
  exactly as `bz_sampler.BATCH` does.

## The other three potentials

`get_potentials` returns nine. Six are rewritten. `SymmetricChainCOMPotential`
(`com_args`), `ContactPotentital` (`union_index`) and
`TemplateReferencePotential` (`ref_args`) are not: each remaps or groups the
atom axis with indices that would collide across records -- COM slots merging,
per-record atom subsets of differing size, union ids merging.

They are *skipped when empty*, checked per record, and the run **raises**
otherwise so the kit falls back to one record per step. So batching is
available for single-chain, template-free, restraint-free inputs, and silently
unavailable (correct, slower, logged) for anything else. A homodimer activates
`SymmetricChainCOMPotential` and loses batching -- that is the likeliest way to
hit this.

NISE never hits it. Its only yaml writer, in both variants, emits one protein
chain, one ligand, `msa: empty`, and no templates or constraints. All three
measured empty on real inputs, zero fallbacks.

## Measurements

8 inputs (4x ~110 aa protein + cholesterol, 4x 199-token complex), one H200
shared with other jobs, `--use_potentials`, `mode fast`, `BOLTZ_BATCH_SIZE=4`
(planned as 2 batches of 4: the two shape buckets).

| | wall | sampler |
|---|---|---|
| batch 1 | 690.8 s | 653.5 s |
| batch 4 | 95.0 / 86.3 s | 61.7 / 53.9 s |

**7.3x.** The `compute_args` memoisation is worth 1.09x alone (632.3 s): it is
bypassed at B=1 and only removes this patch's own per-record overhead.

### Correctness

Per-record energies equal the per-record B=1 runs **exactly**; gradients to
1.2e-07; padding gradients exactly zero. B=2 and B=3, unequal sizes, at
multiplicity 1, 2 and 3 (`pot_check.py`, `pot_check2.py`).

Coordinates cannot be compared that way, because **the guided sampler is not
reproducible**: stock, unbatched, the same input twice differs by 1.995 A RMSD.
`scatter_reduce(..., "sum")` in the guidance loop uses atomics, run 20 times per
denoising step over ~200 steps. Against that floor:

| | RMSD |
|---|---|
| stock unbatched, same input twice | 1.995 A |
| batched, same set twice | 0.160 A |
| batched, probe record with different batch-mates | 0.113 A |

A record moves less when its batch-mates change than when nothing changes, so
no cross-record coupling is detectable (`run_leak.sh`). The batched path is
incidentally ~12x more reproducible than stock, because the per-record
generators pin the noise. Mean `complex_plddt` 0.3968 batched vs 0.3974
unbatched -- though at pLDDT 0.40 these inputs are undetermined and that
comparison is weak.

## Open

- `id(feats)` as a cache key: safe here only because `feats` outlives the
  prediction. Needs a weakref before this is trusted.
- Not folded into the `.sif`; runs as a bind-mounted monkeypatch via
  `BOLTZ_KIT_CODE`.
- Never run inside NISE. Untested at 192 inputs, with MSAs, above ~200 tokens,
  or through the OOM fallback.
- `predict_affinity` forces `alone` independently -- upstream's affinity leg is
  single-record -- so a campaign with affinity on gets no batching regardless.
- The three unbatched potentials are skipped, not batched.

## Files

`opt/boltz2_opt/potentials_batch/patch.py` (the patch), `pot_check.py` /
`pot_check2.py` (per-record energy and gradient equality), `run_potbatch.sh`
(timing), `run_leak.sh` + `cmp_quality.py` (cross-record coupling), and
`list_pots.py` (what `get_potentials` actually returns).
