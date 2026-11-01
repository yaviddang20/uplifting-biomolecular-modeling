# Batching with `--use_potentials`

`BOLTZ_BATCH_SIZE` is honoured but vacuous under `--use_potentials`: the worker
marks every guided record `alone`, so each batch holds one record. This note
covers the work that lifts that, behind `BOLTZ_BATCH_POTENTIALS=1`.

**Status: off by default, and NOT correct across batch sizes.** The potential
math is exact and a record is unaffected by its batch-mates, but a record's
structure changes with batch size and fails the 1.0 A gate the unguided path
passes -- a pre-existing defect, see *Batch size changes a record's structure*.
Run end to end inside NISE and measured to 256 inputs. Not folded into the
image. Untested with affinity on, with MSAs, above ~200 tokens, or on >1 GPU.

Recommended setting: `BOLTZ_BATCH_SIZE=64` (see *Batch size*).

## Why it was off

Three separate things had to be true, and none were.

1. **The potentials read record 0 only.** Every `compute_args` indexes
   `feats[...][0]`, then `compute` broadcasts that one pair list across the whole
   leading axis. Hand it a batch and record 1 is scored with record 0's
   connectivity, reaching into its own padding atoms. Measured before the fix:
   record 1's VDW energy came out 10.588521 against a true 7.707700, with record
   0 exactly right -- so a spot-check passes and the run looks fine.
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

- **Blocks the records on the atom axis.** `compute_args` builds each record's
  topology from its own slice and offsets its atom indices by `b * A`
  (`A` = padded atoms per record); `_blockify` reshapes coords from
  `(L, A, 3)` record-major to `(m, B*A, 3)`. Every pair column then references
  exactly one record's atoms and upstream's own kernels run unchanged. Per-record
  totals come from `_segsum`, a contiguous slice sum per record.
- Rewrites `sample()` from source (not a hand copy, so an upstream change fails
  the substitution instead of going stale) for the nine `multiplicity` sites,
  and forces one sampler chunk per batch when `B > 1` -- a chunk that spans a
  record boundary mismatches records with their conditioning.
- Gives the eager sampler a per-record noise carrier: one generator per record,
  every draw sliced per record (init, per-step `eps`, and the rotation and
  translation inside `compute_random_augmentation`), seeded from the worker
  exactly as `bz_sampler.BATCH` does.

### Why the atom axis and not the pair axis

The first version concatenated the records along the **pair** axis instead, and
masked the reduction. Upstream's `compute_variable` pairs every leading row with
every index column, so that cost `(B*m, B*p)`: every sample met every record's
pairs, only the `B` diagonal blocks meant anything, and `B-1` of every `B`
blocks were computed and then masked away. Work and memory grew as `B**2`.

Measured, at 110-token inputs (~1000 atoms, ~501k pairs per record), that was
~1.54 G energy elements at `B=32` of which ~48 M were used -- 97% discarded --
plus a boolean mask of the same shape. Total GPU memory fit
`0.202*B + 0.00494*B**2` GiB, which predicted the observed OOM at `B=128`
(108.96 GiB). Per-record cost rose with `B`, so `bs=32` looked like an optimum
when it was only the crossover where the waste overtook the amortisation.

Blocking on the atom axis leaves `(m, B*p)`, needs no mask at all, and is linear
in `B`.

`_segsum` sums a contiguous slice per record rather than using `index_add_`.
Measured at `B=32`, `P=16 M`: `index_add_` took 60-71 ms against 0.445 ms, and
outside deterministic mode was wrong run to run (max |d| 5.9e-2). The slice sum
reduces exactly the columns upstream reduces, in the same shape, which is why
energies come out bitwise equal to the per-record call.

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

## Correctness

### Unit level

Per-record energies equal the per-record `B=1` call **bitwise**. Gradients are
bitwise equal under `POTB_DETERMINISTIC=1` and differ by 1.192e-07 (one float32
ULP) without it, from upstream's own `scatter_reduce` atomics -- which the
unbatched path has too. Padding gradients exactly zero. `B=3`, unequal atom
counts (14/9/11, so unequal pair counts), multiplicity 1, 2 and 3
(`pot_check3.py`).

### End to end

Identical 905-atom probe record, `bs=4`, `POTB_DETERMINISTIC=1`, Kabsch-aligned
as well as raw -- Boltz emits each structure in an arbitrary global frame, so a
raw coordinate diff alone conflates "different structure" with "different
orientation" (`run_det2.sh`, `cmp_aligned.py`).

| | raw max abs | aligned RMSD |
|---|---|---|
| same mates, twice | 0.000000 A | 0.000000 A |
| different batch-mates | 0.000000 A | 0.000000 A |
| `bs=1` | 33.516 A | 9.756 A |
| remainder group (`sizes [1, 4]`) | 28.918 A | 12.839 A |

So: exactly reproducible at fixed shape, and **no coupling between
batch-mates** -- the probe is unchanged when its companions change, which is the
property that matters for batching to be sound.

### Batch size changes a record's structure -- UNRESOLVED

A record's structure is **not** invariant to batch size, and guided batching
**fails** the project's own gate for this. Using `BATCHING.md`'s protocol --
same inputs, `--seed 0`, the batched path on both sides so the per-record noise
is held fixed, `b1` = every input alone via `BOLTZ_BATCH_MAX_PAD=1.0` -- on the
confidently predicted Top7 set (pLDDT ~0.85), gate 1.0 A (`run_top7_proper.sh`,
`cmp_sets.py`):

| comparison | mean | max | |
|---|---|---|---|
| unguided `b1` vs `b4` | 0.2078 A | 0.4139 A | PASS |
| unguided `b1` vs `b8` | 0.1880 A | 0.2802 A | PASS |
| **guided** `b1` vs `b4` | **3.9704 A** | 6.3643 A | **FAIL** |
| **guided** `b1` vs `b8` | **3.6264 A** | 5.6186 A | **FAIL** |

The unguided rows reproduce `BATCHING.md`'s published 0.21/0.41 and 0.19/0.28,
so the harness is calibrated against a known-good result.

**There is no variance to hide behind.** The guided path's own floor is
0.0000 A: the same configuration run twice is bitwise identical, with forced
determinism and without it (`run_floor.sh`). The 1.0 A gate therefore applies
here as much as to the unguided path.

**This is not caused by the block-diagonal reduction.** At a matched batch size
the pre-fix masked implementation and this one are bitwise identical -- raw max
0.000000 over 8 records (`cmp_two.py`, `bench_prefix/out_old_b4` vs
`bench_top7p/out_g_b4`). The defect predates both of this change's commits and
lives in the original guided-batching work. It became measurable only now,
because the pre-fix version *crashed* at `b1` (the group-of-one bug) and so
could never be compared against it.

Leading suspect, unverified: the per-record noise carrier draws
`randn((m, A, 3))` where `A` is the **padded** atom width, which depends on the
batch's composition -- so one record draws a different stream at a different
padding. That would also explain why a record is unaffected by batch-mates
whose lengths leave the padding unchanged (0.000000 A, above) while changing the
batch size moves it. The unguided sampler does not show the problem.

Note what is *not* implicated: energies and gradients are bitwise exact per
record (see *Unit level*), and at fixed batch size and padding a record is
unaffected by its batch-mates. The defect is in what the sampler is fed, not in
the potential math.

**Until this is resolved**, treat guided batching as a throughput optimisation
whose structures are not interchangeable with `b1`'s, and do not compare a
batched campaign against an unbatched one structure by structure. The NISE
quality table below was produced under this defect and should be read as "the
optimiser still works", not as evidence that batched and unbatched agree.

## Performance

256 fixed-length inputs (110 aa protein + apixaban, as NISE produces, so no
padding waste), one idle H200, `mode fast`, `--use_potentials`, exactly `bs`
inputs per arm so every arm is one batch (`run_ladder2.sh`).

| batch | sampler s/record | peak GiB | GiB/record |
|---|---|---|---|
| 16 | 1.975 | 4.84 | 0.303 |
| 32 | 1.652 | 7.56 | 0.236 |
| 48 | 1.539 | 10.26 | 0.214 |
| 64 | 1.460 | 12.97 | 0.203 |
| 96 | 1.471 | 18.36 | 0.191 |
| 128 | **1.421** | 23.77 | 0.186 |
| 192 | 1.445 | 34.57 | 0.180 |
| 256 | 1.432 | 45.37 | 0.177 |

No OOM, no fallbacks at any rung.

**Memory is linear in `B`.** Fitting `bs=32` and `bs=96` alone gives
`2.16 + 0.169*B` GiB, which predicts every other rung to within 0.02 GiB
(`bs=256`: 45.36 predicted, 45.37 measured). The `B**2` term is gone. Against
the masked version at matched batch size: 11.52 -> 7.56 GiB at 32,
64.93 -> 18.36 at 96, and `bs=128` now runs at 23.77 GiB where it used to OOM.

**Cost per record falls to `bs=64` and then plateaus.** Marginal cost per added
record is flat from there -- 1.495, 1.272, 1.494, 1.391 s across the 64->96->128
->192->256 steps -- i.e. total sampler time is linear (`2.4 + 1.423*B` s, fitting
`bs=64` and `bs=256`, predicts the middle rungs to ~2 s). Below 64 the GPU is
occupancy-limited and adding records fills idle hardware; at and above 64 it is
saturated and more records simply queue more work. That plateau is the ceiling
of what batching can buy.

The masked version's marginal cost *rose* (1.83, 2.28, 2.71, 3.59 s) because
each added record dragged in `B`x more discarded pair work. A flat marginal
curve is what the fix looks like in the timing data.

Caveat on the old/new timing comparison: the two ladders ran on different (both
H200, both effectively idle) nodes, so treat the time deltas as indicative. The
memory result does not depend on which node ran it.

### Batch size

Use **64**. It is within 2.7% of the best measured point at half the memory
(12.97 vs 23.77 GiB). These inputs are all 110 tokens and pair memory grows
roughly with the square of the atom count, so the ceiling drops for larger
targets and 64 leaves real headroom; hitting the OOM fallback costs far more
than 2.7%. NISE's default 192 records per call also divides into exactly three
groups of 64.

The batch group is capped by the records actually fed to one `boltz predict`
call, so `bs=64` buys nothing when a call has fewer than 64 records -- it is
then on the rising part of the curve whatever `BOLTZ_BATCH_SIZE` says.

### Where the rest of the wall time goes

At `bs=64`, wall 149.0 s:

| | |
|---|---|
| sampler | 93.45 s (63%) |
| trunk | 6.68 s |
| confidence / conditioning | 0.66 s |
| outside the forward pass | 48.2 s (32%) |

The last row is ~21 s of Python start / torch import / CUDA init, ~5.3 s model
build, and featurisation; across rungs it fits ~26 s fixed per `boltz predict`
call plus ~0.36 s/record. Writing is 2.28 s and overlapped.

That fixed 26 s is why wall speedup trails inference speedup, and it is paid
**per iteration** in NISE, which spawns a fresh `boltz predict` per iteration
(~15 min over a 35-iteration campaign). Removing it needs a persistent worker,
which is unrelated to batching.

## NISE

End to end, 2 iterations, 16 then 32 sequences, `bs=16`, one H200
(`run_nise_smoke.sh`, `run_nise_base.sh`). `alone=0`, zero fallbacks.

| | batched | unbatched |
|---|---|---|
| sampler, total | 104.40 s | 929.59 s |
| sampler, per record | 2.175 s | 19.366 s |
| wall | 229 s | 1027 s |
| PDBs | 48 | 48 |

**8.90x on inference, 4.49x on wall** at `bs=16`. At the `bs=64` rate of 1.460
s/record that is 13.3x on inference.

Quality: both campaigns optimise in the same direction and land in the same
place (final mean ligand RMSD 0.691 batched vs 0.702 unbatched; ligand pLDDT
0.975 vs 0.959). The arms are not expected to match run for run -- LASErMPNN's
sequence sampling is unseeded, so the two runs get different starting
populations, and Boltz is batch-size sensitive as above. With one target and two
iterations the per-iteration gaps carry no signal; the claim is only that nothing
degrades.

### Defaults worth knowing

`num_iterations = 35`, `num_top_backbones_per_round = 3`,
`sequences_sampled_per_backbone = 64` -> **192 sequences per iteration**, written
to one directory and predicted in **one** `boltz predict` call per iteration
(35 calls, 6720 records per campaign). `sequences_sampled_at_once = 30` is a
LASErMPNN sub-batch, not a Boltz chunk; the `chunk_N` in input names is the
*device* index. `use_boltz_conformer_potentials = True`, so `--use_potentials` is
always on.

`boltz2_predict_affinity` is True only when the objective contains `pbind`. It
forces `alone` independently of this patch, so a `pbind` campaign gets **no
batching at all**. That path is untested.

## Open

- `id(feats)` as a cache key: safe here only because `feats` outlives the
  prediction. Needs a weakref before this is trusted.
- Not folded into the `.sif`; runs as a bind-mounted monkeypatch via
  `BOLTZ_KIT_CODE`.
- The `compute_args` memoisation was measured at 1.09x, but that was *before*
  the `B**2` fix, when the energy tensor dominated. Its share is larger now and
  it has not been re-measured (`POTB_NO_CACHE=1` switches it off).
- Untested: affinity on, MSAs, >200 tokens, multi-GPU, the OOM fallback path.
- The three unbatched potentials are skipped, not batched.
- Determinism was verified at `bs=4`; not re-verified at 64.
- **The batch-size defect above is open, and blocks using this where structures
  must match an unbatched run.** The noise-carrier padding hypothesis is
  untested.

## Files

`opt/boltz2_opt/potentials_batch/`: `patch.py` (the patch); `pot_check.py`,
`pot_check2.py`, `pot_check3.py` (per-record energy and gradient equality);
`run_ladder.sh`, `run_ladder2.sh` (batch-size ladders); `run_det2.sh` +
`cmp_probe.py`, `cmp_aligned.py` (reproducibility, coupling, batch-size
invariance); `run_bsinv.sh`, `run_unguided_bs.sh` (batch-size sensitivity, and
the unguided control); `run_leak.sh` + `cmp_quality.py` (cross-record coupling);
`run_nise_smoke.sh`, `run_nise_base.sh` (NISE end to end); `list_pots.py` (what
`get_potentials` actually returns).
