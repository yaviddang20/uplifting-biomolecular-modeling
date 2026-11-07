# Batching with `--use_potentials`

`BOLTZ_BATCH_SIZE` is honoured but vacuous under `--use_potentials`: the worker
marks every guided record `alone`, so each batch holds one record. This note
covers the work that lifts that, behind `BOLTZ_BATCH_POTENTIALS=1`.

**Status: opt-in** (`BOLTZ_BATCH_POTENTIALS=1` enables it). It was briefly on by
default; that broke every non-guided run, because the sampler levers own
`AtomDiffusion.sample` on those routes and the source rewrite then matches
nothing. A failed rewrite now costs the speedup, not the run. It does
nothing unless a run asks for batching, since `BOLTZ_BATCH_SIZE` is 1 when
unset; at one record per step the patch moves a structure by 0.018 A aligned (a
0.27 A centring shift, which superposition removes), so a non-batched run is
unaffected in any way that matters. The potential math is exact, a record is
unaffected by its batch-mates, and a fixed batch size is exactly reproducible. A record's
structure does change with the batch size -- a different valid sample of equal
confidence, from amplified floating-point rounding, and the unguided path has
the same property to a lesser degree; see *Batch size changes which sample you
get*. Run end to end inside NISE and measured to 256 inputs. Not folded into the
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

### Batch size changes which sample you get

A record's structure depends on the batch size. It does **not** depend on which
records share the batch, and the potential math is exact; what changes is the
sample you land on.

First, a distinction that matters more than it sounds. `BOLTZ_BATCH_SIZE=1` is
not "batching with one record" -- it leaves the batched planner entirely (no
`BATCH plan` line is printed) and draws a different noise stream, which
`BATCHING.md` states outright: a batched and a non-batched run of the same input
at the same `--seed` "differ the way two seeds do". Measured on 8 equal-length
variants of the Top7 fold, `bs=1` vs `bs=4` is 4.92 A guided and **4.54 A
unguided** -- the shipped path misses it by just as much. So matching
`BOLTZ_BATCH_SIZE=1` is not a thing either path does, by design. `BATCHING.md`'s
harness fakes `b1` with `BOLTZ_BATCH_MAX_PAD=1.0` precisely to stay on the
batched path.

The answerable question is batch size N against batch size M on the batched
path. With padding held identical (8 variants of one fold, all 104 residues,
pLDDT ~0.85, `run_eq2.sh`):

| | `b4` vs `b8` |
|---|---|
| guided | 2.02 A |
| unguided | 0.28 A |

and on mixed lengths with `b1` faked the way their harness does it
(`run_top7_proper.sh`): guided 2.87 A, unguided 0.21 A, against a 1.0 A gate.

**Why.** Changing the batch size changes tensor shapes, so cuBLAS/Triton tile
and reduce differently, and float addition is not associative -- the same sum
computed in a different order differs in the last bits (~1e-7). Diffusion feeds
each step into the next 200 times, and guidance adds 20 gradient corrections per
step, so ~4000 feedback applications amplify that where the unguided path's ~200
do not. The ratio of the two numbers above is that extra amplification.
`torch.use_deterministic_algorithms(True)` does not help: it makes a *given*
shape reproducible, which is why a fixed batch size is exact (0.0000 A), not
different shapes agree.

**Is the quality worse? No, within the power of the test.** Paired
`complex_plddt` over the same 8 records (`cmp_qual.py`, `qual_stat.py`):

| | b1 -> b4 | b1 -> b8 | b4 -> b8 |
|---|---|---|---|
| guided | -0.0058 | -0.0117 (t=-3.50) | -0.0059 |
| unguided | -0.0044 | -0.0065 | -0.0020 |

At most 0.012 on a 0.85 baseline. One of six comparisons reaches p<0.05, which
is about what six tests buy you by chance, and it only sits at the boundary once
corrected. All six deltas are negative, which is itself worth noting (p ~ 0.016
for 6/6 by sign), so a small systematic cost to larger batches may be real --
but **unguided drifts the same way at the same magnitude**, so it would belong
to batching generally, not to guided batching. n=8, one fold, one seed:
underpowered. Settling it needs 30-50 records over a couple of folds.

**Two real bugs were found and fixed here**, both pre-existing (the pre-fix
implementation is bitwise identical to this one at a matched batch size, raw max
0.000000 -- `cmp_two.py`), and both invisible until the group-of-one crash
(054903d) made a `b1` comparison runnable at all:

- The per-record noise was drawn at the **padded** atom width, so a record's
  random stream changed with its batch-mates' lengths. Now drawn at the record's
  own atom count with padding rows on a separate generator. Verified: record 0's
  noise is bitwise identical at `b1` and `b4` (`run_diverge.sh`).
- Centring used upstream's **unmasked** mean over every atom slot, so the
  molecule was translated by a padding-dependent amount. Now a masked mean over
  real atoms -- the convention `BATCHING.md` sets for batched samplers, which
  lives in the `rollout` lever that `--use_potentials` drops by name. It
  measured ~0 on these near-uniform sets and would matter on uneven batches.

Two further hypotheses were tested and rejected: forcing one sampler chunk at
`B=1` moved the result 0.0014 A (reverted, it overrode a user's
`max_parallel_samples` for nothing), and the residual is not in the potential
math, which is bitwise exact per record.

**What to rely on.** At a fixed batch size, guided batching is exactly
reproducible (0.0000 A) and unaffected by batch-mates (0.0000 A), and energies
and gradients match the per-record call bitwise. Across batch sizes you get a
different valid sample of equal confidence. Do not diff a batched campaign
against an unbatched one coordinate by coordinate; compare distributions.
`POTB_DUMP=<dir>` dumps record 0's noise and the coords entering the guidance
loop, which is how the two bugs above were localised.

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
- Whether the small pLDDT drift with batch size is real is unsettled (~0.01,
  same direction in the unguided path); 30-50 records over a couple of folds
  would resolve it.
- The `POTB_DUMP` instrumentation is diagnostic and stays in; it is env-gated
  and inert unless set.

## Files

`opt/boltz2_opt/potentials_batch/`: `patch.py` (the patch); `pot_check.py`,
`pot_check2.py`, `pot_check3.py` (per-record energy and gradient equality);
`run_ladder.sh`, `run_ladder2.sh` (batch-size ladders); `run_det2.sh` +
`cmp_probe.py`, `cmp_aligned.py` (reproducibility, coupling, batch-size
invariance); `run_bsinv.sh`, `run_unguided_bs.sh` (batch-size sensitivity, and
the unguided control); `run_leak.sh` + `cmp_quality.py` (cross-record coupling);
`run_nise_smoke.sh`, `run_nise_base.sh` (NISE end to end); `list_pots.py` (what
`get_potentials` actually returns).
