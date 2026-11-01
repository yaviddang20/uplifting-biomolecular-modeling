# Batched inference (experimental, this fork only)

Stock Boltz-2 predicts **one input per GPU step**. This fork adds an opt-in mode that predicts **several different inputs per step**,
padded to one shape, with every `fast` optimization still on. It is not upstream behaviour and not part of Anthropic's release.

**Status:** runs end to end and matches non-batched predictions on a confidently predicted protein (numbers below). **Not yet tested
with MSAs, ligands, templates, `--diffusion_samples > 1`, large inputs, or the out-of-memory fallback.** Do not rely on it for
production results until it has been validated on inputs like yours (`tools/boltz2_batching/validate.sh`).

## Turn it on

Batching is controlled by environment variables. It is served on `--mode fast`, one GPU PER PROCESS — several GPUs are used by running
one process each (see "Several GPUs" below; `tools/boltz_compat/boltz --devices N` does that for you). Another mode, or the kit's own
model-parallel `--n_gpu > 1`, refuses by name (exit 2).

```bash
BOLTZ_BATCH_SIZE=4 bash run.sh pred --config h100 --mode fast --input <dir of yamls> --out_dir out/x
```

| Variable | Default | Meaning |
|---|---|---|
| `BOLTZ_BATCH_SIZE` | unset (= 1) | maximum inputs per predict step. Unset or `1` runs upstream's one-input-per-step loop, unchanged |
| `BOLTZ_BATCH_MAX_PAD` | `1.25` | the longest / shortest token count allowed in one batch. Inputs are sorted by length and grouped greedily, so a batch can hold fewer than `BOLTZ_BATCH_SIZE` inputs |

A batched run prints `[boltz2-opt] BATCH batch_size=… max_pad=…`, then a `BATCH plan:` line in the worker log
(`<out_dir>/pred_worker.log`) with the batch sizes, and `item_done … batch=<id> batch_n=<n>` per input.

From an image built from `main` (no batching code inside), overlay this branch's kit code at run time; no rebuild is needed:

```bash
BOLTZ_BATCH_SIZE=4 BOLTZ_CACHE=/weights/boltz2 apptainer run --nv \
  --bind <weights>/boltz2:/weights/boltz2 --bind <branch checkout>/boltz2:/kit/boltz2 --bind <branch checkout>/common:/kit/common \
  --bind <your data dir> boltz2.sif pred --config h100 --mode fast --input <dir> --out_dir <out>
```

### Several GPUs

Batching and multiple GPUs compose: each process takes one GPU and batches its own share of the inputs. P GPUs x batch N = P*N inputs in
flight. `tools/boltz_compat/boltz` does the splitting itself — this is one process per GPU, four batches of 8, 32 inputs at once:

```bash
BOLTZ_BATCH_SIZE=8 tools/boltz_compat/boltz predict <input dir> --out_dir <out> --devices 4
```

By hand, without the wrapper, the same thing is a shard per GPU:

```bash
for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g BOLTZ_BATCH_SIZE=8 <pred command> --input shard_$g/ --out_dir out/shard_$g & done; wait
```

Because every input draws its own noise (below), its result does not depend on which GPU or shard it lands on — P GPUs give the same
answers as one.

What does NOT combine with batching is the kit's own `--n_gpu P`, which shards ONE large prediction's pair representation across GPUs
(model-parallel, for inputs too big for one card, `big` mode). That is a different axis and refuses with batching by name.

## What changes

| Piece | Change |
|---|---|
| Buckets (`opt/boltz2_opt/batching.py`) | inputs sorted by token count, cut into batches within `BOLTZ_BATCH_MAX_PAD`; an input with the affinity property, contact / pocket guidance or `--use_potentials` runs alone |
| Features | stock's `PredictionDataset` per input and stock's `collate` (zero padding + masks) for the batch; each input's own unpadded features are kept beside it. The persistent featurizer (`prefetch`) leaves the row by name (`LEVER name=prefetch state=off reason=batch_size`); the batched loop's own DataLoader workers featurize ahead |
| Sampler roll-out (`opt/forward/sampler/src/bz_sampler.py`) | B inputs × m samples per step; per-input noise; record-major sample chunks; centring over real atoms only |
| Fused DiT step (`opt/forward/sampler/src/bz_sampler_dit.py`) | one pair-bias pack per input (its own key mask folded in), the existing kernel launched once per input; no kernel source change |
| Confidence module | run once per input on that input's unpadded features and cropped trunk outputs (stock's single-record statements, so padding cannot reach pLDDT / ipLDDT / PAE) |
| Writer | stock's single-record writer once per input, with that input's samples, masks, confidence outputs and ranking |
| Failure | a batch that raises, or that boltz skips on out of memory, is re-run one input per step, by name (`BATCH batch=… fell back to one record per step`) |
| Kit evidence | the roll-out's capture count is checked per batch instead of per input |

The trunk levers needed no change: an audit of the 28 `fast` levers found them batch-aware already (five of them loop over the batch
inside their wrapper, which is correct but leaves some batch parallelism unused).

### Randomness and reproducibility

Each input's sampler noise and featurizer streams are seeded from (run seed, input name) — `batching.record_seed`. Consequences:

- An input's prediction does **not** depend on its batch-mates, its slot, the batch size, or the GPU, up to floating-point rounding
  (padding moves kernel tile boundaries).
- A batched run of an input and a **non-batched** run of the same input at the same `--seed` draw different noise: they differ the way
  two seeds do. `exact` mode's bit-identity with stock does not extend to batches (batching is `fast` only).

### Centring

Stock's inference sampler centres coordinates with an unmasked mean over every atom slot; training centres over real atoms only. With one
input that is at most 31 padding atoms; in a padded batch a short input would have hundreds. The batched sampler centres over real atoms
(training's convention). The boltz-community fork's `--batch_size` keeps the unmasked mean (checked at its commit `f2e5183`).

## Validation (H200 / H100 NVL, Apptainer 1.5.2, Oct 2026)

`tools/boltz2_batching/validate.sh` and `diagnose.sh` run a set of 8 inputs several ways and compare structures (all-atom RMSD after
superposition) and trunk embeddings.

**Top7 monomers, 97–104 residues, single sequence (predicted confidently, pLDDT ≈ 0.85):**

| Comparison | Structure RMSD mean / max | Note |
|---|---|---|
| alone vs batch of 4 | 0.21 / 0.41 Å | same per-input noise: only padding and rounding differ |
| alone vs batch of 8 | 0.19 / 0.28 Å | |
| batch of 4 vs batch of 8 | 0.21 / 0.46 Å | |
| control: `exact` vs `fast`, non-batched | 0.41 / 1.49 Å | the kit's own rounding difference, for scale |

Confidence scores agree to ~0.002; trunk embeddings differ by 0.2–0.5 % (relative L2), within the `exact` vs `fast` range.

**Barnase / barstar complexes, 178–199 tokens, single sequence (pLDDT ≈ 0.4):** every comparison differs by 5–18 Å, including the
kit's own `exact` vs `fast` (11 Å mean) and two noise streams (14 Å). Without an MSA the model cannot determine these structures and the
diffusion sampler amplifies any rounding difference. This set says nothing about batching correctness; it is a reminder to give
Boltz-2 MSAs and to treat low-pLDDT coordinates as unreliable in any mode.

**Repeatability:** the same batched run twice gives identical structures (0.000 Å).

**Speed** (model seconds per input after warm-up, one GPU per process; single measurements):

| | ~100-residue monomers (H200) | ~190-token complexes (H200) | ~190-token complexes (H100 NVL) |
|---|---|---|---|
| one per step | 0.8–1.5 s | ~0.75 s | 0.78–0.93 s |
| batch of 4 | 0.38 s | 0.60 s | 0.49 s |

## Known limits

- **First batch of each shape is slow.** The CUDA-graph levers record one graph per (batch size, padded length); a batch with a new
  shape pays that capture (5–7 s vs 1.5–2.4 s warm in the runs above). Inputs of many different lengths can lose much of the gain.
  Padding lengths up to a few standard sizes would fix this; not done.
- **Untested:** MSAs (a second padded dimension: MSA depth), ligands / modified residues, templates, `--diffusion_samples > 1`,
  inputs above ~200 tokens, the out-of-memory fallback, `--write_full_pae` / `--write_full_pde`.
- `fast` only. Not combinable with the kit's model-parallel `--n_gpu > 1` (data-parallel multi-GPU is one process per GPU, above).
- The structures of a batched and a non-batched run of one seed differ (see Randomness).

## Files

`opt/boltz2_opt/batching.py` (buckets, dataset, collate, confidence wrapper, per-input prediction), the batched branch of
`opt/forward/trunk_levers/src/bz_worker_levf2.py`, `opt/forward/sampler/src/bz_sampler.py` (`_predraw_records`, `_rollout`),
`opt/forward/sampler/src/bz_sampler_dit.py` (`prepare`, `fused_forward`), `opt/boltz2_opt/worker.py` (mode / GPU check),
`opt/boltz2_opt/modes.py` (`RUN_DROPS["batch_size"]`), `opt/boltz2_opt/stack.py` (roll-out evidence per batch);
tests: `tools/boltz2_batching/` (`make_inputs.py`, `compare.py`, `validate.sh`, `diagnose.sh`).
