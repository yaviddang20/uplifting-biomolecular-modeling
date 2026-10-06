# Protein model kits on CoreHPC — how to use

Optimized Boltz-2, AF3 (af3_torch), BoltzGen, Proteina-Complexa, RFdiffusion3 and ProteinMPNN, packaged as Apptainer containers.
Source: https://github.com/anthropics/uplifting-biomolecular-modeling (+ our patch that builds without Docker).

## Where things are

```
/mnt/scratch/user/daviyang/
├── uplifting-biomolecular-modeling/     the repo
│   ├── boltz2.sif                       Boltz-2 container
│   ├── af3_torch.sif                    AF3 (xfold on OpenFold3-preview2 weights) container
│   ├── boltzgen.sif                     BoltzGen (binder design) container
│   ├── complexa.sif                     Proteina-Complexa (binder design) container
│   ├── rfdiffusion3.sif                 RFdiffusion3 container
│   ├── proteinmpnn.sif                  ProteinMPNN container (weights not installed yet)
│   ├── kits.sh                          defines the *_kit shortcuts
│   ├── KITS_USAGE.md                    this file
│   └── tools/build_sif.sh               rebuilds a container (Apptainer only, no Docker)
├── weights/{boltz2,af3_torch,boltzgen,complexa,rfd3}/   model weights (all installed, checksums verified)
└── kit_out/<kit>/out/                   ALL RESULTS land here (boltzgen: kit_out/boltzgen/runs/)
    kit_out/<kit>/jit/                   compiled-kernel cache (makes later runs start fast)
```

## The shortcuts

| shortcut | container | main command | results (cluster path) |
|---|---|---|---|
| `boltz2_kit` | boltz2.sif | `pred` | kit_out/boltz2/out/ |
| `af3_kit` | af3_torch.sif | `pred` | kit_out/af3_torch/out/ |
| `boltzgen_kit` | boltzgen.sif | `design` | kit_out/boltzgen/runs/ |
| `complexa_kit` | complexa.sif | `design` | kit_out/complexa/out/ |
| `rfd3_kit` | rfdiffusion3.sif | `design` | kit_out/rfdiffusion3/out/ |

All passed `check --config h100 --mode fast` on an H100 NVL (ggpu4-05).

## What the shortcuts are

Shell functions in `kits.sh`, NOT aliases in ~/.bashrc.d. Load them first in every shell/job:

```bash
source /mnt/scratch/user/daviyang/uplifting-biomolecular-modeling/kits.sh
```

Each one is shorthand for `apptainer run --nv --bind <weights> --bind <outputs> --bind /mnt/scratch/user/daviyang <kit>.sif ...`.
See the full expansion with `type boltz2_kit`. Inside the container, `out/...` = `kit_out/<kit>/out/...` on the cluster.

## Rules for this cluster

- **Login node**: internet yes, GPU no, memory is capped per user. Use it only for downloads.
  Heavy CPU work (e.g. converting weights) throttles badly there, so use `srun -p cpu` / `sbatch` instead.
- **Compute nodes**: GPU yes, internet NO. All runs happen here.
- **GPUs**: use H100 (`-p gpu --gres=gpu:nvidia_h100_nvl:1`, `--config h100`) or H200 (`-p large_gpu`, `--config h200`).
  Avoid L40S / RTX PRO 6000: the kits have no config for them.

## Run interactively

```bash
srun -p gpu --gres=gpu:nvidia_h100_nvl:1 -c 8 --mem=64G -t 2:00:00 --pty bash
source /mnt/scratch/user/daviyang/uplifting-biomolecular-modeling/kits.sh

# sanity check (seconds, predicts nothing): should print ACTIVE / DRY-RUN ok
boltz2_kit check --config h100 --mode fast
af3_kit    check --config h100 --mode fast

# Boltz-2: YAML input (a file, or a folder of YAMLs)
boltz2_kit pred --config h100 --mode fast --input /mnt/scratch/user/daviyang/my_inputs/x.yaml --out_dir out/x
#   -> /mnt/scratch/user/daviyang/kit_out/boltz2/out/x/

# AF3: AlphaFold3-format JSON input
af3_kit pred --config h100 --mode fast --json_path /mnt/scratch/user/daviyang/my_inputs/x.json --output_dir out/x
#   -> /mnt/scratch/user/daviyang/kit_out/af3_torch/out/x/

# built-in test inputs (barnase + barstar, ~1 min first time):
boltz2_kit pred --config h100 --mode fast --input inputs/1BRS_barnase_barstar.yaml --out_dir out/test
af3_kit    pred --config h100 --mode fast --json_path inputs/1BRS.json --output_dir out/test
```

## Run as a batch job

```bash
#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -o /mnt/scratch/user/daviyang/kit_out/%x_%j.log
source /mnt/scratch/user/daviyang/uplifting-biomolecular-modeling/kits.sh
boltz2_kit pred --config h100 --mode fast --input /mnt/scratch/user/daviyang/my_inputs/ --out_dir out/batch1
```

## Binder / protein design kits

Same setup as above (`srun` onto an H100/H200, `source kits.sh`). Inside the container, `out/...` / `runs/...` = the results
folder in the table above; your own input files go under `/mnt/scratch/user/daviyang/` with absolute paths.

**BoltzGen** takes a design-spec YAML (boltzgen's own format). Shipped example: PD-L1 binders.
```bash
boltzgen_kit design --config h100 --mode fast /kit/boltzgen/opt/forward/fast_inference/tests/specs/pdl1_ref.yaml \
    --output /kit/boltzgen/runs/pdl1_fast --num_designs 16 --seed 0 --diffusion_batch_size 16
#   -> kit_out/boltzgen/runs/pdl1_fast/     (upstream boltzgen overrides go after `--`)
```

**Proteina-Complexa** takes a targets JSON; upstream's own settings go after `--`. Shipped example: PD-L1, 80-residue binder.
```bash
cat > /mnt/scratch/user/daviyang/pdl1.json <<'EOF'
{"02_PDL1": {"source": "bindcraft_targets", "target_filename": "PD-L1", "target_path": "/opt/pc/assets/target_data/bindcraft_targets/PD-L1.pdb",
             "target_input": "A1-115", "hotspot_residues": ["A37", "A39", "A49", "A98"], "binder_length": [80, 80], "pdb_id": null}}
EOF
complexa_kit design --config h100 --mode fast --input /mnt/scratch/user/daviyang/pdl1.json --out out/pdl1_fast -- \
    ++generation.search.algorithm=single-pass ++generation.reward_model=null ++generation.dataloader.dataset.nres.nsamples=32 ++seed=5
#   -> kit_out/complexa/out/pdl1_fast/      (target_path is a path INSIDE the container; /opt/pc = complexa's code dir;
#                                            for your own target use its /mnt/scratch/... path)
```

**RFdiffusion3** takes rfd3's `key=value` settings. Upstream's example specs are already unpacked at
`kit_out/rfdiffusion3/out/foundry-4010e3e2/models/rfd3/docs/examples/` (protein / small-molecule / nucleic-acid binders,
enzyme, symmetry; each has a `.md` explaining it).
```bash
E=/kit/rfdiffusion3/out/foundry-4010e3e2/models/rfd3/docs/examples
rfd3_kit design --config h100 --mode fast inputs=$E/protein_binder_design.json out_dir=/kit/rfdiffusion3/out/binder_fast n_batches=1 seed=101
#   -> kit_out/rfdiffusion3/out/binder_fast/
```
RFdiffusion3 modes: `off`, `exact`, `fast` (no `big`); `--det 1` with `exact` = deterministic recipe.

## Input formats (Boltz-2 / AF3)

Boltz-2 YAML:
```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: AQVINTFDGVADYLQ...
      msa: empty            # or a path to a precomputed .a3m
  - protein:
      id: B
      sequence: KKAVINGEQIRSISD...
      msa: empty
```

AF3 JSON: standard AlphaFold3 input format. Copy the example at
`/mnt/scratch/user/daviyang/uplifting-biomolecular-modeling/af3_torch/inputs/1BRS.json` and change the sequences.

**MSAs:** compute nodes have no internet, so Boltz's `--use_msa_server` will NOT work there.
Either run single-sequence (`msa: empty`, less accurate), or precompute `.a3m` MSAs on the login node or elsewhere
and point to them (`msa: /path/x.a3m` in YAML; paste into `unpairedMsa` in AF3 JSON).

## Modes

| mode | what |
|---|---|
| `fast` | default, fastest, tiny numeric differences within seed-to-seed noise |
| `exact` | outputs byte-identical to stock Boltz / AF3, still faster |
| `big` | lowest GPU memory, for huge complexes that OOM; `--n_gpu 2/4/8` splits across GPUs |
| `off` | plain stock tool, no optimizations |

Every run prints `ACTIVE mode=...`. `NOT ACTIVE: <reason>` + exit code 3 means it refused to run and tells you why.
The first run of each mode compiles kernels (~1 min); later runs reuse `kit_out/<kit>/jit/`.

## Full docs

Per-kit README (all options, outputs, exit codes):
`/mnt/scratch/user/daviyang/uplifting-biomolecular-modeling/boltz2/README.md`,
`.../af3_torch/README.md`, `.../boltzgen/README.md`, `.../complexa/README.md`, `.../rfdiffusion3/README.md`, `.../proteinmpnn/README.md`.
How the Docker-free build works: `.../tools/README.md`.

## Rebuilding a container

On the login node, in tmux:
```bash
cd /mnt/scratch/user/daviyang/uplifting-biomolecular-modeling
nice bash tools/build_sif.sh boltz2          # ~10–20 min; any kit name works
```
Weights install (login node for the download; conversion-heavy kits: rerun the same command in `srun -p cpu`):
```bash
source kits.sh
boltz2_kit   install --weights /weights/boltz2
af3_kit      install --weights /weights/af3_torch --fetch
boltzgen_kit install --weights /weights/boltzgen
complexa_weights
rfd3_kit     install --weights /weights/rfd3
```
RFdiffusion3's image is special: it needs a CUDA compile (NVIDIA apex), done offline on a CPU node. See
`tools/rfd3_apex/README.md`; `rfd3_pipeline.sh` runs all three steps.

## Troubleshooting

- `*_kit: command not found`: you forgot `source .../kits.sh`.
- `NOT ACTIVE ... gpu`: you're on a node with no GPU, or on an L40S / RTX PRO node.
- Input file not found: use absolute paths under `/mnt/scratch/user/daviyang/`.
- boltz2 note `memory 95830 MiB is not the class's 81559 MiB`: harmless (H100 NVL has more memory than regular H100).
