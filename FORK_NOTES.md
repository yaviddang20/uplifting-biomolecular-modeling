# Fork notes: Apptainer-only builds

This fork of [anthropics/uplifting-biomolecular-modeling](https://github.com/anthropics/uplifting-biomolecular-modeling) builds every
kit image with Apptainer/Singularity alone. Upstream's route B converts an image from a local Docker daemon; here the `.sif` is built
directly from each kit's Dockerfile. The kits themselves (`stock/`, `opt/`, `common/`) are unchanged.

| Added | What |
|---|---|
| `tools/docker2apptainer.py` | generates `<kit>/environment/apptainer.native.def` from the kit's Dockerfile (`Bootstrap: docker` from the public base image) |
| `tools/build_sif.sh <kit>` | one-command build (`--fakeroot`, Dockerfile ARGs as `--build-arg`) |
| `tools/rfd3_apex/`, `rfd3_pipeline.sh` | rfdiffusion3 on clusters whose compute nodes have no internet: download on the login node, compile NVIDIA apex offline in a CPU job |
| `kits.sh`, `KITS_USAGE.md` | run shortcuts and a how-to for UCSF CoreHPC (paths default to that site; `KIT_BASE` relocates them) |

Modified upstream file: `rfdiffusion3/environment/Dockerfile` gains a `WHEELS_FROM=local` branch (installs an apex wheel compiled by
`tools/rfd3_apex` instead of the published one).

**Parallel input parsing** (`BOLTZ_PARSE_PROCS`, boltz2, default: every usable CPU (`sched_getaffinity`, so it follows a Slurm cpuset), capped at the input count). Parsing a pass's inputs — upstream's
`check_inputs` + `process_inputs`, including RDKit conformer generation — was fully serial and overlapped nothing: 29 s of an 82 s
100-record run. `prep.Zygote` serves one parse at a time by construction, so N zygotes are forked at the same pre-CUDA point and the
inputs are spread over them, one thread per zygote. Every input is still parsed in a fresh child of a process that has parsed nothing,
so the RDKit conformer RNG property holds and **outputs are byte-identical to the serial parse** (verified: 200/200 files over 100
inputs). ~29 s → ~3.4 s at 8. `BOLTZ_PARSE_PROCS=1` restores the serial parse. The zygotes are nearly free: they share their pages copy-on-write with the parent, so
measured over 9 of them the marginal cost is ~1 MiB of private memory each (1262 MiB Pss in total, versus the 1589 MiB Rss each process
reports for the shared set). Details and the 2x2 against GPU batching:
[boltz2/BATCHING.md](boltz2/BATCHING.md).

Branch `boltz2-batching` (experimental): batched Boltz-2 inference, several inputs per GPU step on `--mode fast` — see
[boltz2/BATCHING.md](boltz2/BATCHING.md). It modifies kit files under `boltz2/opt/` (listed there); `main` leaves every kit's code as released.

Built and `check`ed on an H100 NVL with Apptainer 1.5.2: boltz2, af3_torch, boltzgen, complexa, rfdiffusion3 (and proteinmpnn built).
Start with `tools/README.md`.
