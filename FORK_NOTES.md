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

Built and `check`ed on an H100 NVL with Apptainer 1.5.2: boltz2, af3_torch, boltzgen, complexa, rfdiffusion3 (and proteinmpnn built).
Start with `tools/README.md`.
