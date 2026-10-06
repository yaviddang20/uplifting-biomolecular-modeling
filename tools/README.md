# Building the kits with Apptainer only (no Docker)

Upstream's `<kit>/environment/apptainer.def` starts with `Bootstrap: docker-daemon`, so it needs a local Docker image first.
`<kit>/environment/apptainer.native.def` is generated from the same Dockerfile by `tools/docker2apptainer.py` and bootstraps
straight from the Dockerfile's public base image (`Bootstrap: docker`). No Docker is needed anywhere. The result is the same
image: the same `/kit`, environment, and `%runscript`, so each kit README's route-B `kit()` lines work unchanged.

Build on the cluster, from the repository root:

    bash tools/build_sif.sh proteinmpnn                                   # -> ./proteinmpnn.sif
    bash tools/build_sif.sh af3_torch --build-arg WHEELS_FROM=prebuilt    # Dockerfile ARGs become --build-arg
    APPTAINER_BUILD_FLAGS= bash tools/build_sif.sh boltz2                 # no --fakeroot (root, or Apptainer >= 1.3 proot builds)

Requirements: Apptainer >= 1.2, internet access on the build node, and permission to build (`--fakeroot` by default).
Kits that compile CUDA extensions (Dockerfiles with `-devel` bases) take a long time and plenty of RAM, so build them in
an interactive or batch job, not on a login node. Pass `--build-arg BUILD_JOBS=<n>` to cap parallel compile jobs.
The build runs on the CPU, so no GPU is needed to build. The GPU is only used at run time (`--nv`).

After editing a Dockerfile, regenerate with `python3 tools/docker2apptainer.py [kit ...]` (`build_sif.sh` does this itself).
