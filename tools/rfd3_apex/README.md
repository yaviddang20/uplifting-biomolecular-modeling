# rfdiffusion3 on a cluster whose compute nodes have no internet

rfdiffusion3's image compiles NVIDIA apex (CUDA extensions) from source: long and memory-hungry, so it should not run on a login node.
The download and the compile are split instead (run from the repository root):

1. Login node (internet, light work): build the compiler image. It downloads everything and compiles nothing.

       nice apptainer build --fakeroot apex_builder.sif tools/rfd3_apex/apex_builder.def

2. CPU compute node (no internet or GPU needed): compile. This writes the wheel into rfdiffusion3/stock/wheels/.

       sbatch -p cpu -c 32 --mem=96G -t 4:00:00 --wrap "apptainer run apex_builder.sif $PWD/rfdiffusion3/stock/wheels 32"

3. Login node: build the kit image from that wheel (no compile).

       nice bash tools/build_sif.sh rfdiffusion3 --build-arg WHEELS_FROM=local

`WHEELS_FROM=local` is a branch added to rfdiffusion3/environment/Dockerfile. It installs the wheel found in stock/wheels/ and prints
its sha256. The wheel is compiled from the pinned commit with the same flags as `WHEELS_FROM=build`, so it does not match the hash
of the published wheel that `prebuilt` checks for.
