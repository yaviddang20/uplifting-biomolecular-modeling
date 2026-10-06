#!/bin/bash
# rfdiffusion3: builder image (login) -> apex compile (cpu job) -> kit image (login). Status lines go to build_status2.txt.
cd "$(cd "$(dirname "$0")" && pwd)" || exit 1
st=build_status2.txt
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$(dirname "$PWD")/apptainer-tmp-$USER}"; mkdir -p "$APPTAINER_TMPDIR"
echo "waiting for boltzgen/complexa builds..."
: # (no wait: runs alongside the other builds)
echo "== step 1: apex builder image"
nice apptainer build --fakeroot -F apex_builder.sif tools/rfd3_apex/apex_builder.def > build_apex_builder.log 2>&1 \
  || { echo "rfd3 step1 (apex_builder) FAILED" >> $st; exit 1; }
echo "== step 2: apex compile on a cpu node"
mkdir -p rfdiffusion3/stock/wheels
sbatch --wait -J apex_compile -p cpu -c 16 --mem=64G -t 4:00:00 -o build_apex_compile.log \
  --wrap "apptainer run $PWD/apex_builder.sif $PWD/rfdiffusion3/stock/wheels 16" \
  || { echo "rfd3 step2 (apex compile) FAILED" >> $st; exit 1; }
echo "== step 3: rfdiffusion3 kit image"
nice bash tools/build_sif.sh rfdiffusion3 --build-arg WHEELS_FROM=local > build_rfdiffusion3.log 2>&1 \
  && echo "rfdiffusion3 OK" >> $st || echo "rfdiffusion3 FAILED" >> $st
