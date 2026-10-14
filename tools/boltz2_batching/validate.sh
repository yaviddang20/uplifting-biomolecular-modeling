#!/bin/bash
# Validate batched Boltz-2 inference (BOLTZ_BATCH_SIZE) on one GPU. Run on a GPU node (H100 / H200), e.g.
#   sbatch -p gpu --gres=gpu:nvidia_h100_nvl:1 -c 8 --mem=64G -t 1:30:00 -o validate_%j.log tools/boltz2_batching/validate.sh
# What it runs (SET=top7, the default: 8 Top7 monomers of 97-104 residues, single-sequence, seed 0 — a fold Boltz-2 predicts CONFIDENTLY without an
# MSA, so rounding is not amplified by the sampler; SET=1brs: 178-199-token complexes it predicts poorly without one, where the kit's own exact and
# fast modes already differ by ~10 A at one seed and no structure threshold is meaningful), each twice — the second pass is the timed one:
#   single   the kit's fast mode as upstream runs it: one input per step
#   b1       the batched code path with every input alone   (BOLTZ_BATCH_MAX_PAD=1.0: no two lengths share a batch)
#   b4, b8   batches of 4 and 8
# What it checks:
#   b1 vs b8, b1 vs b4   the SAME per-record noise: structures must agree up to floating-point rounding (padding only) — the correctness test
#   single vs b8         different noise streams by design: agreement like two seeds of one input (reported, not gated)
# and the throughput of each configuration from the worker's own timings.
# Paths (override by exporting them): REPO = this checkout, SIF = the boltz2 image, WEIGHTS = the Boltz-2 cache, WORK = scratch for outputs.
set -uo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
[ -d "$REPO/boltz2/opt" ] || REPO="${SLURM_SUBMIT_DIR:-$PWD}"
BASE="${KIT_BASE:-$(dirname "$REPO")}"
SIF="${SIF:-$BASE/uplifting-biomolecular-modeling/boltz2.sif}"
WEIGHTS="${WEIGHTS:-$BASE/weights/boltz2}"
WORK="${WORK:-$BASE/kit_out/boltz2_batching/$(date +%Y%m%d_%H%M%S)}"
CARD="${CARD:-h100}"; RMSD_MAX="${RMSD_MAX:-1.0}"; NIN="${NIN:-8}"; SET="${SET:-top7}"
mkdir -p "$WORK/in" "$WORK/jit"
echo "REPO=$REPO  SIF=$SIF  WEIGHTS=$WEIGHTS  WORK=$WORK  CARD=$CARD"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
pyc() { apptainer exec --bind "$BASE" "$SIF" python3 "$@"; }      # the image's python (numpy): compute nodes need none of their own
pyc "$REPO/tools/boltz2_batching/make_inputs.py" "$WORK/in" "$NIN" "$SET"

kit() {   # the image with THIS checkout's kit code over the image's (/kit/boltz2, /kit/common): no rebuild needed to test a change
    BOLTZ_CACHE=/weights/boltz2 MODEL_OPT_JIT_ROOT="$WORK/jit" apptainer run --nv --bind "$WEIGHTS":/weights/boltz2 \
        --bind "$REPO/boltz2":/kit/boltz2 --bind "$REPO/common":/kit/common --bind "$BASE" "$SIF" "$@"
}
run() {   # run <name> [ENV=VALUE ...]: two passes; the second is timed
    local name=$1; shift
    for pass in warm timed; do
        local out="$WORK/$name.$pass"; local t0=$(date +%s)
        ( [ $# -gt 0 ] && export "$@"; kit pred --config "$CARD" --mode fast --input "$WORK/in" --out_dir "$out" --seed 0 ) > "$WORK/$name.$pass.log" 2>&1
        local rc=$?; local t1=$(date +%s)
        echo "RUN $name pass=$pass rc=$rc wall_s=$((t1 - t0)) $(grep -c . <(ls -d "$out"/by_seed/*/s0 2>/dev/null)) outputs   $(grep -E 'ACTIVE mode=|NOT ACTIVE|BATCH plan|PARTIAL' "$WORK/$name.$pass.log" | cut -c1-200 | tr '\n' ' ')"
        [ $rc -eq 0 ] || { echo "--- last lines of $WORK/$name.$pass.log"; tail -25 "$WORK/$name.$pass.log"; }
    done
}
run single
run b1 BOLTZ_BATCH_SIZE=8 BOLTZ_BATCH_MAX_PAD=1.0
run b4 BOLTZ_BATCH_SIZE=4 BOLTZ_BATCH_MAX_PAD=1.25
run b8 BOLTZ_BATCH_SIZE=8 BOLTZ_BATCH_MAX_PAD=1.25

echo; echo "===== throughput (timed pass; the worker's own per-input prediction seconds)"
for name in single b1 b4 b8; do     # the worker's own item_done lines: model_s is the forward of the step (shared by the records of a batch)
    grep -h "item_done" "$WORK/$name.timed/pred_worker.log" 2>/dev/null | sed -E "s/.*'model_s': ([0-9.]*).*/\1/" | awk -v n="$name" -v b="$( [ $name = b4 ] && echo 4 || { [ $name = b8 ] && echo 8 || echo 1; } )" \
        '{ t[NR] = $1 } END { if (NR == 0) { print n ": no timings"; exit } for (i = 1; i <= NR; i++) tot += t[i] / b; last = t[NR] / b; printf "%-8s inputs=%d  model_s per input: mean %.2f  last step %.2f   (a step whose shape is new pays a one-time CUDA-graph capture)\n", n, NR, tot / NR, last }'
done
echo; echo "===== b1 vs b8 (same noise per record: must agree to rounding; limit $RMSD_MAX A)"
pyc "$REPO/tools/boltz2_batching/compare.py" "$WORK/b1.timed" "$WORK/b8.timed" --max-rmsd "$RMSD_MAX"; R1=$?
echo; echo "===== b1 vs b4 (same)"
pyc "$REPO/tools/boltz2_batching/compare.py" "$WORK/b1.timed" "$WORK/b4.timed" --max-rmsd "$RMSD_MAX"; R2=$?
echo; echo "===== b1 warm vs b1 timed (run-to-run repeatability of the batched path)"
pyc "$REPO/tools/boltz2_batching/compare.py" "$WORK/b1.warm" "$WORK/b1.timed"
echo; echo "===== single vs b8 (different noise streams by design: differs like two seeds; not gated)"
pyc "$REPO/tools/boltz2_batching/compare.py" "$WORK/single.timed" "$WORK/b8.timed"
echo; if [ $R1 -eq 0 ] && [ $R2 -eq 0 ]; then echo "VALIDATION PASS: batched structures match the batch-of-one structures within $RMSD_MAX A"; else echo "VALIDATION FAIL (see above; logs in $WORK)"; fi
exit $(( R1 | R2 ))
