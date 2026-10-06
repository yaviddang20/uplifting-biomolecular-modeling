#!/bin/bash
# Validate batched Boltz-2 inference (BOLTZ_BATCH_SIZE) on one GPU. Run on a GPU node (H100 / H200), e.g.
#   sbatch -p gpu --gres=gpu:nvidia_h100_nvl:1 -c 8 --mem=64G -t 1:30:00 -o validate_%j.log tools/boltz2_batching/validate.sh
# What it runs (8 different complexes of 178-199 tokens, single-sequence, seed 0), each twice — the second pass is the timed one (kernels compiled):
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
CARD="${CARD:-h100}"; RMSD_MAX="${RMSD_MAX:-0.5}"; NIN="${NIN:-8}"
mkdir -p "$WORK/in" "$WORK/jit"
echo "REPO=$REPO  SIF=$SIF  WEIGHTS=$WEIGHTS  WORK=$WORK  CARD=$CARD"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
pyc() { apptainer exec --bind "$BASE" "$SIF" python3 "$@"; }      # the image's python (numpy): compute nodes need none of their own
pyc "$REPO/tools/boltz2_batching/make_inputs.py" "$WORK/in" "$NIN"

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
pyc - "$WORK" <<'PY'
import sys, json, glob, os
work = sys.argv[1]
for name in ("single", "b1", "b4", "b8"):
    logs = glob.glob(os.path.join(work, f"{name}.timed", "**", "*worker_log*.json"), recursive=True) + glob.glob(os.path.join(work, f"{name}.timed", "**", "manifest*.json"), recursive=True)
    rows = []
    for p in logs:
        try: d = json.load(open(p))
        except Exception: continue
        rows = d.get("per_item") or ((d.get("worker_log") or {}).get("per_item")) or rows
        if rows: break
    if not rows:
        print(f"{name:<8} no per-item timings found (see {work}/{name}.timed.log)"); continue
    tot = sum(r.get("predict_s") or 0 for r in rows); ms = sum((r.get("model_s") or 0) / max(1, r.get("batch_n") or 1) for r in rows)
    print(f"{name:<8} inputs={len(rows):<3} predict_s_total={tot:8.2f}  per_input={tot / len(rows):6.2f}  model_s_per_input={ms / len(rows):6.2f}  batches={len({r.get('batch_id', i) for i, r in enumerate(rows)})}")
PY
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
