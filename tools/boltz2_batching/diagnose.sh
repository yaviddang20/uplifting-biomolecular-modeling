#!/bin/bash
# Controls for "batched vs batch-of-one structures differ": is it rounding amplified by the sampler on an ill-determined input, or a defect?
#   A  exact vs fast, non-batched, same seed, on the 1BRS set  -> how far rounding ALONE moves these structures (the kit's own two modes)
#   B  trunk embeddings (s, z) of b1 vs b4 / b8                 -> no diffusion in between: rounding ~1e-2, a padding defect O(1)
#   C  the same structure comparison on a confidently predicted set (Top7 monomers, 99-106 residues)
# Run on a GPU node:  bash tools/boltz2_batching/diagnose.sh      (paths as validate.sh)
set -uo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"; BASE="${KIT_BASE:-$(dirname "$REPO")}"
SIF="${SIF:-$BASE/uplifting-biomolecular-modeling/boltz2.sif}"; WEIGHTS="${WEIGHTS:-$BASE/weights/boltz2}"
WORK="${WORK:-$BASE/kit_out/boltz2_batching/diag_$(date +%Y%m%d_%H%M%S)}"; CARD="${CARD:-h100}"
mkdir -p "$WORK/jit" "$WORK/in_1brs" "$WORK/in_top7"; cd "$WORK"
pyc() { apptainer exec --bind "$BASE" "$SIF" python3 "$@"; }
kit() { BOLTZ_CACHE=/weights/boltz2 MODEL_OPT_JIT_ROOT="$WORK/jit" apptainer run --nv --bind "$WEIGHTS":/weights/boltz2 --bind "$REPO/boltz2":/kit/boltz2 --bind "$REPO/common":/kit/common --bind "$BASE" "$SIF" "$@"; }
run() {   # run <name> <input set> <mode> [ENV=VALUE ...]
    local name=$1 set=$2 mode=$3; shift 3
    ( [ $# -gt 0 ] && export "$@"; kit pred --config "$CARD" --mode "$mode" --input "$WORK/in_$set" --out_dir "$WORK/$name" --seed 0 --write_embeddings ) > "$WORK/$name.log" 2>&1
    echo "RUN $name rc=$? outputs=$(ls -d "$WORK/$name"/by_seed/*/s0 2>/dev/null | wc -l) $(grep -h -E 'NOT ACTIVE|fell back' "$WORK/$name.log" "$WORK/$name/pred_worker.log" 2>/dev/null | cut -c1-160 | head -3 | tr '\n' ' ')"
    grep -h "item_done" "$WORK/$name/pred_worker.log" 2>/dev/null | sed -E "s/.*'item': '([^']*)'.*'predict_s': ([0-9.]*).*'model_s': ([0-9.]*).*/    \1 predict_s=\2 model_s=\3/" | tr '\n' ';' | cut -c1-600; echo
}
pyc "$REPO/tools/boltz2_batching/make_inputs.py" "$WORK/in_1brs" 8 1brs > /dev/null
pyc "$REPO/tools/boltz2_batching/make_inputs.py" "$WORK/in_top7" 8 top7 > /dev/null
for set in 1brs top7; do
    run ${set}_exact  $set exact
    run ${set}_single $set fast
    run ${set}_b1     $set fast BOLTZ_BATCH_SIZE=8 BOLTZ_BATCH_MAX_PAD=1.0
    run ${set}_b4     $set fast BOLTZ_BATCH_SIZE=4
    run ${set}_b8     $set fast BOLTZ_BATCH_SIZE=8
    run ${set}_b8rep  $set fast BOLTZ_BATCH_SIZE=8
    echo; echo "##### $set  A: exact vs fast, non-batched (rounding alone)";      pyc "$REPO/tools/boltz2_batching/compare.py" "$WORK/${set}_exact" "$WORK/${set}_single"
    echo; echo "##### $set  A': exact vs fast trunk embeddings";                  pyc "$REPO/tools/boltz2_batching/compare.py" "$WORK/${set}_exact" "$WORK/${set}_single" --emb
    echo; echo "##### $set  B: trunk embeddings b1 vs b4";                         pyc "$REPO/tools/boltz2_batching/compare.py" "$WORK/${set}_b1" "$WORK/${set}_b4" --emb
    echo; echo "##### $set  B: trunk embeddings b1 vs b8";                         pyc "$REPO/tools/boltz2_batching/compare.py" "$WORK/${set}_b1" "$WORK/${set}_b8" --emb
    echo; echo "##### $set  B': trunk embeddings single (non-batched) vs b1";      pyc "$REPO/tools/boltz2_batching/compare.py" "$WORK/${set}_single" "$WORK/${set}_b1" --emb
    echo; echo "##### $set  C: structures b1 vs b4";                               pyc "$REPO/tools/boltz2_batching/compare.py" "$WORK/${set}_b1" "$WORK/${set}_b4"
    echo; echo "##### $set  C: structures b1 vs b8";                               pyc "$REPO/tools/boltz2_batching/compare.py" "$WORK/${set}_b1" "$WORK/${set}_b8"
    echo; echo "##### $set  C: structures b8 vs b8 repeated (determinism)";        pyc "$REPO/tools/boltz2_batching/compare.py" "$WORK/${set}_b8" "$WORK/${set}_b8rep"
    echo; echo "##### $set  C: structures single (non-batched) vs b8 (different noise by design)"; pyc "$REPO/tools/boltz2_batching/compare.py" "$WORK/${set}_single" "$WORK/${set}_b8"
done
echo "DIAG_DONE work=$WORK"
