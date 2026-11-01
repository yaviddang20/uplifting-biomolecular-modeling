#!/bin/bash
# BATCHING.md's actual correctness protocol, applied to the GUIDED path.
#   b1 = the batched code path with every input ALONE (BOLTZ_BATCH_MAX_PAD=1.0:
#        the top7 monomers all differ in length, so no two share a bucket)
#   b4/b8 = real batches
# b1 vs b4 and b1 vs b8 hold the per-record noise FIXED -- same input set, same
# seed, batched path both sides -- so only padding and kernel tiling differ.
# That is the comparison their 0.21 A came from, and it is gated at 1.0 A.
# (single-vs-batched crosses a noise-stream boundary and differs like two seeds
# by design; that is NOT this test.)
set -u
BASE=/mnt/scratch/user/daviyang; BENCH=$BASE/bench_top7p
REPO=$BASE/bz-potbatch
BOLTZ=$BASE/uplifting-biomolecular-modeling/tools/boltz_compat/boltz
rm -rf "$BENCH"; mkdir -p "$BENCH/in"
apptainer exec -B $BASE:$BASE $BASE/uplifting-biomolecular-modeling/boltz2.sif \
    python -I $REPO/tools/boltz2_batching/make_inputs.py "$BENCH/in" 8 top7 > "$BENCH/mk.log" 2>&1
echo "inputs: $(ls $BENCH/in | tr '\n' ' ')"

export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 BOLTZ_KIT_KEEP=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export BOLTZ_KIT_MODE=fast BOLTZ_KIT_CODE=$REPO POTB_DETERMINISTIC=1

run () {  # tag  batch_size  max_pad  guided(1/0)
  local OUT=$BENCH/out_$1; rm -rf "$OUT"; mkdir -p "$OUT"
  local POT=""
  if [ "$4" = 1 ]; then export BOLTZ_BATCH_POTENTIALS=1; POT=--use_potentials
  else unset BOLTZ_BATCH_POTENTIALS; fi
  BOLTZ_BATCH_SIZE=$2 BOLTZ_BATCH_MAX_PAD=$3 "$BOLTZ" predict "$BENCH/in" \
      --out_dir "$OUT" --output_format pdb --override --seed 0 $POT \
      > "$BENCH/log_$1.txt" 2>&1
  echo "RESULT $1 rc=$? pdbs=$(find $OUT -path '*predictions*' -name '*_model_0.pdb'|wc -l) fallbacks=$(grep -hc 'fell back' $OUT/.boltz_kit/out_0/pred_worker.log 2>/dev/null || echo 0)"
  grep -h 'BATCH plan' $OUT/.boltz_kit/out_0/pred_worker.log 2>/dev/null | head -2
}

run g_b1 8 1.0 1
run g_b4 4 1.25 1
run g_b8 8 1.25 1
run u_b1 8 1.0 0
run u_b4 4 1.25 0
run u_b8 8 1.25 0
echo "=== done ==="
