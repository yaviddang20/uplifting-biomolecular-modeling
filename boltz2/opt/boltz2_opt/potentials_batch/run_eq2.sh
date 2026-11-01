#!/bin/bash
# Equal-length variants of a CONFIDENT fold, with true b1 via BOLTZ_BATCH_SIZE=1.
# Padding is identical at every batch size, so anything left is the batch size.
set -u
BASE=/mnt/scratch/user/daviyang; BENCH=$BASE/bench_eq2
REPO=$BASE/bz-potbatch
BOLTZ=$BASE/uplifting-biomolecular-modeling/tools/boltz_compat/boltz
rm -rf "$BENCH"; mkdir -p "$BENCH/in"
apptainer exec -B $BASE:/w -B $BASE:$BASE $BASE/uplifting-biomolecular-modeling/boltz2.sif \
    python -I $BASE/mk_eq.py "$BENCH/in" 8
echo "lengths: $(grep -h 'sequence:' $BENCH/in/*.yaml | awk '{print length($2)}' | sort -u | tr '\n' ' ')"

export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 BOLTZ_KIT_KEEP=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export BOLTZ_KIT_CODE=$REPO BOLTZ_KIT_MODE=fast
export POTB_DETERMINISTIC=1

run () {  # tag bs guided
  local OUT=$BENCH/out_$1; rm -rf "$OUT"; mkdir -p "$OUT"
  local POT=""
  if [ "$3" = 1 ]; then export BOLTZ_BATCH_POTENTIALS=1; POT=--use_potentials
  else unset BOLTZ_BATCH_POTENTIALS; fi
  BOLTZ_BATCH_SIZE=$2 "$BOLTZ" predict "$BENCH/in" --out_dir "$OUT" \
      --output_format pdb --override --seed 0 $POT > "$BENCH/log_$1.txt" 2>&1
  echo "RESULT $1 rc=$? pdbs=$(find $OUT -path '*predictions*' -name '*_model_0.pdb'|wc -l)"
  grep -h 'BATCH plan' $OUT/.boltz_kit/out_0/pred_worker.log 2>/dev/null | head -1
}
run g_b1 1 1
run g_b4 4 1
run g_b8 8 1
run u_b1 1 0
run u_b4 4 0
run u_b8 8 0
echo "=== done ==="
