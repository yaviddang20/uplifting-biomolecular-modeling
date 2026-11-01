#!/bin/bash
# Where do b1 and b4 first differ for the SAME record? Dump record 0's noise
# draw and the coords the guidance loop sees, at both batch sizes, and diff.
set -u
BASE=/mnt/scratch/user/daviyang; BENCH=$BASE/bench_div
REPO=$BASE/bz-potbatch
BOLTZ=$BASE/uplifting-biomolecular-modeling/tools/boltz_compat/boltz
rm -rf "$BENCH"; mkdir -p "$BENCH"; cp -r $BASE/bench_top7p/in "$BENCH/in"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 BOLTZ_KIT_KEEP=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export BOLTZ_KIT_CODE=$REPO BOLTZ_KIT_MODE=fast
export BOLTZ_BATCH_POTENTIALS=1 POTB_DETERMINISTIC=1
for leg in b1 b4; do
  [ $leg = b1 ] && { BS=8; MP=1.0; } || { BS=4; MP=1.25; }
  OUT=$BENCH/out_$leg; rm -rf "$OUT"; mkdir -p "$OUT"
  POTB_DUMP=$BENCH/dump_$leg BOLTZ_BATCH_SIZE=$BS BOLTZ_BATCH_MAX_PAD=$MP "$BOLTZ" predict "$BENCH/in" \
      --out_dir "$OUT" --output_format pdb --override --seed 0 --use_potentials \
      > "$BENCH/log_$leg.txt" 2>&1
  echo "RESULT $leg rc=$? dumps=$(ls $BENCH/dump_$leg 2>/dev/null | wc -l)"
done
echo "=== done ==="
