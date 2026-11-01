#!/bin/bash
# Does the PRE-FIX (masked, O(B^2)) patch fail the same 1.0 A gate? If yes,
# guided batching was never correct and today's commits are a strict
# improvement on a pre-existing defect. If it passes, today regressed it.
set -u
BASE=/mnt/scratch/user/daviyang; BENCH=$BASE/bench_prefix
REPO=$BASE/bz-potbatch
PB=$REPO/boltz2/opt/boltz2_opt/potentials_batch
BOLTZ=$BASE/uplifting-biomolecular-modeling/tools/boltz_compat/boltz
rm -rf "$BENCH"; mkdir -p "$BENCH"; cp -r $BASE/bench_top7p/in "$BENCH/in"

cp "$PB/patch.py" "$BASE/patch_NEW_backup.py"           # restore after
cp "$BASE/patch_OLD_quadratic.py" "$PB/patch.py"        # the pre-fix version
echo "installed pre-fix patch: $(grep -c 'rec_of' $PB/patch.py) rec_of hits (expect >0)"

export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 BOLTZ_KIT_KEEP=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export BOLTZ_KIT_CODE=$REPO BOLTZ_KIT_MODE=fast
export BOLTZ_BATCH_POTENTIALS=1 POTB_DETERMINISTIC=1

run () {  # tag bs maxpad
  local OUT=$BENCH/out_$1; rm -rf "$OUT"; mkdir -p "$OUT"
  BOLTZ_BATCH_SIZE=$2 BOLTZ_BATCH_MAX_PAD=$3 "$BOLTZ" predict "$BENCH/in" \
      --out_dir "$OUT" --output_format pdb --override --seed 0 --use_potentials \
      > "$BENCH/log_$1.txt" 2>&1
  echo "RESULT $1 rc=$? pdbs=$(find $OUT -path '*predictions*' -name '*_model_0.pdb'|wc -l)"
}
run old_b1 8 1.0
run old_b4 4 1.25

cp "$BASE/patch_NEW_backup.py" "$PB/patch.py"           # restore the fixed one
echo "restored: $(grep -c '_blockify' $PB/patch.py) _blockify hits (expect >0)"
echo "=== done ==="
