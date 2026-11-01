#!/bin/bash
# What is the GUIDED path's own noise floor? The 1.0 A gate was calibrated on
# the unguided path, whose run-to-run repeatability is 0.000 A. The guidance
# loop applies ~4000 scatter_reduce updates per prediction, so it amplifies any
# ~1e-7 perturbation. Measure that amplification with perturbations that are
# NOT batching, at b1 throughout (every input alone), so batching is held out.
#   A1 vs A2   same config twice, determinism OFF -> the atomics floor
#   D1 vs D2   same config twice, determinism ON  -> should be 0
#   D1 vs E    fast vs exact, determinism ON      -> the kit's own rounding, the
#              scale control BATCHING.md uses for the unguided path
set -u
BASE=/mnt/scratch/user/daviyang; BENCH=$BASE/bench_floor
REPO=$BASE/bz-potbatch
BOLTZ=$BASE/uplifting-biomolecular-modeling/tools/boltz_compat/boltz
rm -rf "$BENCH"; mkdir -p "$BENCH"
cp -r $BASE/bench_top7p/in "$BENCH/in"
echo "inputs: $(ls $BENCH/in | wc -l)"

export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 BOLTZ_KIT_KEEP=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export BOLTZ_KIT_CODE=$REPO
export BOLTZ_BATCH_POTENTIALS=1

run () {  # tag  mode  det
  local OUT=$BENCH/out_$1; rm -rf "$OUT"; mkdir -p "$OUT"
  if [ "$3" = 1 ]; then export POTB_DETERMINISTIC=1; else unset POTB_DETERMINISTIC; fi
  BOLTZ_KIT_MODE=$2 BOLTZ_BATCH_SIZE=8 BOLTZ_BATCH_MAX_PAD=1.0 "$BOLTZ" predict "$BENCH/in" \
      --out_dir "$OUT" --output_format pdb --override --seed 0 --use_potentials \
      > "$BENCH/log_$1.txt" 2>&1
  echo "RESULT $1 mode=$2 det=$3 rc=$? pdbs=$(find $OUT -path '*predictions*' -name '*_model_0.pdb'|wc -l)"
}

run A1 fast  0
run A2 fast  0
run D1 fast  1
run D2 fast  1
run E  exact 1
echo "=== done ==="
