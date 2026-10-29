#!/bin/bash
set -u
BASE=/mnt/scratch/user/daviyang
BENCH=$BASE/bench_potbatch
IN=$BASE/bench_2x2b/inputs
BOLTZ=$BASE/uplifting-biomolecular-modeling/tools/boltz_compat/boltz
mkdir -p "$BENCH"

export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=8
export BOLTZ_KIT_KEEP=1
export BOLTZ_KIT_MODE=fast
export BOLTZ_KIT_CODE=$BASE/bz-potbatch

run() {  # label  batchsize  potbatch
  local OUT=$BENCH/out_$1_$(date +%s); mkdir -p "$OUT"   # unique: NFS silly-rename leftovers block rm -rf
  local T0=$(date +%s.%N)
  env BOLTZ_BATCH_SIZE=$2 BOLTZ_BATCH_POTENTIALS=$3 \
      "$BOLTZ" predict "$IN" --out_dir "$OUT" --output_format pdb --override --use_potentials \
      > "$BENCH/log_$1.txt" 2>&1
  local RC=$? T1=$(date +%s.%N)
  printf 'RESULT %-12s batch=%s potbatch=%s rc=%s pdbs=%s wall=%.1f\n' \
      "$1" "$2" "$3" "$RC" "$(find "$OUT" -name '*.pdb' | wc -l)" "$(echo "$T1 - $T0" | bc)"
  grep -h 'BATCH plan' "$BENCH"/log_$1.txt $(find "$OUT" -name 'shard_*.log' 2>/dev/null) 2>/dev/null | head -2
}

run batch1_cache 1 1
run batch4_pot   4 1
echo "=== done ==="
