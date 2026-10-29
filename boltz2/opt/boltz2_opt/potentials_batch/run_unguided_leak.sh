#!/bin/bash
set -u
BASE=/mnt/scratch/user/daviyang; BENCH=$BASE/bench_leak
BOLTZ=$BASE/uplifting-biomolecular-modeling/tools/boltz_compat/boltz
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 BOLTZ_KIT_KEEP=1
export BOLTZ_KIT_MODE=fast BOLTZ_KIT_CODE=$BASE/bz-potbatch BOLTZ_BATCH_SIZE=4
unset BOLTZ_BATCH_POTENTIALS     # unguided: the shipped batching path
for s in setA setB setA2; do
  src=$BENCH/${s/setA2/setA}
  OUT=$BENCH/out_ug_$s; rm -rf $OUT; mkdir -p $OUT
  $BOLTZ predict $src --out_dir $OUT --output_format pdb --override > $BENCH/log_ug_$s.txt 2>&1
  echo "RESULT ug_$s rc=$? pdbs=$(find $OUT -name '*.pdb'|wc -l) fallbacks=$(grep -hc 'fell back' $OUT/.boltz_kit/out_0/pred_worker.log 2>/dev/null)"
done
echo '=== done ==='
