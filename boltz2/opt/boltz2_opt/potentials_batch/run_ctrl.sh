#!/bin/bash
set -u
BASE=/mnt/scratch/user/daviyang; BENCH=$BASE/bench_leak
BOLTZ=$BASE/uplifting-biomolecular-modeling/tools/boltz_compat/boltz
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 BOLTZ_KIT_KEEP=1
export BOLTZ_KIT_MODE=fast BOLTZ_KIT_CODE=$BASE/bz-potbatch BOLTZ_BATCH_POTENTIALS=1 BOLTZ_BATCH_SIZE=4 POTB_DETERMINISTIC=1
for r in c1 c2; do
  OUT=$BENCH/out_ctrl_$r; rm -rf $OUT; mkdir -p $OUT
  $BOLTZ predict $BENCH/setA --out_dir $OUT --output_format pdb --override --use_potentials > $BENCH/log_ctrl_$r.txt 2>&1
  echo "RESULT ctrl_$r rc=$? pdbs=$(find $OUT -name '*.pdb'|wc -l) fallbacks=$(grep -hc 'fell back' $OUT/.boltz_kit/out_0/pred_worker.log 2>/dev/null)"
done
echo '=== done ==='
