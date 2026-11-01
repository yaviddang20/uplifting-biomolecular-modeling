#!/bin/bash
# Is a record's result invariant to BATCH SIZE (not just to which mates share
# the batch)? Same probe, batches of 1/2/3/4.
set -u
BASE=/mnt/scratch/user/daviyang; BENCH=$BASE/bench_bsinv
BOLTZ=$BASE/uplifting-biomolecular-modeling/tools/boltz_compat/boltz
rm -rf "$BENCH"; mkdir -p "$BENCH"
A=$BASE/bench_det2/setA
mkdir -p "$BENCH/set1" "$BENCH/set2" "$BENCH/set3" "$BENCH/set4"
cp $A/probe.yaml "$BENCH/set1/"
cp $A/probe.yaml $A/mate1.yaml "$BENCH/set2/"
cp $A/probe.yaml $A/mate1.yaml $A/mate2.yaml "$BENCH/set3/"
cp $A/probe.yaml $A/mate1.yaml $A/mate2.yaml $A/mate3.yaml "$BENCH/set4/"

export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 BOLTZ_KIT_KEEP=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export BOLTZ_KIT_MODE=fast BOLTZ_KIT_CODE=$BASE/bz-potbatch
export BOLTZ_BATCH_POTENTIALS=1 POTB_DETERMINISTIC=1

for n in 1 2 3 4; do
  OUT=$BENCH/out_bs$n; rm -rf "$OUT"; mkdir -p "$OUT"
  BOLTZ_BATCH_SIZE=$n "$BOLTZ" predict "$BENCH/set$n" --out_dir "$OUT" \
      --output_format pdb --override --use_potentials > "$BENCH/log_bs$n.txt" 2>&1
  echo "RESULT bs=$n rc=$? pdbs=$(find $OUT -path '*predictions*' -name '*.pdb'|wc -l)"
  grep -h 'BATCH plan' $OUT/.boltz_kit/out_0/pred_worker.log 2>/dev/null
  grep -h '\[POTB\]' $OUT/.boltz_kit/out_0/pred_worker.log 2>/dev/null | head -1
done
echo "=== done ==="
