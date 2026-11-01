#!/bin/bash
# CONTROL: does the SHIPPED (unguided) batching path also change a record's
# structure with batch size? No --use_potentials, no potentials patch at all.
# If yes, batch-size sensitivity is pre-existing in Boltz batching and not
# something guided batching introduced.
set -u
BASE=/mnt/scratch/user/daviyang; BENCH=$BASE/bench_ungbs
BOLTZ=$BASE/uplifting-biomolecular-modeling/tools/boltz_compat/boltz
rm -rf "$BENCH"; mkdir -p "$BENCH"
A=$BASE/bench_det2/setA
for n in 1 2 3 4; do mkdir -p "$BENCH/set$n"; done
cp $A/probe.yaml "$BENCH/set1/"
cp $A/probe.yaml $A/mate1.yaml "$BENCH/set2/"
cp $A/probe.yaml $A/mate1.yaml $A/mate2.yaml "$BENCH/set3/"
cp $A/probe.yaml $A/mate1.yaml $A/mate2.yaml $A/mate3.yaml "$BENCH/set4/"

export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 BOLTZ_KIT_KEEP=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export BOLTZ_KIT_MODE=fast BOLTZ_KIT_CODE=$BASE/bz-potbatch
unset BOLTZ_BATCH_POTENTIALS          # unguided: the shipped batching path
export POTB_DETERMINISTIC=1

for n in 1 2 3 4; do
  OUT=$BENCH/out_bs$n; rm -rf "$OUT"; mkdir -p "$OUT"
  BOLTZ_BATCH_SIZE=$n "$BOLTZ" predict "$BENCH/set$n" --out_dir "$OUT" \
      --output_format pdb --override > "$BENCH/log_bs$n.txt" 2>&1
  echo "RESULT unguided bs=$n rc=$? pdbs=$(find $OUT -path '*predictions*' -name '*.pdb'|wc -l)"
  grep -h 'BATCH plan' $OUT/.boltz_kit/out_0/pred_worker.log 2>/dev/null
done
echo "=== done ==="
