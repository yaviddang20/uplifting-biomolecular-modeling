#!/bin/bash
# Batch-size scaling INSIDE the batched route: only BOLTZ_BATCH_SIZE varies.
# 16 inputs of one shape bucket so every size divides evenly; arms interleaved
# across reps so GPU contention averages out rather than loading one arm.
set -u
BASE=/mnt/scratch/user/daviyang
BENCH=$BASE/bench_sweep; IN=$BENCH/inputs
rm -rf "$IN"; mkdir -p "$IN"
BOLTZ=$BASE/uplifting-biomolecular-modeling/tools/boltz_compat/boltz
PROT=AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR
CHOL='C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C'
for i in $(seq 1 16); do
  printf 'version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: %s\n      msa: empty\n  - ligand:\n      id: B\n      smiles: '\''%s'\''\n' "${PROT:0:$((110 - i % 3))}" "$CHOL" > "$IN/d$i.yaml"
done

export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8
export BOLTZ_KIT_MODE=fast BOLTZ_KIT_CODE=$BASE/bz-potbatch BOLTZ_BATCH_POTENTIALS=1 BOLTZ_KIT_KEEP=1

for rep in 1 2; do
  for bs in 2 4 8 16; do
    OUT=$BENCH/out_bs${bs}_r${rep}_$(date +%s); mkdir -p "$OUT"
    T0=$(date +%s.%N)
    BOLTZ_BATCH_SIZE=$bs "$BOLTZ" predict "$IN" --out_dir "$OUT" --output_format pdb --override --use_potentials \
        > "$BENCH/log_bs${bs}_r${rep}.txt" 2>&1
    RC=$?; T1=$(date +%s.%N)
    S=$(grep -h 'PHASE item' $OUT/.boltz_kit/out_0/pred_worker.log 2>/dev/null | sed 's/.*sampler_s=\([0-9.]*\).*/\1/' | paste -sd+ | bc)
    printf 'RESULT bs=%-2s rep=%s rc=%s pdbs=%2s wall=%.1f sampler=%s\n' \
        "$bs" "$rep" "$RC" "$(find $OUT -name '*.pdb' | wc -l)" "$(echo "$T1 - $T0" | bc)" "${S:-NA}"
  done
done
echo "=== done ==="
