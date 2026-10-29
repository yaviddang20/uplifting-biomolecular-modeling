#!/bin/bash
# Does a record's result depend on its batch-mates? Run the SAME probe record
# in two batches with different companions. Identical output => no leakage.
set -u
BASE=/mnt/scratch/user/daviyang
BENCH=$BASE/bench_leak; rm -rf "$BENCH"; mkdir -p "$BENCH/setA" "$BENCH/setB"
BOLTZ=$BASE/uplifting-biomolecular-modeling/tools/boltz_compat/boltz

PROT=AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR
CHOL='C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C'
mk() { printf 'version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: %s\n      msa: empty\n  - ligand:\n      id: B\n      smiles: '\''%s'\''\n' "$2" "$CHOL" > "$1"; }

# identical probe in both sets; companions are truncations (different lengths
# still bucket together: all within max_pad 1.25 of each other)
for S in setA setB; do mk "$BENCH/$S/probe.yaml" "$PROT"; done
mk "$BENCH/setA/mate1.yaml" "${PROT:0:104}"
mk "$BENCH/setA/mate2.yaml" "${PROT:0:102}"
mk "$BENCH/setA/mate3.yaml" "${PROT:0:100}"
mk "$BENCH/setB/mate1.yaml" "${PROT:0:98}"
mk "$BENCH/setB/mate2.yaml" "${PROT:0:96}"
mk "$BENCH/setB/mate3.yaml" "${PROT:0:94}"

export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 BOLTZ_KIT_KEEP=1
export BOLTZ_KIT_MODE=fast BOLTZ_KIT_CODE=$BASE/bz-potbatch
export BOLTZ_BATCH_SIZE=4 BOLTZ_BATCH_POTENTIALS=1

for S in setA setB; do
  OUT=$BENCH/out_$S; mkdir -p "$OUT"
  "$BOLTZ" predict "$BENCH/$S" --out_dir "$OUT" --output_format pdb --override --use_potentials \
      > "$BENCH/log_$S.txt" 2>&1
  echo "RESULT $S rc=$? pdbs=$(find $OUT -name '*.pdb' | wc -l)"
  grep -h 'BATCH plan' $OUT/.boltz_kit/out_0/pred_worker.log 2>/dev/null
done
echo "=== done ==="
