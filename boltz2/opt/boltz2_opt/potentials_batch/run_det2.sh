#!/bin/bash
# Block-diagonal patch: same-set reproducibility, batch-mate coupling,
# batched-vs-unbatched, and a remainder group of one.
set -u
BASE=/mnt/scratch/user/daviyang; BENCH=$BASE/bench_det2
BOLTZ=$BASE/uplifting-biomolecular-modeling/tools/boltz_compat/boltz
rm -rf "$BENCH"; mkdir -p "$BENCH"
cp -r $BASE/bench_leak/setA "$BENCH/setA"
cp -r $BASE/bench_leak/setB "$BENCH/setB"
mkdir -p "$BENCH/setS"; cp "$BENCH/setA/probe.yaml" "$BENCH/setS/probe.yaml"   # probe alone
cp -r "$BENCH/setA" "$BENCH/setR"                                              # 5 inputs -> 4 + 1
PROT=AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR
CHOL='C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C'
printf 'version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: %s\n      msa: empty\n  - ligand:\n      id: B\n      smiles: '\''%s'\''\n' \
  "${PROT:0:106}" "$CHOL" > "$BENCH/setR/mate4.yaml"

export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 BOLTZ_KIT_KEEP=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export BOLTZ_KIT_MODE=fast BOLTZ_KIT_CODE=$BASE/bz-potbatch
export BOLTZ_BATCH_POTENTIALS=1 POTB_DETERMINISTIC=1

run () {   # tag  src  batch_size
  local OUT=$BENCH/out_$1; rm -rf "$OUT"; mkdir -p "$OUT"
  BOLTZ_BATCH_SIZE=$3 "$BOLTZ" predict "$BENCH/$2" --out_dir "$OUT" \
      --output_format pdb --override --use_potentials > "$BENCH/log_$1.txt" 2>&1
  local rc=$?
  echo "RESULT $1 rc=$rc pdbs=$(find $OUT -path '*predictions*' -name '*.pdb' | wc -l) fallbacks=$(grep -hc 'fell back' $OUT/.boltz_kit/out_0/pred_worker.log 2>/dev/null || echo 0)"
  grep -h 'BATCH plan' $OUT/.boltz_kit/out_0/pred_worker.log 2>/dev/null
}

run detA1 setA 4
run detA2 setA 4
run detB1 setB 4
run detS  setS 1
run detR  setR 4
echo "=== done ==="
