#!/bin/bash
# Batch-size ladder for the BLOCK-DIAGONAL reduction. Same fixed-length inputs
# as the original ladder so the rungs are directly comparable; extended past 128
# because memory should now be linear in B rather than quadratic.
set -u
BASE=/mnt/scratch/user/daviyang
BENCH=$BASE/bench_ladder2; IN=$BENCH/inputs
mkdir -p "$IN"
BOLTZ=$BASE/uplifting-biomolecular-modeling/tools/boltz_compat/boltz
PROT=AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR
APX='COC1=CC=C(C=C1)N2C3=C(CCN(C3=O)C4=CC=C(C=C4)N5CCCCC5=O)C(=N2)C(=O)N'
if [ "$(ls $IN 2>/dev/null | wc -l)" != 256 ]; then
  rm -f $IN/*.yaml
  for i in $(seq 0 255); do
    p=$(( i % ${#PROT} ))
    s="${PROT:0:$p}A${PROT:$((p+1))}"
    printf 'version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: %s\n      msa: empty\n  - ligand:\n      id: B\n      smiles: '\''%s'\''\n' "$s" "$APX" > "$IN/d$(printf %03d $i).yaml"
  done
fi
echo "inputs: $(ls $IN | wc -l), length ${#PROT}"

export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 BOLTZ_KIT_KEEP=1
export BOLTZ_KIT_MODE=fast BOLTZ_KIT_CODE=$BASE/bz-potbatch BOLTZ_BATCH_POTENTIALS=1

for bs in 16 32 48 64 96 128 192 256; do
  OUT=$BENCH/out_bs${bs}; rm -rf "$OUT"; mkdir -p "$OUT"
  SUB=$BENCH/sub_$bs; rm -rf "$SUB"; mkdir -p "$SUB"
  ls "$IN" | head -$bs | while read f; do ln -s "$IN/$f" "$SUB/$f"; done
  T0=$(date +%s.%N)
  BOLTZ_BATCH_SIZE=$bs "$BOLTZ" predict "$SUB" --out_dir "$OUT" --output_format pdb --override --use_potentials \
      > "$BENCH/log_bs${bs}.txt" 2>&1
  RC=$?; T1=$(date +%s.%N)
  L=$(find "$OUT" -name pred_worker.log | head -1)
  S=$(grep -ho 'sampler_s=[0-9.]*' "$L" 2>/dev/null | cut -d= -f2 | paste -sd+ | bc)
  PK=$(grep -ho 'alloc_gib=[0-9.]*' "$L" 2>/dev/null | cut -d= -f2 | sort -g | tail -1)
  FB=$(grep -c 'fell back' "$L" 2>/dev/null); FB=${FB:-0}
  N=$(find "$OUT" -path '*predictions*' -name '*_model_0.pdb' | wc -l)
  PER=$(echo "scale=3; ${S:-0} / $bs" | bc 2>/dev/null)
  printf 'LADDER2 bs=%-3s rc=%s pdbs=%-3s wall=%.1f sampler=%-8s per_rec=%-7s peak_gib=%-6s fallbacks=%s\n' \
      "$bs" "$RC" "$N" "$(echo "$T1 - $T0" | bc)" "${S:-NA}" "${PER:-NA}" "${PK:-NA}" "$FB"
  if [ "$RC" != 0 ] || [ "$N" != "$bs" ] || [ "${FB:-0}" != 0 ]; then
    echo "LADDER2: stopping at bs=$bs"; grep -m2 -i 'out of memory\|OutOfMemory\|fell back' "$L" 2>/dev/null | head -2; break
  fi
done
echo '=== done ==='
