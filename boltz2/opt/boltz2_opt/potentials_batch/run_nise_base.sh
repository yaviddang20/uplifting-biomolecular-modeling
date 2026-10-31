#!/bin/bash
set -u
BASE=/mnt/scratch/user/daviyang
cd $BASE/NISE
OUT=$BASE/nise_base; rm -rf $OUT; mkdir -p $OUT

# kit: batched guided inference through the compat wrapper
export BOLTZ_KIT_MODE=fast
export BOLTZ_KIT_CODE=$BASE/bz-potbatch
export BOLTZ_BATCH_POTENTIALS=0
export BOLTZ_BATCH_SIZE=1
export BOLTZ_KIT_KEEP=1
export OMP_NUM_THREADS=8
export WANDB_MODE=disabled
# NOTE: no CUDA_VISIBLE_DEVICES here -- NISE sets it for the boltz child itself,
# and LASErMPNN needs to see cuda:3 in the parent.

echo "start $(date)"
.venv/bin/python run_nise_boltz2x_cli.py     --input-pdb example_pdbs/02_apex_NISE_input-pose_00-seq_0980_model_0_rank_01.pdb     --smiles 'COC1=CC=C(C=C1)N2C3=C(CCN(C3=O)C4=CC=C(C=C4)N5CCCCC5=O)C(=N2)C(=O)N'     --output-dir $OUT/out     --boltz-executable $BASE/uplifting-biomolecular-modeling/tools/boltz_compat/boltz     --boltz-devices cuda:3     --num-iterations 2     --num-top-backbones-per-round 2     --sequences-sampled-per-backbone 16     > $OUT/nise.log 2>&1
echo "rc=$? end $(date)"
echo '--- batch plans seen ---'
grep -rh 'BATCH plan' $OUT/out 2>/dev/null | head -4
echo '--- fallbacks ---'
grep -rhc 'fell back' $OUT/out 2>/dev/null | paste -sd+ | bc
echo '--- nise tail ---'
tail -20 $OUT/nise.log
echo '=== done ==='
