# Shortcuts for the kit images. Load with:  source <repo>/kits.sh   (UCSF CoreHPC default paths; set KIT_BASE elsewhere)
# Each function = the README's route-B `kit` for that kit:  boltz2_kit <command> ...   /   af3_kit <command> ...
KIT_BASE="${KIT_BASE:-/mnt/scratch/user/daviyang}"   # where the repo, weights/ and kit_out/ live; export KIT_BASE to relocate
KITS="$KIT_BASE/uplifting-biomolecular-modeling"
KITW="$KIT_BASE/weights"          # model weights, one sub-folder per kit
KITO="$KIT_BASE/kit_out"          # outputs + compile caches, one sub-folder per kit
mkdir -p "$KITW"/boltz2 "$KITW"/af3_torch "$KITO"/boltz2/out "$KITO"/boltz2/jit "$KITO"/af3_torch/out "$KITO"/af3_torch/jit

# --nv only when a GPU is visible (login node has none; downloading weights doesn't need one)
_kit_nv() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && echo --nv; }

boltz2_kit() {
    BOLTZ_CACHE=/weights/boltz2 MODEL_OPT_JIT_ROOT="$KITO/boltz2/jit" \
    apptainer run $(_kit_nv) --bind "$KITW/boltz2":/weights/boltz2 --bind "$KITO/boltz2/out":/kit/boltz2/out \
        --bind "$KIT_BASE" "$KITS/boltz2.sif" "$@"
}

af3_kit() {
    AF3_TORCH_PARAMS_DIR=/weights/af3_torch MODEL_OPT_JIT_ROOT="$KITO/af3_torch/jit" \
    apptainer run $(_kit_nv) --bind "$KITW/af3_torch":/weights/af3_torch --bind "$KITO/af3_torch/out":/kit/af3_torch/out \
        --bind "$KIT_BASE" "$KITS/af3_torch.sif" "$@"
}

# --- boltzgen / complexa (added later)
mkdir -p "$KITW"/boltzgen "$KITW"/complexa "$KITO"/boltzgen/runs "$KITO"/boltzgen/jit "$KITO"/complexa/out "$KITO"/complexa/jit

boltzgen_kit() {   # results: $KITO/boltzgen/runs  (inside: runs/...)
    BOLTZGEN_CACHE=/weights/boltzgen MODEL_OPT_JIT_ROOT="$KITO/boltzgen/jit" \
    apptainer run $(_kit_nv) --bind "$KITW/boltzgen":/weights/boltzgen --bind "$KITO/boltzgen/runs":/kit/boltzgen/runs \
        --bind "$KIT_BASE" "$KITS/boltzgen.sif" "$@"
}

complexa_kit() {   # results: $KITO/complexa/out  (inside: out/...)
    LOCAL_CODE_PATH=/opt/pc CKPT_PATH=/weights/complexa MODEL_OPT_JIT_ROOT="$KITO/complexa/jit" \
    apptainer run $(_kit_nv) --bind "$KITW/complexa":/weights/complexa --bind "$KITO/complexa/out":/kit/complexa/out \
        --bind "$KIT_BASE" "$KITS/complexa.sif" "$@"
}
complexa_weights() {   # complexa's weights are fetched by a separate command (README route B)
    apptainer exec --bind "$KITW/complexa":/weights/complexa "$KITS/complexa.sif" python -m complexa_opt.weights /weights/complexa
}

# --- rfdiffusion3 (image: rfdiffusion3.sif, built via tools/rfd3_apex)
mkdir -p "$KITW"/rfd3 "$KITO"/rfdiffusion3/out "$KITO"/rfdiffusion3/jit

rfd3_kit() {   # results: $KITO/rfdiffusion3/out  (inside: out/...)
    RFD3_CKPT=/weights/rfd3/rfd3_latest.ckpt MODEL_OPT_JIT_ROOT="$KITO/rfdiffusion3/jit" \
    apptainer run $(_kit_nv) --bind "$KITW/rfd3":/weights/rfd3 --bind "$KITO/rfdiffusion3/out":/kit/rfdiffusion3/out \
        --bind "$KIT_BASE" "$KITS/rfdiffusion3.sif" "$@"
}
