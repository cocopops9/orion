#!/bin/bash
# ---
# Build ModelNet10: download -> voxelize -> bin -> HDF5
# Usage: bash build_modelnet10.sh [step]
#   Steps: all | download | voxelize | bin | hdf5 | verify
# ---
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STEP="${1:-all}"
OFF_PATH="datasets/ModelNet10"
POSEPLAN="data_preparation/poseplans/poseplan_MN10.txt"
OUTPUT="datasets"
HDF5_DST="datasets/modelnet10/hdf5"
PLAN_NAME="poseplan_MN10"

echo "Build ModelNet10 -- step: $STEP"

# Validate step name
VALID_STEPS="all download voxelize bin hdf5 verify"
if ! echo "$VALID_STEPS" | grep -qw "$STEP"; then
    echo "ERROR: Unknown step '$STEP'"
    echo "Valid steps: $VALID_STEPS"
    exit 1
fi

# --- download ---
if [ "$STEP" = "all" ] || [ "$STEP" = "download" ]; then
    echo "[download] Downloading ModelNet10..."
    if [ -d "$OFF_PATH" ]; then echo "  Already exists."; else bash datasets/get_modelnet10.sh; fi
fi

# --- voxelize ---
if [ "$STEP" = "all" ] || [ "$STEP" = "voxelize" ]; then
    echo "[voxelize] OFF -> voxel grids (~20 min)..."
    python3 -m orion.data.prepare_modelnet \
        --dataset modelnet10 --off-path "$OFF_PATH" \
        --output-path "$OUTPUT" --pose-plan "$POSEPLAN" --step voxelize
fi

# --- bin ---
if [ "$STEP" = "all" ] || [ "$STEP" = "bin" ]; then
    echo "[bin] Voxel grids -> binary..."
    python3 -m orion.data.prepare_modelnet \
        --dataset modelnet10 --off-path "$OFF_PATH" \
        --output-path "$OUTPUT" --pose-plan "$POSEPLAN" --step bin
fi

# --- hdf5 ---
if [ "$STEP" = "all" ] || [ "$STEP" = "hdf5" ]; then
    echo "[hdf5] Binary -> HDF5..."
    python3 -m orion.data.prepare_modelnet \
        --dataset modelnet10 --off-path "$OFF_PATH" \
        --output-path "$OUTPUT" --pose-plan "$POSEPLAN" --step hdf5

    # Move HDF5 to expected location
    HDF5_SRC="datasets/modelnet10_bin/${PLAN_NAME}/hdf5"
    if [ -d "$HDF5_SRC" ]; then
        mkdir -p "$HDF5_DST"
        for d in "$HDF5_SRC"/*/; do
            name="$(basename "$d")"
            rm -rf "$HDF5_DST/$name" 2>/dev/null
            mv "$d" "$HDF5_DST/$name"
        done
    fi
    # Fix txt to relative paths
    find "$HDF5_DST" -name "*.hdf5.txt" -exec sh -c \
        'tmp="$1.tmp"; while IFS= read -r l; do basename "$l"; done < "$1" > "$tmp"; mv "$tmp" "$1"' _ {} \;
    echo "  HDF5 -> $HDF5_DST/"
fi

# --- verify ---
if [ "$STEP" = "all" ] || [ "$STEP" = "verify" ]; then
    echo "[verify] Checking..."
    python3 -c "
import h5py, os, sys
base = '$HDF5_DST'
for rel in ['train_allrot_shuffled/train.hdf5','test_singlerandrot/test.hdf5','test_allrot/test.hdf5']:
    p = os.path.join(base, rel)
    if os.path.exists(p):
        with h5py.File(p,'r') as f: print(f'  OK: {rel} -> {f[\"data\"].shape[0]} samples')
    else: print(f'  MISSING: {rel}'); sys.exit(1)
print('  ModelNet10 OK!')
"
fi
echo "Done."
