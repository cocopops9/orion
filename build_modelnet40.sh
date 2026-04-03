#!/bin/bash
# ---
# Build ModelNet40: download aligned data -> voxelize -> bin -> HDF5
#
# Usage: bash build_modelnet40.sh [step] [alignment]
#   Steps:     all | download | voxelize | bin | hdf5 | verify
#   Alignment: auto (default) | manual
#
# ModelNet40 requires orientation-aligned OFF files from the ORION authors:
#   auto   = modelnet40_auto_aligned   (paper default, Table 2)
#   manual = modelnet40_manually_aligned
# ---
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STEP="${1:-all}"
ALIGN="${2:-auto}"

if [ "$ALIGN" = "manual" ]; then
    OFF_PATH="datasets/modelnet40_manually_aligned"
    TAR_URL="https://lmb.informatik.uni-freiburg.de/resources/datasets/ORION/modelnet40_manually_aligned.tar"
    TAR_FILE="datasets/modelnet40_manually_aligned.tar"
else
    OFF_PATH="datasets/modelnet40_auto_aligned"
    TAR_URL="https://lmb.informatik.uni-freiburg.de/resources/datasets/ORION/modelnet40_auto_aligned.tar"
    TAR_FILE="datasets/modelnet40_auto_aligned.tar"
fi

POSEPLAN="data_preparation/poseplans/poseplan_MN40.txt"
OUTPUT="datasets"
HDF5_DST="datasets/modelnet40/hdf5"
PLAN_NAME="poseplan_MN40"

# Validate step name
VALID_STEPS="all download voxelize bin hdf5 verify"
if ! echo "$VALID_STEPS" | grep -qw "$STEP"; then
    echo "ERROR: Unknown step '$STEP'"
    echo "Valid steps: $VALID_STEPS"
    exit 1
fi

echo "Build ModelNet40 ($ALIGN-aligned) -- step: $STEP"

# --- download ---
if [ "$STEP" = "all" ] || [ "$STEP" = "download" ]; then
    echo "[download] Getting $ALIGN-aligned ModelNet40..."
    if [ -d "$OFF_PATH" ]; then
        echo "  Already exists at $OFF_PATH"
    else
        mkdir -p datasets
        echo "  Downloading from $TAR_URL ..."
        wget -O "$TAR_FILE" "$TAR_URL"
        echo "  Extracting..."
        mkdir -p "$OFF_PATH"
        tar -xf "$TAR_FILE" -C "$OFF_PATH/"
        rm -f "$TAR_FILE"
        if [ ! -d "$OFF_PATH" ]; then
            echo "  ERROR: Expected directory $OFF_PATH not found after extraction."
            echo "  Check tar contents: tar -tf $TAR_FILE | head"
            exit 1
        fi
    fi
fi

# --- voxelize ---
if [ "$STEP" = "all" ] || [ "$STEP" = "voxelize" ]; then
    if [ ! -d "$OFF_PATH" ]; then
        echo "ERROR: $OFF_PATH not found. Run: bash build_modelnet40.sh download"
        exit 1
    fi
    echo "[voxelize] OFF -> voxel grids (this takes HOURS for MN40)..."
    python3 -m orion.data.prepare_modelnet \
        --dataset modelnet40 --off-path "$OFF_PATH" \
        --output-path "$OUTPUT" --pose-plan "$POSEPLAN" --step voxelize
fi

# --- bin ---
if [ "$STEP" = "all" ] || [ "$STEP" = "bin" ]; then
    echo "[bin] Voxel grids -> binary..."
    python3 -m orion.data.prepare_modelnet \
        --dataset modelnet40 --off-path "$OFF_PATH" \
        --output-path "$OUTPUT" --pose-plan "$POSEPLAN" --step bin
fi

# --- hdf5 ---
if [ "$STEP" = "all" ] || [ "$STEP" = "hdf5" ]; then
    echo "[hdf5] Binary -> HDF5..."
    python3 -m orion.data.prepare_modelnet \
        --dataset modelnet40 --off-path "$OFF_PATH" \
        --output-path "$OUTPUT" --pose-plan "$POSEPLAN" --step hdf5

    HDF5_SRC="datasets/modelnet40_bin/${PLAN_NAME}/hdf5"
    if [ -d "$HDF5_SRC" ]; then
        mkdir -p "$HDF5_DST"
        for d in "$HDF5_SRC"/*/; do
            name="$(basename "$d")"
            rm -rf "$HDF5_DST/$name" 2>/dev/null
            mv "$d" "$HDF5_DST/$name"
        done
    fi
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
print('  ModelNet40 OK!')
"
fi
echo "Done."
