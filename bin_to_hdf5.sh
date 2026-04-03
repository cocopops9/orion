#!/bin/bash
# ---
# Convert binary voxel files to HDF5 (bin -> HDF5 only)
#
# Usage:
#   bash bin_to_hdf5.sh modelnet10
#   bash bin_to_hdf5.sh modelnet40
#   bash bin_to_hdf5.sh all
#
# Prerequisite: binary files must exist in datasets/{dataset}_bin/
# (produced by build_modelnet{10,40}.sh bin)
# ---
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TARGET="${1:-all}"

convert_dataset() {
    local DSNAME="$1"
    local NUM="$2"
    local PLAN="$3"

    local BIN_DIR="datasets/${DSNAME}_bin"
    local HDF5_DST="datasets/${DSNAME}/hdf5"
    local PLAN_NAME="${PLAN%.txt}"

    echo "bin -> HDF5: $DSNAME"

    # Check bins exist
    BIN_COUNT=$(find "$BIN_DIR" -name "*.bin" 2>/dev/null | wc -l)
    if [ "$BIN_COUNT" -eq 0 ]; then
        echo "  ERROR: No .bin files in $BIN_DIR"
        echo "  Run: bash build_${DSNAME}.sh bin"
        return 1
    fi
    echo "  Found $BIN_COUNT binary files."

    # Run HDF5 conversion
    # We need a dummy off-path since --step hdf5 doesn't use it, but argparse requires it
    local OFF_PATH="datasets/ModelNet${NUM}"
    [ ! -d "$OFF_PATH" ] && OFF_PATH="datasets/modelnet${NUM}_auto_aligned"
    [ ! -d "$OFF_PATH" ] && OFF_PATH="dummy"

    python3 -m orion.data.prepare_modelnet \
        --dataset "$DSNAME" --off-path "$OFF_PATH" \
        --output-path datasets --pose-plan "data_preparation/poseplans/${PLAN}" --step hdf5

    # Move to expected structure
    local HDF5_SRC="${BIN_DIR}/${PLAN_NAME}/hdf5"
    if [ -d "$HDF5_SRC" ]; then
        mkdir -p "$HDF5_DST"
        for d in "$HDF5_SRC"/*/; do
            name="$(basename "$d")"
            rm -rf "$HDF5_DST/$name" 2>/dev/null
            mv "$d" "$HDF5_DST/$name"
        done
        echo "  HDF5 -> $HDF5_DST/"
    fi

    # Fix txt to relative paths
    find "$HDF5_DST" -name "*.hdf5.txt" -exec sh -c \
        'tmp="$1.tmp"; while IFS= read -r l; do basename "$l"; done < "$1" > "$tmp"; mv "$tmp" "$1"' _ {} \;

    # Verify
    echo "  Verifying..."
    python3 -c "
import h5py, os
base = '$HDF5_DST'
for rel in ['train_allrot_shuffled/train.hdf5','test_singlerandrot/test.hdf5','test_allrot/test.hdf5']:
    p = os.path.join(base, rel)
    if os.path.exists(p):
        with h5py.File(p,'r') as f: print(f'    OK: {rel} -> {f[\"data\"].shape[0]} samples')
    else: print(f'    MISSING: {rel}')
"
    echo "  $DSNAME done."
}

if [ "$TARGET" = "modelnet10" ] || [ "$TARGET" = "all" ]; then
    convert_dataset "modelnet10" "10" "poseplan_MN10.txt"
fi

if [ "$TARGET" = "modelnet40" ] || [ "$TARGET" = "all" ]; then
    convert_dataset "modelnet40" "40" "poseplan_MN40.txt"
fi

echo "Done."
