#!/bin/bash
# ---
# link_existing_hdf5.sh — Link pre-built HDF5 data into the project
# ---
# Symlinks (or copies) HDF5 files from a previous build into datasets/.
# Handles both single-file and chunked HDF5 layouts.
#
# Usage:
#   bash link_existing_hdf5.sh <source_datasets_dir> [modelnet10|modelnet40|all] [--copy]
#
# Example:
#   bash link_existing_hdf5.sh ~/orion-claude/datasets all
# ---
set -e

if [ -z "$1" ]; then
    echo "Usage: bash link_existing_hdf5.sh <source_datasets_dir> [modelnet10|modelnet40|all] [--copy]"
    exit 1
fi

SOURCE_DIR="$(cd "$1" && pwd)"
DATASET="${2:-all}"
MODE="symlink"
[ "$3" = "--copy" ] && MODE="copy"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Link Existing HDF5 Data"
echo "  Source:  $SOURCE_DIR"
echo "  Target:  $(pwd)/datasets"
echo "  Mode:    $MODE"
echo "  Dataset: $DATASET"
echo ""

link_dataset() {
    local DSNAME="$1"
    local SRC="$SOURCE_DIR/$DSNAME/hdf5"
    local DST="datasets/$DSNAME/hdf5"

    if [ ! -d "$SRC" ]; then
        echo "  ERROR: $SRC not found"; return 1
    fi

    echo "$DSNAME ---"

    for subdir in train_allrot_shuffled test_singlerandrot test_allrot; do
        local SRC_SUB="$SRC/$subdir"
        local DST_SUB="$DST/$subdir"

        if [ ! -d "$SRC_SUB" ]; then
            echo "  SKIP: $SRC_SUB not found"; continue
        fi

        # Clean destination completely to avoid stale chunks
        rm -rf "$DST_SUB"
        mkdir -p "$DST_SUB"

        # Link/copy ALL data files (handles both single and chunked)
        local count=0
        for src_file in "$SRC_SUB"/*; do
            [ -f "$src_file" ] || continue
            local fname="$(basename "$src_file")"

            # Skip txt list files — we'll regenerate them
            [[ "$fname" == *.txt ]] && continue

            if [ "$MODE" = "copy" ]; then
                cp "$src_file" "$DST_SUB/$fname"
            else
                ln -s "$src_file" "$DST_SUB/$fname"
            fi
            count=$((count + 1))
        done

        # Build the .hdf5.txt list from what's actually in the directory
        # Determine base name (train or test)
        local base_name
        if echo "$subdir" | grep -q "train"; then base_name="train"; else base_name="test"; fi

        local txt_file="$DST_SUB/${base_name}.hdf5.txt"
        : > "$txt_file"  # empty it

        # List all HDF5 files in correct order (base first, then chunks)
        for f in $(ls "$DST_SUB" | grep -v '\.txt$' | sort); do
            echo "$f" >> "$txt_file"
        done

        local entries=$(wc -l < "$txt_file")
        echo "  $DSNAME/$subdir: $count files linked, $entries entries in txt"
    done

    echo "  $DSNAME ready."
    echo ""
}

if [ "$DATASET" = "modelnet10" ] || [ "$DATASET" = "all" ]; then
    link_dataset "modelnet10"
fi
if [ "$DATASET" = "modelnet40" ] || [ "$DATASET" = "all" ]; then
    link_dataset "modelnet40"
fi

echo "Done. You can now train with:"
echo "  bash run_experiment.sh modelnet10_basic"
echo "  bash run_experiment.sh modelnet40_fast"
