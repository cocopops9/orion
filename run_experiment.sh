#!/bin/bash
# ---
# Train and evaluate ORION / PointNet models
#
# Usage:
#   bash run_experiment.sh <experiment>
#
# Experiments:
#   modelnet10_basic       ORION Basic on MN10
#   modelnet10_fast        ORION Fast on MN10
#   modelnet10_pointnet    PointNet on MN10
#   modelnet40_basic       ORION Basic on MN40
#   modelnet40_fast        ORION Fast on MN40
#   modelnet40_extended    ORION Extended on MN40
#   modelnet40_pointnet    PointNet on MN40
#   all_mn10               All MN10 experiments
#   all_mn40               All MN40 experiments
#   all                    Everything
#
# Prerequisite: HDF5 data in datasets/ (via build or link scripts)
# ---
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP="${1:?Usage: bash run_experiment.sh <experiment>}"

# Check HDF5 data exists for a dataset
check_data() {
    local ds="$1"
    local train="datasets/${ds}/hdf5/train_allrot_shuffled/train.hdf5"
    local test="datasets/${ds}/hdf5/test_singlerandrot/test.hdf5"
    if [ ! -e "$train" ] || [ ! -e "$test" ]; then
        echo "ERROR: ${ds} HDF5 data not found."
        echo "  Run: bash link_existing_hdf5.sh <source_datasets_dir> ${ds}"
        echo "  Or:  bash build_${ds}.sh"
        exit 1
    fi
}

# Train an ORION model (basic/fast/extended)
train_orion() {
    local config="$1"
    local name="$2"
    echo ""
    echo "Training: $name"
    python3 train.py --config "$config"
    echo "  $name training complete."
}

# Train a PointNet model
train_pointnet() {
    local config="$1"
    local name="$2"
    echo ""
    echo "Training: $name"
    python3 train_pointnet.py --config "$config"
    echo "  $name training complete."
}

# Evaluate ORION model (standard + voting)
eval_orion() {
    local config="$1"
    local save_dir="$2"
    local name="$3"
    local ckpt="$save_dir/best_model.pth"
    if [ ! -f "$ckpt" ]; then
        echo "  SKIP eval: no checkpoint at $ckpt"
        return
    fi
    echo ""
    echo "Evaluate: $name (standard)"
    python3 evaluate.py --config "$config" --checkpoint "$ckpt"

    # Try voting if test_allrot exists
    local ds=$(python3 -c "import yaml; c=yaml.safe_load(open('$config')); print(c['data']['val_hdf5_list'].split('/')[1])")
    local allrot_txt="datasets/${ds}/hdf5/test_allrot/test.hdf5.txt"
    if [ -e "$allrot_txt" ]; then
        echo ""
        echo "Evaluate: $name (voting)"
        python3 evaluate.py --config "$config" --checkpoint "$ckpt" --voting --test_allrot "$allrot_txt"
    fi
}

# Evaluate PointNet model
eval_pointnet() {
    local config="$1"
    local save_dir="$2"
    local name="$3"
    local ckpt="$save_dir/best_model.pth"
    if [ ! -f "$ckpt" ]; then
        echo "  SKIP eval: no checkpoint at $ckpt"
        return
    fi
    local ds=$(python3 -c "import yaml; c=yaml.safe_load(open('$config')); print(c['data']['val_hdf5_list'].split('/')[1])")
    local allrot_txt="datasets/${ds}/hdf5/test_allrot/test.hdf5.txt"

    echo ""
    echo "Evaluate: $name"
    if [ -e "$allrot_txt" ]; then
        python3 evaluate_pointnet.py --config "$config" --checkpoint "$ckpt" --test_allrot "$allrot_txt"
    else
        echo "  WARNING: test_allrot not found, skipping voting evaluation"
    fi
}

# Run a full experiment (train + eval)
run() {
    local config="$1"
    local save_dir="$2"
    local name="$3"
    local kind="$4"  # orion or pointnet

    if [ "$kind" = "pointnet" ]; then
        train_pointnet "$config" "$name"
        eval_pointnet "$config" "$save_dir" "$name"
    else
        train_orion "$config" "$name"
        eval_orion "$config" "$save_dir" "$name"
    fi
}

# --- Dispatch ---
case "$EXP" in
    modelnet10_basic)
        check_data modelnet10
        run configs/modelnet10_basic.yaml checkpoints/modelnet10_basic "MN10 ORION Basic" orion ;;
    modelnet10_fast)
        check_data modelnet10
        run configs/modelnet10_fast.yaml checkpoints/modelnet10_fast "MN10 ORION Fast" orion ;;
    modelnet10_pointnet)
        check_data modelnet10
        run configs/modelnet10_pointnet.yaml checkpoints/modelnet10_pointnet "MN10 PointNet" pointnet ;;
    modelnet40_basic)
        check_data modelnet40
        run configs/modelnet40_basic.yaml checkpoints/modelnet40_basic "MN40 ORION Basic" orion ;;
    modelnet40_fast)
        check_data modelnet40
        run configs/modelnet40_fast.yaml checkpoints/modelnet40_fast "MN40 ORION Fast" orion ;;
    modelnet40_extended)
        check_data modelnet40
        run configs/modelnet40_extended.yaml checkpoints/modelnet40_extended "MN40 ORION Extended" orion ;;
    modelnet40_pointnet)
        check_data modelnet40
        run configs/modelnet40_pointnet.yaml checkpoints/modelnet40_pointnet "MN40 PointNet" pointnet ;;
    all_mn10)
        check_data modelnet10
        run configs/modelnet10_basic.yaml checkpoints/modelnet10_basic "MN10 ORION Basic" orion
        run configs/modelnet10_fast.yaml checkpoints/modelnet10_fast "MN10 ORION Fast" orion
        run configs/modelnet10_pointnet.yaml checkpoints/modelnet10_pointnet "MN10 PointNet" pointnet ;;
    all_mn40)
        check_data modelnet40
        run configs/modelnet40_basic.yaml checkpoints/modelnet40_basic "MN40 ORION Basic" orion
        run configs/modelnet40_fast.yaml checkpoints/modelnet40_fast "MN40 ORION Fast" orion
        run configs/modelnet40_extended.yaml checkpoints/modelnet40_extended "MN40 ORION Extended" orion
        run configs/modelnet40_pointnet.yaml checkpoints/modelnet40_pointnet "MN40 PointNet" pointnet ;;
    all)
        check_data modelnet10
        check_data modelnet40
        run configs/modelnet10_basic.yaml checkpoints/modelnet10_basic "MN10 ORION Basic" orion
        run configs/modelnet10_fast.yaml checkpoints/modelnet10_fast "MN10 ORION Fast" orion
        run configs/modelnet10_pointnet.yaml checkpoints/modelnet10_pointnet "MN10 PointNet" pointnet
        run configs/modelnet40_basic.yaml checkpoints/modelnet40_basic "MN40 ORION Basic" orion
        run configs/modelnet40_fast.yaml checkpoints/modelnet40_fast "MN40 ORION Fast" orion
        run configs/modelnet40_extended.yaml checkpoints/modelnet40_extended "MN40 ORION Extended" orion
        run configs/modelnet40_pointnet.yaml checkpoints/modelnet40_pointnet "MN40 PointNet" pointnet ;;
    *)
        echo "Unknown experiment: $EXP"
        echo "Options: modelnet10_basic, modelnet10_fast, modelnet10_pointnet,"
        echo "         modelnet40_basic, modelnet40_fast, modelnet40_extended, modelnet40_pointnet,"
        echo "         all_mn10, all_mn40, all"
        exit 1 ;;
esac

echo ""
echo "Experiment '$EXP' complete."
