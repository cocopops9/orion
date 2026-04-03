#!/usr/bin/env python
"""
Complete ModelNet40 BIN-to-HDF5 converter for ORION .

This script creates the final HDF5 training/test files from intermediate data in temp_parallel_outputs/:

  - For classes WITH modelnet40_bin/: reads the 36x36x36 .bin files directly
  - For classes WITHOUT modelnet40_bin/: reads 30x30x30 .npy voxels, pads to 36x36x36

Pose labels use ORION cumulative class offsets from poseplan_MN40.txt.
ALL 12 rotations per model are included in training (matching original MATLAB).
The poseplan controls the pose label range per class via modulo:
  - nrot=12: rotations 1-12 map to pose offsets 0-11
  - nrot=3:  rotations 1-12 map to pose offsets 0,1,2,0,1,2,0,1,2,0,1,2
  - nrot=1:  all rotations map to pose offset 0

This is what the original MATLAB does (https://github.com/lmb-freiburg/orion):
  pose_label = cr(class_label+1) + mod(rotation, nrot(class_label+1))
"""
#I had problems with voxelization for mn40 for bin->hdf5 so I created this script to do do bin->hdf5 conversion, without having to do again voxelization

import os
import sys
import random
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Order and rotation counts from poseplan_MN40.txt
CLASSES_AND_NROT = [
    ("airplane",    12),
    ("bathtub",      3),
    ("bed",         12),
    ("bench",        1),
    ("bookshelf",   12),
    ("bottle",       1),
    ("bowl",         1),
    ("car",         12),
    ("chair",       12),
    ("cone",         1),
    ("cup",          1),
    ("curtain",      1),
    ("desk",        12),
    ("door",         1),
    ("dresser",      3),
    ("flower_pot",   1),
    ("glass_box",    1),
    ("guitar",      12),
    ("keyboard",     1),
    ("lamp",         1),
    ("laptop",      12),
    ("mantel",      12),
    ("monitor",      3),
    ("night_stand",  1),
    ("person",       1),
    ("piano",       12),
    ("plant",        1),
    ("radio",        1),
    ("range_hood",  12),
    ("sink",         1),
    ("sofa",        12),
    ("stairs",       1),
    ("stool",        1),
    ("table",        1),
    ("tent",         1),
    ("toilet",      12),
    ("tv_stand",     1),
    ("vase",         1),
    ("wardrobe",     1),
    ("xbox",         1),
]

CLASSES = [c for c, _ in CLASSES_AND_NROT]
NROT = [n for _, n in CLASSES_AND_NROT]
TOTAL_ORIENTATIONS = sum(NROT)  # 189
assert TOTAL_ORIENTATIONS == 189, f"Expected 189, got {TOTAL_ORIENTATIONS}"
assert len(CLASSES) == 40

# Cumulative pose offsets
POSE_OFFSET = np.zeros(len(NROT), dtype=int)
cumsum = np.cumsum(NROT)
POSE_OFFSET[1:] = cumsum[:-1]


def read_binary_voxel(filepath):
    """Read ORION binary voxel file."""
    with open(filepath, 'rb') as f:
        dims = np.fromfile(f, dtype=np.uint16, count=3)
        nvoxels = int(np.prod(dims))
        voxels = np.fromfile(f, dtype=np.uint8, count=nvoxels)
        grid = voxels.reshape(dims, order='F')
    return grid.astype(np.float32)


def read_npy_voxel_and_pad(filepath, pad=3):
    """Read 30x30x30 .npy voxel and pad to 36x36x36."""
    grid = np.load(filepath).astype(np.float32)
    if grid.shape == (30, 30, 30):
        grid = np.pad(grid, pad, mode='constant', constant_values=0)
    return grid


def compute_pose_label(class_idx, rotation_0based):
    """
    Compute ORION cumulative pose label.
    Matches: pose_label = cr(class_label+1) + mod(rotation, nrot(class_label+1))
    """
    nrot = NROT[class_idx]
    return int(POSE_OFFSET[class_idx]) + (rotation_0based % nrot)


def collect_samples_for_class(class_idx, class_name, split, temp_dir):
    """
    Collect all voxel samples for one class/split combination.
    Returns list of (grid_36x36x36, class_idx, pose_label) tuples.
    """
    samples = []

    # Try BIN files first (completed classes)
    bin_dir = temp_dir / f"class_{class_name}" / "modelnet40_bin" / "classes" / class_name / "30" / split
    if bin_dir.exists():
        bin_files = sorted(bin_dir.glob("*.bin"))
        for bf in bin_files:
            parts = bf.stem.split('_')
            rotation = int(parts[-1])  # 1-based
            rotation_0based = rotation - 1
            pose_label = compute_pose_label(class_idx, rotation_0based)
            grid = read_binary_voxel(bf)
            model_key = '_'.join(parts[:-1])
            samples.append((grid, class_idx, pose_label, model_key, rotation))
        return samples

    # Fall back to NPY files (incomplete classes)
    npy_dir = temp_dir / f"class_{class_name}" / "modelnet40_voxelized" / class_name / "30" / split
    if npy_dir.exists():
        npy_files = sorted(npy_dir.glob("*.npy"))
        for nf in npy_files:
            parts = nf.stem.split('_')
            rotation = int(parts[-1])  # 1-based
            rotation_0based = rotation - 1
            pose_label = compute_pose_label(class_idx, rotation_0based)
            grid = read_npy_voxel_and_pad(nf)
            model_key = '_'.join(parts[:-1])
            samples.append((grid, class_idx, pose_label, model_key, rotation))
        return samples

    return samples


def build_split(split_name, temp_dir, output_dir, shuffle=False, single_rand_rot=False, seed=42):
    """Build one HDF5 split from per-class data."""
    print(f"\n{'=' * 70}")
    print(f"Building: {split_name}")
    print(f"{'=' * 70}")

    # Determine which filesystem split to read
    if 'train' in split_name:
        fs_split = 'train'
    else:
        fs_split = 'test'

    all_samples = []
    missing_classes = []

    for class_idx, (class_name, nrot) in enumerate(CLASSES_AND_NROT):
        samples = collect_samples_for_class(class_idx, class_name, fs_split, temp_dir)

        if not samples:
            missing_classes.append(class_name)
            print(f"  WARNING: {class_name} has no data for {fs_split}")
            continue

        # For test_singlerandrot: pick one random rotation per model
        if single_rand_rot:
            rng = random.Random(seed + class_idx)
            model_groups = {}
            for s in samples:
                mk = s[3]  # model_key
                if mk not in model_groups:
                    model_groups[mk] = []
                model_groups[mk].append(s)
            samples = [rng.choice(group) for group in sorted(model_groups.values(), key=lambda g: g[0][3])]

        all_samples.extend(samples)
        pose_labels = [s[2] for s in samples]
        expected_start = POSE_OFFSET[class_idx]
        expected_end = POSE_OFFSET[class_idx] + nrot - 1
        print(f"  Class {class_idx:2d} ({class_name:15s}): {len(samples):5d} samples, "
              f"poses [{min(pose_labels)}-{max(pose_labels)}] "
              f"(expected [{expected_start}-{expected_end}])")

    if not all_samples:
        print("  No samples found!")
        return

    if shuffle:
        random.seed(seed)
        random.shuffle(all_samples)

    total = len(all_samples)
    print(f"\n  Total samples: {total}")
    if missing_classes:
        print(f"  Missing classes: {missing_classes}")

    # Write HDF5
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    hdf5_name = "train.hdf5" if 'train' in split_name else "test.hdf5"
    hdf5_path = split_dir / hdf5_name

    print(f"  Writing {hdf5_path}...")
    with h5py.File(hdf5_path, 'w') as f:
        f.create_dataset('data', shape=(total, 36, 36, 36), dtype='float32', compression='gzip')
        f.create_dataset('label', shape=(total,), dtype='int64')
        f.create_dataset('label_pose', shape=(total,), dtype='int64')

        for i, (grid, cls, pose, _, _) in enumerate(tqdm(all_samples, desc="  Writing")):
            f['data'][i] = grid
            f['label'][i] = cls
            f['label_pose'][i] = pose

    size_mb = hdf5_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {hdf5_path} ({total} samples, {size_mb:.1f} MB)")

    # Write list file
    list_path = split_dir / (hdf5_name + ".txt")
    with open(list_path, 'w') as f:
        f.write(str(hdf5_path) + '\n')

    # Verification
    with h5py.File(hdf5_path, 'r') as f:
        labels = f['label'][:]
        poses = f['label_pose'][:]
        print(f"\n  Verification:")
        print(f"    Data shape: {f['data'].shape}")
        print(f"    Label range: [{labels.min()}, {labels.max()}]")
        print(f"    Pose range:  [{poses.min()}, {poses.max()}]")
        print(f"    Unique poses: {len(np.unique(poses))}")

        for c in range(40):
            mask = labels == c
            if mask.sum() == 0:
                print(f"    Class {c:2d} ({CLASSES[c]:15s}): MISSING")
                continue
            cp = np.unique(poses[mask])
            exp_start = POSE_OFFSET[c]
            exp_end = POSE_OFFSET[c] + NROT[c] - 1
            status = "OK" if cp.min() >= exp_start and cp.max() <= exp_end else "ERROR"
            print(f"    Class {c:2d} ({CLASSES[c]:15s}): {mask.sum():5d} samples, "
                  f"poses [{cp.min()}-{cp.max()}] [{status}]")


def main():
    temp_dir = Path("temp_parallel_outputs")
    output_dir = Path("datasets/modelnet40/hdf5")

    if not temp_dir.exists():
        print(f"Error: {temp_dir} not found")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print(f"ORION ModelNet40 HDF5 Builder")
    print(f"{'=' * 70}")
    print(f"Source:      {temp_dir}")
    print(f"Output:      {output_dir}")
    print(f"Orientations: {TOTAL_ORIENTATIONS}")
    print(f"Pose offsets: {list(POSE_OFFSET)}")

    # Clean old output
    import shutil
    if output_dir.exists():
        shutil.rmtree(output_dir)

    build_split("train_allrot_shuffled", temp_dir, output_dir,
                shuffle=True, single_rand_rot=False, seed=42)

    build_split("test_singlerandrot", temp_dir, output_dir,
                shuffle=False, single_rand_rot=True, seed=42)

    build_split("test_allrot", temp_dir, output_dir,
                shuffle=False, single_rand_rot=False, seed=42)

    print(f"\n{'=' * 70}")
    print(f"COMPLETE")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    main()
