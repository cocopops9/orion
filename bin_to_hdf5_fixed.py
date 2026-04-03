#!/usr/bin/env python
"""
Binary to HDF5 Converter for ORION

Converts existing binary voxel files to HDF5 format, bypassing voxelization.

Binary format (from ORION MATLAB code):
- 3 x uint16 (6 bytes): dimensions [d1, d2, d3]
- d1*d2*d3 x uint8: voxel occupancy data (0 or 1)

IMPORTANT: This script computes pose labels using the ORION cumulative-offset
scheme from poseplan_MN10.txt, NOT raw rotation indices. Each class occupies
a disjoint range of pose label indices:
    bathtub (nrot=6):     poses 0-5
    bed (nrot=12):        poses 6-17
    chair (nrot=12):      poses 18-29
    desk (nrot=12):       poses 30-41
    dresser (nrot=12):    poses 42-53
    monitor (nrot=12):    poses 54-65
    night_stand (nrot=12): poses 66-77
    sofa (nrot=12):       poses 78-89
    table (nrot=3):       poses 90-92
    toilet (nrot=12):     poses 93-104
    Total: 105 orientation classes
"""

import sys
import os
import random
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

CLASSES = [
    'bathtub', 'bed', 'chair', 'desk', 'dresser',
    'monitor', 'night_stand', 'sofa', 'table', 'toilet'
]

# From data_preparation/poseplans/poseplan_MN10.txt
# Must match the original paper and Caffe prototxt (fc8_pose num_output=105)
NROT = [6, 12, 12, 12, 12, 12, 12, 12, 3, 12]

# Precompute cumulative offsets (matches MATLAB circshift + cumsum)
POSE_OFFSET = np.zeros(len(NROT), dtype=int)
cumsum = np.cumsum(NROT)
POSE_OFFSET[1:] = cumsum[:-1]
# Result: [0, 6, 18, 30, 42, 54, 66, 78, 90, 93]

TOTAL_ORIENTATIONS = sum(NROT)  # 105


def read_binary_voxel(filepath):
    """Read ORION binary voxel file (MATLAB format)."""
    with open(filepath, 'rb') as f:
        # Read dimensions: 3 x uint16 (6 bytes)
        dims = np.fromfile(f, dtype=np.uint16, count=3)

        # Read voxel data: d1*d2*d3 x uint8
        nvoxels = int(np.prod(dims))
        voxels = np.fromfile(f, dtype=np.uint8, count=nvoxels)

        # Reshape to 3D grid (Fortran/column-major order for MATLAB compatibility)
        grid = voxels.reshape(dims, order='F')

        # Convert to float32
        grid = grid.astype(np.float32)

    return grid


def compute_pose_label(class_idx, rotation_index):
    """
    Compute the ORION pose label using cumulative class offsets.

    This matches the original MATLAB function:
        modelnet_generate_class_and_pose_labels.m

    The rotation_index is 0-based (rotation file index minus 1).
    The pose label wraps around using modulo with the class-specific nrot.

    Args:
        class_idx: Integer class index (0-9)
        rotation_index: 0-based rotation index

    Returns:
        Integer pose label in [0, 104]
    """
    nrot_for_class = NROT[class_idx]
    wrapped_rotation = rotation_index % nrot_for_class
    pose_label = int(POSE_OFFSET[class_idx]) + wrapped_rotation
    return pose_label


def process_split(bin_dir, output_dir, split_name, classes, chunk_size=1000):
    """Process one data split and create HDF5 file."""

    # Determine parameters based on split
    if split_name == 'train_allrot_shuffled':
        split_subdir = 'train'
        use_all_rotations = True
        shuffle = True
        select_single_random = False
    elif split_name == 'test_singlerandrot':
        split_subdir = 'test'
        use_all_rotations = True  # load all, then pick one random per model
        shuffle = False
        select_single_random = True
    elif split_name == 'test_allrot':
        split_subdir = 'test'
        use_all_rotations = True
        shuffle = False
        select_single_random = False
    else:
        raise ValueError(f"Unknown split: {split_name}")

    print(f"\n  Processing {split_name}...")

    # Collect all binary files with correct labels
    all_files = []

    for class_idx, class_name in enumerate(classes):
        class_dir = bin_dir / 'classes' / class_name / '30' / split_subdir

        if not class_dir.exists():
            print(f"    Warning: {class_dir} not found, skipping")
            continue

        bin_files = sorted(class_dir.glob('*.bin'))

        for bin_file in bin_files:
            # Extract rotation from filename (e.g., bathtub_0001_5.bin -> rotation=5)
            parts = bin_file.stem.split('_')
            rotation = int(parts[-1])

            # Compute 0-based rotation index (files use 1-based indexing)
            rotation_0based = rotation - 1

            # Compute ORION-style pose label with cumulative class offsets
            pose_label = compute_pose_label(class_idx, rotation_0based)

            # Build model identity key (strip rotation suffix) for grouping
            model_key = '_'.join(parts[:-1])

            all_files.append({
                'path': bin_file,
                'class_idx': class_idx,
                'pose_label': pose_label,
                'model_key': model_key,
                'rotation': rotation,
            })

    if not all_files:
        print(f"    No files found")
        return

    # For test_singlerandrot: select ONE random rotation per model
    # This matches the original MATLAB prepare_modelnet_lists.m behavior
    if select_single_random:
        random.seed(42)
        model_groups = {}
        for entry in all_files:
            key = f"{entry['class_idx']}_{entry['model_key']}"
            if key not in model_groups:
                model_groups[key] = []
            model_groups[key].append(entry)

        selected = []
        for key in sorted(model_groups.keys()):
            chosen = random.choice(model_groups[key])
            selected.append(chosen)
        all_files = selected

    # Shuffle training data
    if shuffle:
        random.seed(42)
        random.shuffle(all_files)

    total_samples = len(all_files)
    print(f"    Total samples: {total_samples}")

    # Validate pose labels
    pose_labels_array = np.array([e['pose_label'] for e in all_files])
    print(f"    Pose label range: {pose_labels_array.min()} to {pose_labels_array.max()}")
    print(f"    Unique pose labels: {len(np.unique(pose_labels_array))}")
    assert pose_labels_array.max() < TOTAL_ORIENTATIONS, \
        f"Pose label {pose_labels_array.max()} exceeds total orientations {TOTAL_ORIENTATIONS}"

    # Create output directory
    split_output_dir = output_dir / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)

    # Process in chunks
    num_chunks = (total_samples + chunk_size - 1) // chunk_size
    chunk_files_list = []

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_samples)
        chunk_entries = all_files[start_idx:end_idx]

        print(f"    Chunk {chunk_idx + 1}/{num_chunks}: ", end='', flush=True)

        chunk_data = []
        chunk_labels = []
        chunk_poses = []

        for entry in tqdm(chunk_entries, desc="Loading", leave=False):
            try:
                grid = read_binary_voxel(entry['path'])
                chunk_data.append(grid)
                chunk_labels.append(entry['class_idx'])
                chunk_poses.append(entry['pose_label'])
            except Exception as e:
                print(f"\n      Error reading {entry['path']}: {e}")
                continue

        if not chunk_data:
            continue

        data_array = np.stack(chunk_data, axis=0)
        label_array = np.array(chunk_labels, dtype=np.int64)
        pose_array = np.array(chunk_poses, dtype=np.int64)

        chunk_filename = split_output_dir / f'chunk_{chunk_idx}.h5'

        with h5py.File(chunk_filename, 'w') as f:
            f.create_dataset('data', data=data_array, compression='gzip')
            f.create_dataset('label', data=label_array)
            f.create_dataset('label_pose', data=pose_array)

        chunk_files_list.append(chunk_filename)
        print(f"{len(chunk_data)} samples")

    # Merge chunks into final file
    print(f"    Merging {len(chunk_files_list)} chunks...")

    total = 0
    for chunk_file in chunk_files_list:
        with h5py.File(chunk_file, 'r') as f:
            total += f['data'].shape[0]

    final_filename = split_output_dir / ('train.hdf5' if 'train' in split_name else 'test.hdf5')

    with h5py.File(chunk_files_list[0], 'r') as f:
        grid_shape = f['data'].shape[1:]

    with h5py.File(final_filename, 'w') as out_f:
        out_f.create_dataset('data', shape=(total,) + grid_shape,
                             dtype='float32', compression='gzip')
        out_f.create_dataset('label', shape=(total,), dtype='int64')
        out_f.create_dataset('label_pose', shape=(total,), dtype='int64')

        write_idx = 0
        for chunk_file in chunk_files_list:
            with h5py.File(chunk_file, 'r') as in_f:
                n = in_f['data'].shape[0]
                out_f['data'][write_idx:write_idx + n] = in_f['data'][:]
                out_f['label'][write_idx:write_idx + n] = in_f['label'][:]
                out_f['label_pose'][write_idx:write_idx + n] = in_f['label_pose'][:]
                write_idx += n

    size_mb = final_filename.stat().st_size / (1024 * 1024)
    print(f"    Created: {final_filename.name} ({total} samples, {size_mb:.1f} MB)")

    # Write the .txt list file that the training code expects
    txt_filename = split_output_dir / ('train.hdf5.txt' if 'train' in split_name else 'test.hdf5.txt')
    with open(txt_filename, 'w') as f:
        f.write(str(final_filename) + '\n')
    print(f"    Created: {txt_filename.name}")

    # Cleanup temporary chunks
    for chunk_file in chunk_files_list:
        chunk_file.unlink()


def main():
    """Main conversion function."""

    base_dir = Path('.')
    bin_base = base_dir / 'datasets' / 'modelnet10_bin'
    output_dir = base_dir / 'datasets' / 'modelnet10' / 'hdf5'

    if not bin_base.exists():
        print(f"Error: {bin_base} not found")
        print("Make sure you run this from the orion project root directory.")
        sys.exit(1)

    # Verify binary files exist
    test_bins = list((bin_base / 'classes').glob('**/30/**/*.bin'))
    if not test_bins:
        print(f"Error: No .bin files found under {bin_base / 'classes'}")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print(f"ORION Binary to HDF5 Converter (Fixed Pose Labels)")
    print(f"{'=' * 70}")
    print(f"Input:  {bin_base}")
    print(f"Output: {output_dir}")
    print(f"Total orientation classes: {TOTAL_ORIENTATIONS}")
    print(f"Per-class rotations: {dict(zip(CLASSES, NROT))}")
    print(f"Cumulative offsets:  {list(POSE_OFFSET)}")
    print(f"{'=' * 70}\n")

    # Clean old HDF5 files
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
        print("Removed old HDF5 directory")

    # Process all splits
    splits = [
        'train_allrot_shuffled',
        'test_singlerandrot',
        'test_allrot'
    ]

    for split in splits:
        process_split(bin_base, output_dir, split, CLASSES)

    # Final verification
    print(f"\n{'=' * 70}")
    print(f"VERIFICATION")
    print(f"{'=' * 70}")

    for split in splits:
        hdf5_name = 'train.hdf5' if 'train' in split else 'test.hdf5'
        hdf5_file = output_dir / split / hdf5_name
        if hdf5_file.exists():
            with h5py.File(hdf5_file, 'r') as f:
                labels = f['label'][:]
                poses = f['label_pose'][:]
                print(f"\n  {split}/{hdf5_name}:")
                print(f"    Samples: {len(labels)}")
                print(f"    Data shape: {f['data'].shape}")
                print(f"    Label range: [{labels.min()}, {labels.max()}]")
                print(f"    Pose range:  [{poses.min()}, {poses.max()}]")
                print(f"    Unique poses: {len(np.unique(poses))}")

                for c in range(10):
                    class_mask = labels == c
                    class_poses = np.unique(poses[class_mask])
                    expected_start = POSE_OFFSET[c]
                    expected_end = POSE_OFFSET[c] + NROT[c] - 1
                    status = "OK" if (len(class_poses) == 0 or
                                      (class_poses.min() >= expected_start and
                                       class_poses.max() <= expected_end)) else "ERROR"
                    print(f"    Class {c} ({CLASSES[c]:>11s}): "
                          f"poses {class_poses} "
                          f"(expected [{expected_start}-{expected_end}]) [{status}]")

    print(f"\n{'=' * 70}")
    print(f"CONVERSION COMPLETE")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    main()
