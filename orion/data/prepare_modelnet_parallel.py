"""
Python port of the ModelNet data preparation pipeline.

Port of:
    - data_preparation/Modelnet/main.m
    - data_preparation/Modelnet/modelnet_off2mat.m
    - data_preparation/Modelnet/modelnet_mat2bin.m
    - data_preparation/Modelnet/modelnet_binlist2hdf5.m
    - data_preparation/Modelnet/functions/*.m

Pipeline:
    1. OFF -> voxel grids (with rotations)
    2. Voxel grids -> padded binary files
    3. Binary files -> labeled HDF5 chunks for training
"""

import os
import glob
import random
import struct
import argparse

import numpy as np
import h5py
from tqdm import tqdm

from .off_loader import load_off
from .voxelization import polygon2voxel
from .binary_io import save_voxel_grid_as_bin, load_binary_voxelgrid
from .label_utils import (
    load_pose_plan,
    modelnet_extract_classname_and_rot_from_filename,
    modelnet_generate_class_and_pose_labels,
)

MODELNET10_CLASSES = [
    'bathtub', 'bed', 'chair', 'desk', 'dresser',
    'monitor', 'night_stand', 'sofa', 'table', 'toilet'
]

MODELNET40_CLASSES = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
    'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
    'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
    'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
    'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
    'wardrobe', 'xbox'
]


def off_to_voxel(off_path, mat_path, classes, volume_size=24, pad_size=3, angle_inc=30):
    """
    Convert OFF mesh files to voxelized representations.

    Port of modelnet_off2mat.m. For each model, generates multiple rotated
    voxel grids and saves them as .npy files (replacing MATLAB .mat files).

    Args:
        off_path: Root path to OFF files (e.g., datasets/ModelNet10)
        mat_path: Output path for voxelized data
        classes: List of class names to process
        volume_size: Base voxel grid dimension (default 24)
        pad_size: Padding on each side (default 3)
        angle_inc: Rotation angle increment in degrees (default 30)
    """
    data_size = pad_size * 2 + volume_size
    phases = ['train', 'test']

    for cls in classes:
        print(f"Processing class: {cls}")
        for phase in phases:
            off_dir = os.path.join(off_path, cls, phase)
            dest_dir = os.path.join(mat_path, cls, str(data_size), phase)
            os.makedirs(dest_dir, exist_ok=True)

            if not os.path.isdir(off_dir):
                print(f"  Skipping {off_dir} (not found)")
                continue

            off_files = sorted([f for f in os.listdir(off_dir) if f.endswith('.off')])
            for off_file in tqdm(off_files, desc=f"  {cls}/{phase}"):
                filename = os.path.join(off_dir, off_file)
                base_name = off_file[:-4]  # strip .off

                for viewpoint in range(1, 360 // angle_inc + 1):
                    dest_name = os.path.join(dest_dir, f"{base_name}_{viewpoint}.npy")

                    # Load OFF with rotation
                    theta = (viewpoint - 1) * angle_inc
                    off_data = load_off(filename, theta=theta)

                    # Voxelize
                    instance = polygon2voxel(
                        off_data['vertices'], off_data['faces'],
                        [volume_size, volume_size, volume_size],
                        mode='auto'
                    )

                    # Pad
                    instance = np.pad(instance, pad_size, mode='constant', constant_values=0)
                    instance = instance.astype(np.int8)

                    np.save(dest_name, instance)


def voxel_to_bin(D, source_folder, dest_folder, classes):
    """
    Convert voxelized .npy files to padded binary format.

    Port of modelnet_mat2bin.m. Adds additional padding and saves in the
    ORION binary format.

    Args:
        D: Additional padding on each side (default 3)
        source_folder: Path to voxelized .npy files
        dest_folder: Output path for binary files
        classes: List of class names
    """
    phases = ['train', 'test']

    for cls in classes:
        for phase in phases:
            # Find all .npy files for this class/phase
            # Look in the size subdirectory
            size_dirs = glob.glob(os.path.join(source_folder, cls, '*', phase))
            for src_dir in size_dirs:
                size_str = os.path.basename(os.path.dirname(src_dir))
                dest_dir = os.path.join(dest_folder, 'classes', cls, size_str, phase)
                os.makedirs(dest_dir, exist_ok=True)

                npy_files = sorted(glob.glob(os.path.join(src_dir, '*.npy')))
                for npy_file in tqdm(npy_files, desc=f"  {cls}/{phase} -> bin"):
                    v = np.load(npy_file)

                    # Add padding
                    padded = np.zeros(np.array(v.shape) + 2 * D)
                    padded[D:-D, D:-D, D:-D] = v

                    # Save as binary
                    base_name = os.path.splitext(os.path.basename(npy_file))[0]
                    dest_file = os.path.join(dest_dir, f"{base_name}.bin")
                    save_voxel_grid_as_bin(padded, dest_file)


def create_labeled_list(bin_folder, pose_file, output_list, rotation_start_index=1,
                        collate_singlerots=False):
    """
    Create a labeled file list from binary voxel files.

    Port of modelnet_add_labels_to_list.m + supporting functions.

    Args:
        bin_folder: Root folder containing .bin files
        pose_file: Path to pose plan file
        output_list: Output file path for the labeled list
        rotation_start_index: Starting rotation index (default 1)
        collate_singlerots: Whether to collate single rotations
    """
    classes, nrot = load_pose_plan(pose_file)

    # Find all .bin files
    bin_files = sorted(glob.glob(os.path.join(bin_folder, '**', '*.bin'), recursive=True))

    lines = []
    for bin_file in bin_files:
        class_name, rotation = modelnet_extract_classname_and_rot_from_filename(bin_file)
        class_label, pose_label = modelnet_generate_class_and_pose_labels(
            class_name, rotation - rotation_start_index, classes, nrot, collate_singlerots
        )
        lines.append(f"{bin_file} 1 {class_label} {pose_label}")

    with open(output_list, 'w') as f:
        for line in lines:
            f.write(line + '\n')


def prepare_lists(all_list, dest_path, first_rot_index=1):
    """
    Create train/test split lists with various rotation configurations.

    Port of prepare_modelnet_lists.m.

    Args:
        all_list: Path to the full labeled file list
        dest_path: Destination directory for split lists
        first_rot_index: First rotation index
    """
    os.makedirs(dest_path, exist_ok=True)

    with open(all_list, 'r') as f:
        all_lines = [line.strip() for line in f if line.strip()]

    for split in ['train', 'test']:
        # All rotations
        split_lines = [l for l in all_lines if f'/{split}/' in l]
        allrot_file = os.path.join(dest_path, f'{split}_allrot.txt')
        with open(allrot_file, 'w') as f:
            f.write('\n'.join(split_lines) + '\n')

        # First rotation only
        firstrot_lines = [l for l in split_lines if f'_{first_rot_index}.bin' in l]
        firstrot_file = os.path.join(dest_path, f'{split}_firstrot.txt')
        with open(firstrot_file, 'w') as f:
            f.write('\n'.join(firstrot_lines) + '\n')

        # Single random rotation per model
        model_groups = {}
        for line in split_lines:
            bin_path = line.split()[0]
            # Strip rotation suffix to get model identity
            import re
            model_key = re.sub(r'_\d+\.bin', '', bin_path)
            if model_key not in model_groups:
                model_groups[model_key] = []
            model_groups[model_key].append(line)

        singlerand_lines = [random.choice(v) for v in model_groups.values()]
        singlerand_file = os.path.join(dest_path, f'{split}_singlerandrot.txt')
        with open(singlerand_file, 'w') as f:
            f.write('\n'.join(singlerand_lines) + '\n')

    # Create shuffled versions
    shuffled_dir = os.path.join(dest_path, 'shuffled')
    os.makedirs(shuffled_dir, exist_ok=True)

    for txt_file in glob.glob(os.path.join(dest_path, '*.txt')):
        with open(txt_file, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]

        perm = list(range(len(lines)))
        random.shuffle(perm)
        shuffled_lines = [lines[i] for i in perm]

        dest_file = os.path.join(shuffled_dir, os.path.basename(txt_file))
        with open(dest_file, 'w') as f:
            f.write('\n'.join(shuffled_lines) + '\n')

        # Save permutation
        perm_file = dest_file + '.perm'
        with open(perm_file, 'w') as f:
            f.write('\n'.join(str(p + 1) for p in perm) + '\n')  # 1-based to match MATLAB


def binlist_to_hdf5(bins_list, h5_filename, h5_text_filename, chunk_size=1000):
    """
    Convert a labeled binary file list to HDF5 chunks.

    Port of binlist_to_hdf5.m. Creates chunked HDF5 files with voxel data,
    class labels, and pose labels.

    Args:
        bins_list: Path to labeled file list (format: path dummy class_label pose_label)
        h5_filename: Output HDF5 file path (base name, chunks will be numbered)
        h5_text_filename: Output text file listing all HDF5 chunks
        chunk_size: Number of samples per HDF5 chunk (default 1000)
    """
    # Parse the list file
    files = []
    labels_class = []
    labels_pose = []
    with open(bins_list, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            files.append(parts[0])
            labels_class.append(int(parts[2]))
            labels_pose.append(int(parts[3]))

    N = len(files)
    if N == 0:
        print(f"  WARNING: No samples found in {bins_list}. Skipping HDF5 creation.")
        return
    nchunks = (N + chunk_size - 1) // chunk_size

    # Create output directories
    os.makedirs(os.path.dirname(h5_filename), exist_ok=True)
    os.makedirs(os.path.dirname(h5_text_filename), exist_ok=True)

    # Get dimensions from first file
    temp_v = load_binary_voxelgrid(files[0])
    d1, d2, d3 = temp_v.shape

    # Create text file listing all chunks
    with open(h5_text_filename, 'w') as f:
        for c in range(1, nchunks + 1):
            chunk_name = h5_filename if c == 1 else f"{h5_filename}.{c:04d}"
            f.write(chunk_name + '\n')

    # Create HDF5 chunks
    for c in range(1, nchunks + 1):
        print(f"  Chunk {c}/{nchunks}")
        chunk_name = h5_filename if c == 1 else f"{h5_filename}.{c:04d}"

        written_sofar = (c - 1) * chunk_size
        this_chunk_size = min(chunk_size, N - written_sofar)
        ind1 = written_sofar
        ind2 = ind1 + this_chunk_size

        # Load binary voxel files
        data = np.zeros((d1, d2, d3, 1, this_chunk_size), dtype=np.uint8)
        for i in tqdm(range(this_chunk_size), desc="    Loading"):
            v = load_binary_voxelgrid(files[ind1 + i])
            data[:, :, :, 0, i] = np.uint8(v)

        # Write HDF5
        if os.path.exists(chunk_name):
            os.remove(chunk_name)

        with h5py.File(chunk_name, 'w') as hf:
            # Matches MATLAB: h5create with shape [d3 d2 d1 1 N]
            hf.create_dataset('data', data=data, dtype='uint8')
            hf.create_dataset('label', data=np.array(labels_class[ind1:ind2]).reshape(1, -1))
            hf.create_dataset('label_pose', data=np.array(labels_pose[ind1:ind2]).reshape(1, -1))


def main():
    """Main entry point for ModelNet data preparation."""
    parser = argparse.ArgumentParser(description='ORION ModelNet Data Preparation')
    parser.add_argument('--dataset', choices=['modelnet10', 'modelnet40'], default='modelnet10',
                        help='Dataset to prepare')
    parser.add_argument('--off-path', type=str, required=True,
                        help='Path to raw OFF files (e.g., datasets/ModelNet10)')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Base output path for processed data')
    parser.add_argument('--pose-plan', type=str, required=True,
                        help='Path to pose plan file')
    parser.add_argument('--volume-size', type=int, default=24,
                        help='Base voxel grid dimension (default: 24)')
    parser.add_argument('--pad-size', type=int, default=3,
                        help='Padding on each side (default: 3)')
    parser.add_argument('--angle-inc', type=int, default=30,
                        help='Rotation angle increment in degrees (default: 30)')
    parser.add_argument('--extra-pad', type=int, default=3,
                        help='Additional padding for bin conversion (default: 3)')
    parser.add_argument('--chunk-size', type=int, default=1000,
                        help='Samples per HDF5 chunk (default: 1000)')
    parser.add_argument('--classes', type=str, nargs='*', default=None,
                        help='Specific classes to process (default: all classes)')
    parser.add_argument('--step', choices=['all', 'voxelize', 'bin', 'hdf5'], default='all',
                        help='Which step to run')
    args = parser.parse_args()

    classes = MODELNET10_CLASSES if args.dataset == 'modelnet10' else MODELNET40_CLASSES
    # Filter classes if --classes argument provided (enables parallelization)
    if args.classes:
        classes = [c for c in classes if c in args.classes]
        if not classes:
            print(f"ERROR: No valid classes found in --classes argument")
            return
        print(f"Processing subset: {classes}")
    mat_path = os.path.join(args.output_path, f'{args.dataset}_voxelized')
    bin_path = os.path.join(args.output_path, f'{args.dataset}_bin')
    pose_plan_name = os.path.splitext(os.path.basename(args.pose_plan))[0]

    if args.step in ('all', 'voxelize'):
        print("=== Step 1: OFF -> Voxelized grids ===")
        off_to_voxel(args.off_path, mat_path, classes,
                     args.volume_size, args.pad_size, args.angle_inc)

    if args.step in ('all', 'bin'):
        print("=== Step 2: Voxelized grids -> Binary files ===")
        voxel_to_bin(args.extra_pad, mat_path, bin_path, classes)

    if args.step in ('all', 'hdf5'):
        print("=== Step 3: Binary files -> HDF5 chunks ===")
        plan_dir = os.path.join(bin_path, pose_plan_name)
        os.makedirs(plan_dir, exist_ok=True)

        # Create labeled list
        all_list = os.path.join(plan_dir, 'all.txt')
        create_labeled_list(bin_path, args.pose_plan, all_list)

        # Check if any data was produced
        with open(all_list, 'r') as f:
            num_entries = sum(1 for line in f if line.strip())
        if num_entries == 0:
            print("ERROR: No binary files were found. Make sure the dataset was downloaded")
            print(f"  and extracted to: {args.off_path}")
            print("  Run the download script first: bash datasets/get_modelnet10.sh")
            return

        # Create train/test splits
        lists_dir = os.path.join(plan_dir, 'lists')
        prepare_lists(all_list, lists_dir)

        # Create HDF5 datasets
        hdf5_base = os.path.join(plan_dir, 'hdf5')

        train_list = os.path.join(lists_dir, 'shuffled', 'train_allrot.txt')
        binlist_to_hdf5(
            train_list,
            os.path.join(hdf5_base, 'train_allrot_shuffled', 'train.hdf5'),
            os.path.join(hdf5_base, 'train_allrot_shuffled', 'train.hdf5.txt'),
            args.chunk_size
        )

        test_singlerand = os.path.join(lists_dir, 'test_singlerandrot.txt')
        binlist_to_hdf5(
            test_singlerand,
            os.path.join(hdf5_base, 'test_singlerandrot', 'test.hdf5'),
            os.path.join(hdf5_base, 'test_singlerandrot', 'test.hdf5.txt'),
            args.chunk_size
        )

        test_allrot = os.path.join(lists_dir, 'test_allrot.txt')
        binlist_to_hdf5(
            test_allrot,
            os.path.join(hdf5_base, 'test_allrot', 'test.hdf5'),
            os.path.join(hdf5_base, 'test_allrot', 'test.hdf5.txt'),
            args.chunk_size
        )

    print("=== Done ===")


if __name__ == '__main__':
    main()
