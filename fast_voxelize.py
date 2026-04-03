#!/usr/bin/env python3
"""
Fast parallel voxelization for ModelNet datasets.

Uses multiprocessing across all CPU cores with an optional Numba JIT-compiled
voxelizer for significant speedup over the sequential pure-Python pipeline.
Supports resume: already-completed files are skipped automatically.

Usage:
    python3 fast_voxelize.py --off-path datasets/modelnet40_auto_aligned \\
                             --output datasets/modelnet40_voxelized \\
                             --dataset modelnet40 --workers 10
"""
import os
import sys
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orion.data.off_loader import load_off
try:
    from orion.data.voxelization_fast import polygon2voxel
    print("Using numba-accelerated voxelizer (~6x faster)")
except ImportError:
    from orion.data.voxelization import polygon2voxel
    print("WARNING: numba not available, using slow pure-Python voxelizer")

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

def voxelize_one_file(args):
    """Process one OFF file: 12 rotations -> 12 .npy voxel grids.
    Uses the original ORION voxelizer (polygon2voxel) for exact reproduction.
    """
    off_path, dest_dir, base_name, volume_size, pad_size, angle_inc = args
    n_rotations = 360 // angle_inc
    count = 0
    for viewpoint in range(1, n_rotations + 1):
        dest_name = os.path.join(dest_dir, f"{base_name}_{viewpoint}.npy")
        if os.path.exists(dest_name):
            count += 1
            continue
        try:
            theta = (viewpoint - 1) * angle_inc
            off_data = load_off(off_path, theta=theta)
            instance = polygon2voxel(
                off_data['vertices'], off_data['faces'],
                [volume_size, volume_size, volume_size],
                mode='auto'
            )
            instance = np.pad(instance, pad_size, mode='constant')
            np.save(dest_name, instance.astype(np.int8))
            count += 1
        except Exception:
            pass
    return count


def main():
    parser = argparse.ArgumentParser(description='Fast parallel voxelization')
    parser.add_argument('--off-path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dataset', choices=['modelnet10', 'modelnet40'], default='modelnet40')
    parser.add_argument('--volume-size', type=int, default=24)
    parser.add_argument('--pad-size', type=int, default=3)
    parser.add_argument('--angle-inc', type=int, default=30)
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() - 2))
    args = parser.parse_args()

    classes = MODELNET10_CLASSES if args.dataset == 'modelnet10' else MODELNET40_CLASSES

    data_size = args.pad_size * 2 + args.volume_size

    # Collect all work items
    work_items = []
    already_done = 0
    for cls in classes:
        for phase in ['train', 'test']:
            off_dir = os.path.join(args.off_path, cls, phase)
            dest_dir = os.path.join(args.output, cls, str(data_size), phase)
            os.makedirs(dest_dir, exist_ok=True)
            if not os.path.isdir(off_dir):
                continue
            for off_file in sorted(os.listdir(off_dir)):
                if not off_file.endswith('.off'):
                    continue
                base = off_file[:-4]
                # Check if all rotations already exist
                n_rot = 360 // args.angle_inc
                existing = sum(1 for vp in range(1, n_rot+1)
                             if os.path.exists(os.path.join(dest_dir, f"{base}_{vp}.npy")))
                if existing == n_rot:
                    already_done += 1
                    continue
                work_items.append((
                    os.path.join(off_dir, off_file),
                    dest_dir, base,
                    args.volume_size, args.pad_size, args.angle_inc,
                ))

    n_rot = 360 // args.angle_inc
    print(f"OFF files to process: {len(work_items)} (already done: {already_done})")
    print(f"Rotations per file: {n_rot}")
    print(f"Total voxel grids remaining: {len(work_items) * n_rot}")
    print(f"Workers: {args.workers}")
    print()

    if not work_items:
        print("Nothing to do!")
        return

    # Warmup: JIT-compile the voxelizer in the main process
    # Workers inherit the compiled code via fork()
    print("JIT warmup...")
    voxelize_one_file(work_items[0])
    print("Warmup done, starting parallel processing...\n")

    done = 0
    with Pool(args.workers) as pool:
        for count in pool.imap_unordered(voxelize_one_file, work_items, chunksize=4):
            done += 1
            if done % 50 == 0 or done == len(work_items):
                pct = done * 100 // len(work_items)
                print(f"  [{pct:3d}%] {done}/{len(work_items)} files")

    print(f"\nDone! Output: {args.output}/")


if __name__ == '__main__':
    main()
