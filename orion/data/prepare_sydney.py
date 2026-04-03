"""
Python port of the Sydney Urban Objects dataset preparation pipeline.

Port of data_preparation/sydney/main.m and supporting functions.

Pipeline:
    1. Load PLY point clouds
    2. Rotate around Z axis (multiple viewpoints)
    3. Voxelize to NxNxN grids
    4. Pad and save as binary files
    5. Generate labeled lists and HDF5 datasets
"""

import os
import glob
import random
import re
import argparse

import numpy as np
from tqdm import tqdm

try:
    import trimesh
except ImportError:
    trimesh = None

from .pointcloud_to_voxels import pointcloud_to_voxels
from .binary_io import save_voxel_grid_as_bin, load_binary_voxelgrid
from .label_utils import (
    sydney_extract_classname_and_rot_from_filename,
    sydney_generate_class_and_pose_labels,
)


def load_ply_as_model(ply_path):
    """
    Load a PLY file and return a model dict compatible with pointcloud_to_voxels.

    Replaces the MATLAB nplyRead + import_3D_model functions.

    Args:
        ply_path: Path to PLY file

    Returns:
        dict with 'x', 'y', 'z' arrays
    """
    if trimesh is None:
        raise ImportError("trimesh is required. Install with: pip install trimesh")

    mesh_or_cloud = trimesh.load(ply_path)

    # Handle both meshes and point clouds
    if hasattr(mesh_or_cloud, 'vertices'):
        points = np.array(mesh_or_cloud.vertices)
    else:
        points = np.array(mesh_or_cloud)

    return {
        'x': points[:, 0],
        'y': points[:, 1],
        'z': points[:, 2],
        'nx': np.array([]),
        'ny': np.array([]),
        'nz': np.array([]),
    }


def rotate_model_around_z(model, phi_degrees):
    """
    Rotate a point cloud model around the Z axis.

    Port of rotate_3D_model_around_z.m.

    Args:
        model: dict with 'x', 'y', 'z' arrays
        phi_degrees: Rotation angle in degrees

    Returns:
        New model dict with rotated coordinates
    """
    phi = np.radians(phi_degrees)
    R = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi),  np.cos(phi), 0],
        [0,            0,           1]
    ])

    xyz = np.column_stack([model['x'], model['y'], model['z']])
    xyz_rot = xyz @ R.T

    new_model = dict(model)
    new_model['x'] = xyz_rot[:, 0]
    new_model['y'] = xyz_rot[:, 1]
    new_model['z'] = xyz_rot[:, 2]

    # Rotate normals if present
    if len(model.get('nx', [])) > 0:
        nxyz = np.column_stack([model['nx'], model['ny'], model['nz']])
        nxyz_rot = nxyz @ R.T
        new_model['nx'] = nxyz_rot[:, 0]
        new_model['ny'] = nxyz_rot[:, 1]
        new_model['nz'] = nxyz_rot[:, 2]

    return new_model


def prepare_sydney(ply_folder, destination_path, n_rotations=18,
                   basic_dimension=28, D=4, voxelization_type='binary',
                   binary_max=1, rotation_start_index=0,
                   rotation_randomization=False, collate_singlerots=False):
    """
    Full Sydney dataset preparation pipeline.

    Port of data_preparation/sydney/main.m.

    Args:
        ply_folder: Path to aligned PLY files
        destination_path: Output directory
        n_rotations: Number of rotation viewpoints (default 18)
        basic_dimension: Base voxel grid dimension (default 28)
        D: Padding on each side (default 4)
        voxelization_type: 'binary' or 'density'
        binary_max: Max value for binary voxelization
        rotation_start_index: Starting rotation index
        rotation_randomization: Whether to add random rotation perturbation
        collate_singlerots: Whether to collate single rotations
    """
    os.makedirs(destination_path, exist_ok=True)

    # Find all PLY files
    ply_files = sorted(glob.glob(os.path.join(ply_folder, '*.ply')))
    if not ply_files:
        print(f"No PLY files found in {ply_folder}")
        return

    print(f"Found {len(ply_files)} PLY files")

    for ply_path in tqdm(ply_files, desc="Processing PLY files"):
        base_name = os.path.splitext(os.path.basename(ply_path))[0]
        model = load_ply_as_model(ply_path)

        for r in range(n_rotations):
            rotation = r * 360.0 / n_rotations

            if rotation_randomization:
                margin = 360.0 / n_rotations * 0.80
                rotation += random.random() * margin - margin / 2.0

            rmodel = rotate_model_around_z(model, rotation)

            # Voxelize
            voxel = pointcloud_to_voxels(
                rmodel, basic_dimension, voxelization_type=voxelization_type,
                binary_max=binary_max
            )

            # Pad
            padded = np.zeros(np.array(voxel.shape) + 2 * D)
            padded[D:-D, D:-D, D:-D] = voxel

            # Save as binary
            dest_filename = os.path.join(
                destination_path, f"{base_name}_r{r + rotation_start_index:02d}.bin"
            )
            save_voxel_grid_as_bin(padded, dest_filename)

    # Create labeled list
    all_list_file = os.path.join(destination_path, 'all.txt')
    bin_files = sorted(glob.glob(os.path.join(destination_path, '*.bin')))

    lines = []
    for bin_file in bin_files:
        class_name, rotation = sydney_extract_classname_and_rot_from_filename(bin_file)
        class_label, pose_label = sydney_generate_class_and_pose_labels(
            class_name, rotation - rotation_start_index, collate_singlerots
        )
        lines.append(f"{bin_file} 1 {class_label} {pose_label}")

    with open(all_list_file, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"Created labeled list: {all_list_file}")


def main():
    """Main entry point for Sydney data preparation."""
    parser = argparse.ArgumentParser(description='ORION Sydney Data Preparation')
    parser.add_argument('--ply-folder', type=str, required=True,
                        help='Path to aligned PLY files')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--n-rotations', type=int, default=18,
                        help='Number of rotation viewpoints (default: 18)')
    parser.add_argument('--basic-dimension', type=int, default=28,
                        help='Base voxel grid dimension (default: 28)')
    parser.add_argument('--padding', type=int, default=4,
                        help='Padding on each side (default: 4)')
    args = parser.parse_args()

    prepare_sydney(
        args.ply_folder, args.output_path,
        n_rotations=args.n_rotations,
        basic_dimension=args.basic_dimension,
        D=args.padding,
    )


if __name__ == '__main__':
    main()
