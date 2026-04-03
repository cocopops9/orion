"""
Point Cloud Dataset for PointNet.

Reads the existing ORION voxel HDF5 files and converts occupied voxels
to point cloud coordinates on-the-fly. This ensures identical train/test
splits for a fair comparison between voxel-based (ORION) and point-based
(PointNet) representations.

Uses lazy HDF5 access — voxels are read one-at-a-time and converted to
point clouds on the fly, keeping memory usage minimal.

Conversion pipeline:
    1. Extract (x, y, z) coordinates of occupied voxels
    2. Center at origin (subtract centroid)
    3. Normalize to unit sphere
    4. Subsample or pad to fixed point count
"""

import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset


def voxel_to_pointcloud(voxel_grid, num_points=1024):
    """
    Convert a binary voxel grid to a normalized point cloud.

    Args:
        voxel_grid: (D, H, W) or (D, H, W, 1) binary numpy array.
        num_points: Fixed number of output points.

    Returns:
        points: (num_points, 3) float32 array, zero-centered,
                unit-sphere normalized.
    """
    # Squeeze channel dim if present
    if voxel_grid.ndim == 4 and voxel_grid.shape[-1] == 1:
        voxel_grid = voxel_grid[..., 0]
    elif voxel_grid.ndim == 4 and voxel_grid.shape[0] == 1:
        voxel_grid = voxel_grid[0]

    occupied = np.argwhere(voxel_grid > 0).astype(np.float32)

    if len(occupied) == 0:
        return np.zeros((num_points, 3), dtype=np.float32)

    # Center at origin
    centroid = occupied.mean(axis=0)
    occupied -= centroid

    # Normalize to unit sphere
    max_dist = np.max(np.linalg.norm(occupied, axis=1))
    if max_dist > 0:
        occupied /= max_dist

    # Subsample or pad to fixed number of points
    n = len(occupied)
    if n >= num_points:
        indices = np.random.choice(n, num_points, replace=False)
        points = occupied[indices]
    else:
        pad_indices = np.random.choice(n, num_points - n, replace=True)
        points = np.concatenate([occupied, occupied[pad_indices]], axis=0)

    return points.astype(np.float32)


class PointCloudHDF5Dataset(Dataset):
    """
    Dataset that loads ORION voxel HDF5 files and converts to point clouds.

    Uses lazy HDF5 access — voxels read on-the-fly, converted to point
    clouds per-sample. RAM usage is minimal.

    Args:
        hdf5_list_file: Path to a text file listing HDF5 files.
        num_points: Number of points per sample (default: 1024).
        augment: Whether to apply random jitter (training).
    """

    def __init__(self, hdf5_list_file, num_points=1024, augment=False):
        self.num_points = num_points
        self.augment = augment

        # Resolve HDF5 file paths
        list_dir = os.path.dirname(os.path.abspath(hdf5_list_file))
        with open(hdf5_list_file, 'r') as f:
            raw_paths = [line.strip() for line in f if line.strip()]

        hdf5_files = []
        for p in raw_paths:
            if os.path.isabs(p):
                hdf5_files.append(p)
            else:
                hdf5_files.append(os.path.join(list_dir, p))

        # Open HDF5 files lazily
        self._h5_handles = []
        self._file_ranges = []
        all_labels = []
        all_label_poses = []
        offset = 0

        for hdf5_path in hdf5_files:
            hf = h5py.File(hdf5_path, 'r')
            self._h5_handles.append(hf)
            n = hf['data'].shape[0]
            self._file_ranges.append((offset, offset + n))
            offset += n

            labels = np.array(hf['label']).flatten().astype(np.int64)
            if 'label_pose' in hf:
                label_poses = np.array(hf['label_pose']).flatten().astype(np.int64)
            else:
                label_poses = np.zeros_like(labels)
            all_labels.append(labels)
            all_label_poses.append(label_poses)

        self.labels = np.concatenate(all_labels)
        self.label_poses = np.concatenate(all_label_poses)
        self._total = len(self.labels)

        print(f"PointCloud dataset: {self._total} samples from "
              f"{len(hdf5_files)} HDF5 files, {num_points} points each "
              f"(lazy mode)")

    def _get_file_and_local_idx(self, idx):
        for fh, (start, end) in zip(self._h5_handles, self._file_ranges):
            if start <= idx < end:
                return fh, idx - start
        raise IndexError(f"Index {idx} out of range")

    def __len__(self):
        return self._total

    def __getitem__(self, idx):
        fh, local_idx = self._get_file_and_local_idx(idx)
        voxel = np.array(fh['data'][local_idx])

        points = voxel_to_pointcloud(voxel, self.num_points)

        if self.augment:
            points += np.random.normal(0, 0.02, size=points.shape).astype(np.float32)
            np.random.shuffle(points)

        points = torch.from_numpy(points).float()
        label = int(self.labels[idx])
        label_pose = int(self.label_poses[idx])
        return points, label, label_pose

    def __del__(self):
        for fh in getattr(self, '_h5_handles', []):
            try:
                fh.close()
            except Exception:
                pass
