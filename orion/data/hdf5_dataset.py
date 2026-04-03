"""
PyTorch Dataset for loading ORION voxel data from HDF5 files.

The HDF5 files contain pre-voxelized 3D object data from ModelNet.
Each sample has:
- data: voxel grid (36x36x36 typically, gets cropped to 32x32x32)
- label: object class (0 to num_classes-1)
- label_pose: orientation bin (0 to num_orientations-1)

Uses lazy HDF5 access to avoid loading entire datasets into RAM.
Only the labels are loaded upfront (tiny); voxel grids are read
on-the-fly from the memory-mapped HDF5 file.
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class VoxelHDF5Dataset(Dataset):
    """
    Dataset that loads voxel grids from HDF5 files with lazy I/O.

    Voxel grids are read from disk on each __getitem__ call via HDF5
    direct indexing, keeping RAM usage minimal. Labels are small and
    loaded upfront for fast length/indexing.

    Args:
        hdf5_list_file: Path to a text file that lists HDF5 file paths,
            one per line. Relative paths are resolved relative to the
            directory containing the list file.
        transform: Optional transform to apply to voxel grids.
    """

    def __init__(self, hdf5_list_file, transform=None):
        self.transform = transform

        # Read the list of HDF5 files to load
        list_dir = os.path.dirname(os.path.abspath(hdf5_list_file))
        with open(hdf5_list_file, 'r') as f:
            raw_paths = [line.strip() for line in f if line.strip()]

        # Resolve relative paths w.r.t. the directory of the list file
        hdf5_files = []
        for p in raw_paths:
            if os.path.isabs(p):
                hdf5_files.append(p)
            else:
                hdf5_files.append(os.path.join(list_dir, p))

        # Open HDF5 files and collect metadata
        # Keep file handles open for lazy reading during training
        self._h5_handles = []
        self._file_ranges = []  # (start_idx, end_idx) per file

        all_labels = []
        all_label_poses = []
        offset = 0

        for hdf5_path in hdf5_files:
            hf = h5py.File(hdf5_path, 'r')
            self._h5_handles.append(hf)

            n = hf['data'].shape[0]
            self._file_ranges.append((offset, offset + n))
            offset += n

            # Labels are tiny — load them into memory
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

        print(f"Loaded {self._total} samples from {len(hdf5_files)} HDF5 files "
              f"(lazy mode — voxels read on-the-fly)")

    def _get_file_and_local_idx(self, idx):
        """Map a global index to (file_handle, local_index)."""
        for fh, (start, end) in zip(self._h5_handles, self._file_ranges):
            if start <= idx < end:
                return fh, idx - start
        raise IndexError(f"Index {idx} out of range [0, {self._total})")

    def __len__(self):
        return self._total

    def __getitem__(self, idx):
        fh, local_idx = self._get_file_and_local_idx(idx)

        # Read single voxel grid from HDF5 (lazy — no full dataset in RAM)
        voxel = fh['data'][local_idx]

        # Convert to float tensor and add channel dimension
        voxel = torch.from_numpy(np.array(voxel, dtype=np.float32)).unsqueeze(0)

        if self.transform:
            voxel = self.transform(voxel)

        label = int(self.labels[idx])
        label_pose = int(self.label_poses[idx])

        return voxel, label, label_pose

    def __del__(self):
        """Close HDF5 file handles on cleanup."""
        for fh in getattr(self, '_h5_handles', []):
            try:
                fh.close()
            except Exception:
                pass
