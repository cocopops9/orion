"""
Python port of binary voxel grid I/O functions.

Port of save_voxel_grid_as_bin.m and load_binary_voxelgrid.m.

Binary format:
    [uint16 x_size][uint16 y_size][uint16 z_size][uint8 voxel_data...]
"""

import struct
import numpy as np


def save_voxel_grid_as_bin(voxel, filename):
    """
    Save a 3D voxel grid in the ORION binary format.

    Port of save_voxel_grid_as_bin.m.

    Args:
        voxel: 3D numpy array (voxel grid)
        filename: Output file path
    """
    shape = voxel.shape
    with open(filename, 'wb') as f:
        # Write dimensions as uint16
        f.write(struct.pack('<HHH', shape[0], shape[1], shape[2]))
        # Write voxel data as uint8
        f.write(np.uint8(voxel).tobytes(order='F'))  # Fortran order to match MATLAB


def load_binary_voxelgrid(filename):
    """
    Load a 3D voxel grid from the ORION binary format.

    Port of load_binary_voxelgrid.m.

    Args:
        filename: Path to binary voxel file

    Returns:
        3D numpy array (voxel grid)
    """
    with open(filename, 'rb') as f:
        # Read dimensions (3 x uint16)
        sz = struct.unpack('<HHH', f.read(6))
        # Read voxel data (uint8)
        data = np.frombuffer(f.read(sz[0] * sz[1] * sz[2]), dtype=np.uint8)
        # Reshape in Fortran order to match MATLAB
        voxel = data.reshape(sz, order='F')
    return voxel
