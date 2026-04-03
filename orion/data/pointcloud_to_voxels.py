"""
Python port of pointcloud_to_voxels_3.m.

Converts a 3D point cloud to an NxNxN voxel grid while preserving
aspect ratio. Supports binary and density voxelization modes.
"""

import numpy as np


def pointcloud_to_voxels(points, N, bbox=None, voxelization_type='binary', binary_max=1):
    """
    Convert a point cloud to an NxNxN voxel grid.

    Port of pointcloud_to_voxels_3.m. Preserves aspect ratio by scaling
    uniformly and zero-padding to make a cube.

    Args:
        points: dict with 'x', 'y', 'z' arrays, or (M, 3) numpy array
        N: Target cube dimension
        bbox: Optional (7,) array [x1, y1, z1, w, h, d, phi] for bounding box
        voxelization_type: 'binary' or 'density'
        binary_max: Maximum value for binary voxelization (default 1)

    Returns:
        NxNxN numpy array (voxel grid)
    """
    # Handle dict input (MATLAB model struct equivalent)
    if isinstance(points, dict):
        X = np.asarray(points['x']).flatten()
        Y = np.asarray(points['y']).flatten()
        Z = np.asarray(points['z']).flatten()
    else:
        points = np.asarray(points)
        X, Y, Z = points[:, 0], points[:, 1], points[:, 2]

    # Handle empty point cloud
    if len(X) == 0:
        return np.zeros((N, N, N))

    # Compute bounding limits
    if bbox is None:
        mX, MX = X.min(), X.max()
        mY, MY = Y.min(), Y.max()
        mZ, MZ = Z.min(), Z.max()
    else:
        x1, y1, z1, w, h, d, phi = bbox
        x2 = x1 + w
        y2 = y1 + h
        z2 = z1 + d

        corners_x = np.array([x1, x1, x2, x2])
        corners_y = np.array([y1, y2, y1, y2])
        xc = (x1 + x2) / 2.0
        yc = (y1 + y2) / 2.0

        # Rotate corners around center
        corners_x -= xc
        corners_y -= yc
        cos_phi = np.cos(np.radians(phi))
        sin_phi = np.sin(np.radians(phi))
        rot_x = cos_phi * corners_x - sin_phi * corners_y + xc
        rot_y = sin_phi * corners_x + cos_phi * corners_y + yc

        mX, MX = rot_x.min(), rot_x.max()
        mY, MY = rot_y.min(), rot_y.max()
        mZ, MZ = min(z1, z2), max(z1, z2)

    # Compute per-axis extent and uniform scale factor
    maxmax = max(MX - mX, MY - mY, MZ - mZ)
    if maxmax == 0:
        maxmax = 1  # Single-point case

    # Compute per-axis voxel grid sizes (preserving aspect ratio)
    N1 = max(1, round(N * (MX - mX) / maxmax))
    N2 = max(1, round(N * (MY - mY) / maxmax))
    N3 = max(1, round(N * (MZ - mZ) / maxmax))

    voxel_model = np.zeros((N1, N2, N3))

    # Compute mapping from world coordinates to voxel indices
    ax = (N1 - 1) / (MX - mX) if MX != mX else 0
    bx = -mX
    ay = (N2 - 1) / (MY - mY) if MY != mY else 0
    by = -mY
    az = (N3 - 1) / (MZ - mZ) if MZ != mZ else 0
    bz = -mZ

    # Handle NaN from division by zero
    if np.isnan(ax):
        ax = 0
    if np.isnan(ay):
        ay = 0
    if np.isnan(az):
        az = 0

    X2 = np.round(ax * (X + bx)).astype(int)
    Y2 = np.round(ay * (Y + by)).astype(int)
    Z2 = np.round(az * (Z + bz)).astype(int)

    # Clamp to valid range
    X2 = np.clip(X2, 0, N1 - 1)
    Y2 = np.clip(Y2, 0, N2 - 1)
    Z2 = np.clip(Z2, 0, N3 - 1)

    if voxelization_type == 'binary':
        voxel_model[X2, Y2, Z2] = binary_max
    elif voxelization_type == 'density':
        coords = np.column_stack([X2, Y2, Z2])
        unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        voxel_model[unique_coords[:, 0], unique_coords[:, 1], unique_coords[:, 2]] = counts
        # Normalize to [0, 255]
        max_val = voxel_model.max()
        if max_val > 0:
            voxel_model = voxel_model / max_val * 255
    else:
        raise ValueError(f"Unsupported voxelization type: {voxelization_type}")

    # Pad to NxNxN cube with symmetric zero padding (preserves centering)
    nzx = (N - N1) // 2
    nzy = (N - N2) // 2
    nzz = (N - N3) // 2
    final = np.zeros((N, N, N))
    final[nzx:nzx + N1, nzy:nzy + N2, nzz:nzz + N3] = voxel_model

    return final
