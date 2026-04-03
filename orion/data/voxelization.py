"""
Python port of polygon2voxel (originally polygon2voxel_double.c + polygon2voxel.m).

Converts a triangulated mesh into a 3D voxel volume by recursive triangle
subdivision until edges are smaller than 0.5 voxels, then setting voxels at
vertex positions.

Original: D. Kroon, University of Twente (May 2009), modified by Jianxiong.
"""

import numpy as np
from scipy import ndimage


def _mindex3(x, y, z, sizx, sizy, sizz, wrap):
    """Compute linear index into 3D volume with optional wrapping/clamping."""
    if wrap == 1:
        # Positive modulo (wrap mode)
        x = x % sizx
        y = y % sizy
        z = z % sizz
    elif wrap > 1:
        # Clamp mode
        x = max(0, min(x, sizx - 1))
        y = max(0, min(y, sizy - 1))
        z = max(0, min(z, sizz - 1))
    return z * sizx * sizy + y * sizx + x


def _draw_or_split(volume, ax, ay, az, bx, by, bz, cx, cy, cz, volume_size, wrap):
    """
    Recursively subdivide triangle and rasterize into voxel volume.

    Splits the longest edge of the triangle until all edges are < 0.5 voxels,
    then marks the voxel at each vertex position.
    """
    sizx, sizy, sizz = volume_size

    # Check if vertices are outside the volume (only in non-wrap mode)
    if wrap == 0:
        check_a = (ax < 0 or ay < 0 or az < 0 or
                   ax > sizx - 1 or ay > sizy - 1 or az > sizz - 1)
        check_b = (bx < 0 or by < 0 or bz < 0 or
                   bx > sizx - 1 or by > sizy - 1 or bz > sizz - 1)
        check_c = (cx < 0 or cy < 0 or cz < 0 or
                   cx > sizx - 1 or cy > sizy - 1 or cz > sizz - 1)

        # Early rejection: all vertices outside on the same side
        if ((ax < 0 and bx < 0 and cx < 0) or
            (ay < 0 and by < 0 and cy < 0) or
            (az < 0 and bz < 0 and cz < 0) or
            (ax > sizx - 1 and bx > sizx - 1 and cx > sizx - 1) or
            (ay > sizy - 1 and by > sizy - 1 and cy > sizy - 1) or
            (az > sizz - 1 and bz > sizz - 1 and cz > sizz - 1)):
            return
    else:
        check_a = check_b = check_c = False

    # Compute squared edge lengths
    dist1 = (ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2
    dist2 = (cx - bx) ** 2 + (cy - by) ** 2 + (cz - bz) ** 2
    dist3 = (ax - cx) ** 2 + (ay - cy) ** 2 + (az - cz) ** 2

    # Find longest edge and split if > 0.5
    if dist1 > dist2:
        if dist1 > dist3:
            maxdist = dist1
            if maxdist > 0.5:
                dx, dy, dz = (ax + bx) / 2, (ay + by) / 2, (az + bz) / 2
                _draw_or_split(volume, dx, dy, dz, bx, by, bz, cx, cy, cz, volume_size, wrap)
                _draw_or_split(volume, ax, ay, az, dx, dy, dz, cx, cy, cz, volume_size, wrap)
        else:
            maxdist = dist3
            if maxdist > 0.5:
                dx, dy, dz = (ax + cx) / 2, (ay + cy) / 2, (az + cz) / 2
                _draw_or_split(volume, dx, dy, dz, bx, by, bz, cx, cy, cz, volume_size, wrap)
                _draw_or_split(volume, ax, ay, az, bx, by, bz, dx, dy, dz, volume_size, wrap)
    else:
        if dist2 > dist3:
            maxdist = dist2
            if maxdist > 0.5:
                dx, dy, dz = (cx + bx) / 2, (cy + by) / 2, (cz + bz) / 2
                _draw_or_split(volume, ax, ay, az, dx, dy, dz, cx, cy, cz, volume_size, wrap)
                _draw_or_split(volume, ax, ay, az, bx, by, bz, dx, dy, dz, volume_size, wrap)
        else:
            maxdist = dist3
            if maxdist > 0.5:
                dx, dy, dz = (ax + cx) / 2, (ay + cy) / 2, (az + cz) / 2
                _draw_or_split(volume, dx, dy, dz, bx, by, bz, cx, cy, cz, volume_size, wrap)
                _draw_or_split(volume, ax, ay, az, bx, by, bz, dx, dy, dz, volume_size, wrap)

    # Mark voxels at vertex positions
    if wrap == 0:
        if not check_a:
            idx = _mindex3(int(ax + 0.5), int(ay + 0.5), int(az + 0.5), sizx, sizy, sizz, wrap)
            volume[idx] = 1
        if not check_b:
            idx = _mindex3(int(bx + 0.5), int(by + 0.5), int(bz + 0.5), sizx, sizy, sizz, wrap)
            volume[idx] = 1
        if not check_c:
            idx = _mindex3(int(cx + 0.5), int(cy + 0.5), int(cz + 0.5), sizx, sizy, sizz, wrap)
            volume[idx] = 1
    else:
        volume[_mindex3(int(ax + 0.5), int(ay + 0.5), int(az + 0.5), sizx, sizy, sizz, wrap)] = 1
        volume[_mindex3(int(bx + 0.5), int(by + 0.5), int(bz + 0.5), sizx, sizy, sizz, wrap)] = 1
        volume[_mindex3(int(cx + 0.5), int(cy + 0.5), int(cz + 0.5), sizx, sizy, sizz, wrap)] = 1


def polygon2voxel(vertices, faces, volume_size, mode='auto', yxz=True, hollow=False):
    """
    Convert a triangulated mesh to a voxel volume.

    Port of the original MATLAB polygon2voxel function.

    Args:
        vertices: (N, 3) array of vertex coordinates
        faces: (M, 3) array of face indices (1-based, as in MATLAB)
        volume_size: int or (3,) array specifying output volume dimensions
        mode: Coordinate handling mode:
            'auto' - Scale and translate vertices to fit volume
            'center' - Place coordinate origin at volume center
            'wrap' - Circular wrapping for out-of-bounds coordinates
            'clamp' - Clamp out-of-bounds to volume edges
            'none' - Use vertex coordinates directly
        yxz: If True, use MATLAB convention (1st dim=Y, 2nd=X, 3rd=Z)
        hollow: If True, skip hole-filling step

    Returns:
        volume: 3D boolean numpy array
    """
    vertices = np.array(vertices, dtype=np.float64).copy()
    faces = np.array(faces, dtype=np.int64).copy()

    if isinstance(volume_size, (int, float)):
        volume_size = np.array([volume_size, volume_size, volume_size])
    else:
        volume_size = np.array(volume_size, dtype=np.float64)

    volume_size = np.round(volume_size).astype(int)

    if vertices.shape[1] != 3:
        raise ValueError("Vertices must be an Nx3 array")
    if faces.shape[1] != 3:
        raise ValueError("Faces must be an Mx3 array")
    if np.max(faces) > vertices.shape[0]:
        raise ValueError("Face list contains undefined vertex index")
    if np.min(faces) < 1:
        raise ValueError("Face list contains vertex index < 1")

    # MATLAB dimension convention: swap X and Y
    if yxz:
        vertices = vertices[:, [1, 0, 2]]

    mode_key = mode[:2].lower()
    if mode_key == 'au':  # auto
        # Translate vertices to origin
        vertices -= vertices.min(axis=0)
        # Scale uniformly to fit volume
        scaling = np.min((volume_size - 1) / vertices.max())
        vertices = vertices * scaling + 1
        # Center in volume
        offset = volume_size / 2.0 - np.max(vertices, axis=0) / 2.0
        vertices += offset
        wrap = 0
    elif mode_key == 'ce':  # center
        vertices += volume_size / 2.0
        wrap = 0
    elif mode_key == 'wr':  # wrap
        wrap = 1
    elif mode_key == 'cl':  # clamp
        wrap = 2
    else:  # none
        wrap = 0

    # Create flat volume array
    vol_flat = np.zeros(int(np.prod(volume_size)), dtype=np.bool_)

    faces_a = faces[:, 0].astype(np.float64)
    faces_b = faces[:, 1].astype(np.float64)
    faces_c = faces[:, 2].astype(np.float64)
    vx = vertices[:, 0].astype(np.float64)
    vy = vertices[:, 1].astype(np.float64)
    vz = vertices[:, 2].astype(np.float64)

    vol_size_tuple = (int(volume_size[0]), int(volume_size[1]), int(volume_size[2]))

    # Rasterize each face
    for i in range(len(faces_a)):
        # Convert to 0-based indexing (faces are 1-based from MATLAB)
        ai = int(faces_a[i]) - 1
        bi = int(faces_b[i]) - 1
        ci = int(faces_c[i]) - 1
        _draw_or_split(
            vol_flat,
            vx[ai] - 1.0, vy[ai] - 1.0, vz[ai] - 1.0,
            vx[bi] - 1.0, vy[bi] - 1.0, vz[bi] - 1.0,
            vx[ci] - 1.0, vy[ci] - 1.0, vz[ci] - 1.0,
            vol_size_tuple, wrap
        )

    # Reshape to 3D
    volume = vol_flat.reshape(vol_size_tuple[2], vol_size_tuple[1], vol_size_tuple[0])
    volume = volume.transpose(2, 1, 0)  # Match MATLAB column-major ordering

    # Fill holes (matches MATLAB's imfill('holes'))
    if not hollow:
        volume = ndimage.binary_fill_holes(volume)

    return volume
