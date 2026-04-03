"""
Numba-accelerated polygon2voxel — drop-in replacement for voxelization.py.
Same algorithm, same output, ~20-50x faster via JIT compilation.
First call has ~2s compilation overhead, subsequent calls are fast.
"""
import numpy as np
from numba import njit, types
from numba.core.types import int64, float64, boolean
from scipy import ndimage


@njit(cache=True)
def _draw_or_split_jit(volume, ax, ay, az, bx, by, bz, cx, cy, cz,
                        sizx, sizy, sizz, wrap):
    """Recursive triangle rasterization — JIT compiled."""
    if wrap == 0:
        check_a = (ax < 0 or ay < 0 or az < 0 or
                   ax > sizx - 1 or ay > sizy - 1 or az > sizz - 1)
        check_b = (bx < 0 or by < 0 or bz < 0 or
                   bx > sizx - 1 or by > sizy - 1 or bz > sizz - 1)
        check_c = (cx < 0 or cy < 0 or cz < 0 or
                   cx > sizx - 1 or cy > sizy - 1 or cz > sizz - 1)
        if ((ax < 0 and bx < 0 and cx < 0) or
            (ay < 0 and by < 0 and cy < 0) or
            (az < 0 and bz < 0 and cz < 0) or
            (ax > sizx - 1 and bx > sizx - 1 and cx > sizx - 1) or
            (ay > sizy - 1 and by > sizy - 1 and cy > sizy - 1) or
            (az > sizz - 1 and bz > sizz - 1 and cz > sizz - 1)):
            return
    else:
        check_a = False
        check_b = False
        check_c = False

    dist1 = (ax - bx)**2 + (ay - by)**2 + (az - bz)**2
    dist2 = (cx - bx)**2 + (cy - by)**2 + (cz - bz)**2
    dist3 = (ax - cx)**2 + (ay - cy)**2 + (az - cz)**2

    if dist1 > dist2:
        if dist1 > dist3:
            if dist1 > 0.5:
                dx = (ax + bx) * 0.5; dy = (ay + by) * 0.5; dz = (az + bz) * 0.5
                _draw_or_split_jit(volume, dx, dy, dz, bx, by, bz, cx, cy, cz, sizx, sizy, sizz, wrap)
                _draw_or_split_jit(volume, ax, ay, az, dx, dy, dz, cx, cy, cz, sizx, sizy, sizz, wrap)
        else:
            if dist3 > 0.5:
                dx = (ax + cx) * 0.5; dy = (ay + cy) * 0.5; dz = (az + cz) * 0.5
                _draw_or_split_jit(volume, dx, dy, dz, bx, by, bz, cx, cy, cz, sizx, sizy, sizz, wrap)
                _draw_or_split_jit(volume, ax, ay, az, bx, by, bz, dx, dy, dz, sizx, sizy, sizz, wrap)
    else:
        if dist2 > dist3:
            if dist2 > 0.5:
                dx = (cx + bx) * 0.5; dy = (cy + by) * 0.5; dz = (cz + bz) * 0.5
                _draw_or_split_jit(volume, ax, ay, az, dx, dy, dz, cx, cy, cz, sizx, sizy, sizz, wrap)
                _draw_or_split_jit(volume, ax, ay, az, bx, by, bz, dx, dy, dz, sizx, sizy, sizz, wrap)
        else:
            if dist3 > 0.5:
                dx = (ax + cx) * 0.5; dy = (ay + cy) * 0.5; dz = (az + cz) * 0.5
                _draw_or_split_jit(volume, dx, dy, dz, bx, by, bz, cx, cy, cz, sizx, sizy, sizz, wrap)
                _draw_or_split_jit(volume, ax, ay, az, bx, by, bz, dx, dy, dz, sizx, sizy, sizz, wrap)

    # Mark voxels
    if wrap == 0:
        if not check_a:
            ix = int(ax + 0.5); iy = int(ay + 0.5); iz = int(az + 0.5)
            volume[iz * sizx * sizy + iy * sizx + ix] = True
        if not check_b:
            ix = int(bx + 0.5); iy = int(by + 0.5); iz = int(bz + 0.5)
            volume[iz * sizx * sizy + iy * sizx + ix] = True
        if not check_c:
            ix = int(cx + 0.5); iy = int(cy + 0.5); iz = int(cz + 0.5)
            volume[iz * sizx * sizy + iy * sizx + ix] = True
    else:
        if wrap == 1:
            ax2=ax%sizx; ay2=ay%sizy; az2=az%sizz
            bx2=bx%sizx; by2=by%sizy; bz2=bz%sizz
            cx2=cx%sizx; cy2=cy%sizy; cz2=cz%sizz
        else:
            ax2=max(0,min(int(ax),sizx-1)); ay2=max(0,min(int(ay),sizy-1)); az2=max(0,min(int(az),sizz-1))
            bx2=max(0,min(int(bx),sizx-1)); by2=max(0,min(int(by),sizy-1)); bz2=max(0,min(int(bz),sizz-1))
            cx2=max(0,min(int(cx),sizx-1)); cy2=max(0,min(int(cy),sizy-1)); cz2=max(0,min(int(cz),sizz-1))
        volume[int(az2)*sizx*sizy + int(ay2)*sizx + int(ax2)] = True
        volume[int(bz2)*sizx*sizy + int(by2)*sizx + int(bx2)] = True
        volume[int(cz2)*sizx*sizy + int(cy2)*sizx + int(cx2)] = True


@njit(cache=True)
def _rasterize_faces(vol_flat, vertices, faces, sizx, sizy, sizz, wrap):
    """Rasterize all faces into the volume — JIT compiled loop."""
    for i in range(faces.shape[0]):
        ai = int(faces[i, 0]) - 1
        bi = int(faces[i, 1]) - 1
        ci = int(faces[i, 2]) - 1
        _draw_or_split_jit(
            vol_flat,
            vertices[ai, 0] - 1.0, vertices[ai, 1] - 1.0, vertices[ai, 2] - 1.0,
            vertices[bi, 0] - 1.0, vertices[bi, 1] - 1.0, vertices[bi, 2] - 1.0,
            vertices[ci, 0] - 1.0, vertices[ci, 1] - 1.0, vertices[ci, 2] - 1.0,
            sizx, sizy, sizz, wrap
        )


def polygon2voxel(vertices, faces, volume_size, mode='auto', yxz=True, hollow=False):
    """Drop-in replacement for voxelization.polygon2voxel, accelerated with numba."""
    vertices = np.array(vertices, dtype=np.float64).copy()
    faces = np.array(faces, dtype=np.int64).copy()

    if isinstance(volume_size, (int, float)):
        volume_size = np.array([volume_size, volume_size, volume_size])
    else:
        volume_size = np.array(volume_size, dtype=np.float64)
    volume_size = np.round(volume_size).astype(np.int64)

    if yxz:
        vertices = vertices[:, [1, 0, 2]]

    mode_key = mode[:2].lower()
    if mode_key == 'au':
        vertices -= vertices.min(axis=0)
        scaling = np.min((volume_size - 1) / vertices.max())
        vertices = vertices * scaling + 1
        offset = volume_size / 2.0 - np.max(vertices, axis=0) / 2.0
        vertices += offset
        wrap = 0
    elif mode_key == 'ce':
        vertices += volume_size / 2.0
        wrap = 0
    elif mode_key == 'wr':
        wrap = 1
    elif mode_key == 'cl':
        wrap = 2
    else:
        wrap = 0

    sizx, sizy, sizz = int(volume_size[0]), int(volume_size[1]), int(volume_size[2])
    vol_flat = np.zeros(sizx * sizy * sizz, dtype=np.bool_)

    _rasterize_faces(vol_flat, vertices, faces, sizx, sizy, sizz, wrap)

    volume = vol_flat.reshape(sizz, sizy, sizx).transpose(2, 1, 0)
    if not hollow:
        volume = ndimage.binary_fill_holes(volume)
    return volume
