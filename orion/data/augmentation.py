"""
3D voxel augmentation for ORION.

The main augmentation used in ORION is random cropping. The voxel grids
are stored at 36x36x36, and we crop to 32x32x32. During training, the
crop location is randomized; during testing, we use center crop.

This is derived from: CreateDeformation + ApplyDeformation behavior.
"""

import torch


class Voxel3DTransform:
    """
    Random/fixed cropping for 3D voxel grids.

    During training, randomly samples offsets in the specified range.
    For testing, use fixed offsets (typically center crop with offset=2).

    Args:
        offset_from: Tuple of (z, y, x) minimum offsets.
        offset_to: Tuple of (z, y, x) maximum offsets (inclusive).
    """

    def __init__(self, offset_from=(0, 0, 0), offset_to=(4, 4, 4)):
        self.offset_from = offset_from
        self.offset_to = offset_to
        self.crop_size = 32

    def __call__(self, voxel):
        # Handle case where voxel doesn't have channel dimension
        if voxel.ndim == 3:
            voxel = voxel.unsqueeze(0)

        # Sample offsets for each axis
        offsets = []
        for i in range(3):
            if self.offset_from[i] == self.offset_to[i]:
                # Fixed offset (for test-time center crop)
                offsets.append(self.offset_from[i])
            else:
                # Random offset (for training augmentation)
                offsets.append(
                    torch.randint(self.offset_from[i], self.offset_to[i] + 1, (1,)).item()
                )

        z, y, x = offsets
        cs = self.crop_size

        return voxel[:, z:z+cs, y:y+cs, x:x+cs]
