"""
Python port of the OFF file loader from modelnet_off2mat.m.

Loads 3D mesh data from Object File Format (.off) files and applies
centering and Z-axis rotation, matching the original MATLAB implementation.
"""

import numpy as np


def load_off(filename, theta=0.0, axis=None, stretch=1.0):
    """
    Load a 3D mesh from an OFF file.

    Port of the off_loader function from modelnet_off2mat.m.

    Args:
        filename: Path to .off file
        theta: Rotation angle in degrees around the Z axis
        axis: Optional axis ('x', 'y', or 'z') for stretching
        stretch: Stretch factor along the specified axis

    Returns:
        dict with keys:
            'vertices': (N, 3) numpy array of vertex coordinates
            'faces': (M, 3) numpy array of face vertex indices (1-based)
    """
    with open(filename, 'r') as f:
        # Read and verify OFF signature
        header = f.readline().strip()
        if not header.startswith('OFF'):
            raise ValueError(f"Not a valid OFF file: {filename}")

        # If 'OFF' is on the same line as counts (e.g., "OFF 100 200 0")
        parts = header.split()
        if len(parts) == 4:
            n_vertices = int(parts[1])
            n_faces = int(parts[2])
        else:
            # Read counts from next line
            info_line = f.readline().strip()
            while info_line == '' or info_line.startswith('#'):
                info_line = f.readline().strip()
            info = info_line.split()
            n_vertices = int(info[0])
            n_faces = int(info[1])

        # Read vertices
        vertices = np.zeros((n_vertices, 3), dtype=np.float64)
        for i in range(n_vertices):
            line = f.readline().strip()
            while line == '' or line.startswith('#'):
                line = f.readline().strip()
            coords = line.split()
            vertices[i] = [float(coords[0]), float(coords[1]), float(coords[2])]

        # Read faces (OFF format: num_verts idx0 idx1 idx2 ...)
        faces = np.zeros((n_faces, 3), dtype=np.int64)
        for i in range(n_faces):
            line = f.readline().strip()
            while line == '' or line.startswith('#'):
                line = f.readline().strip()
            vals = line.split()
            # Skip the first value (number of vertices per face, usually 3)
            faces[i] = [int(vals[1]), int(vals[2]), int(vals[3])]

    # Center the mesh at origin
    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2.0
    vertices = vertices - center

    # Apply stretch along axis if specified
    if axis is not None:
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        if axis.lower() not in axis_map:
            raise ValueError(f"Invalid axis: {axis}")
        vertices[:, axis_map[axis.lower()]] *= stretch

    # Apply Z-axis rotation
    theta_rad = theta * np.pi / 180.0
    R = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad), 0],
        [np.sin(theta_rad),  np.cos(theta_rad), 0],
        [0,                  0,                  1]
    ])
    vertices = vertices @ R  # Same as MATLAB's vertices * R

    # Convert faces to 1-based indexing (OFF uses 0-based, MATLAB polygon2voxel expects 1-based)
    faces = faces + 1

    return {
        'vertices': vertices,
        'faces': faces,
    }
