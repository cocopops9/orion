"""
Python port of the OFF file loader from modelnet_off2mat.m.

Loads 3D mesh data from Object File Format (.off) files and applies
centering and Z-axis rotation, matching the original MATLAB implementation.

FIXED VERSION - Handles various OFF file format variations robustly.
"""

import numpy as np


def load_off(filename, theta=0.0, axis=None, stretch=1.0):
    """
    Load a 3D mesh from an OFF file.

    Port of the off_loader function from modelnet_off2mat.m with enhanced
    robustness to handle various OFF file format variations.

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
    try:
        # Read all lines and strip whitespace, filter out empty lines
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        if not lines:
            raise ValueError(f"Empty file: {filename}")
        
        # Find the OFF header (skip any leading comments or blank lines)
        off_line_idx = None
        for i, line in enumerate(lines):
            if line.startswith('OFF'):
                off_line_idx = i
                break
            if not line.startswith('#'):
                # Non-comment, non-OFF line before header
                raise ValueError(f"Invalid OFF file (no OFF header): {filename}")
        
        if off_line_idx is None:
            raise ValueError(f"Invalid OFF file (no OFF header found): {filename}")
        
        # Check if counts are on the same line as OFF (e.g., "OFF 100 200 0")
        off_line = lines[off_line_idx]
        parts = off_line.split()
        
        if len(parts) == 4 and parts[0] == 'OFF':
            # Counts are inline: OFF n_vertices n_faces n_edges
            try:
                n_vertices = int(parts[1])
                n_faces = int(parts[2])
                vertex_start_idx = off_line_idx + 1
            except (ValueError, IndexError) as e:
                raise ValueError(f"Invalid inline counts in {filename}: {off_line}") from e
        
        elif len(parts) == 1 and parts[0] == 'OFF':
            # Counts are on the next line
            # Skip any comments between OFF and counts
            count_line_idx = off_line_idx + 1
            while count_line_idx < len(lines) and lines[count_line_idx].startswith('#'):
                count_line_idx += 1
            
            if count_line_idx >= len(lines):
                raise ValueError(f"No count line found after OFF header in {filename}")
            
            count_line = lines[count_line_idx]
            count_parts = count_line.split()
            
            # Validate that we have at least 2 integers (vertices and faces)
            if len(count_parts) < 2:
                raise ValueError(f"Invalid count line in {filename}: '{count_line}'")
            
            # Check if the first value is actually an integer (not a float)
            try:
                n_vertices = int(count_parts[0])
                n_faces = int(count_parts[1])
            except ValueError as e:
                # This is where the error was occurring!
                # The count line is actually a vertex coordinate
                raise ValueError(
                    f"Invalid count line in {filename}. Expected integers, got: '{count_line}'. "
                    f"First value '{count_parts[0]}' cannot be parsed as integer. "
                    f"This likely means the file is corrupted or has an invalid format."
                ) from e
            
            vertex_start_idx = count_line_idx + 1
        
        else:
            raise ValueError(f"Invalid OFF header format in {filename}: '{off_line}'")
        
        # Read vertices (skip comments)
        vertices = []
        current_idx = vertex_start_idx
        
        while len(vertices) < n_vertices and current_idx < len(lines):
            line = lines[current_idx]
            if not line.startswith('#'):
                try:
                    coords = line.split()
                    if len(coords) >= 3:
                        vertices.append([float(coords[0]), float(coords[1]), float(coords[2])])
                except (ValueError, IndexError) as e:
                    raise ValueError(f"Invalid vertex at line {current_idx} in {filename}: '{line}'") from e
            current_idx += 1
        
        if len(vertices) != n_vertices:
            raise ValueError(
                f"Expected {n_vertices} vertices but found {len(vertices)} in {filename}"
            )
        
        vertices = np.array(vertices, dtype=np.float64)
        
        # Read faces (skip comments)
        faces = []
        
        while len(faces) < n_faces and current_idx < len(lines):
            line = lines[current_idx]
            if not line.startswith('#'):
                try:
                    vals = line.split()
                    # OFF format: num_verts idx0 idx1 idx2 ...
                    # We expect triangular faces (3 vertices)
                    num_face_verts = int(vals[0])
                    if num_face_verts >= 3:
                        # Take the first 3 vertices (triangulate if needed)
                        faces.append([int(vals[1]), int(vals[2]), int(vals[3])])
                except (ValueError, IndexError) as e:
                    raise ValueError(f"Invalid face at line {current_idx} in {filename}: '{line}'") from e
            current_idx += 1
        
        if len(faces) != n_faces:
            raise ValueError(
                f"Expected {n_faces} faces but found {len(faces)} in {filename}"
            )
        
        faces = np.array(faces, dtype=np.int64)
        
    except Exception as e:
        # Re-raise with filename context
        raise ValueError(f"Error loading OFF file {filename}: {str(e)}") from e
    
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
