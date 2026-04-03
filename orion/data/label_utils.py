"""
Python port of label generation utilities.

Port of:
    - modelnet_generate_class_and_pose_labels.m
    - modelnet_extract_classname_and_rot_from_filename.m
    - sydney_generate_class_and_pose_labels.m
    - sydney_extract_classname_and_rot_from_filename.m
"""

import os
import re


def load_pose_plan(pose_file):
    """
    Load a pose plan file (e.g., poseplan_MN10.txt).

    Args:
        pose_file: Path to pose plan file (tab-separated: class_name\\tnum_rotations)

    Returns:
        classes: list of class names
        nrot: list of rotation counts per class
    """
    classes = []
    nrot = []
    with open(pose_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                parts = line.split()
            classes.append(parts[0])
            nrot.append(int(parts[1]))
    return classes, nrot


def modelnet_extract_classname_and_rot_from_filename(filename):
    """
    Extract class name and rotation index from a ModelNet binary filename.

    Port of modelnet_extract_classname_and_rot_from_filename.m.

    The filename path is expected to be like:
        .../classes/<class_name>/30/<phase>/<model_name>_<rotation>.bin

    The MATLAB code does:
        s = strsplit(filename, {'.', '/'});
        class_name = s{end-4};
        r = strsplit(s{end-1}, '_');
        rotation = str2double(r(end));

    Args:
        filename: Full path to a .bin file

    Returns:
        class_name: The object class name
        rotation: The rotation index (integer)
    """
    # Split by '/' and '.'
    parts = re.split(r'[/.]', filename)
    # Filter out empty strings
    parts = [p for p in parts if p]

    # class_name is 4 positions from the end (before size/phase/filename/ext)
    class_name = parts[-5]

    # rotation is extracted from the base filename (before extension)
    base_name = parts[-2]  # filename without extension
    rot_parts = base_name.split('_')
    rotation = int(rot_parts[-1])

    return class_name, rotation


def modelnet_generate_class_and_pose_labels(class_name, rotation, classes, nrot,
                                             collate_singlerots=False):
    """
    Generate class and pose labels for a ModelNet sample.

    Port of modelnet_generate_class_and_pose_labels.m.

    Both class_label and pose_label start from 0.

    Args:
        class_name: Object class name string
        rotation: Rotation index (0-based, after subtracting rotation_start_index)
        classes: List of class names (from pose plan)
        nrot: List of rotation counts per class (from pose plan)
        collate_singlerots: If True, single-rotation classes get pose_label=0

    Returns:
        class_label: Integer class label (0-based), -1 if invalid
        pose_label: Integer pose label (0-based), -1 if invalid
    """
    nrot_arr = list(nrot)
    if collate_singlerots:
        nrot_arr = [0 if n == 1 else n for n in nrot_arr]

    if class_name not in classes:
        return -1, -1

    class_label = classes.index(class_name)

    # Cumulative sum with shift (matches MATLAB circshift)
    import numpy as np
    cr = np.cumsum(nrot_arr)
    cr = np.roll(cr, 1)
    cr[0] = 0

    pose_label = int(cr[class_label] + rotation % nrot_arr[class_label])

    if collate_singlerots:
        pose_label += 1
        if nrot_arr[class_label] == 0:
            pose_label = 0

    return class_label, pose_label


# ---- Sydney-specific functions ----

SYDNEY_CLASSES = [
    '4wd', 'building', 'bus', 'car', 'pedestrian', 'pillar',
    'pole', 'traffic_lights', 'traffic_sign', 'tree',
    'truck', 'trunk', 'ute', 'van'
]
SYDNEY_NROT = [18, 9, 9, 18, 1, 1, 1, 9, 18, 1, 18, 1, 18, 18]


def sydney_extract_classname_and_rot_from_filename(filename):
    """
    Extract class name and rotation from a Sydney dataset binary filename.

    Port of sydney_extract_classname_and_rot_from_filename.m.

    Expected path structure: .../<class_name>/<base_name>_r<NN>.bin

    Args:
        filename: Full path to a .bin file

    Returns:
        class_name: The object class name
        rotation: The rotation index (integer)
    """
    parts = re.split(r'[/.]', filename)
    parts = [p for p in parts if p]

    class_name = parts[-4]
    # Rotation is the last 2 characters of the base name part
    base_name = parts[-2]
    rotation = int(base_name[-2:])

    return class_name, rotation


def sydney_generate_class_and_pose_labels(class_name, rotation, collate_singlerots=False):
    """
    Generate class and pose labels for a Sydney dataset sample.

    Port of sydney_generate_class_and_pose_labels.m.

    Args:
        class_name: Object class name string
        rotation: Rotation index (0-based)
        collate_singlerots: If True, single-rotation classes get pose_label=0

    Returns:
        class_label: Integer class label (0-based), -1 if invalid
        pose_label: Integer pose label (0-based), -1 if invalid
    """
    return modelnet_generate_class_and_pose_labels(
        class_name, rotation, SYDNEY_CLASSES, SYDNEY_NROT, collate_singlerots
    )
