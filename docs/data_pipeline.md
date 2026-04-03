# Data Pipeline

This document describes the complete data preparation flow from raw ModelNet meshes to training-ready HDF5 files. Understanding this pipeline is important for reproducing results.


## Overview

The pipeline transforms 3D triangle meshes (OFF format) into fixed-size voxel grids stored in HDF5 format. The key steps are:

```
OFF meshes --> voxel grids (numpy) --> binary files --> HDF5 chunks
```

Each step is designed to match the original ORION MATLAB/C pipeline exactly, so that the resulting voxel representations are identical to those used in the paper.


## Step 1: Voxelization (OFF to numpy)

**Input**: `.off` mesh files organized as `<dataset>/<class>/<split>/<name>.off`

**Output**: `.npy` files, one per mesh per rotation

For each mesh, the pipeline:

1. Loads the vertices and face indices from the OFF file.
2. Applies a z-axis rotation (0, 30, 60, ..., 330 degrees, giving 12 orientations per mesh).
3. Normalizes the rotated mesh to fit within a 24x24x24 voxel grid.
4. Rasterizes each triangle into the voxel grid using recursive subdivision. The algorithm subdivides each triangle until its extent in each dimension is smaller than one voxel, then marks the corresponding voxel as occupied.
5. Pads the 24x24x24 grid with 6 voxels on each side, producing a 36x36x36 grid. The extra padding provides room for the random crop augmentation during training.

The voxelization algorithm (`polygon2voxel`) is the most computationally expensive step. The pure-Python version takes about 2 seconds per rotation per mesh. For ModelNet40 (12,311 meshes x 12 rotations = ~148k voxelizations), this adds up to roughly 10 hours on a single core.

The Numba-accelerated version reduces this to about 0.02 seconds per rotation after the initial JIT compilation, and the parallel script distributes work across all available cores.


## Step 2: Binary conversion (numpy to bin)

**Input**: `.npy` voxel grids

**Output**: `.bin` files in ORION binary format

The binary format stores each voxel grid as:
- 6 bytes: three uint16 values for the grid dimensions (d1, d2, d3)
- d1 x d2 x d3 bytes: flattened occupancy data as uint8 (0 or 1)

This format matches the original MATLAB code exactly and serves as an intermediate representation that the HDF5 packing step reads.


## Step 3: HDF5 packing (bin to HDF5)

**Input**: Binary voxel files organized by class and split

**Output**: Chunked HDF5 files with labels

This step reads all binary files, assigns class and pose labels, and writes them into HDF5 files. Each HDF5 file contains three datasets:

- `data`: float32 array of shape (N, 36, 36, 36) containing the voxel grids
- `label`: int array of shape (N,) containing class indices
- `label_pose`: int array of shape (N,) containing orientation indices

Three output directories are created:

**train_allrot_shuffled**: Contains all training samples (every mesh at every rotation) shuffled randomly. Shuffling across classes and orientations is important because the network should not see all rotations of the same object in sequence, which would make the orientation prediction task trivially easy. The data is split into chunks of 1000 samples each to keep individual file sizes manageable and enable partial loading. A `.txt` index file lists all chunk filenames in order.

**test_singlerandrot**: Contains one randomly selected rotation per test object. This is the standard evaluation set used for quick accuracy checks during training.

**test_allrot**: Contains all rotations of all test objects, stored in a specific order: all rotations of object 0, then all rotations of object 1, and so on. This sequential ordering is critical for the voting evaluation, which needs to identify which samples belong to the same physical object.


## Orientation scheme

The pose plan files define the set of discrete orientations. For ModelNet40, `poseplan_MN40.txt` specifies 189 orientation bins derived from sampling the rotation space. However, the actual voxelization only uses 12 rotations (every 30 degrees around z). The 189-bin label space comes from the full rotation discretization used in the paper, where each 30-degree z rotation is further decomposed into finer orientation bins based on the pose plan.

For ModelNet10, the pose plan has 105 bins.


## Compression characteristics

The voxel grids are extremely sparse: a typical object occupies only 2-5% of the grid volume. This means that uncompressed HDF5 files (which store float32 zeros and ones) compress to less than 1% of their original size with gzip. Specifically, the 21 GB ModelNet40 training set compresses to approximately 70 MB. This property is exploited for efficient transfer to remote training environments like Google Colab.


## Data sizes (ModelNet40)

| Dataset | Samples | Uncompressed | Compressed (gzip) |
|---------|---------|-------------|-------------------|
| train_allrot_shuffled | 118,116 | 21 GB | ~70 MB |
| test_singlerandrot | 2,468 | 440 MB | ~1.4 MB |
| test_allrot | 29,616 | 5.2 GB | ~17 MB |

For ModelNet10, the sizes are proportionally smaller (10 classes instead of 40, fewer meshes per class).
