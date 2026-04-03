# Code Documentation

This document describes each script and module in the project. It is intended as a reference for understanding what each file does, how it fits into the overall pipeline, and what its key design decisions are.


## Top-level scripts

### train.py

Main training script for the three ORION architectures (basic, fast, extended).

The training loop jointly optimizes two objectives: object classification and orientation prediction. The total loss is a weighted combination of the two cross-entropy terms, controlled by a gamma parameter (default 0.5, meaning equal weight). This dual-task formulation is the central contribution of the ORION paper: predicting orientation as an auxiliary task forces the network to learn viewpoint-aware features, which improves classification accuracy.

The script reads all hyperparameters from a YAML config file. It supports two learning rate schedules: step decay (default, matches the original paper) and cosine annealing with warm restarts (useful for longer training runs). Training progress is logged to a CSV file for later plotting, and the best model checkpoint is saved whenever validation accuracy improves.

Data loading uses lazy HDF5 access, meaning only the voxel grid for the current sample is read from disk at each iteration. This keeps memory usage low even for large datasets like ModelNet40 (118k training samples). The tradeoff is that `num_workers` must be set to 0 because h5py file handles cannot be shared across forked processes.


### train_pointnet.py

Training script for the PointNet baseline. It uses the same HDF5 data as ORION but converts each voxel grid to a 1024-point cloud on the fly. This conversion samples points uniformly from occupied voxel positions, giving an apples-to-apples comparison: same data, same train/test splits, different input representation.

PointNet is trained with Adam optimizer (rather than SGD) and includes the feature transform regularization from the original PointNet paper, which penalizes the feature transformation matrix for deviating from orthogonality. This regularization helps stabilize training and is weighted by a configurable parameter (default 0.001).


### evaluate.py

Evaluation script for ORION models with two modes.

In standard mode, it runs inference on the single-rotation test set (`test_singlerandrot`) and reports overall accuracy, mean per-class accuracy, and pose prediction accuracy.

In voting mode, it implements the test-time aggregation procedure from the ORION paper (Equation 3). Each test object has 12 rotations stored in the all-rotation test set (`test_allrot`). The script runs all 12 through the network, sums the softmax probability vectors, and takes the argmax as the final prediction. This voting procedure typically improves accuracy by 1-2 percentage points over single-rotation evaluation. The voting evaluation loads all HDF5 chunks listed in the `.txt` index file and concatenates them before processing.


### evaluate_pointnet.py

Evaluation script for the PointNet baseline using the same 12-rotation voting protocol. Each rotation's voxel grid is converted to a point cloud and passed through PointNet, and the softmax scores are summed across rotations before taking argmax. This ensures a fair comparison between PointNet and ORION under identical evaluation conditions.


### run_experiment.sh

Shell script that dispatches training and evaluation for any combination of models and datasets. It accepts experiment names like `modelnet40_fast` or `all_mn40` and runs the corresponding training followed by evaluation (both standard and voting when the all-rotation data is available). The script verifies that the required HDF5 data exists before starting, to avoid wasting time on runs that would fail at data loading.


### fast_voxelize.py

Parallel voxelization script for building the ModelNet datasets faster than the sequential pipeline. It distributes OFF mesh files across multiple worker processes using Python's `multiprocessing.Pool`. Each worker loads a mesh, rotates it to each of the 12 orientations, voxelizes it using the original polygon-to-voxel algorithm, pads the result, and saves it as a compressed numpy array.

If `numba` is installed, the script imports the JIT-compiled voxelizer from `orion/data/voxelization_fast.py`, which runs approximately 100 times faster per voxelization call than the pure-Python version. The script performs a warmup call in the main process before forking workers, so the compiled code is shared via copy-on-write memory and each worker does not need to recompile.

The script supports resume: before processing each file, it checks whether all 12 rotation outputs already exist and skips completed files.


### plot_losses.py

Reads the CSV training logs produced by the training scripts and generates loss curve plots using matplotlib. Produces separate plots for training loss, validation loss, and accuracy over epochs.


### build_modelnet10.sh and build_modelnet40.sh

Shell scripts that automate the full data preparation pipeline from raw meshes to training-ready HDF5 files. The pipeline has four stages:

1. **download**: Fetches the raw OFF mesh files. ModelNet10 comes from the standard Princeton repository. ModelNet40 uses the orientation-aligned meshes provided by the ORION authors (available in auto-aligned or manually-aligned variants).

2. **voxelize**: Converts each OFF mesh to a 24x24x24 voxel grid at each of 12 rotation angles (every 30 degrees around the z-axis), then pads to 30x30x30. The voxelization uses the polygon2voxel algorithm ported from the original MATLAB/C code.

3. **bin**: Converts the numpy voxel arrays to the ORION binary format (6-byte dimension header followed by flattened uint8 occupancy data).

4. **hdf5**: Packs the binary files into labeled HDF5 chunks of 1000 samples each, with shuffled training data and separate test sets for single-rotation and all-rotation evaluation.

Each stage can be run independently by passing its name as an argument. A final `verify` stage checks that the generated HDF5 files can be opened and contain the expected number of samples.


### link_existing_hdf5.sh

Utility script for linking pre-built HDF5 data into the project without copying. It creates symlinks from a source directory to the expected locations under `datasets/`, handling both single-file and chunked layouts. It also regenerates the `.txt` index files to match whatever chunks are present. The `--copy` flag can be used instead of symlinks when the source is on a different filesystem.


## Core library (orion/)

### orion/models/orion_basic.py

PyTorch implementation of the original ORION architecture from the paper. The network is a compact 3D CNN with two convolutional layers followed by one fully connected layer and two output heads (classification and orientation).

The architecture processes a 1x32x32x32 input voxel grid through:
- Conv1: 32 filters of size 5x5x5 with stride 2, followed by batch normalization, LeakyReLU (slope 0.1), and 20% dropout.
- Conv2: 32 filters of size 3x3x3 with stride 1, followed by batch normalization, LeakyReLU, 2x2x2 max pooling, and 30% dropout.
- FC6: 128 units with ReLU and 40% dropout.
- Two linear heads: one for class prediction, one for orientation prediction.

Weight initialization follows the original Caffe implementation: Kaiming (MSRA) initialization for convolutional layers and Gaussian (std=0.01) for fully connected layers.


### orion/models/orion_fast.py

Lightweight ORION variant with the same parameter count as Basic but optimized for faster convergence. The architectural differences are minor; the main improvement comes from using label smoothing during training (configured in the YAML, not hard-coded in the model). This variant achieves 93.50% on ModelNet10 and 86.63% on ModelNet40.


### orion/models/orion_extended.py

Deeper 4-layer variant described in Table 3 of the paper's supplementary material. It adds two additional convolutional layers with progressively more filters (32, 64, 64, 128), which increases the model capacity and improves performance on the larger ModelNet40 dataset at the cost of more parameters (~1.2M vs ~482K for Basic/Fast).


### orion/models/pointnet.py

Standard PointNet implementation for comparison. Takes Nx3 point clouds as input and applies shared MLPs (implemented as 1D convolutions over the point dimension), a max-pooling global feature aggregation, and a classification head. Optionally includes the feature transform network, which learns a 64x64 transformation matrix applied to the per-point features. The `feature_transform_regularization` static method computes the orthogonality penalty used during training.


### orion/data/hdf5_dataset.py

PyTorch Dataset for loading voxel grids from chunked HDF5 files. The constructor reads a `.txt` index file listing the HDF5 chunks, opens all of them, and loads only the label arrays into memory. Voxel grids are read lazily from disk on each `__getitem__` call using HDF5 direct indexing, keeping RAM usage proportional to the label arrays rather than the full voxel data.

A global-to-local index mapping translates dataset indices to the correct file handle and offset within that file. Because h5py file handles cannot be safely shared across forked processes, `num_workers` must be set to 0 in the DataLoader when using this dataset.


### orion/data/pointcloud_dataset.py

Dataset wrapper that converts voxel grids to point clouds for PointNet. On each access, it reads a voxel grid from the underlying HDF5 files and converts it to a fixed-size point cloud by sampling `num_points` coordinates (default 1024) from the occupied voxel positions. If the voxel grid has fewer occupied voxels than requested, points are resampled with replacement. Optional augmentation adds small random jitter to point positions during training.


### orion/data/augmentation.py

Implements the 3D random crop augmentation used in ORION. The voxel grids are stored at 36x36x36 (24 voxels plus 6 padding on each side) and cropped to 32x32x32 for the network input. During training, the crop offset is sampled uniformly from a specified range (default 0 to 4 on each axis), providing translation augmentation. During testing, a fixed center crop at offset (2, 2, 2) is used. This corresponds to the `CreateDeformation` and `ApplyDeformation` operations in the original Caffe pipeline.


### orion/data/voxelization.py

Pure-Python implementation of the polygon-to-voxel conversion algorithm. This is a direct port of the original MATLAB/C `polygon2voxel` function from the ORION repository. Given a triangle mesh (vertices and face indices), it rasterizes each triangle into a 3D voxel grid using recursive subdivision. The mesh is first normalized to fit within the target volume, then each triangle is subdivided until it is small enough to fill individual voxels.

The algorithm operates in "auto" mode by default, which selects between solid and surface voxelization based on the mesh properties. For closed meshes, it fills the interior; for open meshes, it only marks the surface voxels.


### orion/data/voxelization_fast.py

Numba JIT-compiled version of the polygon-to-voxel algorithm. Produces bit-identical output to the pure-Python version but runs approximately 100 times faster per call by compiling the inner triangle rasterization loop with `@njit(cache=True)`. The first call incurs a compilation overhead of a few seconds, but subsequent calls (including across different sessions when caching is enabled) run at native speed.


### orion/data/off_loader.py

Reader for the OFF (Object File Format) mesh format used by ModelNet. Parses the vertex coordinates and face indices from `.off` files, with support for applying a z-axis rotation before returning the data. The rotation angle is specified in degrees and is used during data preparation to generate the multiple orientations required by the ORION training procedure.


### orion/data/binary_io.py

I/O functions for the ORION binary voxel format. Each file stores a 3D voxel grid as a 6-byte header (three uint16 dimension values) followed by the flattened occupancy data as uint8 values. This format matches the original MATLAB code and is used as an intermediate step between voxelization and HDF5 packing.


### orion/data/label_utils.py

Utilities for parsing pose plan files and generating class/orientation labels. The pose plan files (`poseplan_MN10.txt` and `poseplan_MN40.txt`) define the discrete set of orientations used during voxelization. Each line specifies a rotation angle, and the total number of lines determines the number of orientation bins for the network output. This module maps each voxelized sample to its corresponding class label (derived from the directory name) and pose label (derived from the rotation angle index).


### orion/data/prepare_modelnet.py

The complete data preparation pipeline invoked by the build scripts. It orchestrates three stages: voxelization (OFF meshes to numpy arrays), binary conversion (numpy to ORION binary format), and HDF5 packing (binary files to labeled, chunked HDF5). The HDF5 packing stage creates three output directories:

- `train_allrot_shuffled`: all training samples with all rotations, randomly shuffled across classes and orientations, split into chunks of 1000 samples.
- `test_singlerandrot`: one randomly selected rotation per test object, used for standard evaluation.
- `test_allrot`: all rotations of all test objects in sequential order (object 0 rotation 0, object 0 rotation 1, ...), used for voting evaluation.


## Configuration files

The YAML configs under `configs/` control all aspects of training. They follow a common structure with three sections: `model` (architecture and output dimensions), `data` (paths to HDF5 index files), `augmentation` (crop offset ranges), and `training` (optimizer settings, schedule, and checkpoint paths). The configs are designed to be self-contained: changing a config file is sufficient to switch between architectures, datasets, or hyperparameter settings without modifying any code.
