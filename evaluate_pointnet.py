"""
PointNet Evaluation Script with Test-Time Voting.

Implements the same 12-rotation voting protocol as ORION (Equation 5 in
the paper) for fair cross-architecture comparison:

    c* = argmax_k sum_{r=0}^{11} S_k(x_r)

Each test object is evaluated at all 12 rotation angles. The voxel grid
at each rotation is converted to a point cloud, passed through PointNet,
and the softmax scores are summed across rotations before taking argmax.

This way we can directly compare PointNet vs ORION on the same data
with the same evaluation protocol.
"""

import os
import argparse
import yaml
import numpy as np
import h5py
import torch
from collections import defaultdict

from orion.models.pointnet import PointNet
from orion.data.pointcloud_dataset import voxel_to_pointcloud


def evaluate_voting(model, test_hdf5_path, device, num_points=1024, batch_size=256):
    """
    12-rotation voting evaluation matching ORION protocol.

    We do the same thing as ORION: run all 12 rotations of each test object,
    sum up the softmax scores, and pick the class with highest total.
    This gives us a fair comparison between architectures.

    Args:
        model: Trained PointNet model.
        test_hdf5_path: Path to test_allrot HDF5 file (12 rotations per
            object, stored sequentially).
        device: torch device.
        num_points: Number of points per sample.
        batch_size: Batch size for inference.

    Returns:
        overall_acc: Overall voting accuracy (%).
        mean_class_acc: Mean per-class accuracy (%).
        per_class_acc: List of per-class accuracies (%).
    """
    with h5py.File(test_hdf5_path, 'r') as hf:
        data = hf['data'][:]
        labels = hf['label'][:]

    n_samples = len(labels)
    n_rotations = 12
    n_objects = n_samples // n_rotations

    print(f"Total samples: {n_samples}, Objects: {n_objects}, "
          f"Rotations: {n_rotations}")

    # Convert all voxels to point clouds
    # This is the expensive part but we only do it once
    print("Converting voxels to point clouds...")
    all_points = np.zeros((n_samples, num_points, 3), dtype=np.float32)
    for i in range(n_samples):
        all_points[i] = voxel_to_pointcloud(data[i], num_points)

    # Run inference in batches
    all_scores = []
    model.eval()
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = torch.from_numpy(all_points[start:end]).float().to(device)
            logits, _ = model(batch)
            scores = torch.softmax(logits, dim=1)
            all_scores.append(scores.cpu())

    all_scores = torch.cat(all_scores, dim=0)  # (n_samples, num_classes)

    # Aggregate by object: sum scores across rotations
    num_classes = int(labels.max()) + 1
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    correct = 0

    for obj_id in range(n_objects):
        start = obj_id * n_rotations
        end = start + n_rotations

        gt_class = int(labels[start])
        summed_scores = all_scores[start:end].sum(dim=0)
        pred_class = summed_scores.argmax().item()

        class_total[gt_class] += 1
        if pred_class == gt_class:
            correct += 1
            class_correct[gt_class] += 1

    overall_acc = 100.0 * correct / n_objects

    per_class_acc = []
    for c in range(num_classes):
        if class_total[c] > 0:
            per_class_acc.append(100.0 * class_correct[c] / class_total[c])
        else:
            per_class_acc.append(0.0)

    mean_class_acc = np.mean(per_class_acc)
    return overall_acc, mean_class_acc, per_class_acc


def main():
    parser = argparse.ArgumentParser(description='PointNet Evaluation')
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--checkpoint', required=True, help='Path to .pth')
    parser.add_argument('--test_allrot', required=True,
                        help='Path to test_allrot HDF5 .txt file or .hdf5')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_classes = config['model']['num_classes']
    num_points = config['model'].get('num_points', 1024)

    model = PointNet(
        num_classes=num_classes,
        use_feature_transform=config['model'].get('use_feature_transform', True)
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"PointNet ({total_params:,} params), "
          f"epoch {ckpt.get('epoch', '?')}, "
          f"val acc {ckpt.get('class_accuracy', 0):.2f}%")

    # Resolve test HDF5 path
    # Sometimes we get a .txt file that points to the real HDF5
    test_path = args.test_allrot
    if test_path.endswith('.txt'):
        with open(test_path) as f:
            resolved = f.readline().strip()
            if not os.path.isabs(resolved):
                resolved = os.path.join(os.path.dirname(test_path), resolved)
            test_path = resolved

    print(f"\n=== Voting Evaluation (12 rotations) ===")
    print(f"Test file: {test_path}")
    print(f"Points per sample: {num_points}")

    overall, mean_class, per_class = evaluate_voting(
        model, test_path, device, num_points
    )

    print(f"\nOverall accuracy:        {overall:.2f}%")
    print(f"Mean per-class acc:      {mean_class:.2f}%")
    print(f"\nPer-class:")
    for i, acc in enumerate(per_class):
        print(f"  Class {i:2d}: {acc:.2f}%")


if __name__ == '__main__':
    main()
