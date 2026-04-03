"""
ORION Evaluation Script with Test-Time Voting

This implements the voting procedure from the paper (Equation 3):
    c_final = argmax_k sum_r S_k(x_r)

Basically, given a test object, rotate it 12 times, run inference
on each rotation, sum up the softmax scores, and pick the class with highest
total score. This is how the paper reports their numbers, so I comply to this
to have a fair comparison.

Two modes:
  - Standard: just run on test_singlerandrot
  - Voting: use all 12 rotations per object (proper evaluation)
"""

import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
import h5py
import os

from orion.models import ORIONBasic, ORIONFast, ORIONExtended
from orion.data import VoxelHDF5Dataset, Voxel3DTransform


# same registry as in train.py 
MODEL_REGISTRY = {
    "basic": ORIONBasic,
    "fast": ORIONFast,
    "extended": ORIONExtended,
}


def evaluate_standard(model, test_loader, device):
    #Basic single-shot evaluation.
    model.eval()
    correct_c = correct_p = total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    with torch.no_grad():
        for voxels, lc, lp in test_loader:
            voxels = voxels.to(device)
            lc = torch.tensor(lc, dtype=torch.long, device=device)
            lp = torch.tensor(lp, dtype=torch.long, device=device)

            cl, pl = model(voxels)
            pc = cl.argmax(1)

            correct_c += pc.eq(lc).sum().item()
            correct_p += pl.argmax(1).eq(lp).sum().item()
            total += lc.size(0)

            # also track per-class accuracy for more detailed analysis
            for i in range(lc.size(0)):
                c = lc[i].item()
                per_class_total[c] += 1
                if pc[i].item() == c:
                    per_class_correct[c] += 1

    overall = 100.0 * correct_c / total if total else 0.0
    pose = 100.0 * correct_p / total if total else 0.0
    per_class_acc = {c: 100.0 * per_class_correct[c] / per_class_total[c]
                     for c in sorted(per_class_total)}
    mean_class = np.mean(list(per_class_acc.values())) if per_class_acc else 0.0

    return overall, mean_class, pose, per_class_acc


def evaluate_voting(model, test_allrot_path, num_rotations, num_classes, device, batch_size=128):
    
    #test-time voting as described in the paper (Equation 3).

    #Each test object has 12 rotations stored in the HDF5. I run all of them
    #through the network, sum up the softmax scores, and pick the winner.
    #This is how the paper reports their main results, so we need this
    #or a fair comparison.

    #The idea is that if the network is confident about a class across multiple
    #rotations, that confidence adds up and wins out over noise.
    
    model.eval()

    # first figure out where the actual HDF5 file is
    # the path we get is usually a .txt file that points to the real HDF5
    with open(test_allrot_path, 'r') as f:
        hdf5_path = f.read().strip()

    # resolve relative paths w.r.t. the directory of the txt file
    if not os.path.isabs(hdf5_path):
        txt_dir = os.path.dirname(os.path.abspath(test_allrot_path))
        hdf5_path = os.path.join(txt_dir, hdf5_path)

    print(f"Loading test_allrot from: {hdf5_path}")

    # load all the data at once
    with h5py.File(hdf5_path, 'r') as hf:
        voxels_all = hf['data'][:]
        labels_class = hf['label'][:]
        labels_pose = hf['label_pose'][:]

    total_samples = len(labels_class)
    num_objects = total_samples // num_rotations

    print(f"Total samples: {total_samples}, Objects: {num_objects}, Rotations: {num_rotations}")

    # sanity check , make sure the data is structured correctly
    # if this fails, something's wrong with the data generation
    if total_samples % num_rotations != 0:
        print(f"Warning: {total_samples} not divisible by {num_rotations}")
        print("Falling back to standard evaluation")
        return None

    # we'll use center crop for test (no random offset)
    transform = Voxel3DTransform(offset_from=(2, 2, 2), offset_to=(2, 2, 2))

    all_class_scores = []
    all_pose_preds = []

    # run inference in batches to avoid Out of Memory
    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_end = min(i + batch_size, total_samples)
            batch_voxels = torch.from_numpy(voxels_all[i:batch_end]).float().unsqueeze(1)

            # apply the crop transform to each sample
            transformed = []
            for v in batch_voxels:
                transformed.append(transform(v))
            batch_voxels = torch.stack(transformed).to(device)

            class_out, pose_out = model(batch_voxels)

            # we use softmax scores for voting, not raw logits
            # this is important because we want probabilities that sum to 1
            class_scores = torch.softmax(class_out, dim=1)
            all_class_scores.append(class_scores.cpu())
            all_pose_preds.append(pose_out.argmax(1).cpu())

    all_class_scores = torch.cat(all_class_scores, dim=0)
    all_pose_preds = torch.cat(all_pose_preds, dim=0)

    # now do the actual voting , aggregate scores across rotations for each object
    correct_class = 0
    correct_pose_per_sample = 0
    correct_pose_per_object = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    for obj_idx in range(num_objects):
        start = obj_idx * num_rotations
        end = start + num_rotations

        # ground truth is the same for all rotations of this object
        gt_class = labels_class[start]
        gt_poses = labels_pose[start:end]

        # double-checking the data is correct , all rotations should have same class
        assert all(labels_class[start:end] == gt_class), f"Class mismatch at object {obj_idx}"

        # this is the key part , sum scores across all rotations (Equation 3 from paper)
        summed_scores = all_class_scores[start:end].sum(dim=0)
        pred_class = summed_scores.argmax().item()

        # count class accuracy
        if pred_class == gt_class:
            correct_class += 1
            per_class_correct[gt_class] += 1
        per_class_total[gt_class] += 1

        # pose accuracy , check each rotation individually since each has its own pose
        pose_preds = all_pose_preds[start:end]
        correct_pose_per_sample += (pose_preds.numpy() == gt_poses).sum()

        # also compute per-object pose accuracy (majority vote)
        pose_correct_count = (pose_preds.numpy() == gt_poses).sum()
        if pose_correct_count >= num_rotations // 2:
            correct_pose_per_object += 1

    # calculate final metrics
    overall = 100.0 * correct_class / num_objects
    pose_sample = 100.0 * correct_pose_per_sample / total_samples
    pose_object = 100.0 * correct_pose_per_object / num_objects
    per_class_acc = {c: 100.0 * per_class_correct[c] / per_class_total[c]
                     for c in sorted(per_class_total)}
    mean_class = np.mean(list(per_class_acc.values())) if per_class_acc else 0.0

    return overall, mean_class, pose_sample, pose_object, per_class_acc


def main():
    parser = argparse.ArgumentParser(description='ORION Evaluation')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--voting', action='store_true',
                        help='Use test-time voting (requires test_allrot data)')
    parser.add_argument('--test_allrot', type=str, default=None,
                        help='Path to test_allrot.hdf5.txt for voting mode')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    arch = config['model']['arch']
    model = MODEL_REGISTRY[arch](
        num_classes=config['model']['num_classes'],
        num_orientations=config['model']['num_orientations']
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {arch} ({params:,} params), epoch {ckpt.get('epoch', '?')}")

    if args.voting:
        # try to find the test_allrot file if not specified
        if args.test_allrot is None:
            val_path = config['data']['val_hdf5_list']
            test_allrot = val_path.replace('test_singlerandrot', 'test_allrot')
            if os.path.exists(test_allrot):
                args.test_allrot = test_allrot
            else:
                print(f"Error: Cannot find test_allrot at {test_allrot}")
                print("Please specify --test_allrot path")
                return

        num_classes = config['model']['num_classes']
        num_rotations = 12  # standard for ModelNet

        print(f"\n Voting Evaluation (paper method) ")
        print(f"Using {num_rotations} rotations per object")

        results = evaluate_voting(
            model, args.test_allrot, num_rotations, num_classes, device
        )

        if results is None:
            print("Voting failed, falling back to standard evaluation")
            args.voting = False

    if not args.voting:
        # standard single-shot evaluation, faster but less accurate
        test_transform = Voxel3DTransform(
            offset_from=tuple(config['augmentation']['test']['offset_from']),
            offset_to=tuple(config['augmentation']['test']['offset_to'])
        )
        test_dataset = VoxelHDF5Dataset(
            config['data']['val_hdf5_list'], transform=test_transform)
        test_loader = DataLoader(
            test_dataset, batch_size=config['training'].get('val_batch_size', 128),
            shuffle=False, num_workers=config['training'].get('num_workers', 4),
            pin_memory=True)

        print(f"\nStandard Evaluation")
        results = evaluate_standard(model, test_loader, device)

    # print results
    if args.voting:
        overall, mean_class, pose_sample, pose_object, per_class_acc = results

        print(f"\nOverall accuracy:        {overall:.2f}%")
        print(f"Mean per-class acc:      {mean_class:.2f}%")
        print(f"Pose acc (per-sample):   {pose_sample:.2f}%")
        print(f"Pose acc (per-object):   {pose_object:.2f}%  <- paper metric")
    else:
        overall, mean_class, pose, per_class_acc = results

        print(f"\nOverall accuracy:     {overall:.2f}%")
        print(f"Mean per-class acc:   {mean_class:.2f}%")
        print(f"Pose accuracy:        {pose:.2f}%")

    print(f"\nPer-class:")
    for c in sorted(per_class_acc):
        print(f"  Class {c:2d}: {per_class_acc[c]:.2f}%")


if __name__ == '__main__':
    main()
