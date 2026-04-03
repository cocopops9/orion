"""
ORION Training Script

This is the main training loop for PyTorch reimplementation of ORION.
I am training on two tasks at once: object classification + orientation prediction.
The loss is just a weighted sum of both (equal weights by default, as per paper).

"""

import os
import time
import argparse
import yaml
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from orion.models import ORIONBasic, ORIONFast, ORIONExtended
from orion.data import VoxelHDF5Dataset, Voxel3DTransform


# quick lookup so we can pick the model from config without a bunch of if-else
MODEL_REGISTRY = {
    "basic": ORIONBasic,
    "fast": ORIONFast,
    "extended": ORIONExtended,
}


def evaluate(model, data_loader, device):
    """
    Accuracy check on a dataset. Returns class acc and pose acc.
    """
    model.eval()
    correct_class = 0
    correct_pose = 0
    total = 0

    with torch.no_grad():
        for voxels, labels_class, labels_pose in data_loader:
            voxels = voxels.to(device)
            labels_class = labels_class.clone().detach().to(device)
            labels_pose = labels_pose.clone().detach().to(device)

            output_class, output_pose = model(voxels)
            _, pred_class = output_class.max(1)
            _, pred_pose = output_pose.max(1)

            total += labels_class.size(0)
            correct_class += pred_class.eq(labels_class).sum().item()
            correct_pose += pred_pose.eq(labels_pose).sum().item()

    # avoid division by zero if somehow we have no samples
    class_acc = 100.0 * correct_class / total if total > 0 else 0
    pose_acc = 100.0 * correct_pose / total if total > 0 else 0

    return class_acc, pose_acc




def evaluate_with_loss(model, data_loader, device, criterion_class, criterion_orient, gamma):
    """
    Same as evaluate() but also computes the loss.
    We need this for the CSV logging , otherwise we'd have no val loss to plot.
    """
    model.eval()
    correct_class = 0
    correct_pose = 0
    total = 0
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for voxels, labels_class, labels_pose in data_loader:
            voxels = voxels.to(device)
            labels_class = labels_class.clone().detach().to(device)
            labels_pose = labels_pose.clone().detach().to(device)

            output_class, output_pose = model(voxels)

            # compute both losses exactly like in training
            loss_class = criterion_class(output_class, labels_class)
            loss_orient = criterion_orient(output_pose, labels_pose)
            loss = (1 - gamma) * loss_class + gamma * loss_orient

            total_loss += loss.item()
            num_batches += 1

            _, pred_class = output_class.max(1)
            _, pred_pose = output_pose.max(1)

            total += labels_class.size(0)
            correct_class += pred_class.eq(labels_class).sum().item()
            correct_pose += pred_pose.eq(labels_pose).sum().item()

    class_acc = 100.0 * correct_class / total if total > 0 else 0
    pose_acc = 100.0 * correct_pose / total if total > 0 else 0
    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    return class_acc, pose_acc, avg_loss



def train(config):
    """
    The actual training loop. It does:
    - sets up model, data loaders, optimizer
    - runs epochs with progress bars
    - saves checkpoints and logs to CSV

    Most of the code here is just setup, the actual training part is pretty short.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # figure out which architecture we're using from the config
    arch = config['model']['arch']
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {arch}. Available: {list(MODEL_REGISTRY.keys())}")

    model = MODEL_REGISTRY[arch](
        num_classes=config['model']['num_classes'],
        num_orientations=config['model']['num_orientations']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {arch} ({total_params:,} parameters)")

    # set up the augmentation transforms
    # training gets random crops (offset range), test uses center crop (fixed offset)
    train_transform = Voxel3DTransform(
        offset_from=tuple(config['augmentation']['train']['offset_from']),
        offset_to=tuple(config['augmentation']['train']['offset_to'])
    )
    test_transform = Voxel3DTransform(
        offset_from=tuple(config['augmentation']['test']['offset_from']),
        offset_to=tuple(config['augmentation']['test']['offset_to'])
    )

    # load datasets from HDF5 files — lazy mode reads voxels on-the-fly
    # to avoid loading entire dataset into RAM
    train_dataset = VoxelHDF5Dataset(
        config['data']['train_hdf5_list'], transform=train_transform)
    val_dataset = VoxelHDF5Dataset(
        config['data']['val_hdf5_list'], transform=test_transform)

    # Note: num_workers=0 is required for lazy HDF5 loading because
    # h5py file handles cannot be shared across forked processes
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training'].get('val_batch_size', 128),
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # loss functions, class loss can have label smoothing if we want it, pose loss is plain CE
    label_smoothing = config['training'].get('label_smoothing', 0.0)
    criterion_class = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    criterion_orient = nn.CrossEntropyLoss()

    # gamma controls the balance between the two losses
    # 0.5 means equal weight which is what the paper uses
    gamma = config['training'].get('gamma', 0.5)
    print(f"Loss weighting: L = {1-gamma:.1f}*L_class + {gamma:.1f}*L_orient")

    # standard SGD with momentum, works well for CNNs and this is no exception
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['training']['lr'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay']
    )

    # scheduler choice: cosine annealing or simple step decay
    # cosine annealing decays the LR smoothly and can escape local minima on restarts,
	# while step decay is simpler and matches the original paper
    scheduler_type = config['training'].get('scheduler', 'step')
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config['training'].get('cosine_t0', 50),
            T_mult=config['training'].get('cosine_t_mult', 2)
        )
        cosine_mode = True
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training'].get('lr_step_size', 20),
            gamma=config['training'].get('lr_gamma', 0.1)
        )
        cosine_mode = False

    train_start = time.time()
    best_class_acc = 0.0
    num_epochs = config['training']['num_epochs']
    save_dir = config['training'].get('save_dir', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    # set up CSV logging to make plots later
    csv_path = os.path.join(save_dir, 'training_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'class_acc', 'pose_acc', 'lr'])

    #  main training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        train_loss_class = 0.0
        train_loss_orient = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")

        for batch_idx, (voxels, labels_class, labels_pose) in enumerate(pbar):
            voxels = voxels.to(device)
            labels_class = labels_class.clone().detach().to(device)
            labels_pose = labels_pose.clone().detach().to(device)

            optimizer.zero_grad()
            output_class, output_orient = model(voxels)

            # compute both losses and combine them with gamma weighting
            loss_class = criterion_class(output_class, labels_class)
            loss_orient = criterion_orient(output_orient, labels_pose)
            loss = (1 - gamma) * loss_class + gamma * loss_orient

            loss.backward()
            optimizer.step()

            # cosine scheduler updates every step, not every epoch
            if cosine_mode:
                scheduler.step(epoch - 1 + batch_idx / len(train_loader))

            train_loss += loss.item()
            train_loss_class += loss_class.item()
            train_loss_orient += loss_orient.item()
            num_batches += 1

            # update the progress bar with running averages
            pbar.set_postfix({
                'loss': f'{train_loss / num_batches:.4f}',
                'L_cls': f'{train_loss_class / num_batches:.4f}',
                'L_ori': f'{train_loss_orient / num_batches:.4f}'
            })

        # step scheduler at end of epoch for non-cosine
        if not cosine_mode:
            scheduler.step()

        # validation time
        avg_train_loss = train_loss / num_batches
        class_acc, pose_acc, val_loss = evaluate_with_loss(
            model, val_loader, device, criterion_class, criterion_orient, gamma
        )
        current_lr = optimizer.param_groups[0]['lr']

        # write to CSV to have a record
        csv_writer.writerow([epoch, avg_train_loss, val_loss, class_acc, pose_acc, current_lr])
        csv_file.flush()  # make sure it's written to disk

        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"Class Acc={class_acc:.2f}%, Pose Acc={pose_acc:.2f}%, LR={current_lr:.6f}")

        # save best model whenever we beat the previous best
        if class_acc > best_class_acc:
            best_class_acc = class_acc
            checkpoint_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'class_accuracy': class_acc,
                'pose_accuracy': pose_acc,
                'config': config,
            }, checkpoint_path)
            print(f"  Saved best model (class acc={class_acc:.2f}%)")

        # also save periodic checkpoints in case we need to resume or debug
        if epoch % config['training'].get('save_every', 10) == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'class_accuracy': class_acc,
                'pose_accuracy': pose_acc,
                'config': config,
            }, checkpoint_path)

    csv_file.close()
    total_time = time.time() - train_start

    # print some final stats
    print(f"\nTraining complete. Best class accuracy: {best_class_acc:.2f}%")
    print(f"Total training time: {total_time/3600:.2f}h ({total_time:.1f}s)")
    print(f"Training log saved to: {csv_path}")



def main():
    parser = argparse.ArgumentParser(description='Train ORION network')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == '__main__':
    main()
