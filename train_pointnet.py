"""
PointNet Training Script

Trains the classic PointNet on ModelNet using the same HDF5 data as ORION.
Voxels get converted to point clouds, so we can do an
apples-to-apples comparison: same data, same splits, different representation.

This is a check to see how a known architecture performs
on preprocessed data.
"""

import os
import time
import csv
import argparse
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from orion.models.pointnet import PointNet
from orion.data.pointcloud_dataset import PointCloudHDF5Dataset


def evaluate(model, loader, device, criterion):
    """
    Quick validation pass. Returns accuracy and loss.
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for points, labels, _ in loader:
            points = points.to(device)
            labels = labels.to(device)

            logits, feat_t = model(points)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            n_batches += 1
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / max(n_batches, 1)
    return accuracy, avg_loss


def main():
    parser = argparse.ArgumentParser(description='PointNet Training')
    parser.add_argument('--config', required=True, help='Path to YAML config')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # set up model
    num_classes = config['model']['num_classes']
    num_points = config['model'].get('num_points', 1024)
    use_feat_transform = config['model'].get('use_feature_transform', True)

    model = PointNet(
        num_classes=num_classes,
        use_feature_transform=use_feat_transform
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: PointNet ({total_params:,} parameters, "
          f"{num_points} points, feat_transform={use_feat_transform})")

    # load data , voxels get converted to point clouds automatically
    train_dataset = PointCloudHDF5Dataset(
        config['data']['train_hdf5_list'],
        num_points=num_points,
        augment=True
    )
    val_dataset = PointCloudHDF5Dataset(
        config['data']['val_hdf5_list'],
        num_points=num_points,
        augment=False
    )

    tc = config['training']

    # Note: num_workers=0 required for lazy HDF5 loading
    train_loader = DataLoader(
        train_dataset,
        batch_size=tc['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True  # helps with batchnorm when batch size is small
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=tc.get('val_batch_size', tc['batch_size']),
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # loss and optimizer - Adam works well for PointNet
    label_smoothing = tc.get('label_smoothing', 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    optimizer = optim.Adam(
        model.parameters(),
        lr=tc['lr'],
        weight_decay=tc.get('weight_decay', 0.0001)
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=tc.get('lr_step_size', 20),
        gamma=tc.get('lr_gamma', 0.5)
    )

    # the original PointNet uses a regularizer to keep feature transform close to orthogonal
    feat_reg_weight = tc.get('feature_reg_weight', 0.001)

    # set up logging
    save_dir = tc.get('save_dir', 'checkpoints/pointnet')
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, 'training_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'lr'])

    # training loop
    num_epochs = tc['num_epochs']
    best_acc = 0.0
    train_start = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")

        for points, labels, _ in pbar:
            points = points.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits, feat_t = model(points)

            # classification loss
            loss = criterion(logits, labels)

            # add feature transform regularization if we're using it
            # this keeps the transform close to orthogonal which helps with training
            if feat_t is not None:
                reg_loss = PointNet.feature_transform_regularization(feat_t)
                loss = loss + feat_reg_weight * reg_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': f'{train_loss / n_batches:.4f}'})

        scheduler.step()

        # validation
        avg_train_loss = train_loss / n_batches
        val_acc, val_loss = evaluate(model, val_loader, device, criterion)
        current_lr = optimizer.param_groups[0]['lr']

        csv_writer.writerow([epoch, avg_train_loss, val_loss, val_acc, current_lr])
        csv_file.flush()

        print(f"Epoch {epoch}: Train={avg_train_loss:.4f}, Val={val_loss:.4f}, "
              f"Acc={val_acc:.2f}%, LR={current_lr:.6f}")

        # save best model
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'class_accuracy': val_acc,
                'config': config,
            }, checkpoint_path)
            print(f"  Saved best model (acc={val_acc:.2f}%)")

        # periodic checkpoints
        if epoch % tc.get('save_every', 10) == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'class_accuracy': val_acc,
                'config': config,
            }, checkpoint_path)

    csv_file.close()
    total_time = time.time() - train_start

    print(f"\nTraining complete. Best accuracy: {best_acc:.2f}%")
    print(f"Total time: {total_time/3600:.2f}h ({total_time:.1f}s)")


if __name__ == '__main__':
    main()
