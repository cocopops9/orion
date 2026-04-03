"""
Generate training/validation loss plots for ORION models.

A simple script to visualize the training curves.
Run this after training to see how things went.
"""

import os
import csv
import matplotlib.pyplot as plt
import numpy as np


def load_csv(csv_path):
    """Load training log CSV and extract the columns we care about."""
    epochs = []
    train_losses = []
    val_losses = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_losses.append(float(row['train_loss']))
            val_losses.append(float(row['val_loss']))

    return epochs, train_losses, val_losses


def plot_losses(csv_path, title, output_path):
    """Create a single loss plot with train and val curves."""
    epochs, train_losses, val_losses = load_csv(csv_path)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    # Paths match the save_dir values in config files.
	# Only change these if non-default save_dir settings are used.
    models = [
        ('checkpoints/modelnet10_basic/training_log.csv', 'ModelNet10 Basic', 'plots/mn10_basic_loss.png'),
        ('checkpoints/modelnet10_fast/training_log.csv', 'ModelNet10 Fast', 'plots/mn10_fast_loss.png'),
        ('checkpoints/modelnet40_basic/training_log.csv', 'ModelNet40 Basic', 'plots/mn40_basic_loss.png'),
        ('checkpoints/modelnet40_fast/training_log.csv', 'ModelNet40 Fast', 'plots/mn40_fast_loss.png'),
    ]

    os.makedirs('plots', exist_ok=True)

    for csv_path, title, output_path in models:
        if os.path.exists(csv_path):
            plot_losses(csv_path, title, output_path)
        else:
            print(f"Warning: {csv_path} not found")


if __name__ == '__main__':
    main()
