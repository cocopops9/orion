"""
PointNet for 3D Object Classification.

Reference: Qi et al., "PointNet: Deep Learning on Point Sets for 3D
Classification and Segmentation", CVPR 2017.

Architecture:
    Input (B, N, 3) -> T-Net(3) -> MLP(64,64) -> T-Net(64) ->
    MLP(64,128,1024) -> MaxPool -> FC(512,256,C)

The key idea is that point clouds are just unordered sets of points, so
we need a network that's invariant to permutation. PointNet achieves this
by using symmetric functions (max pooling) to aggregate features.

The T-Nets are these cool mini-networks that learn to align the input
coordinates and features, which helps the network be more robust to
different orientations of the same object.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """
    Spatial transformer network that predicts a KxK alignment matrix.

    Uses a mini-PointNet architecture: shared MLPs -> max pool -> FC layers.
    Output is initialized to identity so the network starts with no-op
    alignment and learns deviations as needed.

    Args:
        k: Dimensionality of the transform (3 for input, 64 for features).
    """

    def __init__(self, k=3):
        super().__init__()
        self.k = k

        # Shared MLPs (implemented as 1D convolutions)
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # FC layers after global pooling
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # Initialize final layer so output starts as identity matrix
        # This is important - without it, training is unstable at the start
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        self.fc3.bias.data.copy_(torch.eye(k).flatten())

    def forward(self, x):
        """
        Args:
            x: (B, K, N) - K-dimensional features for N points.
        Returns:
            transform: (B, K, K) transformation matrix.
        """
        batch_size = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Symmetric function: global max pooling over points
        x = torch.max(x, dim=2)[0]  # (B, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        transform = x.view(batch_size, self.k, self.k)
        return transform


class PointNet(nn.Module):
    """
    PointNet classification network.

    Processes unordered point sets through shared MLPs with spatial
    transformer alignment, then aggregates via global max pooling
    (symmetric function) to achieve permutation invariance.

    Args:
        num_classes: Number of output categories.
        use_feature_transform: If True, applies a 64x64 T-Net after the
            first MLP block. Adds a regularization term to training.
    """

    def __init__(self, num_classes, use_feature_transform=True):
        super().__init__()
        self.use_feature_transform = use_feature_transform

        # Input spatial transform (3x3)
        self.input_transform = TNet(k=3)

        # First shared MLP block: 3 -> 64 -> 64
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        # Feature spatial transform (64x64)
        if use_feature_transform:
            self.feature_transform = TNet(k=64)

        # Second shared MLP block: 64 -> 128 -> 1024
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        # Classification head
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        """
        Args:
            x: (B, N, 3) point cloud coordinates.
        Returns:
            logits: (B, num_classes) raw class scores.
            feat_transform: (B, 64, 64) feature transform matrix, or None.
        """
        # Transpose to channel-first for Conv1d: (B, 3, N)
        x = x.transpose(1, 2)

        # Input alignment
        t_input = self.input_transform(x)  # (B, 3, 3)
        x = torch.bmm(t_input, x)          # (B, 3, N)

        # First shared MLP
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 64, N)

        # Feature alignment
        t_feat = None
        if self.use_feature_transform:
            t_feat = self.feature_transform(x)  # (B, 64, 64)
            x = torch.bmm(t_feat, x)            # (B, 64, N)

        # Second shared MLP
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 64, N)
        x = F.relu(self.bn4(self.conv4(x)))  # (B, 128, N)
        x = F.relu(self.bn5(self.conv5(x)))  # (B, 1024, N)

        # Symmetric function: global max pooling
        x = torch.max(x, dim=2)[0]  # (B, 1024)

        # Classification head with dropout
        x = F.relu(self.bn6(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn7(self.fc2(x)))
        x = self.dropout(x)
        logits = self.fc3(x)  # (B, num_classes)

        return logits, t_feat

    @staticmethod
    def feature_transform_regularization(transform):
        """
        Orthogonality regularization for the feature transform.

        The idea is that a good feature transform should be close to orthogonal,
        i.e., A * A^T should be close to identity. If the transform squishes
        or stretches features too much, it can hurt generalization.

        Penalizes deviation from orthogonal matrix:
            L_reg = ||I - A * A^T||_F

        This is averaged over the batch.

        Args:
            transform: (B, K, K) predicted transform matrix.
        Returns:
            Scalar regularization loss.
        """
        if transform is None:
            return torch.tensor(0.0)

        k = transform.size(1)
        identity = torch.eye(k, device=transform.device).unsqueeze(0)
        product = torch.bmm(transform, transform.transpose(1, 2))

        return torch.mean(torch.norm(identity - product, dim=(1, 2)))
