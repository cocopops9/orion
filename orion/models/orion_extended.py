"""
ORION Extended network architecture.

This is the deeper 4-layer version from the ORION paper's supplementary material
(Table 3). It's designed for ModelNet40 which has more classes and harder examples.

Architecture (input: 1x32x32x32 voxel grid):
    Conv1: 32 filters, 3x3x3, stride=2 -> BatchNorm -> LeakyReLU(0.1) -> Dropout(0.2)
    Conv2: 64 filters, 3x3x3, stride=1 -> BatchNorm -> LeakyReLU(0.1) -> Dropout(0.3)
    Conv3: 128 filters, 3x3x3, stride=1 -> BatchNorm -> LeakyReLU(0.1) -> Dropout(0.4)
    Conv4: 256 filters, 3x3x3, stride=1 -> BatchNorm -> LeakyReLU(0.1) -> MaxPool(2) -> Dropout(0.6)
    FC1: 128 -> ReLU -> Dropout(0.4)
    Dual heads:
        fc_class: num_classes (object classification)
        fc_orientation: num_orientations (pose classification)

The paper reports 89.7% on ModelNet40 with manual alignment using this architecture, and 89.4% with automatic alignment .
"""

import torch
import torch.nn as nn


class ORIONExtended(nn.Module):
    """
    ORION Extended 3D CNN for joint object classification and orientation prediction.

    This is the deeper 4-layer architecture from the paper. Key differences from basic:
    - 4 conv layers instead of 2
    - Progressive filter increase: 32 -> 64 -> 128 -> 256
    - Higher dropout in deeper layers (0.6 for conv4)
    - All layers use batch normalization
    """

    def __init__(self, num_classes=40, num_orientations=189):
        super().__init__()

        # 1st conv layer
        # Input: 1x32x32x32 -> Output: 32x15x15x15
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=0)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.drop1 = nn.Dropout3d(p=0.2)

        # 2nd conv layer
        # Input: 32x15x15x15 -> Output: 64x13x13x13
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm3d(64)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.drop2 = nn.Dropout3d(p=0.3)

        # 3rd conv layer
        # Input: 64x13x13x13 -> Output: 128x11x11x11
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm3d(128)
        self.relu3 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.drop3 = nn.Dropout3d(p=0.4)

        # 4th conv layer
        # Input: 128x11x11x11 -> Output: 256x9x9x9
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=0)
        self.bn4 = nn.BatchNorm3d(256)
        self.relu4 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout3d(p=0.6)

        # FC layers
        # After conv4 (256x9x9x9) and pool4 (k=2,s=2): 256x4x4x4 = 16384 features
        self.fc1 = nn.Linear(256 * 4 * 4 * 4, 128)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.drop_fc1 = nn.Dropout(p=0.4)

        # Dual output heads
        self.fc_class = nn.Linear(128, num_classes)
        self.fc_orientation = nn.Linear(128, num_orientations)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # MSRA (Kaiming) initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Gaussian(std=0.01) initialization
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.drop3(x)

        # Conv block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = self.drop4(x)

        # FC layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.drop_fc1(x)

        # Dual heads
        class_out = self.fc_class(x)
        orientation_out = self.fc_orientation(x)

        return class_out, orientation_out
