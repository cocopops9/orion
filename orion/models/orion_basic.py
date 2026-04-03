"""
ORION Basic network architecture - PyTorch port.

This is a faithful reproduction of the architecture from:
    net_archs/modelnet10/ORION_basic_trainval.prototxt (https://github.com/lmb-freiburg/orion)

Architecture (input: 1x32x32x32 voxel grid):
    Conv1: 32 filters, 5x5x5, stride=2, pad=0 -> BatchNorm -> LeakyReLU(0.1) -> Dropout(0.2)
    Conv2: 32 filters, 3x3x3, stride=1, pad=0 -> BatchNorm -> LeakyReLU(0.1) -> MaxPool(2) -> Dropout(0.3)
    FC6: 128 -> ReLU -> Dropout(0.4)
    Dual heads:
        fc8: num_classes (object classification)
        fc8_pose: num_orientations (pose classification)

The key insight from the ORION paper is that jointly training for classification
and orientation estimation gives you better features than classification alone.
"""

import torch
import torch.nn as nn


class ORIONBasic(nn.Module):
    """
    ORION Basic 3D CNN for joint object classification and orientation prediction.

    Matches the original Caffe prototxt:
    - Conv layers use MSRA (Kaiming) weight initialization
    - FC layers use Gaussian(std=0.01) initialization
    - BatchNorm + Scale (learnable affine) after each conv
    - LeakyReLU with negative_slope=0.1 after conv layers
    - Standard ReLU after FC6 (matching original prototxt)
    - Dropout rates: 0.2, 0.3, 0.4 per layer group
    """

    def __init__(self, num_classes=10, num_orientations=105):
        super(ORIONBasic, self).__init__()

        # First layer group
        # Conv1: 32 filters, 5x5x5, stride=2, pad=0
        self.conv1 = nn.Conv3d(1, 32, kernel_size=5, stride=2, padding=0)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.drop1 = nn.Dropout3d(p=0.2)

        # Second layer group
        # Conv2: 32 filters, 3x3x3, stride=1, pad=0
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm3d(32)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout3d(p=0.3)

        # FC layers
        # The spatial dimensions after conv+pool:
        #   Input: 1x32x32x32
        #   After conv1 (k=5,s=2,p=0): 32x14x14x14
        #   After conv2 (k=3,s=1,p=0): 32x12x12x12
        #   After pool2 (k=2,s=2):     32x6x6x6
        self.fc6 = nn.Linear(32 * 6 * 6 * 6, 128)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout(p=0.4)

        # Dual output heads, this is the key ORION contribution
        self.fc_class = nn.Linear(128, num_classes)
        self.fc_orientation = nn.Linear(128, num_orientations)

        # Initialize weights according to original initialization
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
                # Gaussian(std=0.01)
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # First layer group
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        # Second layer group
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # FC layers
        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.drop6(x)

        # Dual output heads
        class_out = self.fc_class(x)
        orientation_out = self.fc_orientation(x)

        return class_out, orientation_out
