"""
ORION Fast: Optimized for training speed with comparable accuracy.

The idea here is pretty simple: the FC layers are where most of the parameters
are in this network. FC6 alone has 885K params (94% of total). By cutting
the FC6 output from 128 to 64, we save about 443K params and get a noticeable
speedup in the backward pass.

Architecture:
- Same conv layers as basic (32 filters each)
- FC6: 6912 -> 64 (was 128) - 50% fewer FC params
- Dual heads: 64 -> num_classes, 64 -> num_orientations

In practice this gives us ~15-20% faster training with maybe 0.5-1pp accuracy

"""

import torch
import torch.nn as nn


class ORIONFast(nn.Module):
    """
    ORION Fast: Speed-optimized variant.

    Changes from basic:
    - FC6 output: 128 -> 64 (halves FC computation)
    - Total params: ~500K (vs 941K for basic, -47%)

    The conv layers are kept identical because:
    1. They're already efficient (only 32 filters)
    2. Reducing them hurts spatial feature extraction
    3. FC layers dominate parameter count anyway
    """

    def __init__(self, num_classes=10, num_orientations=105):
        super().__init__()

        # Conv layers (same as basic)
        self.conv1 = nn.Conv3d(1, 32, kernel_size=5, stride=2, padding=0)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.drop1 = nn.Dropout3d(p=0.2)

        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm3d(32)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout3d(p=0.3)

        # FC layers (reduced)
        # 32 * 6 * 6 * 6 = 6912 input features
        self.fc6 = nn.Linear(32 * 6 * 6 * 6, 64)  # Was 128, now 64
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout(p=0.4)

        # Dual output heads
        self.fc_class = nn.Linear(64, num_classes)
        self.fc_orientation = nn.Linear(64, num_orientations)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
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
        x = self.pool2(x)
        x = self.drop2(x)

        # FC layers
        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.drop6(x)

        # Dual heads
        class_out = self.fc_class(x)
        orientation_out = self.fc_orientation(x)

        return class_out, orientation_out
