"""
Utility for loading original Caffe weights into the PyTorch models.

Converts Caffe .caffemodel weights to PyTorch state dict format.
Requires the pycaffe Python bindings to be installed (only needed
for weight conversion, not for training or inference).

Layer name mapping (Caffe -> PyTorch):
    Basic model:
        conv1a -> conv1, bn1 -> bn1, fc6 -> fc6, fc8 -> fc_class, fc8_pose -> fc_orientation
    Extended model:
        conv1a -> conv1, conv2a -> conv2, conv3a -> conv3, conv4a -> conv4,
        bn1 -> bn1, bn2 -> bn2, bn3 -> bn3, bn4 -> bn4,
        fc6 -> fc6, fc8 -> fc_class, fc8_pose -> fc_orientation
"""

import numpy as np
import torch


# Layer name mapping from Caffe to PyTorch
BASIC_LAYER_MAP = {
    'conv1a': 'conv1',
    'conv2a': 'conv2',
    'bn1': 'bn1',
    'bn2': 'bn2',
    'scale1': 'bn1',  # Caffe Scale -> PyTorch BN affine params
    'scale2': 'bn2',
    'fc6': 'fc6',
    'fc8': 'fc_class',
    'fc8_pose': 'fc_orientation',
}

EXTENDED_LAYER_MAP = {
    'conv1a': 'conv1',
    'conv2a': 'conv2',
    'conv3a': 'conv3',
    'conv4a': 'conv4',
    'bn1': 'bn1',
    'bn2': 'bn2',
    'bn3': 'bn3',
    'bn4': 'bn4',
    'scale1': 'bn1',
    'scale2': 'bn2',
    'scale3': 'bn3',
    'scale4': 'bn4',
    'fc6': 'fc6',
    'fc8': 'fc_class',
    'fc8_pose': 'fc_orientation',
}


def load_caffe_weights(prototxt_path, caffemodel_path, pytorch_model, model_type='basic'):
    """
    Load weights from a Caffe model into a PyTorch model.

    Args:
        prototxt_path: Path to the Caffe prototxt file
        caffemodel_path: Path to the Caffe .caffemodel file
        pytorch_model: PyTorch model instance (ORIONBasic or ORIONExtended)
        model_type: 'basic' or 'extended'

    Returns:
        pytorch_model with loaded weights
    """
    try:
        import caffe
    except ImportError:
        raise ImportError(
            "pycaffe is required for weight conversion. "
            "This is only needed for converting original Caffe weights. "
            "For training from scratch, this is not needed."
        )

    caffe.set_mode_cpu()
    net = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)

    layer_map = BASIC_LAYER_MAP if model_type == 'basic' else EXTENDED_LAYER_MAP

    state_dict = pytorch_model.state_dict()
    converted = 0

    for caffe_name, params in net.params.items():
        if caffe_name not in layer_map:
            print(f"  Skipping unmapped Caffe layer: {caffe_name}")
            continue

        pytorch_name = layer_map[caffe_name]

        # Handle Scale layers (Caffe separates BN and Scale; PyTorch combines them)
        if caffe_name.startswith('scale'):
            weight_key = f'{pytorch_name}.weight'
            bias_key = f'{pytorch_name}.bias'
            if weight_key in state_dict:
                state_dict[weight_key] = torch.from_numpy(params[0].data.copy())
                converted += 1
            if len(params) > 1 and bias_key in state_dict:
                state_dict[bias_key] = torch.from_numpy(params[1].data.copy())
                converted += 1
        elif caffe_name.startswith('bn'):
            # BatchNorm running mean and variance
            mean_key = f'{pytorch_name}.running_mean'
            var_key = f'{pytorch_name}.running_var'
            if len(params) >= 3:
                # Caffe BN stores: mean, variance, scale_factor
                scale_factor = params[2].data[0] if params[2].data[0] != 0 else 1
                if mean_key in state_dict:
                    state_dict[mean_key] = torch.from_numpy(params[0].data.copy() / scale_factor)
                    converted += 1
                if var_key in state_dict:
                    state_dict[var_key] = torch.from_numpy(params[1].data.copy() / scale_factor)
                    converted += 1
        else:
            # Conv and FC layers
            weight_key = f'{pytorch_name}.weight'
            bias_key = f'{pytorch_name}.bias'
            if weight_key in state_dict:
                state_dict[weight_key] = torch.from_numpy(params[0].data.copy())
                converted += 1
            if len(params) > 1 and bias_key in state_dict:
                state_dict[bias_key] = torch.from_numpy(params[1].data.copy())
                converted += 1

    pytorch_model.load_state_dict(state_dict)
    print(f"Converted {converted} parameter tensors from Caffe to PyTorch")
    return pytorch_model


def convert_and_save(prototxt_path, caffemodel_path, output_path, model_type='basic',
                     num_classes=10, num_orientations=105):
    """
    Convert Caffe weights and save as a PyTorch checkpoint.

    Args:
        prototxt_path: Path to Caffe prototxt
        caffemodel_path: Path to Caffe .caffemodel
        output_path: Output path for PyTorch .pth file
        model_type: 'basic' or 'extended'
        num_classes: Number of object classes
        num_orientations: Number of orientation classes
    """
    from orion.models import ORIONBasic, ORIONExtended

    if model_type == 'basic':
        model = ORIONBasic(num_classes=num_classes, num_orientations=num_orientations)
    else:
        model = ORIONExtended(num_classes=num_classes, num_orientations=num_orientations)

    model = load_caffe_weights(prototxt_path, caffemodel_path, model, model_type)

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'num_classes': num_classes,
        'num_orientations': num_orientations,
    }, output_path)

    print(f"Saved PyTorch checkpoint to: {output_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert Caffe weights to PyTorch')
    parser.add_argument('--prototxt', type=str, required=True, help='Caffe prototxt path')
    parser.add_argument('--caffemodel', type=str, required=True, help='Caffe .caffemodel path')
    parser.add_argument('--output', type=str, required=True, help='Output .pth path')
    parser.add_argument('--model-type', choices=['basic', 'extended'], default='basic')
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--num-orientations', type=int, default=105)
    args = parser.parse_args()

    convert_and_save(
        args.prototxt, args.caffemodel, args.output,
        args.model_type, args.num_classes, args.num_orientations
    )
