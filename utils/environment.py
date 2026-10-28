"""记录解释器与依赖版本，并实际验证设备上的 U-Net 前向和反向计算。"""
import json
import platform
import sys
from importlib.metadata import version

import torch

from models.unet import UNet
from training.engine import get_device
from training.losses import segmentation_loss
from .io import save_json


def check_environment(device_name='auto', output='outputs/environment.json'):
    torch.set_num_threads(4)
    device = get_device(device_name)
    model = UNet(base=4).to(device)
    images = torch.randn(2, 3, 32, 32, device=device)
    masks = torch.zeros(2, 1, 32, 32, device=device)
    logits = model(images)
    loss = segmentation_loss(logits, masks)
    loss.backward()
    if not torch.isfinite(loss) or not all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    ):
        raise RuntimeError('Non-finite loss or gradients during environment check.')
    if device.type == 'cuda':
        torch.cuda.synchronize()
    report = {
        'python': platform.python_version(),
        'executable': sys.executable,
        'packages': {name: version(name) for name in ('torch', 'numpy', 'Pillow', 'matplotlib')},
        'cuda_runtime': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(),
        'device_tested': str(device),
        'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        'unet_output_shape': list(logits.shape),
        'forward_backward_passed': True,
    }
    save_json(output, report)
    print(json.dumps(report, indent=2), flush=True)
    print(f'Environment report saved to {output}', flush=True)
