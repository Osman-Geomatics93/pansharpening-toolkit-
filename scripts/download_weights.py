#!/usr/bin/env python
"""Download pre-trained model weights."""

import argparse
import urllib.request
from pathlib import Path


WEIGHTS_URL = "https://github.com/Osman-Geomatics93/pansharpening-toolkit-/releases/download"

AVAILABLE_WEIGHTS = {
    'panformer_lite': 'v1.0.0/panformer_lite_weights.pth',
    'pannet_cbam': 'v1.0.0/pannet_cbam_weights.pth',
    'pannet': 'v1.0.0/pannet_weights.pth',
}


def download_weights(model_name: str, output_dir: str = 'checkpoints'):
    """Download pre-trained weights for a model."""
    if model_name not in AVAILABLE_WEIGHTS:
        print(f"Available models: {list(AVAILABLE_WEIGHTS.keys())}")
        raise ValueError(f"Unknown model: {model_name}")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    url = f"{WEIGHTS_URL}/{AVAILABLE_WEIGHTS[model_name]}"
    output_path = output_dir / f"{model_name}_weights.pth"

    print(f"Downloading {model_name} weights...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=list(AVAILABLE_WEIGHTS.keys()))
    parser.add_argument('--output-dir', '-o', default='checkpoints')
    args = parser.parse_args()

    download_weights(args.model, args.output_dir)
