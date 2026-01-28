#!/usr/bin/env python
"""Export trained model weights for release."""

import argparse
import torch
from pathlib import Path


def export_weights(checkpoint_path: str, output_path: str = None):
    """Export model weights, removing optimizer state."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract only model weights
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            weights = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            weights = checkpoint['state_dict']
        else:
            weights = checkpoint
    else:
        weights = checkpoint

    # Save clean weights
    if output_path is None:
        output_path = checkpoint_path.replace('.pth', '_weights.pth')

    torch.save(weights, output_path)
    print(f"Exported weights to: {output_path}")

    # Print size
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='Path to checkpoint')
    parser.add_argument('--output', '-o', help='Output path')
    args = parser.parse_args()

    export_weights(args.checkpoint, args.output)
