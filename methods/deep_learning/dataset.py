"""
PyTorch Dataset for Pansharpening Training
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple


class PansharpeningDataset(Dataset):
    """Dataset for pansharpening with data augmentation."""

    def __init__(self, ms_patches: np.ndarray, pan_patches: np.ndarray,
                 target_patches: np.ndarray, augment: bool = True):
        """
        Args:
            ms_patches: MS patches (N, bands, H, W)
            pan_patches: PAN patches (N, 1, H, W)
            target_patches: Target patches (N, bands, H, W)
            augment: Whether to apply augmentation
        """
        self.ms = torch.from_numpy(ms_patches).float()
        self.pan = torch.from_numpy(pan_patches).float()
        self.target = torch.from_numpy(target_patches).float()
        self.augment = augment

    def __len__(self) -> int:
        return len(self.ms)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ms, pan, target = self.ms[idx], self.pan[idx], self.target[idx]

        if self.augment:
            ms, pan, target = self._augment(ms, pan, target)

        return ms, pan, target

    def _augment(self, ms: torch.Tensor, pan: torch.Tensor,
                 target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply random augmentation."""
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            ms = torch.flip(ms, [-1])
            pan = torch.flip(pan, [-1])
            target = torch.flip(target, [-1])

        # Random vertical flip
        if torch.rand(1) > 0.5:
            ms = torch.flip(ms, [-2])
            pan = torch.flip(pan, [-2])
            target = torch.flip(target, [-2])

        # Random 90-degree rotation
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            ms = torch.rot90(ms, k, [-2, -1])
            pan = torch.rot90(pan, k, [-2, -1])
            target = torch.rot90(target, k, [-2, -1])

        return ms, pan, target
