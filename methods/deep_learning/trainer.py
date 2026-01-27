"""
Training Pipeline for Deep Learning Pansharpening
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from typing import Dict, Optional, Tuple

from .dataset import PansharpeningDataset


class PansharpeningTrainer:
    """Trainer for pansharpening neural networks."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        device: str = 'cuda',
        lr: float = 0.001,
        checkpoint_dir: str = 'checkpoints'
    ):
        """
        Args:
            model: Neural network model
            criterion: Loss function
            device: 'cuda' or 'cpu'
            lr: Learning rate
            checkpoint_dir: Directory to save checkpoints
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')

    def create_dataloaders(
        self,
        ms_patches: np.ndarray,
        pan_patches: np.ndarray,
        target_patches: np.ndarray,
        batch_size: int = 16,
        val_split: float = 0.2
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders."""
        n = len(ms_patches)
        idx = np.random.permutation(n)
        split = int((1 - val_split) * n)

        train_dataset = PansharpeningDataset(
            ms_patches[idx[:split]],
            pan_patches[idx[:split]],
            target_patches[idx[:split]],
            augment=True
        )

        val_dataset = PansharpeningDataset(
            ms_patches[idx[split:]],
            pan_patches[idx[split:]],
            target_patches[idx[split:]],
            augment=False
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Print data statistics
        print(f"\nDataset Statistics:")
        print(f"  Total patches: {n}")
        print(f"  Training patches: {len(train_dataset)}")
        print(f"  Validation patches: {len(val_dataset)}")

        # Compute and print the difference between input and target
        diff = target_patches - ms_patches
        print(f"  Input MS range: [{ms_patches.min():.4f}, {ms_patches.max():.4f}]")
        print(f"  Target range: [{target_patches.min():.4f}, {target_patches.max():.4f}]")
        print(f"  Residual (target-input) stats:")
        print(f"    Mean: {diff.mean():.6f}, Std: {diff.std():.6f}")
        print(f"    Range: [{diff.min():.4f}, {diff.max():.4f}]")

        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc='Training', leave=False)
        for ms, pan, target in pbar:
            ms = ms.to(self.device)
            pan = pan.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(ms, pan)
            loss, _ = self.criterion(output, target)

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / len(train_loader)

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0

        for ms, pan, target in val_loader:
            ms = ms.to(self.device)
            pan = pan.to(self.device)
            target = target.to(self.device)

            output = self.model(ms, pan)
            loss, _ = self.criterion(output, target)
            total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 200,
        save_every: int = 20,
        warmup_epochs: int = 10
    ):
        """Main training loop with warmup."""
        # Create scheduler with warmup
        # Adjust warmup if epochs is too small
        effective_warmup = min(warmup_epochs, max(1, epochs - 1))

        def lr_lambda(epoch):
            if epoch < effective_warmup:
                # Linear warmup
                return (epoch + 1) / effective_warmup
            else:
                # Cosine annealing after warmup
                remaining = epochs - effective_warmup
                if remaining <= 0:
                    return 1.0
                progress = (epoch - effective_warmup) / remaining
                return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        print(f"Training for {epochs} epochs on {self.device}")
        print(f"  Warmup: {effective_warmup} epochs")
        print(f"  Initial LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        print("=" * 50)

        no_improve_count = 0
        patience = 30  # Early stopping patience

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            scheduler.step()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            # Print progress
            if (epoch + 1) % save_every == 0 or epoch == 0 or epoch < 5:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train: {train_loss:.6f}, Val: {val_loss:.6f}, LR: {lr:.2e}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best.pth')
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Early stopping check
            if no_improve_count >= patience and epoch > warmup_epochs + 20:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

        # Save final model
        self.save_checkpoint('final.pth')
        self.save_history()

        print(f"\nTraining complete! Best val loss: {self.best_val_loss:.6f}")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, filepath: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Loaded checkpoint from {filepath}")

    def save_history(self):
        """Save training history."""
        with open(self.checkpoint_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

    @torch.no_grad()
    def predict(self, ms: np.ndarray, pan: np.ndarray) -> np.ndarray:
        """
        Run inference on full image.

        Args:
            ms: MS image (bands, H, W)
            pan: PAN image (1, H, W)

        Returns:
            Fused image (bands, H, W)
        """
        self.model.eval()

        ms_tensor = torch.from_numpy(ms).unsqueeze(0).float().to(self.device)
        pan_tensor = torch.from_numpy(pan).unsqueeze(0).float().to(self.device)

        output = self.model(ms_tensor, pan_tensor)
        return np.clip(output.squeeze(0).cpu().numpy(), 0, 1)
