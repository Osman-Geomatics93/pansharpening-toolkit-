"""
Loss Functions for Deep Learning Pansharpening

Includes:
- GradientLoss: Edge/spatial fidelity loss
- CombinedLoss: L1 + MSE + Gradient
- SpectralAngleLoss: Spectral Angle Mapper (SAM) loss
- SSIMLoss: Structural Similarity loss
- PerceptualLoss: VGG feature-based loss
- AdvancedCombinedLoss: Configurable multi-loss combination
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


class GradientLoss(nn.Module):
    """
    Gradient/Edge loss for spatial fidelity.
    Uses Sobel filters to compute image gradients.
    """

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient loss.

        Args:
            pred: Predicted image (B, bands, H, W)
            target: Target image (B, bands, H, W)

        Returns:
            Gradient loss value
        """
        sobel_x = self.sobel_x.to(pred.device)
        sobel_y = self.sobel_y.to(pred.device)

        # Average across bands
        pred_gray = pred.mean(dim=1, keepdim=True)
        target_gray = target.mean(dim=1, keepdim=True)

        # Compute gradients
        pred_gx = F.conv2d(pred_gray, sobel_x, padding=1)
        pred_gy = F.conv2d(pred_gray, sobel_y, padding=1)
        target_gx = F.conv2d(target_gray, sobel_x, padding=1)
        target_gy = F.conv2d(target_gray, sobel_y, padding=1)

        loss = F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss: L1 + MSE + Gradient.
    Balanced for stable training.
    """

    def __init__(self, l1_weight: float = 1.0, mse_weight: float = 1.0,
                 gradient_weight: float = 0.1):
        """
        Args:
            l1_weight: Weight for L1 loss
            mse_weight: Weight for MSE loss
            gradient_weight: Weight for gradient loss
        """
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.gradient_weight = gradient_weight

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.gradient_loss = GradientLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Compute combined loss.

        Args:
            pred: Predicted image
            target: Target image

        Returns:
            total_loss, loss_dict
        """
        l1 = self.l1_loss(pred, target)
        mse = self.mse_loss(pred, target)
        gradient = self.gradient_loss(pred, target)

        total = (self.l1_weight * l1 +
                 self.mse_weight * mse +
                 self.gradient_weight * gradient)

        loss_dict = {
            'l1': l1.item(),
            'mse': mse.item(),
            'gradient': gradient.item()
        }

        return total, loss_dict


class SpectralAngleLoss(nn.Module):
    """
    Spectral Angle Mapper (SAM) Loss.

    Measures spectral fidelity by computing the angle between spectral vectors.
    Lower angle = better spectral preservation.
    """

    def __init__(self, eps: float = 1e-8):
        """
        Args:
            eps: Small value for numerical stability
        """
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute SAM loss.

        Args:
            pred: Predicted image (B, bands, H, W)
            target: Target image (B, bands, H, W)

        Returns:
            Mean spectral angle (in radians)
        """
        # Compute dot product along spectral dimension
        dot = (pred * target).sum(dim=1)

        # Compute norms
        pred_norm = torch.sqrt((pred ** 2).sum(dim=1) + self.eps)
        target_norm = torch.sqrt((target ** 2).sum(dim=1) + self.eps)

        # Compute cosine similarity, clamp for numerical stability
        cos_sim = dot / (pred_norm * target_norm + self.eps)
        cos_sim = torch.clamp(cos_sim, -1 + self.eps, 1 - self.eps)

        # Compute angle
        angle = torch.acos(cos_sim)

        return angle.mean()


class SSIMLoss(nn.Module):
    """
    Structural Similarity (SSIM) Loss.

    Measures structural similarity between images.
    SSIM = 1 means identical; loss = 1 - SSIM.
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5,
                 channel: int = None, reduction: str = 'mean'):
        """
        Args:
            window_size: Size of the Gaussian window
            sigma: Standard deviation of Gaussian window
            channel: Number of channels (auto-detected if None)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channel = channel
        self.reduction = reduction
        self._window = None

    def _create_window(self, channel: int, device: torch.device) -> torch.Tensor:
        """Create Gaussian window."""
        # Create 1D Gaussian
        coords = torch.arange(self.window_size, device=device).float()
        coords -= self.window_size // 2
        gauss = torch.exp(-coords ** 2 / (2 * self.sigma ** 2))
        gauss = gauss / gauss.sum()

        # Create 2D window
        window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0)
        window = window.expand(channel, 1, self.window_size, self.window_size)

        return window.contiguous()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM loss (1 - SSIM).

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            SSIM loss value
        """
        channel = pred.size(1)

        # Create or reuse window
        if self._window is None or self._window.size(0) != channel:
            self._window = self._create_window(channel, pred.device)
        window = self._window.to(pred.device)

        # Constants for stability
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Compute means
        mu_pred = F.conv2d(pred, window, padding=self.window_size // 2, groups=channel)
        mu_target = F.conv2d(target, window, padding=self.window_size // 2, groups=channel)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        # Compute variances and covariance
        sigma_pred_sq = F.conv2d(pred ** 2, window, padding=self.window_size // 2, groups=channel) - mu_pred_sq
        sigma_target_sq = F.conv2d(target ** 2, window, padding=self.window_size // 2, groups=channel) - mu_target_sq
        sigma_pred_target = F.conv2d(pred * target, window, padding=self.window_size // 2, groups=channel) - mu_pred_target

        # SSIM formula
        ssim = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

        if self.reduction == 'mean':
            return 1 - ssim.mean()
        elif self.reduction == 'sum':
            return (1 - ssim).sum()
        else:
            return 1 - ssim


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using VGG features.

    Computes loss in feature space rather than pixel space.
    Captures high-level structural and textural similarity.
    """

    def __init__(self, layers: list = None, weights: list = None):
        """
        Args:
            layers: VGG layers to use (default: relu1_2, relu2_2, relu3_3)
            weights: Weights for each layer's loss
        """
        super().__init__()
        self.layers = layers or ['relu1_2', 'relu2_2', 'relu3_3']
        self.weights = weights or [1.0, 1.0, 1.0]
        self._vgg = None
        self._mean = None
        self._std = None

    def _load_vgg(self, device: torch.device):
        """Lazy load VGG model."""
        if self._vgg is not None:
            return

        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        except ImportError:
            # Fallback for older torchvision
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True).features.eval()

        # Freeze VGG
        for param in vgg.parameters():
            param.requires_grad = False

        # Layer indices for VGG16
        layer_indices = {
            'relu1_1': 1, 'relu1_2': 3,
            'relu2_1': 6, 'relu2_2': 8,
            'relu3_1': 11, 'relu3_2': 13, 'relu3_3': 15,
            'relu4_1': 18, 'relu4_2': 20, 'relu4_3': 22,
            'relu5_1': 25, 'relu5_2': 27, 'relu5_3': 29
        }

        # Build feature extractor
        max_idx = max(layer_indices[l] for l in self.layers)
        self._vgg = nn.Sequential(*list(vgg.children())[:max_idx + 1]).to(device)
        self._layer_indices = [layer_indices[l] for l in self.layers]

        # ImageNet normalization
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input to ImageNet statistics."""
        return (x - self._mean) / self._std

    def _extract_features(self, x: torch.Tensor) -> list:
        """Extract features from specified VGG layers."""
        # Convert to 3 channels if needed
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.size(1) == 4:
            # Use first 3 bands (typically R, G, B)
            x = x[:, :3]
        elif x.size(1) > 4:
            x = x[:, :3]

        x = self._normalize(x)

        features = []
        for i, layer in enumerate(self._vgg):
            x = layer(x)
            if i in self._layer_indices:
                features.append(x)

        return features

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            Perceptual loss value
        """
        self._load_vgg(pred.device)

        pred_features = self._extract_features(pred)
        target_features = self._extract_features(target)

        loss = 0
        for w, pf, tf in zip(self.weights, pred_features, target_features):
            loss = loss + w * F.l1_loss(pf, tf)

        return loss


class AdvancedCombinedLoss(nn.Module):
    """
    Advanced Combined Loss with configurable components.

    Supports: L1, MSE, Gradient, SSIM, SAM, Perceptual
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        mse_weight: float = 0.5,
        gradient_weight: float = 0.1,
        ssim_weight: float = 0.0,
        sam_weight: float = 0.0,
        perceptual_weight: float = 0.0
    ):
        """
        Args:
            l1_weight: Weight for L1 loss
            mse_weight: Weight for MSE loss
            gradient_weight: Weight for gradient loss
            ssim_weight: Weight for SSIM loss
            sam_weight: Weight for SAM (spectral angle) loss
            perceptual_weight: Weight for perceptual loss
        """
        super().__init__()

        self.weights = {
            'l1': l1_weight,
            'mse': mse_weight,
            'gradient': gradient_weight,
            'ssim': ssim_weight,
            'sam': sam_weight,
            'perceptual': perceptual_weight
        }

        # Initialize loss functions based on weights
        self.l1_loss = nn.L1Loss() if l1_weight > 0 else None
        self.mse_loss = nn.MSELoss() if mse_weight > 0 else None
        self.gradient_loss = GradientLoss() if gradient_weight > 0 else None
        self.ssim_loss = SSIMLoss() if ssim_weight > 0 else None
        self.sam_loss = SpectralAngleLoss() if sam_weight > 0 else None
        self.perceptual_loss = PerceptualLoss() if perceptual_weight > 0 else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            pred: Predicted image (B, C, H, W)
            target: Target image (B, C, H, W)

        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Dictionary with individual loss values
        """
        total = torch.tensor(0.0, device=pred.device)
        loss_dict = {}

        if self.l1_loss is not None:
            l1 = self.l1_loss(pred, target)
            total = total + self.weights['l1'] * l1
            loss_dict['l1'] = l1.item()

        if self.mse_loss is not None:
            mse = self.mse_loss(pred, target)
            total = total + self.weights['mse'] * mse
            loss_dict['mse'] = mse.item()

        if self.gradient_loss is not None:
            gradient = self.gradient_loss(pred, target)
            total = total + self.weights['gradient'] * gradient
            loss_dict['gradient'] = gradient.item()

        if self.ssim_loss is not None:
            ssim = self.ssim_loss(pred, target)
            total = total + self.weights['ssim'] * ssim
            loss_dict['ssim'] = ssim.item()

        if self.sam_loss is not None:
            sam = self.sam_loss(pred, target)
            total = total + self.weights['sam'] * sam
            loss_dict['sam'] = sam.item()

        if self.perceptual_loss is not None:
            perceptual = self.perceptual_loss(pred, target)
            total = total + self.weights['perceptual'] * perceptual
            loss_dict['perceptual'] = perceptual.item()

        return total, loss_dict
