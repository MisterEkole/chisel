"""
chisel.perception.depth_estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Monocular depth estimation for initializing dense reconstruction.
Uses a DPT/MiDaS-style encoder-decoder architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from pathlib import Path


class DepthDecoderBlock(nn.Module):
    """Upsampling + refinement block for depth decoder."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None):
        x = self.up(x)
        if skip is not None:
            # Align sizes
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                                  align_corners=True)
            x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class SimpleDepthNet(nn.Module):
    """
    Simplified encoder-decoder for monocular depth estimation.
    Uses ResNet-style encoder with skip connections.
    """

    def __init__(self, encoder_channels=(64, 128, 256, 512)):
        super().__init__()

        # Encoder (simplified ResNet-style)
        c1, c2, c3, c4 = encoder_channels

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, c1, 7, stride=2, padding=3),
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(True),
            nn.Conv2d(c2, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(True),
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(True),
            nn.Conv2d(c3, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(True),
        )
        self.enc4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c3, c4, 3, padding=1), nn.BatchNorm2d(c4), nn.ReLU(True),
            nn.Conv2d(c4, c4, 3, padding=1), nn.BatchNorm2d(c4), nn.ReLU(True),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c4, c4, 3, padding=1), nn.BatchNorm2d(c4), nn.ReLU(True),
        )

        # Decoder
        self.dec4 = DepthDecoderBlock(c4, c4, c3)
        self.dec3 = DepthDecoderBlock(c3, c3, c2)
        self.dec2 = DepthDecoderBlock(c2, c2, c1)
        self.dec1 = DepthDecoderBlock(c1, c1, c1)

        # Depth head (outputs inverse depth for numerical stability)
        self.depth_head = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, 1, 1),
            nn.Sigmoid(),  # output in [0, 1] → scale to depth range
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) RGB images normalized to [0, 1]
        Returns:
            (B, 1, H, W) inverse depth map in [0, 1]
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        bn = self.bottleneck(e4)

        # Decoder with skip connections
        d4 = self.dec4(bn, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        # Upsample to input resolution
        d1 = F.interpolate(d1, size=x.shape[2:], mode="bilinear", align_corners=True)

        return self.depth_head(d1)


class MonocularDepthEstimator:
    """
    Monocular depth estimation for initializing dense MVS.

    Provides relative depth priors that help guide the dense matching
    cost volume in the reconstruction layer.

    Usage:
        estimator = MonocularDepthEstimator()
        depth, confidence = estimator.estimate(image_bgr)
    """

    def __init__(
        self,
        min_depth: float = 0.1,
        max_depth: float = 100.0,
        target_size: Tuple[int, int] = (384, 512),  # (H, W) for inference
        weights_path: Optional[str] = None,
        device: str = "auto",
    ):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.target_size = target_size

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = SimpleDepthNet().to(self.device)
        self.model.eval()

        if weights_path and Path(weights_path).exists():
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"[perception] Loaded depth model from {weights_path}")
        else:
            print("[perception] Depth model initialized with random weights "
                  "(provide pretrained weights for metric depth)")

    def _preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Preprocess image for depth estimation."""
        import cv2
        H, W = image.shape[:2]

        # Convert BGR → RGB and normalize
        if len(image.shape) == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Resize to target size
        resized = cv2.resize(rgb, (self.target_size[1], self.target_size[0]))

        tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        return tensor.to(self.device), (H, W)

    @torch.no_grad()
    def estimate(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate monocular depth from a single image.

        Args:
            image: (H, W, 3) BGR uint8 image

        Returns:
            depth: (H, W) depth map in meters
            confidence: (H, W) confidence map in [0, 1]
        """
        import cv2

        inp, orig_size = self._preprocess(image)

        inv_depth = self.model(inp)  # (1, 1, Ht, Wt) in [0, 1]
        inv_depth = inv_depth[0, 0].cpu().numpy()

        # Convert inverse depth → metric depth
        # inv_depth ∈ [0, 1] → depth ∈ [min_depth, max_depth]
        inv_min = 1.0 / self.max_depth
        inv_max = 1.0 / self.min_depth
        inv_depth_metric = inv_min + inv_depth * (inv_max - inv_min)
        depth = 1.0 / (inv_depth_metric + 1e-8)

        # Resize back to original resolution
        depth = cv2.resize(depth, (orig_size[1], orig_size[0]),
                           interpolation=cv2.INTER_LINEAR)

        # Simple confidence: based on gradient magnitude (edges = uncertain)
        grad_x = np.abs(np.gradient(depth, axis=1))
        grad_y = np.abs(np.gradient(depth, axis=0))
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        confidence = 1.0 / (1.0 + grad_mag / np.mean(grad_mag + 1e-8))
        confidence = confidence.astype(np.float32)

        return depth.astype(np.float32), confidence

    def estimate_batch(
        self, images: list
    ) -> list:
        """Estimate depth for a batch of images."""
        results = []
        for img in images:
            depth, conf = self.estimate(img)
            results.append((depth, conf))
        return results
