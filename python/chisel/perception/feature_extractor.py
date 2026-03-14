"""chisel.perception.feature_extractor — SuperPoint and SIFT feature extraction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import cv2


@dataclass
class FeatureData:
    """Container for extracted features from a single image."""
    keypoints: np.ndarray       # (N, 2) pixel coordinates
    descriptors: np.ndarray     # (N, D) descriptor vectors
    scores: np.ndarray          # (N,) detection confidence
    image_size: Tuple[int, int] # (H, W)

    @property
    def num_features(self) -> int:
        return len(self.keypoints)

    def to_cv_keypoints(self) -> List[cv2.KeyPoint]:
        """Convert to OpenCV KeyPoint format."""
        return [
            cv2.KeyPoint(x=float(kp[0]), y=float(kp[1]), size=1.0, response=float(s))
            for kp, s in zip(self.keypoints, self.scores)
        ]


# ─── SuperPoint Network ─────────────────────

class SuperPointEncoder(nn.Module):
    """Shared VGG-style encoder backbone."""

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        c1, c2, c3, c4 = 64, 64, 128, 128

        self.conv1a = nn.Conv2d(1, c1, 3, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, 3, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, 3, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, 3, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, 3, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, 3, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, 3, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        return x


class SuperPointDetectorHead(nn.Module):
    """Interest point detection head: outputs 8×8 grid cells."""

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(128, 256, 3, padding=1)
        self.out = nn.Conv2d(256, 65, 1)  # 8×8 + dustbin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv(x))
        x = self.out(x)
        return x


class SuperPointDescriptorHead(nn.Module):
    """Descriptor head: outputs semi-dense descriptors."""

    def __init__(self, desc_dim: int = 256):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(128, 256, 3, padding=1)
        self.out = nn.Conv2d(256, desc_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv(x))
        x = self.out(x)
        return F.normalize(x, p=2, dim=1)


class SuperPointNet(nn.Module):
    """Full SuperPoint network."""

    def __init__(self, desc_dim: int = 256):
        super().__init__()
        self.encoder = SuperPointEncoder()
        self.detector = SuperPointDetectorHead()
        self.descriptor = SuperPointDescriptorHead(desc_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: (B, 1, H, W) grayscale. Returns dict with 'scores_map' and 'descriptors_map'."""
        features = self.encoder(x)
        scores = self.detector(features)
        descriptors = self.descriptor(features)

        return {
            "scores_map": scores,
            "descriptors_map": descriptors,
        }


class SuperPointExtractor:
    """SuperPoint feature extractor."""

    def __init__(
        self,
        max_keypoints: int = 2048,
        nms_radius: int = 4,
        detection_threshold: float = 0.005,
        desc_dim: int = 256,
        weights_path: Optional[str] = None,
        device: str = "auto",
    ):
        self.max_keypoints = max_keypoints
        self.nms_radius = nms_radius
        self.detection_threshold = detection_threshold
        self.desc_dim = desc_dim

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = SuperPointNet(desc_dim).to(self.device)
        self.model.eval()

        if weights_path and Path(weights_path).exists():
            state = torch.load(weights_path, map_location=self.device)
            state = self._remap_state_dict(state)
            self.model.load_state_dict(state)
            print(f"[perception] Loaded SuperPoint weights from {weights_path}")
        else:
            print("[perception] SuperPoint initialized with random weights "
                  "(provide pretrained weights for production use)")

    @staticmethod
    def _remap_state_dict(state: dict) -> dict:
        """Remap MagicLeap pretrained keys to SuperPointNet layout."""
        key_map = {
            "conv1a": "encoder.conv1a", "conv1b": "encoder.conv1b",
            "conv2a": "encoder.conv2a", "conv2b": "encoder.conv2b",
            "conv3a": "encoder.conv3a", "conv3b": "encoder.conv3b",
            "conv4a": "encoder.conv4a", "conv4b": "encoder.conv4b",
            "convPa": "detector.conv",  "convPb": "detector.out",
            "convDa": "descriptor.conv","convDb": "descriptor.out",
        }
        remapped = {}
        for k, v in state.items():
            prefix = k.split(".")[0]
            suffix = k[len(prefix):]          # e.g. ".weight"
            remapped[key_map.get(prefix, prefix) + suffix] = v
        return remapped

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Convert BGR image → (1, 1, H, W) normalized grayscale tensor."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        tensor = torch.from_numpy(gray.astype(np.float32) / 255.0)
        return tensor.unsqueeze(0).unsqueeze(0).to(self.device)

    def _nms(self, scores: np.ndarray, radius: int) -> np.ndarray:
        """Non-maximum suppression on score map."""
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(scores, size=2 * radius + 1)
        mask = (scores == local_max) & (scores > self.detection_threshold)
        return mask

    @torch.no_grad()
    def extract(self, image: np.ndarray) -> FeatureData:
        """Extract SuperPoint features. image: (H,W,3) BGR or (H,W) grayscale uint8."""
        H, W = image.shape[:2]
        inp = self._preprocess(image)

        # Pad to multiple of 8
        pH = (8 - H % 8) % 8
        pW = (8 - W % 8) % 8
        if pH > 0 or pW > 0:
            inp = F.pad(inp, (0, pW, 0, pH))

        output = self.model(inp)

        # Process detection scores
        scores_map = output["scores_map"][0]  # (65, Hc, Wc)
        scores_map = F.softmax(scores_map, dim=0)
        scores_map = scores_map[:-1]  # remove dustbin channel: (64, Hc, Wc)

        Hc, Wc = scores_map.shape[1:]
        # Reshape from (64, Hc, Wc) → (H, W)
        scores = scores_map.permute(1, 2, 0).reshape(Hc, Wc, 8, 8)
        scores = scores.permute(0, 2, 1, 3).reshape(Hc * 8, Wc * 8)
        scores = scores[:H, :W].cpu().numpy()

        # NMS
        try:
            nms_mask = self._nms(scores, self.nms_radius)
        except ImportError:
            # Fallback without scipy
            nms_mask = scores > self.detection_threshold

        # Get keypoint locations
        ys, xs = np.where(nms_mask)
        kp_scores = scores[ys, xs]

        # Sort by score and keep top-k
        order = np.argsort(-kp_scores)[:self.max_keypoints]
        keypoints = np.stack([xs[order], ys[order]], axis=1).astype(np.float32)
        kp_scores = kp_scores[order]

        # Interpolate descriptors at keypoint locations
        desc_map = output["descriptors_map"][0]  # (D, Hc, Wc)
        D = desc_map.shape[0]

        # Normalize keypoint coords to [-1, 1] for grid_sample
        kp_norm = torch.from_numpy(keypoints).float().to(self.device)
        kp_norm[:, 0] = kp_norm[:, 0] / (W - 1) * 2 - 1
        kp_norm[:, 1] = kp_norm[:, 1] / (H - 1) * 2 - 1

        grid = kp_norm.unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)
        desc_map_4d = desc_map.unsqueeze(0)  # (1, D, Hc, Wc)
        descs = F.grid_sample(desc_map_4d, grid, align_corners=True)
        descs = descs[0, :, 0, :].T  # (N, D)
        descs = F.normalize(descs, p=2, dim=1).cpu().numpy()

        return FeatureData(
            keypoints=keypoints,
            descriptors=descs,
            scores=kp_scores,
            image_size=(H, W),
        )


class SIFTExtractor:
    """OpenCV SIFT feature extractor."""

    def __init__(
        self,
        max_keypoints: int = 4096,
        octave_layers: int = 3,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10,
    ):
        self.max_keypoints = max_keypoints
        self.sift = cv2.SIFT_create(
            nfeatures=max_keypoints,
            nOctaveLayers=octave_layers,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
        )

    def extract(self, image: np.ndarray) -> FeatureData:
        """Extract SIFT features from image."""
        H, W = image.shape[:2]

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        kps, descs = self.sift.detectAndCompute(gray, None)

        if len(kps) == 0:
            return FeatureData(
                keypoints=np.zeros((0, 2), dtype=np.float32),
                descriptors=np.zeros((0, 128), dtype=np.float32),
                scores=np.zeros(0, dtype=np.float32),
                image_size=(H, W),
            )

        keypoints = np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.float32)
        scores = np.array([kp.response for kp in kps], dtype=np.float32)

        # L2-normalize descriptors
        if descs is not None:
            descs = descs.astype(np.float32)
            norms = np.linalg.norm(descs, axis=1, keepdims=True) + 1e-8
            descs = descs / norms
        else:
            descs = np.zeros((len(kps), 128), dtype=np.float32)

        return FeatureData(
            keypoints=keypoints,
            descriptors=descs,
            scores=scores,
            image_size=(H, W),
        )
