"""
chisel.perception.feature_matcher
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Feature matching: classical NN and learned (LightGlue-style) approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from .feature_extractor import FeatureData


@dataclass
class MatchResult:
    """Container for feature matches between two images."""
    matches: np.ndarray          # (M, 2) index pairs (idx_in_img1, idx_in_img2)
    match_scores: np.ndarray     # (M,) confidence
    num_inliers: int = 0
    fundamental: Optional[np.ndarray] = None  # 3×3 F matrix if verified

    @property
    def num_matches(self) -> int:
        return len(self.matches)


class NNMatcher:
    """
    Nearest-neighbor descriptor matching with ratio test and
    optional geometric verification via fundamental matrix RANSAC.
    """

    def __init__(
        self,
        ratio_threshold: float = 0.80,
        mutual_check: bool = True,
        ransac_threshold: float = 3.0,
        verify_geometry: bool = True,
        device: str = "auto",
    ):
        self.ratio_threshold = ratio_threshold
        self.mutual_check = mutual_check
        self.ransac_threshold = ransac_threshold
        self.verify_geometry = verify_geometry

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def match(self, feat1: FeatureData, feat2: FeatureData) -> MatchResult:
        """Match features between two images."""
        if feat1.num_features == 0 or feat2.num_features == 0:
            return MatchResult(
                matches=np.zeros((0, 2), dtype=np.int64),
                match_scores=np.zeros(0, dtype=np.float32),
            )

        # Compute distance matrix on GPU
        d1 = torch.from_numpy(feat1.descriptors).float().to(self.device)
        d2 = torch.from_numpy(feat2.descriptors).float().to(self.device)

        # Cosine similarity → distance
        sim = torch.mm(d1, d2.t())  # (N1, N2)
        dist = 1 - sim  # lower = better match

        # Forward kNN (k=2 for ratio test)
        topk_dist, topk_idx = dist.topk(2, dim=1, largest=False)

        # Ratio test
        ratio = topk_dist[:, 0] / (topk_dist[:, 1] + 1e-8)
        ratio_mask = ratio < self.ratio_threshold

        # Forward matches: query_idx → train_idx
        fwd_matches = topk_idx[:, 0]

        if self.mutual_check:
            # Backward kNN
            bwd_topk_dist, bwd_topk_idx = dist.topk(2, dim=0, largest=False)
            bwd_matches = bwd_topk_idx[0]  # best match for each column

            # Check mutual consistency
            n1 = d1.shape[0]
            mutual_mask = torch.zeros(n1, dtype=torch.bool, device=self.device)
            for i in range(n1):
                if ratio_mask[i] and bwd_matches[fwd_matches[i]] == i:
                    mutual_mask[i] = True
            valid_mask = mutual_mask
        else:
            valid_mask = ratio_mask

        # Collect matches
        valid_indices = torch.where(valid_mask)[0].cpu().numpy()
        fwd_matches_np = fwd_matches.cpu().numpy()

        matches = np.stack([
            valid_indices,
            fwd_matches_np[valid_indices]
        ], axis=1)

        scores = (1 - topk_dist[:, 0]).cpu().numpy()[valid_indices]

        result = MatchResult(matches=matches, match_scores=scores)

        # Geometric verification
        if self.verify_geometry and len(matches) >= 8:
            result = self._verify_fundamental(feat1, feat2, result)

        return result

    def _verify_fundamental(
            self,
            feat1: FeatureData,
            feat2: FeatureData,
            result: MatchResult) -> MatchResult:
        """RANSAC fundamental matrix estimation for geometric verification."""
        import cv2

        pts1 = feat1.keypoints[result.matches[:, 0]]
        pts2 = feat2.keypoints[result.matches[:, 1]]

        F_mat, mask = cv2.findFundamentalMat(
            pts1, pts2, cv2.FM_RANSAC,
            self.ransac_threshold, 0.999)

        if F_mat is None or mask is None:
            return result

        inlier_mask = mask.ravel().astype(bool)
        result.matches = result.matches[inlier_mask]
        result.match_scores = result.match_scores[inlier_mask]
        result.num_inliers = int(inlier_mask.sum())
        result.fundamental = F_mat

        return result


# ─── Learned Matcher (LightGlue-style) ──────

class AttentionBlock(nn.Module):
    """Self/cross-attention block for learned matching."""

    def __init__(self, d_model: int = 256, n_heads: int = 4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor):
        # Self-attention
        x0 = x0 + self.self_attn(self.norm1(x0), self.norm1(x0), self.norm1(x0))[0]
        x1 = x1 + self.self_attn(self.norm1(x1), self.norm1(x1), self.norm1(x1))[0]

        # Cross-attention
        x0_new = x0 + self.cross_attn(self.norm2(x0), self.norm2(x1), self.norm2(x1))[0]
        x1_new = x1 + self.cross_attn(self.norm2(x1), self.norm2(x0), self.norm2(x0))[0]

        # FFN
        x0_new = x0_new + self.ffn(self.norm3(x0_new))
        x1_new = x1_new + self.ffn(self.norm3(x1_new))

        return x0_new, x1_new


class LightGlueNet(nn.Module):
    """Simplified LightGlue-style learned feature matcher."""

    def __init__(self, d_model: int = 256, n_layers: int = 6, n_heads: int = 4):
        super().__init__()

        # Positional encoding from keypoint coordinates
        self.pos_enc = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

        # Descriptor projection
        self.desc_proj = nn.Linear(d_model, d_model)

        # Attention layers
        self.layers = nn.ModuleList([
            AttentionBlock(d_model, n_heads) for _ in range(n_layers)
        ])

        # Final matching head
        self.match_head = nn.Linear(d_model, d_model)

    def forward(
        self,
        desc0: torch.Tensor,   # (B, N, D)
        desc1: torch.Tensor,   # (B, M, D)
        kpt0: torch.Tensor,    # (B, N, 2) normalized coords
        kpt1: torch.Tensor,    # (B, M, 2) normalized coords
    ) -> torch.Tensor:
        """Returns (B, N, M) log-assignment matrix."""

        # Combine descriptor + positional features
        x0 = self.desc_proj(desc0) + self.pos_enc(kpt0)
        x1 = self.desc_proj(desc1) + self.pos_enc(kpt1)

        # Attention layers
        for layer in self.layers:
            x0, x1 = layer(x0, x1)

        # Matching scores
        x0 = self.match_head(x0)
        x1 = self.match_head(x1)

        scores = torch.bmm(x0, x1.transpose(1, 2))  # (B, N, M)
        return scores


class LightGlueMatcher:
    """
    Learned feature matcher using attention-based architecture.
    Falls back to NN matching if model is not loaded.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        weights_path: Optional[str] = None,
        match_threshold: float = 0.2,
        device: str = "auto",
    ):
        self.match_threshold = match_threshold

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = LightGlueNet(d_model, n_layers).to(self.device)
        self.model.eval()

        if weights_path:
            from pathlib import Path
            if Path(weights_path).exists():
                state = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state)
                print(f"[perception] Loaded LightGlue weights: {weights_path}")

        # Fallback matcher
        self._nn_matcher = NNMatcher(device=device)

    @torch.no_grad()
    def match(self, feat1: FeatureData, feat2: FeatureData) -> MatchResult:
        """Match features using the learned model."""
        if feat1.num_features == 0 or feat2.num_features == 0:
            return MatchResult(
                matches=np.zeros((0, 2), dtype=np.int64),
                match_scores=np.zeros(0, dtype=np.float32),
            )

        # Prepare inputs
        desc0 = torch.from_numpy(feat1.descriptors).float().unsqueeze(0).to(self.device)
        desc1 = torch.from_numpy(feat2.descriptors).float().unsqueeze(0).to(self.device)

        H1, W1 = feat1.image_size
        H2, W2 = feat2.image_size
        kpt0 = torch.from_numpy(feat1.keypoints).float().unsqueeze(0).to(self.device)
        kpt1 = torch.from_numpy(feat2.keypoints).float().unsqueeze(0).to(self.device)

        # Normalize coordinates to [-1, 1]
        kpt0[:, :, 0] = kpt0[:, :, 0] / W1 * 2 - 1
        kpt0[:, :, 1] = kpt0[:, :, 1] / H1 * 2 - 1
        kpt1[:, :, 0] = kpt1[:, :, 0] / W2 * 2 - 1
        kpt1[:, :, 1] = kpt1[:, :, 1] / H2 * 2 - 1

        try:
            scores = self.model(desc0, desc1, kpt0, kpt1)  # (1, N, M)
            scores = scores[0]

            # Dual softmax for assignment
            p0 = F.softmax(scores, dim=1)
            p1 = F.softmax(scores, dim=0)
            mutual_scores = (p0 * p1)

            # Extract mutual nearest neighbors above threshold
            max0 = mutual_scores.max(dim=1)
            max1 = mutual_scores.max(dim=0)

            mutual_mask = torch.zeros_like(mutual_scores, dtype=torch.bool)
            for i in range(scores.shape[0]):
                j = max0.indices[i]
                if max1.indices[j] == i and max0.values[i] > self.match_threshold:
                    mutual_mask[i, j] = True

            idx0, idx1 = torch.where(mutual_mask)
            match_scores = mutual_scores[idx0, idx1]

            matches = np.stack([
                idx0.cpu().numpy(),
                idx1.cpu().numpy()
            ], axis=1)

            return MatchResult(
                matches=matches,
                match_scores=match_scores.cpu().numpy(),
                num_inliers=len(matches),
            )
        except Exception as e:
            print(f"[perception] LightGlue failed, falling back to NN: {e}")
            return self._nn_matcher.match(feat1, feat2)
