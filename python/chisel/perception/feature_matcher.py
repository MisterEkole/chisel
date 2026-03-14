"""chisel.perception.feature_matcher — NN and LightGlue feature matching."""

import torch
import torch.nn as nn
import torch.nn.functional as F  # used in feature_extractor; kept for _apply_rope
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

        pts1 = feat1.keypoints[result.matches[:, 0]].astype(np.float32)
        pts2 = feat2.keypoints[result.matches[:, 1]].astype(np.float32)

        pts1= np.ascontiguousarray(pts1)
        pts2 = np.ascontiguousarray(pts2)

        if pts1.shape[0] < 8:
            return result
        
        try:
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

        except cv2.error as e:
            print("Perception Error: Geometric Verification failed")

        return result

            




# ─── Learned Matcher (LightGlue — ETH CVG architecture) ──────

def _make_ffn(d_model: int) -> nn.Sequential:
    """FFN: cat([x, msg]) (2*d_model) → d_model."""
    return nn.Sequential(
        nn.Linear(2 * d_model, 2 * d_model),
        nn.LayerNorm(2 * d_model),
        nn.GELU(),
        nn.Linear(2 * d_model, d_model),
    )


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional encoding to Q or K."""
    cos = torch.cos(freqs).unsqueeze(1)   # (B, 1, N, head_dim//2)
    sin = torch.sin(freqs).unsqueeze(1)
    x1, x2 = x[..., ::2], x[..., 1::2]  # split even / odd dims
    x_rot = torch.stack([x1 * cos - x2 * sin,
                         x1 * sin + x2 * cos], dim=-1)
    return x_rot.flatten(-2)              # (B, H, N, head_dim)


class RotaryPositionalEncoding(nn.Module):
    """Learned rotary positional encoding; cos/sin applied inside attention."""

    def __init__(self, n_heads: int = 4, head_dim: int = 64):
        super().__init__()
        self.Wr = nn.Linear(2, head_dim // 2, bias=False)

    def forward(self, kpts: torch.Tensor) -> torch.Tensor:
        return self.Wr(kpts)   # (B, N, head_dim//2) — raw angles


class SelfAttentionBlock(nn.Module):
    """Per-image self-attention with fused QKV and RoPE on Q/K."""

    def __init__(self, d_model: int = 256, n_heads: int = 4):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.Wqkv     = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn      = _make_ffn(d_model)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q, k, v = self.Wqkv(x).chunk(3, dim=-1)         # each (B, N, D)

        def to_heads(t):
            return t.reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)  # (B, H, N, d)
        q = _apply_rope(q, freqs)
        k = _apply_rope(k, freqs)

        scale = self.head_dim ** -0.5
        attn  = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
        msg   = (attn @ v).transpose(1, 2).reshape(B, N, D)
        msg   = self.out_proj(msg)
        return x + self.ffn(torch.cat([x, msg], dim=-1))


class CrossAttentionBlock(nn.Module):
    """Cross-image attention with shared to_qk projection."""

    def __init__(self, d_model: int = 256, n_heads: int = 4):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.to_qk    = nn.Linear(d_model, d_model)
        self.to_v     = nn.Linear(d_model, d_model)
        self.to_out   = nn.Linear(d_model, d_model)
        self.ffn      = _make_ffn(d_model)

    def _cross(self, x: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q = self.to_qk(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.to_qk(src).reshape(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(src).reshape(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        scale = self.head_dim ** -0.5
        attn  = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
        msg   = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return x + self.ffn(torch.cat([x, self.to_out(msg)], dim=-1))

    def forward(self, x0: torch.Tensor,
                x1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Use pre-update features for both directions (simultaneous update)
        return self._cross(x0, x1), self._cross(x1, x0)


class LogAssignment(nn.Module):
    """Per-layer assignment head: produces (B, N, M) matching scores."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.r            = nn.Parameter(torch.ones(1))
        self.matchability = nn.Linear(d_model, 1)
        self.final_proj   = nn.Linear(d_model, d_model)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> torch.Tensor:
        m0 = self.matchability(desc0)                              # (B, N, 1)
        m1 = self.matchability(desc1)                              # (B, M, 1)
        scores = self.r * torch.bmm(
            self.final_proj(desc0), self.final_proj(desc1).transpose(1, 2)
        )                                                          # (B, N, M)
        return scores + m0 + m1.transpose(1, 2)


class TokenConfidence(nn.Module):
    """Early-exit confidence token (one per layer except the last)."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.token = nn.Sequential(nn.Linear(d_model, 1))

    def forward(self, desc0: torch.Tensor,
                desc1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.token(desc0.detach()).squeeze(-1),
                self.token(desc1.detach()).squeeze(-1))


class LightGlueNet(nn.Module):
    """LightGlue feature matcher (ETH CVG checkpoint layout)."""

    def __init__(self, d_model: int = 256, n_layers: int = 9, n_heads: int = 4):
        super().__init__()
        head_dim = d_model // n_heads
        self.posenc           = RotaryPositionalEncoding(n_heads, head_dim)
        self.self_attn        = nn.ModuleList([SelfAttentionBlock(d_model, n_heads)
                                               for _ in range(n_layers)])
        self.cross_attn       = nn.ModuleList([CrossAttentionBlock(d_model, n_heads)
                                               for _ in range(n_layers)])
        self.log_assignment   = nn.ModuleList([LogAssignment(d_model)
                                               for _ in range(n_layers)])
        self.token_confidence = nn.ModuleList([TokenConfidence(d_model)
                                               for _ in range(n_layers - 1)])

    def forward(
        self,
        desc0: torch.Tensor,   # (B, N, D)
        desc1: torch.Tensor,   # (B, M, D)
        kpt0:  torch.Tensor,   # (B, N, 2) normalised coords
        kpt1:  torch.Tensor,   # (B, M, 2) normalised coords
    ) -> torch.Tensor:
        """Returns (B, N, M) assignment scores from the final layer."""
        freqs0 = self.posenc(kpt0)
        freqs1 = self.posenc(kpt1)
        for sa, ca in zip(self.self_attn, self.cross_attn):
            desc0 = sa(desc0, freqs0)
            desc1 = sa(desc1, freqs1)
            desc0, desc1 = ca(desc0, desc1)
        return self.log_assignment[-1](desc0, desc1)


class LightGlueMatcher:
    """
    Learned feature matcher using attention-based architecture.
    Falls back to NN matching if model is not loaded.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 9,
        weights_path: Optional[str] = None,
        match_threshold: float = 1e-4,
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
                state = torch.load(weights_path, map_location=self.device,
                                   weights_only=True)
                # The checkpoint may be wrapped under a top-level key
                if "model" in state:
                    state = state["model"]
                elif "state_dict" in state:
                    state = state["state_dict"]
                self.model.load_state_dict(state, strict=True)
                print(f"[perception] Loaded LightGlue weights: {weights_path}")
            else:
                print(f"[perception] LightGlue weights not found at {weights_path}, "
                      "using random weights")

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

        desc0 = torch.from_numpy(feat1.descriptors).float().unsqueeze(0).to(self.device)
        desc1 = torch.from_numpy(feat2.descriptors).float().unsqueeze(0).to(self.device)

        H1, W1 = feat1.image_size
        H2, W2 = feat2.image_size
        kpt0 = torch.from_numpy(feat1.keypoints).float().unsqueeze(0).to(self.device)
        kpt1 = torch.from_numpy(feat2.keypoints).float().unsqueeze(0).to(self.device)

        # Normalize: centre on image, scale by max(H,W)/2 (preserves aspect ratio for RoPE).
        kpt0 = kpt0.clone()
        scale0 = max(H1, W1) / 2.0
        kpt0[:, :, 0] = (kpt0[:, :, 0] - W1 / 2.0) / scale0
        kpt0[:, :, 1] = (kpt0[:, :, 1] - H1 / 2.0) / scale0
        kpt1 = kpt1.clone()
        scale1 = max(H2, W2) / 2.0
        kpt1[:, :, 0] = (kpt1[:, :, 0] - W2 / 2.0) / scale1
        kpt1[:, :, 1] = (kpt1[:, :, 1] - H2 / 2.0) / scale1

        try:
            scores = self.model(desc0, desc1, kpt0, kpt1)[0]  # (N, M)
            N = scores.shape[0]

            idx0 = scores.argmax(dim=1)  # (N,) best col for each row
            idx1 = scores.argmax(dim=0)  # (M,) best row for each col
            arange = torch.arange(N, device=scores.device)
            mutual  = arange == idx1[idx0]        # True where i→j and j→i agree
            raw_val = scores[arange, idx0]
            valid   = mutual & (raw_val > self.match_threshold)

            valid_i = torch.where(valid)[0]
            if len(valid_i) == 0:
                return MatchResult(
                    matches=np.zeros((0, 2), dtype=np.int64),
                    match_scores=np.zeros(0, dtype=np.float32),
                )

            matches = np.stack([
                valid_i.cpu().numpy(),
                idx0[valid_i].cpu().numpy(),
            ], axis=1)
            match_scores = raw_val[valid_i].cpu().numpy()

            result = MatchResult(matches=matches, match_scores=match_scores)

            # Geometric verification
            if len(matches) >= 8:
                result = self._nn_matcher._verify_fundamental(feat1, feat2, result)

            return result

        except Exception as e:
            print(f"[perception] LightGlue failed, falling back to NN: {e}")
            return self._nn_matcher.match(feat1, feat2)
