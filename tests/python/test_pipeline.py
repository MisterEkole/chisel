"""
tests/python/test_pipeline.py  –  Unit tests for Chisel pipeline.
"""

import sys
import numpy as np
import pytest
from pathlib import Path




try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


@requires_torch
class TestFeatureExtraction:
    """Test feature extraction modules."""

    def test_sift_extraction(self):
        from chisel.perception.feature_extractor import SIFTExtractor
        extractor = SIFTExtractor(max_keypoints=500)

        # Create synthetic test image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Add some structure for SIFT to detect
        image[100:200, 100:200] = 255
        image[300:400, 400:500] = 128

        features = extractor.extract(image)
        assert features.num_features > 0
        assert features.keypoints.shape[1] == 2
        assert features.descriptors.shape[1] == 128
        assert features.image_size == (480, 640)

    def test_superpoint_init(self):
        from chisel.perception.feature_extractor import SuperPointExtractor
        extractor = SuperPointExtractor(max_keypoints=256, device="cpu")
        assert extractor.max_keypoints == 256

    def test_sift_empty_image(self):
        from chisel.perception.feature_extractor import SIFTExtractor
        extractor = SIFTExtractor(max_keypoints=100)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        features = extractor.extract(image)
        assert features.keypoints.shape[1] == 2


@requires_torch
class TestFeatureMatching:
    """Test feature matching."""

    def test_nn_matcher(self):
        from chisel.perception.feature_extractor import SIFTExtractor, FeatureData
        from chisel.perception.feature_matcher import NNMatcher

        extractor = SIFTExtractor(max_keypoints=500)
        matcher = NNMatcher(verify_geometry=False, device="cpu")

        # Two related images
        img1 = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        img2 = img1.copy()
        # Shift slightly to simulate camera motion
        img2 = np.roll(img2, 10, axis=1)

        feat1 = extractor.extract(img1)
        feat2 = extractor.extract(img2)

        if feat1.num_features > 0 and feat2.num_features > 0:
            result = matcher.match(feat1, feat2)
            # Should find some matches between nearly-identical images
            assert result.matches.shape[1] == 2

    def test_empty_match(self):
        from chisel.perception.feature_extractor import FeatureData
        from chisel.perception.feature_matcher import NNMatcher

        matcher = NNMatcher(device="cpu")
        empty = FeatureData(
            keypoints=np.zeros((0, 2)),
            descriptors=np.zeros((0, 128)),
            scores=np.zeros(0),
            image_size=(480, 640),
        )
        result = matcher.match(empty, empty)
        assert result.num_matches == 0


class TestEvalMetrics:
    """Test evaluation metrics."""

    def test_perfect_reconstruction(self):
        from chisel.eval.metrics import evaluate_reconstruction

        # Perfect reconstruction: points match exactly
        points = np.random.randn(1000, 3)
        metrics = evaluate_reconstruction(points, points)

        assert metrics.mean_accuracy > 99.0
        assert metrics.mean_completeness > 99.0
        assert metrics.mean_f1 > 99.0

    def test_noisy_reconstruction(self):
        from chisel.eval.metrics import evaluate_reconstruction

        rng = np.random.RandomState(42)
        gt = rng.randn(1000, 3)
        noise = rng.randn(1000, 3) * 0.01  # 1cm noise
        recon = gt + noise

        metrics = evaluate_reconstruction(recon, gt, thresholds_cm=[2.0, 5.0])
        # With 1cm noise, most should be within 2cm
        assert metrics.accuracy[0] > 60.0
        assert metrics.completeness[0] > 60.0

    def test_empty_reconstruction(self):
        from chisel.eval.metrics import evaluate_reconstruction

        metrics = evaluate_reconstruction(np.zeros((0, 3)), np.random.randn(100, 3))
        assert metrics.mean_f1 == 0.0

    def test_pose_evaluation(self):
        from chisel.eval.metrics import evaluate_poses

        # Perfect poses
        centers = np.random.randn(10, 3)
        metrics = evaluate_poses(centers, centers)
        assert metrics.ate_rmse < 0.01

    def test_pose_with_noise(self):
        from chisel.eval.metrics import evaluate_poses

        gt = np.cumsum(np.random.randn(20, 3) * 0.5, axis=0)
        est = gt + np.random.randn(20, 3) * 0.1

        metrics = evaluate_poses(est, gt)
        assert metrics.ate_rmse > 0
        assert metrics.num_aligned == 20

    def test_depth_evaluation(self):
        from chisel.eval.metrics import evaluate_depth

        gt = np.random.uniform(1, 50, (100, 100)).astype(np.float32)
        pred = gt + np.random.randn(100, 100).astype(np.float32) * 0.5

        metrics = evaluate_depth(pred, gt)
        assert metrics.abs_rel > 0
        assert metrics.delta_1 > 0


@requires_torch
class TestDepthEstimator:
    """Test monocular depth estimation."""

    def test_init(self):
        from chisel.perception.depth_estimator import MonocularDepthEstimator
        estimator = MonocularDepthEstimator(device="cpu")
        assert estimator.min_depth == 0.1

    def test_inference_shape(self):
        from chisel.perception.depth_estimator import MonocularDepthEstimator
        estimator = MonocularDepthEstimator(device="cpu")
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth, conf = estimator.estimate(image)
        assert depth.shape == (480, 640)
        assert conf.shape == (480, 640)
        assert depth.min() >= 0


class TestVisualization:
    """Test visualization utilities."""

    def test_depth_colorize(self):
        from chisel.utils.visualization import visualize_depth
        depth = np.random.uniform(0.5, 10, (100, 200)).astype(np.float32)
        colored = visualize_depth(depth)
        assert colored.shape == (100, 200, 3)

    def test_pointcloud_export(self, tmp_path):
        from chisel.utils.visualization import visualize_pointcloud
        points = np.random.randn(100, 3)
        colors = np.random.rand(100, 3)
        output = str(tmp_path / "test.ply")
        visualize_pointcloud(points, colors, output)
        assert Path(output).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
