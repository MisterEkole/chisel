"""Perception layer — lazy imports to handle optional torch dependency."""


def __getattr__(name):
    if name in ("SuperPointExtractor", "SIFTExtractor", "FeatureData"):
        from .feature_extractor import SuperPointExtractor, SIFTExtractor, FeatureData
        return locals()[name]
    if name in ("NNMatcher", "LightGlueMatcher", "MatchResult"):
        from .feature_matcher import NNMatcher, LightGlueMatcher, MatchResult
        return locals()[name]
    if name in ("MonocularDepthEstimator",):
        from .depth_estimator import MonocularDepthEstimator
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
