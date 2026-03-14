#!/usr/bin/env python3
"""
run_pipeline.py  –  Run the Chisel pipeline on an ETH3D scene.

Usage:
    python scripts/run_pipeline.py --scene /path/to/eth3d/courtyard
    python scripts/run_pipeline.py --dataset /path/to/eth3d --scene-name courtyard
    python scripts/run_pipeline.py --config configs/default.yaml --scene /path/to/scene

Options:
    --scene         Path to a single scene directory
    --dataset       Path to ETH3D root directory
    --scene-name    Name of scene within dataset
    --config        YAML config file (optional)
    --output        Output directory (default: ./output)
    --extractor     Feature extractor: sift, superpoint (default: sift)
    --matcher       Feature matcher: nn, lightglue (default: nn)
    --optimizer     BA optimizer: ceres, gtsam (default: ceres)
    --max-dim       Max image dimension (default: 1600)
    --no-dense      Skip dense reconstruction
    --eval-only     Only run evaluation (expects existing results)
"""

import argparse
import sys
import json
from pathlib import Path



from chisel.pipeline import ReconstructionPipeline, PipelineConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chisel: Multi-view 3D Reconstruction Pipeline")

    # Input
    parser.add_argument("--scene", type=str,
                        help="Path to scene directory")
    parser.add_argument("--dataset", type=str,
                        help="Path to ETH3D dataset root")
    parser.add_argument("--scene-name", type=str, default="courtyard",
                        help="Scene name within dataset")

    # Pipeline options
    parser.add_argument("--extractor", type=str, default="sift",
                        choices=["sift", "superpoint"])
    parser.add_argument("--matcher", type=str, default="nn",
                        choices=["nn", "lightglue"])
    parser.add_argument("--optimizer", type=str, default="ceres",
                        choices=["ceres", "gtsam"])
    parser.add_argument("--max-dim", type=int, default=1600)
    parser.add_argument("--max-keypoints", type=int, default=4096)
    parser.add_argument("--min-tri-angle", type=float, default=2.0,
                        help="Min triangulation angle in degrees (default: 2.0)")
    parser.add_argument("--pnp-threshold", type=float, default=2.0,
                        help="PnP RANSAC reprojection threshold in pixels (default: 2.0)")
    parser.add_argument("--min-pnp-inliers", type=int, default=15,
                        help="Min PnP inliers to accept a pose (default: 15)")
    parser.add_argument("--max-reproj-tri", type=float, default=4.0,
                        help="Max reprojection error for triangulated points in pixels (default: 4.0)")
    parser.add_argument("--ratio-threshold", type=float, default=0.75,
                        help="NN ratio test threshold (default: 0.75, SIFT+NN only)")
    parser.add_argument("--ba-outlier-threshold", type=float, default=3.0,
                        help="Cull 3D points above this reproj error after each BA (default: 3.0px)")
    parser.add_argument("--ba-max-iterations", type=int, default=300,
                        help="Max iterations for periodic bundle adjustment (default: 300)")
    parser.add_argument("--ba-final-iterations", type=int, default=500,
                        help="Max iterations for final global bundle adjustment (default: 500)")
    parser.add_argument("--ba-frequency", type=int, default=3,
                        help="Run periodic BA every N registered cameras (default: 3)")
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--weights-dir", type=str, default=None,
                        help="Directory containing superpoint_v1.pth and "
                             "superpoint_lightglue.pth (default: weights/)")

    # Output
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--verbose", action="store_true", default=True)

    # Config file (overrides CLI args)
    parser.add_argument("--config", type=str, help="YAML config file")

    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve weights paths
    weights_dir = Path(args.weights_dir) if args.weights_dir else Path("weights")
    sp_weights = str(weights_dir / "superpoint_v1.pth")
    lg_weights = str(weights_dir / "superpoint_lightglue.pth")

    # Build config
    config = PipelineConfig(
        feature_extractor=args.extractor,
        feature_matcher=args.matcher,
        optimizer=args.optimizer,
        max_keypoints=args.max_keypoints,
        max_image_dim=args.max_dim,
        run_dense=not args.no_dense,
        device=args.device,
        output_dir=args.output,
        verbose=args.verbose,
        superpoint_weights=sp_weights if Path(sp_weights).exists() else None,
        lightglue_weights=lg_weights if Path(lg_weights).exists() else None,
        min_triangulation_angle=args.min_tri_angle,
        pnp_reprojection_threshold=args.pnp_threshold,
        min_pnp_inliers=args.min_pnp_inliers,
        max_reproj_for_triangulation=args.max_reproj_tri,
        match_ratio_threshold=args.ratio_threshold,
        ba_max_iterations=args.ba_max_iterations,
        ba_final_iterations=args.ba_final_iterations,
        ba_frequency=args.ba_frequency,
        ba_outlier_threshold=args.ba_outlier_threshold,
    )

    # Load YAML config if provided
    if args.config:
        import yaml
        with open(args.config) as f:
            yaml_cfg = yaml.safe_load(f)
        for key, value in yaml_cfg.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Determine scene path
    if args.scene:
        scene_path = args.scene
    elif args.dataset:
        scene_path = str(Path(args.dataset) / "training" / args.scene_name)
    else:
        print("Error: provide either --scene or --dataset + --scene-name")
        sys.exit(1)

    if not Path(scene_path).exists():
        print(f"Error: scene path does not exist: {scene_path}")
        print(f"\nTo download ETH3D data, run:")
        print(f"  python scripts/download_eth3d.py --output {args.dataset or '/data/eth3d'}")
        sys.exit(1)

    # Run pipeline
    pipeline = ReconstructionPipeline(config)
    results = pipeline.run(scene_path)

    # Print summary
    print(results.summary())

    # Save results
    output_path = Path(args.output) / results.scene_name / "results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
