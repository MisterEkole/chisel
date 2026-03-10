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
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--device", type=str, default="auto")

    # Output
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--verbose", action="store_true", default=True)

    # Config file (overrides CLI args)
    parser.add_argument("--config", type=str, help="YAML config file")

    return parser.parse_args()


def main():
    args = parse_args()

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
