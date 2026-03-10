#!/usr/bin/env python3
"""
run_eval.py  –  Evaluate reconstruction results against ETH3D ground truth.

Usage:
    python scripts/run_eval.py --results ./output --dataset /data/eth3d
    python scripts/run_eval.py --recon-ply ./output/courtyard/reconstruction.ply \
                               --gt-ply /data/eth3d/training/courtyard/points3D.txt
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path


from chisel.eval.metrics import (
    evaluate_reconstruction, evaluate_poses, evaluate_depth,
    ReconstructionMetrics,
)
from chisel.data.eth3d_dataset import ETH3DDataset


def load_ply_points(path: str) -> np.ndarray:
    """Load points from PLY file."""
    points = []
    in_data = False
    with open(path) as f:
        for line in f:
            if line.strip() == "end_header":
                in_data = True
                continue
            if not in_data:
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(points) if points else np.zeros((0, 3))


def evaluate_scene(recon_points: np.ndarray, gt_points: np.ndarray,
                   scene_name: str) -> dict:
    """Evaluate a single scene."""
    metrics = evaluate_reconstruction(
        recon_points, gt_points,
        thresholds_cm=[1.0, 2.0, 5.0, 10.0],
    )
    print(f"\n── {scene_name} ──")
    print(metrics.summary())
    return {
        "scene": scene_name,
        "mean_accuracy": metrics.mean_accuracy,
        "mean_completeness": metrics.mean_completeness,
        "mean_f1": metrics.mean_f1,
        "num_recon_points": len(recon_points),
        "num_gt_points": len(gt_points),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate reconstruction results")
    parser.add_argument("--results", type=str, help="Results directory")
    parser.add_argument("--dataset", type=str, help="ETH3D dataset root")
    parser.add_argument("--recon-ply", type=str, help="Single reconstruction PLY")
    parser.add_argument("--gt-ply", type=str, help="Single GT point cloud")
    parser.add_argument("--output", type=str, default="./eval_results.json")
    args = parser.parse_args()

    all_results = []

    if args.recon_ply and args.gt_ply:
        # Single-scene evaluation
        recon_pts = load_ply_points(args.recon_ply)
        gt_pts = load_ply_points(args.gt_ply)
        result = evaluate_scene(recon_pts, gt_pts, "single")
        all_results.append(result)

    elif args.results and args.dataset:
        # Multi-scene evaluation
        results_dir = Path(args.results)
        dataset = ETH3DDataset(args.dataset, split="training")

        for scene_name in dataset.list_scenes():
            recon_ply = results_dir / scene_name / "reconstruction.ply"
            if not recon_ply.exists():
                print(f"Skipping {scene_name}: no reconstruction.ply found")
                continue

            scene = dataset.get_scene(scene_name, load_images=False)
            gt_pts, _ = scene.load_gt_points()
            if len(gt_pts) == 0:
                print(f"Skipping {scene_name}: no GT points")
                continue

            recon_pts = load_ply_points(str(recon_ply))
            result = evaluate_scene(recon_pts, gt_pts, scene_name)
            all_results.append(result)

        # Summary
        if all_results:
            print(f"\n{'='*60}")
            print(f"  Overall Summary ({len(all_results)} scenes)")
            print(f"{'='*60}")
            mean_f1 = np.mean([r["mean_f1"] for r in all_results])
            mean_acc = np.mean([r["mean_accuracy"] for r in all_results])
            mean_comp = np.mean([r["mean_completeness"] for r in all_results])
            print(f"  Mean Accuracy:     {mean_acc:.2f}%")
            print(f"  Mean Completeness: {mean_comp:.2f}%")
            print(f"  Mean F1:           {mean_f1:.2f}%")

    else:
        parser.print_help()
        sys.exit(1)

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
