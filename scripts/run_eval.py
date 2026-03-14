#!/usr/bin/env python3
"""
run_eval.py  –  Evaluate reconstruction results against ETH3D ground truth.

Usage:
    # Multi-scene (auto-finds sparse.ply in each output subdirectory)
    python scripts/run_eval.py --results ./output --dataset ./data/eth3d

    # Single scene
    python scripts/run_eval.py \
        --recon-ply ./output/courtyard/sparse.ply \
        --gt-ply ./data/eth3d/training/courtyard/dslr_calibration_undistorted/points3D.txt

    # With visualisation (saves <output>/eval_<scene>.png)
    python scripts/run_eval.py --results ./output --dataset ./data/eth3d --plot
"""

import argparse
import struct
import sys
import json
import numpy as np
from pathlib import Path

from chisel.eval.metrics import evaluate_reconstruction, ReconstructionMetrics
from chisel.data.eth3d_dataset import ETH3DDataset


# ─── Point-cloud loaders ────────────────────────────────────────────────────

def load_ply_points(path: str) -> np.ndarray:
    """
    Load XYZ points from a PLY file.
    Handles both ASCII and binary_little_endian formats.
    """
    path = Path(path)
    with open(path, "rb") as f:
        # --- parse header (always ASCII) ---
        fmt = "ascii"
        n_verts = 0
        props = []          # ordered list of (name, type)
        in_vertex = False

        while True:
            raw = f.readline()
            line = raw.decode("ascii", errors="ignore").strip()
            if line == "end_header":
                break
            if line.startswith("format"):
                if "binary_little_endian" in line:
                    fmt = "binary_le"
                elif "binary_big_endian" in line:
                    fmt = "binary_be"
            elif line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
                in_vertex = True
            elif line.startswith("element") and not line.startswith("element vertex"):
                in_vertex = False           # stop collecting props after vertex block
            elif line.startswith("property") and in_vertex:
                parts = line.split()        # e.g. ["property", "float", "x"]
                props.append((parts[2], parts[1]))

        if n_verts == 0:
            return np.zeros((0, 3))

        # build struct format string for one vertex
        _ply_type = {
            "float": ("f", 4), "float32": ("f", 4),
            "double": ("d", 8), "float64": ("d", 8),
            "int": ("i", 4), "uint": ("I", 4),
            "short": ("h", 2), "ushort": ("H", 2),
            "char": ("b", 1), "uchar": ("B", 1),
            "int8": ("b", 1), "uint8": ("B", 1),
            "int16": ("h", 2), "uint16": ("H", 2),
            "int32": ("i", 4), "uint32": ("I", 4),
        }
        prop_fmts = []
        prop_names = []
        for name, ptype in props:
            code, size = _ply_type.get(ptype, ("f", 4))
            prop_fmts.append((code, size))
            prop_names.append(name)

        endian = "<" if fmt in ("ascii", "binary_le") else ">"
        struct_fmt = endian + "".join(c for c, _ in prop_fmts)
        row_size   = sum(s for _, s in prop_fmts)

        # find x/y/z indices
        try:
            xi, yi, zi = prop_names.index("x"), prop_names.index("y"), prop_names.index("z")
        except ValueError:
            return np.zeros((0, 3))

        # --- read data ---
        if fmt == "ascii":
            points = []
            for _ in range(n_verts):
                parts = f.readline().decode("ascii").split()
                vals  = [float(p) for p in parts]
                points.append([vals[xi], vals[yi], vals[zi]])
            return np.array(points, dtype=np.float64)
        else:
            raw_data = f.read(n_verts * row_size)
            rows = struct.iter_unpack(struct_fmt, raw_data)
            points = [[r[xi], r[yi], r[zi]] for r in rows]
            return np.array(points, dtype=np.float64)


def load_colmap_points3d(path: str) -> np.ndarray:
    """
    Load XYZ from a COLMAP points3D.txt file.
    Format per line: POINT3D_ID X Y Z R G B ERROR TRACK[]
    """
    points = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                points.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(points, dtype=np.float64) if points else np.zeros((0, 3))


def load_points(path: str) -> np.ndarray:
    """Auto-detect PLY vs COLMAP txt and load XYZ."""
    p = Path(path)
    if p.suffix.lower() == ".ply":
        return load_ply_points(path)
    return load_colmap_points3d(path)


# ─── Visualisation ──────────────────────────────────────────────────────────

def plot_comparison(recon: np.ndarray, gt: np.ndarray,
                    scene_name: str, output_path: str = None,
                    subsample: int = 20_000):
    """
    Interactive 3-panel figure (top / front / side) comparing
    reconstruction (blue) against ground-truth (grey).
    Pass output_path to also save a PNG; omit to only show the window.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [eval] matplotlib not installed — skipping plot")
        return

    def _sub(pts, n):
        if len(pts) > n:
            idx = np.random.default_rng(0).choice(len(pts), n, replace=False)
            return pts[idx]
        return pts

    r = _sub(recon, subsample)
    g = _sub(gt,    subsample)

    gt_center = g.mean(axis=0)
    r = r - gt_center
    g = g - gt_center

    views = [
        ("Top  (X–Z)",  0, 2, "X", "Z"),
        ("Front (X–Y)", 0, 1, "X", "Y"),
        ("Side  (Z–Y)", 2, 1, "Z", "Y"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Reconstruction vs GT — {scene_name}\n"
                 f"Recon: {len(recon):,} pts (blue)   GT: {len(gt):,} pts (grey)",
                 fontsize=11)

    for ax, (title, a, b, xl, yl) in zip(axes, views):
        ax.scatter(g[:, a], g[:, b], s=0.3, c="lightgrey", alpha=0.4, label="GT")
        ax.scatter(r[:, a], r[:, b], s=0.5, c="steelblue", alpha=0.7, label="Recon")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_aspect("equal")
        ax.legend(markerscale=8, fontsize=7)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  [eval] Plot saved → {output_path}")

    plt.show()   # opens interactive window; blocks until closed


# ─── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_scene(recon_points: np.ndarray, gt_points: np.ndarray,
                   scene_name: str) -> dict:
    metrics = evaluate_reconstruction(recon_points, gt_points,
                                      thresholds_cm=[1.0, 2.0, 5.0, 10.0])
    print(f"\n── {scene_name} ──")
    print(metrics.summary())
    return {
        "scene":             scene_name,
        "mean_accuracy":     metrics.mean_accuracy,
        "mean_completeness": metrics.mean_completeness,
        "mean_f1":           metrics.mean_f1,
        "num_recon_points":  len(recon_points),
        "num_gt_points":     len(gt_points),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate reconstruction results")
    parser.add_argument("--results",   type=str, help="Results directory (output/)")
    parser.add_argument("--dataset",   type=str, help="ETH3D dataset root")
    parser.add_argument("--recon-ply", type=str, help="Single reconstruction PLY")
    parser.add_argument("--gt-ply",    type=str,
                        help="Single GT file (.ply or COLMAP points3D.txt)")
    parser.add_argument("--output",    type=str, default="./eval_results.json")
    parser.add_argument("--plot",      action="store_true",
                        help="Show interactive comparison plot (requires matplotlib)")
    parser.add_argument("--save-plot", action="store_true",
                        help="Also save plot as PNG alongside results")
    args = parser.parse_args()

    all_results = []

    if args.recon_ply and args.gt_ply:
        # ── single-scene mode ──────────────────────────────────────────────
        recon_pts = load_points(args.recon_ply)
        gt_pts    = load_points(args.gt_ply)
        print(f"  Recon: {len(recon_pts):,} pts   GT: {len(gt_pts):,} pts")
        result = evaluate_scene(recon_pts, gt_pts, "single")
        all_results.append(result)

        if args.plot:
            out_png = str(Path(args.output).with_suffix(".png")) if args.save_plot else None
            plot_comparison(recon_pts, gt_pts, "single", out_png)

    elif args.results and args.dataset:
        # ── multi-scene mode ───────────────────────────────────────────────
        results_dir = Path(args.results)
        dataset     = ETH3DDataset(args.dataset, split="training")

        for scene_name in dataset.list_scenes():
            # Pipeline saves sparse.ply (and optionally dense.ply)
            recon_ply = results_dir / scene_name / "sparse.ply"
            if not recon_ply.exists():
                print(f"  Skipping {scene_name}: sparse.ply not found in {results_dir / scene_name}")
                continue

            scene = dataset.get_scene(scene_name, load_images=False)
            gt_pts, _ = scene.load_gt_points()
            if len(gt_pts) == 0:
                print(f"  Skipping {scene_name}: no GT points")
                continue

            recon_pts = load_ply_points(str(recon_ply))
            print(f"  {scene_name}: recon {len(recon_pts):,} pts, GT {len(gt_pts):,} pts")
            result = evaluate_scene(recon_pts, gt_pts, scene_name)
            all_results.append(result)

            if args.plot:
                out_png = str(results_dir / scene_name / f"eval_{scene_name}.png") if args.save_plot else None
                plot_comparison(recon_pts, gt_pts, scene_name, out_png)

        if all_results:
            print(f"\n{'='*55}")
            print(f"  Overall ({len(all_results)} scenes)")
            print(f"{'='*55}")
            print(f"  Mean Accuracy:     {np.mean([r['mean_accuracy']     for r in all_results]):.2f}%")
            print(f"  Mean Completeness: {np.mean([r['mean_completeness'] for r in all_results]):.2f}%")
            print(f"  Mean F1:           {np.mean([r['mean_f1']           for r in all_results]):.2f}%")

    else:
        parser.print_help()
        sys.exit(1)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
