#!/usr/bin/env python3
"""
visualize.py  –  Interactive 3D viewer: reconstruction vs ground truth.

Requires: pip install open3d

Usage:
    python scripts/visualize.py \
        --recon ./output/courtyard/sparse.ply \
        --gt    ./data/eth3d/training/courtyard/dslr_calibration_undistorted/points3D.txt

Controls in the viewer window:
    Left-drag     rotate
    Right-drag    pan
    Scroll        zoom
    R             reset view
    Q / Esc       quit
"""

import argparse
import struct
import numpy as np
from pathlib import Path


# ─── Loaders ────────────────────────────────────────────────────────────────

def load_ply(path: str):
    """
    Load a PLY file. Returns (points, colors).
    points: (N, 3) float64
    colors: (N, 3) float64 in [0, 1], or None
    Handles binary_little_endian and ASCII.
    """
    _type_map = {
        "float": ("f", 4), "float32": ("f", 4),
        "double": ("d", 8), "float64": ("d", 8),
        "uchar": ("B", 1), "uint8": ("B", 1),
        "char": ("b", 1), "int8": ("b", 1),
        "short": ("h", 2), "int16": ("h", 2),
        "ushort": ("H", 2), "uint16": ("H", 2),
        "int": ("i", 4), "int32": ("i", 4),
        "uint": ("I", 4), "uint32": ("I", 4),
    }
    with open(path, "rb") as f:
        fmt, n_verts, props, in_vert = "ascii", 0, [], False
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line == "end_header":
                break
            if "binary_little_endian" in line:
                fmt = "binary_le"
            elif "binary_big_endian" in line:
                fmt = "binary_be"
            elif line.startswith("element vertex"):
                n_verts = int(line.split()[-1]); in_vert = True
            elif line.startswith("element") and not line.startswith("element vertex"):
                in_vert = False
            elif line.startswith("property") and in_vert:
                p = line.split()
                props.append((p[2], p[1]))

        if n_verts == 0:
            return np.zeros((0, 3)), None

        names  = [p[0] for p in props]
        fmts   = [_type_map.get(p[1], ("f", 4)) for p in props]
        endian = "<" if fmt in ("ascii", "binary_le") else ">"
        sfmt   = endian + "".join(c for c, _ in fmts)
        rsize  = sum(s for _, s in fmts)

        def _get(row, name):
            return row[names.index(name)] if name in names else None

        if fmt == "ascii":
            rows = [f.readline().decode().split() for _ in range(n_verts)]
            pts    = np.array([[float(r[names.index("x")]),
                                float(r[names.index("y")]),
                                float(r[names.index("z")])] for r in rows])
            has_rgb = all(c in names for c in ("red", "green", "blue"))
            if has_rgb:
                ri, gi, bi = names.index("red"), names.index("green"), names.index("blue")
                cols = np.array([[float(r[ri]), float(r[gi]), float(r[bi])] for r in rows]) / 255.0
            else:
                cols = None
        else:
            raw  = f.read(n_verts * rsize)
            rows = list(struct.iter_unpack(sfmt, raw))
            xi, yi, zi = names.index("x"), names.index("y"), names.index("z")
            pts = np.array([[r[xi], r[yi], r[zi]] for r in rows], dtype=np.float64)
            has_rgb = all(c in names for c in ("red", "green", "blue"))
            if has_rgb:
                ri, gi, bi = names.index("red"), names.index("green"), names.index("blue")
                cols = np.array([[r[ri], r[gi], r[bi]] for r in rows], dtype=np.float64) / 255.0
            else:
                cols = None

    return pts, cols


def load_colmap_points3d(path: str):
    """Load XYZ + RGB from a COLMAP points3D.txt file."""
    pts, cols = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) >= 7:
                pts.append([float(p[1]), float(p[2]), float(p[3])])
                cols.append([float(p[4]) / 255.0,
                             float(p[5]) / 255.0,
                             float(p[6]) / 255.0])
    if not pts:
        return np.zeros((0, 3)), None
    return np.array(pts, dtype=np.float64), np.array(cols, dtype=np.float64)


def load_pointcloud(path: str):
    """Auto-detect format and load (points, colors)."""
    p = Path(path)
    if p.suffix.lower() == ".ply":
        return load_ply(path)
    return load_colmap_points3d(path)


# ─── Viewer ─────────────────────────────────────────────────────────────────

def view_plotly(recon_pts, recon_cols, gt_pts, gt_cols,
                scene_name: str, subsample: int = 50_000):
    """
    Interactive 3D scatter in the browser via plotly.
    Full rotate / pan / zoom. No OpenGL dependency.
    """
    import plotly.graph_objects as go

    def _sub(pts, cols, n):
        if len(pts) > n:
            idx = np.random.default_rng(0).choice(len(pts), n, replace=False)
            return pts[idx], (cols[idx] if cols is not None else None)
        return pts, cols

    r, rc = _sub(recon_pts, recon_cols, subsample)
    g, gc = _sub(gt_pts,    gt_cols,    subsample)

    def _rgb_strings(pts, cols, default):
        if cols is not None and len(cols) == len(pts):
            return [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})"
                    for c in cols]
        return default

    traces = [
        go.Scatter3d(
            x=g[:, 0], y=g[:, 1], z=g[:, 2],
            mode="markers",
            name=f"GT ({len(gt_pts):,} pts)",
            marker=dict(
                size=1.2,
                color=_rgb_strings(g, gc, "lightgrey"),
                opacity=0.4,
            ),
        ),
        go.Scatter3d(
            x=r[:, 0], y=r[:, 1], z=r[:, 2],
            mode="markers",
            name=f"Recon ({len(recon_pts):,} pts)",
            marker=dict(
                size=2.0,
                color=_rgb_strings(r, rc, "steelblue"),
                opacity=0.9,
            ),
        ),
    ]

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Chisel — {scene_name}  |  reconstruction vs ground truth",
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            bgcolor="rgb(20,20,20)",
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
        ),
        paper_bgcolor="rgb(30,30,30)",
        font_color="white",
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    print(f"\n  Recon : {len(recon_pts):,} pts  |  GT : {len(gt_pts):,} pts")
    print("  Opening browser — left-drag=rotate  right-drag=pan  scroll=zoom\n")
    fig.show()   # opens default browser


def view_matplotlib(recon_pts, gt_pts, scene_name: str, subsample: int = 15_000):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    def _sub(pts, n):
        if len(pts) > n:
            idx = np.random.default_rng(0).choice(len(pts), n, replace=False)
            return pts[idx]
        return pts

    r = _sub(recon_pts, subsample)
    g = _sub(gt_pts,    subsample)
    c = g.mean(axis=0)
    r, g = r - c, g - c

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")
    ax.scatter(g[:, 0], g[:, 1], g[:, 2], s=0.3, c="lightgrey", alpha=0.3, label=f"GT ({len(gt_pts):,})")
    ax.scatter(r[:, 0], r[:, 1], r[:, 2], s=1.0, c="steelblue", alpha=0.8, label=f"Recon ({len(recon_pts):,})")
    ax.set_title(f"{scene_name} — reconstruction vs GT")
    ax.legend(markerscale=6)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout()
    print("  [viz] Showing matplotlib 3D plot (install open3d for a better viewer)")
    plt.show()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Interactive 3D viewer: reconstruction vs ground truth")
    parser.add_argument("--recon", required=True,
                        help="Reconstructed point cloud (.ply)")
    parser.add_argument("--gt",    required=True,
                        help="Ground-truth point cloud (.ply or COLMAP points3D.txt)")
    parser.add_argument("--scene",    default="scene",
                        help="Scene name shown in window title")
    parser.add_argument("--fallback", action="store_true",
                        help="Force matplotlib fallback (skip open3d)")
    args = parser.parse_args()

    print(f"Loading reconstruction: {args.recon}")
    recon_pts, recon_cols = load_pointcloud(args.recon)
    print(f"Loading ground truth:   {args.gt}")
    gt_pts, gt_cols = load_pointcloud(args.gt)

    if len(recon_pts) == 0:
        print("ERROR: reconstruction point cloud is empty"); return
    if len(gt_pts) == 0:
        print("ERROR: ground-truth point cloud is empty"); return

    if not args.fallback:
        try:
            view_plotly(recon_pts, recon_cols, gt_pts, gt_cols, args.scene)
            return
        except ImportError:
            print("  [viz] plotly not found — using matplotlib fallback")
            print("        pip install plotly  for a better interactive viewer")

    view_matplotlib(recon_pts, gt_pts, args.scene)


if __name__ == "__main__":
    main()
