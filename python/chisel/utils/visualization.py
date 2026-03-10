"""
chisel.utils.visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Visualization utilities for features, matches, depth maps, and point clouds.
"""

import numpy as np
import cv2
from typing import Optional, List, Tuple
from pathlib import Path


def visualize_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    kpts1: np.ndarray,
    kpts2: np.ndarray,
    matches: np.ndarray,
    max_display: int = 100,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """
    Draw feature matches between two images.

    Args:
        img1, img2: BGR images
        kpts1, kpts2: (N, 2) keypoint coordinates
        matches: (M, 2) index pairs
        max_display: max matches to draw
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Side-by-side canvas
    H = max(h1, h2)
    canvas = np.zeros((H, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:] = img2

    # Subsample if too many
    if len(matches) > max_display:
        indices = np.random.choice(len(matches), max_display, replace=False)
        matches = matches[indices]

    # Draw matches
    for i, (idx1, idx2) in enumerate(matches):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        pt1 = tuple(kpts1[idx1].astype(int))
        pt2 = (int(kpts2[idx2][0] + w1), int(kpts2[idx2][1]))

        cv2.circle(canvas, pt1, 3, color, -1)
        cv2.circle(canvas, pt2, 3, color, -1)
        cv2.line(canvas, pt1, pt2, color, 1)

    if output_path:
        cv2.imwrite(output_path, canvas)

    return canvas


def visualize_depth(
    depth: np.ndarray,
    min_depth: float = 0.1,
    max_depth: float = 50.0,
    colormap: int = cv2.COLORMAP_MAGMA,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """
    Colorize a depth map for visualization.

    Args:
        depth: (H, W) depth map
        min_depth, max_depth: depth range for normalization
    """
    # Clip and normalize
    d = np.clip(depth, min_depth, max_depth)
    d_norm = (d - min_depth) / (max_depth - min_depth)

    # Invert so near=bright
    d_norm = 1.0 - d_norm

    # Apply colormap
    d_uint8 = (d_norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(d_uint8, colormap)

    # Mark invalid pixels
    invalid = depth <= 0
    colored[invalid] = [0, 0, 0]

    if output_path:
        cv2.imwrite(output_path, colored)

    return colored


def visualize_pointcloud(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    output_path: str = "pointcloud.ply",
):
    """
    Save point cloud as PLY file for viewing in MeshLab/CloudCompare.

    Args:
        points: (N, 3) XYZ coordinates
        colors: (N, 3) RGB colors in [0, 1] or [0, 255]
    """
    n = len(points)
    has_color = colors is not None and len(colors) == n

    with open(output_path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")

        for i in range(n):
            line = f"{points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f}"
            if has_color:
                c = colors[i]
                if c.max() <= 1.0:
                    c = c * 255
                line += f" {int(c[0])} {int(c[1])} {int(c[2])}"
            f.write(line + "\n")

    print(f"[viz] Saved {n} points to {output_path}")
