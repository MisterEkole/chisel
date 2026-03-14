"""chisel.eval.metrics — ETH3D reconstruction, pose, and depth evaluation metrics."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.spatial import cKDTree


@dataclass
class ReconstructionMetrics:
    """ETH3D-style reconstruction quality metrics."""
    # Point cloud metrics at multiple thresholds (in cm)
    thresholds: List[float]

    accuracy: List[float]       # % of recon pts within τ of GT
    completeness: List[float]   # % of GT pts within τ of recon
    f1_score: List[float]       # harmonic mean

    # Summary
    mean_accuracy: float = 0.0
    mean_completeness: float = 0.0
    mean_f1: float = 0.0

    # Raw distances
    accuracy_distances: Optional[np.ndarray] = None
    completeness_distances: Optional[np.ndarray] = None

    def summary(self) -> str:
        lines = ["╔══════════════════════════════════════════════╗",
                 "║     ETH3D Reconstruction Evaluation          ║",
                 "╠══════════════════════════════════════════════╣"]
        lines.append(f"║ {'Thresh (cm)':>12} {'Acc %':>8} {'Comp %':>8} {'F1 %':>8} ║")
        lines.append("╠══════════════════════════════════════════════╣")
        for i, tau in enumerate(self.thresholds):
            lines.append(
                f"║ {tau:>12.1f} {self.accuracy[i]:>8.2f} "
                f"{self.completeness[i]:>8.2f} {self.f1_score[i]:>8.2f} ║")
        lines.append("╠══════════════════════════════════════════════╣")
        lines.append(
            f"║ {'MEAN':>12} {self.mean_accuracy:>8.2f} "
            f"{self.mean_completeness:>8.2f} {self.mean_f1:>8.2f} ║")
        lines.append("╚══════════════════════════════════════════════╝")
        return "\n".join(lines)


def evaluate_reconstruction(
    recon_points: np.ndarray,      # (N, 3)
    gt_points: np.ndarray,         # (M, 3)
    thresholds_cm: List[float] = [1.0, 2.0, 5.0, 10.0],
    max_dist: float = 1.0,
) -> ReconstructionMetrics:
    """Evaluate reconstruction vs GT using ETH3D accuracy/completeness/F1 metrics."""
    if len(recon_points) == 0 or len(gt_points) == 0:
        return ReconstructionMetrics(
            thresholds=thresholds_cm,
            accuracy=[0.0] * len(thresholds_cm),
            completeness=[0.0] * len(thresholds_cm),
            f1_score=[0.0] * len(thresholds_cm),
        )

    # Build KD-trees
    gt_tree = cKDTree(gt_points)
    recon_tree = cKDTree(recon_points)

    # Accuracy: for each recon point, find nearest GT point
    acc_dists, _ = gt_tree.query(recon_points)
    acc_dists = acc_dists[acc_dists < max_dist]  # filter outliers

    # Completeness: for each GT point, find nearest recon point
    comp_dists, _ = recon_tree.query(gt_points)
    comp_dists = comp_dists[comp_dists < max_dist]

    # Compute metrics at each threshold
    thresholds_m = [t / 100.0 for t in thresholds_cm]
    accuracy_list = []
    completeness_list = []
    f1_list = []

    for tau in thresholds_m:
        acc = float(np.mean(acc_dists < tau)) * 100.0 if len(acc_dists) > 0 else 0.0
        comp = float(np.mean(comp_dists < tau)) * 100.0 if len(comp_dists) > 0 else 0.0

        if acc + comp > 0:
            f1 = 2 * acc * comp / (acc + comp)
        else:
            f1 = 0.0

        accuracy_list.append(acc)
        completeness_list.append(comp)
        f1_list.append(f1)

    result = ReconstructionMetrics(
        thresholds=thresholds_cm,
        accuracy=accuracy_list,
        completeness=completeness_list,
        f1_score=f1_list,
        mean_accuracy=float(np.mean(accuracy_list)),
        mean_completeness=float(np.mean(completeness_list)),
        mean_f1=float(np.mean(f1_list)),
        accuracy_distances=acc_dists,
        completeness_distances=comp_dists,
    )

    return result


# ─── Pose evaluation ────────────────────────

@dataclass
class PoseMetrics:
    """Camera pose evaluation metrics."""
    ate_rmse: float = 0.0       # Absolute Trajectory Error (RMSE, meters)
    ate_mean: float = 0.0
    ate_median: float = 0.0
    rpe_trans: float = 0.0      # Relative Pose Error (translation, meters)
    rpe_rot: float = 0.0        # Relative Pose Error (rotation, degrees)
    num_aligned: int = 0

    def summary(self) -> str:
        return (
            f"ATE RMSE: {self.ate_rmse:.4f} m | "
            f"ATE Mean: {self.ate_mean:.4f} m | "
            f"ATE Median: {self.ate_median:.4f} m | "
            f"RPE Trans: {self.rpe_trans:.4f} m | "
            f"RPE Rot: {self.rpe_rot:.2f}°"
        )


def align_trajectories_umeyama(
    estimated: np.ndarray,  # (N, 3)
    ground_truth: np.ndarray,  # (N, 3)
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Umeyama Sim3 alignment: minimizes ||GT - (s·R·Est + t)||². Returns (R, s, t)."""
    valid = (np.all(np.isfinite(estimated), axis=1) &
             np.all(np.isfinite(ground_truth), axis=1))
    if valid.sum() < 3:
        return np.eye(3), 1.0, np.zeros(3)
    estimated   = estimated[valid]
    ground_truth = ground_truth[valid]

    n = estimated.shape[0]

    mu_est = estimated.mean(axis=0)
    mu_gt = ground_truth.mean(axis=0)

    est_centered = estimated - mu_est
    gt_centered = ground_truth - mu_gt

    sigma_est = np.sum(est_centered ** 2) / n
    if sigma_est < 1e-10:
        return np.eye(3), 1.0, mu_gt - mu_est

    cov = gt_centered.T @ est_centered / n

    U, S, Vt = np.linalg.svd(cov)

    d = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[2, 2] = -1

    R = U @ d @ Vt
    s = np.trace(np.diag(S) @ d) / sigma_est
    t = mu_gt - s * R @ mu_est

    return R, s, t


def evaluate_poses(
    estimated_centers: np.ndarray,  # (N, 3) camera centers
    gt_centers: np.ndarray,         # (N, 3) ground truth centers
    estimated_rotations: Optional[np.ndarray] = None,  # (N, 3, 3)
    gt_rotations: Optional[np.ndarray] = None,         # (N, 3, 3)
) -> PoseMetrics:
    """Evaluate camera poses (ATE, RPE) with Umeyama alignment."""
    if len(estimated_centers) < 3 or len(gt_centers) < 3:
        return PoseMetrics()

    n = min(len(estimated_centers), len(gt_centers))
    est = estimated_centers[:n]
    gt = gt_centers[:n]

    R, s, t = align_trajectories_umeyama(est, gt)
    est_aligned = (s * (R @ est.T).T) + t

    # ATE
    errors = np.linalg.norm(est_aligned - gt, axis=1)

    metrics = PoseMetrics(
        ate_rmse=float(np.sqrt(np.mean(errors ** 2))),
        ate_mean=float(np.mean(errors)),
        ate_median=float(np.median(errors)),
        num_aligned=n,
    )

    # RPE
    if estimated_rotations is not None and gt_rotations is not None:
        n_rot = min(len(estimated_rotations), len(gt_rotations))
        rpe_rot_errors = []
        rpe_trans_errors = []

        for i in range(n_rot - 1):
            dR_est = estimated_rotations[i+1] @ estimated_rotations[i].T
            dR_gt = gt_rotations[i+1] @ gt_rotations[i].T

            dR_err = dR_gt @ dR_est.T
            angle = np.arccos(np.clip((np.trace(dR_err) - 1) / 2, -1, 1))
            rpe_rot_errors.append(np.degrees(angle))

            dt_est = est_aligned[i+1] - est_aligned[i]
            dt_gt = gt[i+1] - gt[i]
            rpe_trans_errors.append(np.linalg.norm(dt_est - dt_gt))

        metrics.rpe_rot = float(np.mean(rpe_rot_errors))
        metrics.rpe_trans = float(np.mean(rpe_trans_errors))

    return metrics


# ─── Depth map evaluation ───────────────────

@dataclass
class DepthMetrics:
    """Monocular/MVS depth evaluation metrics."""
    abs_rel: float = 0.0    # |d* - d| / d
    sq_rel: float = 0.0     # (d* - d)^2 / d
    rmse: float = 0.0
    rmse_log: float = 0.0
    delta_1: float = 0.0    # % with max(d*/d, d/d*) < 1.25
    delta_2: float = 0.0    # < 1.25^2
    delta_3: float = 0.0    # < 1.25^3
    num_valid: int = 0

    def summary(self) -> str:
        return (
            f"AbsRel: {self.abs_rel:.4f} | SqRel: {self.sq_rel:.4f} | "
            f"RMSE: {self.rmse:.4f} | "
            f"δ<1.25: {self.delta_1:.3f} | δ<1.25²: {self.delta_2:.3f} | "
            f"δ<1.25³: {self.delta_3:.3f}"
        )


def evaluate_depth(
    pred_depth: np.ndarray,    # (H, W)
    gt_depth: np.ndarray,      # (H, W)
    min_depth: float = 0.1,
    max_depth: float = 80.0,
) -> DepthMetrics:
    """Evaluate predicted depth map against ground truth."""

    # Valid mask
    mask = (gt_depth > min_depth) & (gt_depth < max_depth) & \
           (pred_depth > min_depth) & (pred_depth < max_depth)

    pred = pred_depth[mask]
    gt = gt_depth[mask]

    if len(pred) == 0:
        return DepthMetrics()

    # Standard depth metrics
    thresh = np.maximum(pred / gt, gt / pred)
    abs_diff = np.abs(pred - gt)

    metrics = DepthMetrics(
        abs_rel=float(np.mean(abs_diff / gt)),
        sq_rel=float(np.mean((abs_diff ** 2) / gt)),
        rmse=float(np.sqrt(np.mean(abs_diff ** 2))),
        rmse_log=float(np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2))),
        delta_1=float(np.mean(thresh < 1.25)),
        delta_2=float(np.mean(thresh < 1.25 ** 2)),
        delta_3=float(np.mean(thresh < 1.25 ** 3)),
        num_valid=int(mask.sum()),
    )

    return metrics
