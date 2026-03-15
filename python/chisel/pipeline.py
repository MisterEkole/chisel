"""chisel.pipeline — end-to-end 3D reconstruction pipeline."""

import time
import json
import struct
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import cv2

from .data.eth3d_dataset import ETH3DScene
from .perception.feature_extractor import SuperPointExtractor, SIFTExtractor, FeatureData
from .perception.feature_matcher import NNMatcher, LightGlueMatcher, MatchResult
from .perception.depth_estimator import MonocularDepthEstimator
from .eval.metrics import (
    evaluate_reconstruction, evaluate_poses, evaluate_depth,
    align_trajectories_umeyama,
    ReconstructionMetrics, PoseMetrics, DepthMetrics,
)


# ---------------------------------------------------------------------------
# PLY helper
# ---------------------------------------------------------------------------

def _save_ply(path: Path, points: np.ndarray, colors: Optional[np.ndarray] = None):
    """Write a binary-little-endian PLY point cloud."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(points)
    has_color = colors is not None and len(colors) == n
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
    )
    if has_color:
        header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    header += "end_header\n"

    with open(path, "wb") as f:
        f.write(header.encode())
        for i in range(n):
            f.write(struct.pack("<fff", *points[i].astype(np.float32)))
            if has_color:
                rgb = np.clip(colors[i], 0, 255).astype(np.uint8)
                f.write(struct.pack("BBB", int(rgb[2]), int(rgb[1]), int(rgb[0])))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    # Perception
    feature_extractor: str = "sift"       # "sift" | "superpoint"
    feature_matcher: str = "nn"           # "nn"   | "lightglue"
    max_keypoints: int = 4096
    match_ratio_threshold: float = 0.80
    use_depth_prior: bool = False
    superpoint_weights: Optional[str] = None
    lightglue_weights: Optional[str] = None

    # Pair selection
    pair_window: int = 4             # sequential matching window

    # Geometry
    min_match_inliers: int = 30
    ransac_threshold: float = 3.0
    ba_max_iterations: int = 300         # periodic BA iterations
    ba_final_iterations: int = 500       # final global BA gets more budget
    ba_frequency: int = 3                # run BA every N new cameras (lower = more frequent)
    optimizer: str = "ceres"             # "ceres" | "gtsam"

    # Geometry thresholds
    min_triangulation_angle: float = 2.0    # degrees; filters near-degenerate points
    pnp_reprojection_threshold: float = 2.0 # pixels; solvePnPRansac inlier threshold
    min_pnp_inliers: int = 15               # minimum PnP inliers to accept a pose
    max_reproj_for_triangulation: float = 4.0  # pixels; filter triangulated points
    ba_outlier_threshold: float = 3.0       # pixels; cull 3D points above this after each BA

    # Reconstruction
    run_dense: bool = True
    num_depth_samples: int = 128
    num_source_images: int = 5
    depth_confidence_threshold: float = 0.3
    voxel_size: float = 0.01

    # General
    max_image_dim: int = 1600
    device: str = "auto"
    output_dir: str = "./output"
    verbose: bool = True


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class PipelineResults:
    """Results from a full pipeline run."""
    scene_name: str = ""
    num_images: int = 0
    num_registered: int = 0
    num_sparse_points: int = 0
    num_dense_points: int = 0

    time_features: float = 0.0
    time_matching: float = 0.0
    time_sfm: float = 0.0
    time_dense: float = 0.0
    time_total: float = 0.0

    recon_metrics: Optional[ReconstructionMetrics] = None
    pose_metrics: Optional[PoseMetrics] = None
    depth_metrics: Optional[DepthMetrics] = None

    ba_mean_reproj_error: Optional[float] = None  # None = BA not run

    def summary(self) -> str:
        if self.ba_mean_reproj_error is None:
            ba_str = "n/a (need >2 registered cameras)"
        elif isinstance(self.ba_mean_reproj_error, float) and (
                self.ba_mean_reproj_error != self.ba_mean_reproj_error):  # NaN check
            ba_str = "NaN (degenerate reconstruction)"
        else:
            ba_str = f"{self.ba_mean_reproj_error:.3f} px"
        lines = [
            "",
            "╔══════════════════════════════════════════════════╗",
            f"║  Pipeline Results: {self.scene_name:<30}║",
            "╠══════════════════════════════════════════════════╣",
            f"║  Images: {self.num_registered}/{self.num_images} registered",
            f"║  Sparse points: {self.num_sparse_points:,}",
            f"║  Dense points:  {self.num_dense_points:,}",
            "╠──────────────────────────────────────────────────╣",
            f"║  Feature extraction: {self.time_features:.1f}s",
            f"║  Matching:           {self.time_matching:.1f}s",
            f"║  SfM:                {self.time_sfm:.1f}s",
            f"║  Dense recon:        {self.time_dense:.1f}s",
            f"║  Total:              {self.time_total:.1f}s",
            "╠──────────────────────────────────────────────────╣",
            f"║  BA reproj error:    {ba_str}",
        ]
        if self.pose_metrics:
            lines.append(f"║  ATE RMSE:           {self.pose_metrics.ate_rmse:.4f} m")
        if self.recon_metrics:
            lines.append(f"║  Mean F1:            {self.recon_metrics.mean_f1:.2f}%")
        if self.depth_metrics:
            lines.append(f"║  Depth AbsRel:       {self.depth_metrics.abs_rel:.4f}")
        lines.append("╚══════════════════════════════════════════════════╝")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = {
            "scene_name": self.scene_name,
            "num_images": self.num_images,
            "num_registered": self.num_registered,
            "num_sparse_points": self.num_sparse_points,
            "num_dense_points": self.num_dense_points,
            "timing": {
                "features": self.time_features,
                "matching": self.time_matching,
                "sfm": self.time_sfm,
                "dense": self.time_dense,
                "total": self.time_total,
            },
            "ba_mean_reproj_error": self.ba_mean_reproj_error,  # null when BA skipped
        }
        if self.pose_metrics:
            d["pose_ate_rmse"] = self.pose_metrics.ate_rmse
        if self.recon_metrics:
            d["recon_mean_f1"] = self.recon_metrics.mean_f1
        if self.depth_metrics:
            d["depth_abs_rel"] = self.depth_metrics.abs_rel
        return d


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ReconstructionPipeline:
    """
    End-to-end multi-view 3D reconstruction pipeline.

    Phases:
      1. Feature extraction (SIFT / SuperPoint)
      2. Sequential window-based feature matching
      3. Incremental SfM with PnP + track management + bundle adjustment
      4. Evaluation (poses, point cloud, depth)
      5. Dense reconstruction (SGBM stereo fallback; C++ MVS if built)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg = config or PipelineConfig()
        self._setup_modules()
        self._cpp = None
        try:
           
            from chisel import _chisel_cpp as _cpp 
            self._cpp = _cpp
            print("[pipeline] C++ backend loaded")
        except ImportError:
            print("[pipeline] C++ backend not available — Python fallbacks active")

    # ------------------------------------------------------------------
    # Module init
    # ------------------------------------------------------------------

    def _setup_modules(self):
        sp_max_kp = self.cfg.max_keypoints
        self.extractor = (
            SuperPointExtractor(max_keypoints=sp_max_kp,
                                weights_path=self.cfg.superpoint_weights,
                                device=self.cfg.device)
            if self.cfg.feature_extractor == "superpoint"
            else SIFTExtractor(max_keypoints=self.cfg.max_keypoints)
        )
        self.matcher = (
            LightGlueMatcher(weights_path=self.cfg.lightglue_weights,
                             device=self.cfg.device)
            if self.cfg.feature_matcher == "lightglue"
            else NNMatcher(ratio_threshold=self.cfg.match_ratio_threshold,
                           ransac_threshold=self.cfg.ransac_threshold,
                           device=self.cfg.device)
        )
        self.depth_estimator = (
            MonocularDepthEstimator(device=self.cfg.device)
            if self.cfg.use_depth_prior else None
        )
        if self.cfg.verbose:
            print(f"[pipeline] extractor={self.cfg.feature_extractor}  "
                  f"matcher={self.cfg.feature_matcher}  "
                  f"window={self.cfg.pair_window}  dense={self.cfg.run_dense}")

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, scene_path: str) -> PipelineResults:
        """Run the full pipeline on an ETH3D scene directory."""
        t_total = time.time()
        results = PipelineResults()

        print(f"\n{'='*60}")
        print(f"  Scene: {scene_path}")
        print(f"{'='*60}\n")

        scene = ETH3DScene(scene_path, load_images=True,
                           max_image_dim=self.cfg.max_image_dim)
        results.scene_name = scene.name
        results.num_images = len(scene.images)
        image_ids = scene.get_image_ids()

        out = Path(self.cfg.output_dir) / results.scene_name
        out.mkdir(parents=True, exist_ok=True)

        # Phase 1 — Feature extraction
        print("\n── Phase 1: Feature Extraction ──")
        t0 = time.time()
        features: Dict[int, FeatureData] = {}
        for iid in image_ids:
            img = scene.get_image(iid)
            if img is None:
                continue
            features[iid] = self.extractor.extract(img)
            if self.cfg.verbose:
                print(f"  [{iid}] {scene.images[iid].name}: "
                      f"{features[iid].num_features} kps")
        results.time_features = time.time() - t0
        print(f"  {len(features)} images in {results.time_features:.1f}s")

        # Phase 2 — Sequential matching
        print(f"\n── Phase 2: Feature Matching (window={self.cfg.pair_window}) ──")
        t0 = time.time()
        pairs = self._select_pairs(sorted(features.keys()), self.cfg.pair_window)
        matches: Dict[tuple, MatchResult] = {}
        for id1, id2 in pairs:
            mr = self.matcher.match(features[id1], features[id2])
            if mr.num_matches >= self.cfg.min_match_inliers:
                matches[(id1, id2)] = mr
                if self.cfg.verbose:
                    print(f"  {scene.images[id1].name} <-> "
                          f"{scene.images[id2].name}: {mr.num_matches}")
        results.time_matching = time.time() - t0
        print(f"  {len(matches)}/{len(pairs)} pairs in {results.time_matching:.1f}s")

        # Phase 3 — SfM
        print("\n── Phase 3: Structure from Motion ──")
        t0 = time.time()
        sfm = self._run_sfm(scene, features, matches)
        results.num_registered = sfm["num_registered"]
        results.num_sparse_points = sfm["num_points"]
        results.ba_mean_reproj_error = sfm.get("mean_reproj_error", 0.0)
        results.time_sfm = time.time() - t0
        print(f"  {results.num_registered} cameras | "
              f"{results.num_sparse_points:,} points | {results.time_sfm:.1f}s")

        if results.num_sparse_points > 0:
            _save_ply(out / "sparse.ply", sfm["points3d"])
            print(f"  Sparse cloud → {out / 'sparse.ply'}")

        # Phase 4 — Evaluation
        print("\n── Phase 4: Evaluation ──")
        sim3_R, sim3_s, sim3_t = None, None, None  # Umeyama Sim3 to GT frame
        if len(sfm.get("poses", {})) >= 3:
            results.pose_metrics = self._evaluate_poses(scene, sfm)
            if results.pose_metrics:
                print(f"  Poses: {results.pose_metrics.summary()}")
            # Compute Sim3 for point-cloud alignment (used in F1 eval below)
            est_c, gt_c = [], []
            for iid, (R, t) in sfm["poses"].items():
                if iid in scene.images:
                    est_c.append(-R.T @ t)
                    gt_c.append(scene.images[iid].center)
            if len(est_c) >= 3:
                sim3_R, sim3_s, sim3_t = align_trajectories_umeyama(
                    np.array(est_c), np.array(gt_c))

        gt_pts, _ = scene.load_gt_points()
        if len(gt_pts) > 0 and results.num_sparse_points > 0:
            pts = sfm["points3d"]
            # Align reconstruction to GT coordinate frame before computing F1.
            # Without this, every estimated point is in an arbitrary frame and
            # all distances to GT exceed max_dist → F1 = 0% unconditionally.
            if sim3_R is not None:
                pts = (sim3_s * (sim3_R @ pts.T).T) + sim3_t
            results.recon_metrics = evaluate_reconstruction(pts, gt_pts)
            print(results.recon_metrics.summary())

        if self.depth_estimator is not None:
            results.depth_metrics = self._evaluate_depth(scene, sfm)
            if results.depth_metrics:
                print(f"  Depth: {results.depth_metrics.summary()}")

        # Phase 5 — Dense reconstruction
        if self.cfg.run_dense and results.num_registered >= 2:
            print("\n── Phase 5: Dense Reconstruction ──")
            t0 = time.time()
            dense_pts = self._dense_reconstruction(scene, sfm, out)
            results.num_dense_points = len(dense_pts)
            results.time_dense = time.time() - t0
            print(f"  {results.num_dense_points:,} points in {results.time_dense:.1f}s")
            if results.num_dense_points > 0:
                print(f"  Dense cloud → {out / 'dense.ply'}")

        results.time_total = time.time() - t_total
        print(results.summary())
        with open(out / "results.json", "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        return results

    # ------------------------------------------------------------------
    # Sequential pair selection
    # ------------------------------------------------------------------

    def _select_pairs(self, image_ids: List[int],
                      window: int) -> List[Tuple[int, int]]:
        """Exhaustive for ≤60 images, window-based otherwise."""
        pairs = []
        n = len(image_ids)
        if n <= 60:
            # Exhaustive: try every pair
            for i in range(n):
                for j in range(i + 1, n):
                    pairs.append((image_ids[i], image_ids[j]))
        else:
            for i, id1 in enumerate(image_ids):
                for j in range(i + 1, min(i + 1 + window, n)):
                    pairs.append((id1, image_ids[j]))
        return pairs

    # ------------------------------------------------------------------
    # Incremental SfM
    # ------------------------------------------------------------------

    def _run_sfm(self, scene: ETH3DScene,
                 features: Dict[int, FeatureData],
                 matches: Dict[tuple, MatchResult]) -> dict:
        """Incremental SfM: two-view init → PnP registration → triangulation → BA."""


        _empty = {"num_registered": 0, "num_points": 0,
                  "points3d": np.zeros((0, 3)), "poses": {},
                  "track": {}, "points3d_list": [],
                  "mean_reproj_error": 0.0}

        if not matches:
            return _empty

        # track[(img_id, feat_idx)] = index into pts3d
        track: Dict[Tuple[int, int], int] = {}
        pts3d: List[Dict] = []          # [{'xyz': np.ndarray(3)}, ...]
        registered: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        # ---- Two-view init: prefer largest match-graph component --------
        from collections import deque

        # Build adjacency list from matched pairs
        adj: Dict[int, set] = {}
        for (a, b) in matches:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)

        # BFS to find connected components
        def _bfs_component(start: int) -> set:
            seen, q = {start}, deque([start])
            while q:
                n = q.popleft()
                for nb in adj.get(n, []):
                    if nb not in seen:
                        seen.add(nb); q.append(nb)
            return seen

        visited: set = set()
        components: List[set] = []
        for node in adj:
            if node not in visited:
                c = _bfs_component(node)
                components.append(c)
                visited |= c
        largest_cc = max(components, key=len) if components else set()
        if self.cfg.verbose:
            print(f"  Match graph: {len(adj)} images, "
                  f"{len(components)} component(s), "
                  f"largest={len(largest_cc)} images")

        # Sort candidates: pairs inside the largest CC first, then by match count
        def _pair_key(kv):
            (a, b), mr = kv
            in_lcc = int(a in largest_cc and b in largest_cc)
            return (in_lcc, mr.num_matches)

        sorted_pairs = sorted(matches.items(), key=_pair_key, reverse=True)

        id1, id2, init_mr = None, None, None
        R0, t0 = None, None
        pts3d_init_accepted = []
        m_in_accepted = None

        for (cid1, cid2), cand_mr in sorted_pairs[:10]:
            K1c = scene.cameras[scene.images[cid1].camera_id].K
            K2c = scene.cameras[scene.images[cid2].camera_id].K
            cp1 = features[cid1].keypoints[cand_mr.matches[:, 0]]
            cp2 = features[cid2].keypoints[cand_mr.matches[:, 1]]

            Ec, mEc = cv2.findEssentialMat(cp1, cp2, K1c,
                                           method=cv2.RANSAC, prob=0.999,
                                           threshold=1.0)
            if Ec is None:
                continue
            mEc = mEc.ravel().astype(bool)
            if mEc.sum() < 15:
                continue

            cp1i, cp2i = cp1[mEc], cp2[mEc]
            m_ini = cand_mr.matches[mEc]

            _, Rc, tc, _ = cv2.recoverPose(Ec, cp1i, cp2i, K1c)
            tc = tc.ravel()

            P1c = K1c @ np.hstack([np.eye(3), np.zeros((3, 1))])
            P2c = K2c @ np.hstack([Rc, tc.reshape(3, 1)])
            h4c = cv2.triangulatePoints(P1c, P2c, cp1i.T, cp2i.T)
            wc = h4c[3:4].copy(); wc[np.abs(wc) < 1e-10] = 1e-10
            xyz_c = (h4c[:3] / wc).T

            # Per-point triangulation angle; exclude near-degenerate points.
            c1 = np.zeros(3)
            c2 = -Rc.T @ tc
            accepted, angles = [], []
            for m, xyz in zip(m_ini, xyz_c):
                if xyz[2] <= 0 or (Rc @ xyz + tc)[2] <= 0:
                    continue
                r1 = xyz - c1; r1 /= (np.linalg.norm(r1) + 1e-10)
                r2 = xyz - c2; r2 /= (np.linalg.norm(r2) + 1e-10)
                angle = np.degrees(np.arccos(np.clip(r1 @ r2, -1, 1)))
                angles.append(angle)
                if angle >= self.cfg.min_triangulation_angle:
                    accepted.append((m, xyz))

            if len(angles) == 0:
                continue
            median_angle = float(np.median(angles))

            if self.cfg.verbose:
                print(f"  Candidate pair ({cid1},{cid2}): "
                      f"{cand_mr.num_matches} matches, "
                      f"{len(accepted)} valid pts, "
                      f"median angle={median_angle:.2f}°")

            # Require at least 1° median triangulation angle
            if median_angle >= 1.0 and len(accepted) >= 20:
                id1, id2, init_mr = cid1, cid2, cand_mr
                R0, t0 = Rc, tc
                pts3d_init_accepted = accepted
                m_in_accepted = m_ini
                K1 = K1c
                break  # best viable pair found

        if id1 is None:
            print("  [SfM] No viable initialization pair found "
                  "(all pairs have degenerate baseline). "
                  "Try a scene with more camera motion.")
            return _empty

        registered[id1] = (np.eye(3), np.zeros(3))
        registered[id2] = (R0, t0)

        for m, xyz in pts3d_init_accepted:
            idx = len(pts3d)
            pts3d.append({"xyz": xyz.copy()})
            track[(id1, int(m[0]))] = idx
            track[(id2, int(m[1]))] = idx

        if self.cfg.verbose:
            print(f"  Init pair ({id1},{id2}): {len(pts3d)} points triangulated")

        # ---- Multi-pass incremental registration ----
        n_reg = 2
        max_passes = 3
        for sfm_pass in range(max_passes):
            newly_registered = 0
            for nid in sorted(features.keys()):
                if nid in registered:
                    continue

                p2d, p3d, fi_list, pi_list = [], [], [], []
                for rid in list(registered.keys()):
                    pk = (min(nid, rid), max(nid, rid))
                    if pk not in matches:
                        continue
                    mr = matches[pk]
                    for k in range(mr.num_matches):
                        if pk[0] == nid:
                            fi_n, fi_r = int(mr.matches[k, 0]), int(mr.matches[k, 1])
                        else:
                            fi_n, fi_r = int(mr.matches[k, 1]), int(mr.matches[k, 0])
                        if (rid, fi_r) in track:
                            pi = track[(rid, fi_r)]
                            p2d.append(features[nid].keypoints[fi_n])
                            p3d.append(pts3d[pi]["xyz"])
                            fi_list.append(fi_n)
                            pi_list.append(pi)

                if len(p2d) < 6:
                    if self.cfg.verbose and sfm_pass == 0:
                        print(f"  Skip [{nid}] {scene.images[nid].name}: "
                              f"only {len(p2d)} 2D-3D correspondences (need ≥6)")
                    continue

                K_n = scene.cameras[scene.images[nid].camera_id].K
                ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    np.array(p3d, dtype=np.float64),
                    np.array(p2d, dtype=np.float32),
                    K_n, None,
                    iterationsCount=200,
                    reprojectionError=self.cfg.pnp_reprojection_threshold,
                    confidence=0.999, flags=cv2.SOLVEPNP_EPNP,
                )
                if not ok or inliers is None or len(inliers) < self.cfg.min_pnp_inliers:
                    if self.cfg.verbose and sfm_pass == 0:
                        n_inl = len(inliers) if inliers is not None else 0
                        print(f"  Skip [{nid}] {scene.images[nid].name}: "
                              f"PnP failed (ok={ok}, inliers={n_inl})")
                    continue

                R_n, _ = cv2.Rodrigues(rvec)
                t_n = tvec.ravel()
                registered[nid] = (R_n, t_n)
                n_reg += 1
                newly_registered += 1

                for inl in inliers.ravel():
                    track[(nid, fi_list[inl])] = pi_list[inl]

                if self.cfg.verbose:
                    print(f"  [pass {sfm_pass+1}] Registered [{nid}] "
                          f"{scene.images[nid].name}  "
                          f"inliers={len(inliers)}  pts={len(pts3d)}")

                # Triangulate new points with all registered cameras
                for rid in list(registered.keys()):
                    if rid == nid:
                        continue
                    pk = (min(nid, rid), max(nid, rid))
                    if pk not in matches or matches[pk].num_matches < 8:
                        continue

                    R_r, t_r = registered[rid]
                    K_r = scene.cameras[scene.images[rid].camera_id].K
                    Pn = K_n @ np.hstack([R_n, t_n.reshape(3, 1)])
                    Pr = K_r @ np.hstack([R_r, t_r.reshape(3, 1)])

                    mr = matches[pk]
                    new_pairs = []
                    for k in range(mr.num_matches):
                        if pk[0] == nid:
                            fn, fr = int(mr.matches[k, 0]), int(mr.matches[k, 1])
                        else:
                            fn, fr = int(mr.matches[k, 1]), int(mr.matches[k, 0])
                        if (nid, fn) not in track and (rid, fr) not in track:
                            new_pairs.append((fn, fr))

                    if len(new_pairs) < 5:
                        continue

                    an = np.array([features[nid].keypoints[p[0]] for p in new_pairs],
                                  dtype=np.float32)
                    ar = np.array([features[rid].keypoints[p[1]] for p in new_pairs],
                                  dtype=np.float32)
                    h4n = cv2.triangulatePoints(Pn, Pr, an.T, ar.T)
                    wn = h4n[3:4]; wn[np.abs(wn) < 1e-10] = 1e-10
                    new_xyz = (h4n[:3] / wn).T

                    c_n = -R_n.T @ t_n
                    c_r = -R_r.T @ t_r
                    max_e = self.cfg.max_reproj_for_triangulation
                    for (fn, fr), xyz in zip(new_pairs, new_xyz):
                        if not np.all(np.isfinite(xyz)):
                            continue
                        pt_n = R_n @ xyz + t_n
                        pt_r = R_r @ xyz + t_r
                        if pt_n[2] <= 0 or pt_r[2] <= 0:
                            continue
                        r1 = xyz - c_n; r1 /= (np.linalg.norm(r1) + 1e-10)
                        r2 = xyz - c_r; r2 /= (np.linalg.norm(r2) + 1e-10)
                        if np.degrees(np.arccos(np.clip(r1 @ r2, -1, 1))) < self.cfg.min_triangulation_angle:
                            continue
                        # Reprojection error filter
                        e_n = np.linalg.norm((K_n @ pt_n)[:2] / pt_n[2] - features[nid].keypoints[fn])
                        e_r = np.linalg.norm((K_r @ pt_r)[:2] / pt_r[2] - features[rid].keypoints[fr])
                        if e_n > max_e or e_r > max_e:
                            continue
                        pi = len(pts3d)
                        pts3d.append({"xyz": xyz.copy()})
                        track[(nid, fn)] = pi
                        track[(rid, fr)] = pi

                # Periodic BA
                if n_reg % self.cfg.ba_frequency == 0:
                    registered_pre = {k: (R.copy(), t.copy())
                                      for k, (R, t) in registered.items()}
                    registered, pts3d = self._bundle_adjust(
                        registered, pts3d, track, features, scene,
                        max_obs=2000, max_nfev=self.cfg.ba_max_iterations)
                    # Revert any poses that went NaN (Ceres diverged on bad points)
                    for iid in list(registered.keys()):
                        R_out, t_out = registered[iid]
                        if not (np.all(np.isfinite(R_out)) and
                                np.all(np.isfinite(t_out))):
                            registered[iid] = registered_pre[iid]
                    # Mark NaN 3D points as invalid (skipped by _bundle_adjust pre-filter)
                    for pt in pts3d:
                        if not np.all(np.isfinite(pt["xyz"])):
                            pt["xyz"] = np.full(3, np.nan)
                    # Cull high-reproj outliers so they don't poison future BAs
                    n_culled = self._cull_outlier_points(
                        pts3d, track, registered, features, scene,
                        max_reproj=self.cfg.ba_outlier_threshold)
                    if self.cfg.verbose and n_culled:
                        print(f"    [cull] {n_culled} outlier points removed "
                              f"(>{self.cfg.ba_outlier_threshold:.1f}px)")

            # After each pass: retriangulate between ALL registered pairs so
            # that later passes see a denser track.
            print(f"  [pass {sfm_pass+1}] {newly_registered} new cameras registered "
                  f"({n_reg} total). Retriangulating…")
            for (pid1, pid2), mr in matches.items():
                if pid1 not in registered or pid2 not in registered:
                    continue
                R_a, t_a = registered[pid1]
                R_b, t_b = registered[pid2]
                K_a = scene.cameras[scene.images[pid1].camera_id].K
                K_b = scene.cameras[scene.images[pid2].camera_id].K
                Pa = K_a @ np.hstack([R_a, t_a.reshape(3, 1)])
                Pb = K_b @ np.hstack([R_b, t_b.reshape(3, 1)])

                new_pairs = []
                for k in range(mr.num_matches):
                    fa, fb = int(mr.matches[k, 0]), int(mr.matches[k, 1])
                    if (pid1, fa) not in track and (pid2, fb) not in track:
                        new_pairs.append((fa, fb))

                if len(new_pairs) < 5:
                    continue

                an = np.array([features[pid1].keypoints[p[0]] for p in new_pairs],
                              dtype=np.float32)
                bn = np.array([features[pid2].keypoints[p[1]] for p in new_pairs],
                              dtype=np.float32)
                h4 = cv2.triangulatePoints(Pa, Pb, an.T, bn.T)
                w4 = h4[3:4].copy(); w4[np.abs(w4) < 1e-10] = 1e-10
                xyz_all = (h4[:3] / w4).T

                c_a = -R_a.T @ t_a
                c_b = -R_b.T @ t_b
                max_e = self.cfg.max_reproj_for_triangulation
                for (fa, fb), xyz in zip(new_pairs, xyz_all):
                    if not np.all(np.isfinite(xyz)):
                        continue
                    pt_a = R_a @ xyz + t_a
                    pt_b = R_b @ xyz + t_b
                    if pt_a[2] <= 0 or pt_b[2] <= 0:
                        continue
                    r1 = xyz - c_a; r1 /= (np.linalg.norm(r1) + 1e-10)
                    r2 = xyz - c_b; r2 /= (np.linalg.norm(r2) + 1e-10)
                    if np.degrees(np.arccos(np.clip(r1 @ r2, -1, 1))) < self.cfg.min_triangulation_angle:
                        continue
                    # Reprojection error filter
                    e_a = np.linalg.norm((K_a @ pt_a)[:2] / pt_a[2] - features[pid1].keypoints[fa])
                    e_b = np.linalg.norm((K_b @ pt_b)[:2] / pt_b[2] - features[pid2].keypoints[fb])
                    if e_a > max_e or e_b > max_e:
                        continue
                    pi = len(pts3d)
                    pts3d.append({"xyz": xyz.copy()})
                    track[(pid1, fa)] = pi
                    track[(pid2, fb)] = pi

            if self.cfg.verbose:
                print(f"  After pass {sfm_pass+1} retriangulation: "
                      f"{len(pts3d)} total points")

            # Stop early if no progress was made this pass
            if newly_registered == 0:
                break

        # Final BA
        mean_err = None
        if n_reg > 2:
            print("  Running final bundle adjustment…")
            registered_pre = {k: (R.copy(), t.copy())
                              for k, (R, t) in registered.items()}
            registered, pts3d, mean_err = self._bundle_adjust(
                registered, pts3d, track, features, scene,
                max_obs=5000, max_nfev=self.cfg.ba_final_iterations,
                return_cost=True)
            for iid in list(registered.keys()):
                R_out, t_out = registered[iid]
                if not (np.all(np.isfinite(R_out)) and np.all(np.isfinite(t_out))):
                    registered[iid] = registered_pre[iid]
            if mean_err is not None and not np.isfinite(mean_err):
                mean_err = None
            # Final outlier cull with tighter threshold
            n_culled = self._cull_outlier_points(
                pts3d, track, registered, features, scene,
                max_reproj=max(1.5, self.cfg.ba_outlier_threshold / 2.0))
            if self.cfg.verbose and n_culled:
                print(f"  [final cull] {n_culled} outlier points removed")

        # Filter NaN/Inf points before building final array
        valid_xyz = [p["xyz"] for p in pts3d if np.all(np.isfinite(p["xyz"]))]
        arr = np.array(valid_xyz, dtype=np.float64) if valid_xyz else np.zeros((0, 3))

        return {"num_registered": len(registered), "num_points": len(arr),
                "points3d": arr, "poses": registered,
                "track": track, "points3d_list": pts3d,
                "mean_reproj_error": mean_err}

    # ------------------------------------------------------------------
    # Bundle Adjustment
    # ------------------------------------------------------------------

    def _cull_outlier_points(self, pts3d, track, registered, features, scene,
                             max_reproj: float = 3.0) -> int:
        """Remove high-reproj-error 3D points and their track entries. Returns count."""
        pid_obs: Dict[int, list] = {}
        for (iid, fidx), pid in track.items():
            if iid in registered:
                pid_obs.setdefault(pid, []).append((iid, fidx))

        bad_pids: set = set()
        for pid, obs in pid_obs.items():
            xyz = pts3d[pid]["xyz"]
            if not np.all(np.isfinite(xyz)):
                bad_pids.add(pid)
                continue
            errors = []
            for iid, fidx in obs:
                R, t = registered[iid]
                cam = scene.cameras[scene.images[iid].camera_id]
                pt_c = R @ xyz + t
                if pt_c[2] <= 0:
                    continue  # behind camera — skip this observation
                px = cam.fx * pt_c[0] / pt_c[2] + cam.cx
                py = cam.fy * pt_c[1] / pt_c[2] + cam.cy
                kp = features[iid].keypoints[fidx]
                errors.append(float(np.sqrt((px - kp[0])**2 + (py - kp[1])**2)))
            if not errors:
                bad_pids.add(pid)  # no valid observations → cull
            elif float(np.mean(errors)) > max_reproj:
                bad_pids.add(pid)

        if not bad_pids:
            return 0

        # Free track entries so features can be retriangulated
        for k in [k for k, v in track.items() if v in bad_pids]:
            del track[k]
        for pid in bad_pids:
            pts3d[pid]["xyz"] = np.full(3, np.nan)

        return len(bad_pids)

    def _bundle_adjust(self, registered, pts3d, track, features, scene,
                           max_nfev=50, return_cost= False, max_obs=3000, **kwargs):
        """Call C++ Ceres/GTSAM bundle adjustment via pybind11."""
        import numpy as np

        pt_obs: dict[int, list] = {}
        for (iid, fidx), pid in track.items():
            if iid in registered and pid < len(pts3d):
                pt_obs.setdefault(pid, []).append((iid, fidx))

        cpp_scene = self._cpp.Scene()

        # Cameras
        for cid, cam in scene.cameras.items():
            cpp_cam = self._cpp.CameraIntrinsics()
            cpp_cam.id = int(cid)
            cpp_cam.fx, cpp_cam.fy = float(cam.fx), float(cam.fy)
            cpp_cam.cx, cpp_cam.cy = float(cam.cx), float(cam.cy)
            cpp_scene.cameras[int(cid)] = cpp_cam

        # Images
        for iid, (R, t) in registered.items():
            cpp_img = self._cpp.Image()
            cpp_img.id = int(iid)
            cpp_img.camera_id = int(scene.images[iid].camera_id)
            
            pose = self._cpp.CameraPose()
            pose.R = np.asarray(R, dtype=np.float64)
            pose.translation = np.asarray(t, dtype=np.float64)
            cpp_img.pose = pose
            cpp_img.pose_valid = True
            
            img_kps_arr = np.asarray(features[iid].keypoints, dtype=np.float64)
            #cpp_img.keypoints = [self._cpp.Keypoint(kp[:2]) for kp in img_kps_arr]
            kp_list = []
            for kp_xy in img_kps_arr:
                kp = self._cpp.Keypoint()
                kp.xy = kp_xy[:2].copy() # Explicitly set the Eigen vector
                kp_list.append(kp)
            cpp_img.keypoints = kp_list
            cpp_scene.images[int(iid)] = cpp_img

        # 3D points
        for pid, pt in enumerate(pts3d):
            obs = pt_obs.get(pid, [])
            if not obs: continue
            if not np.all(np.isfinite(pt["xyz"])): continue  # skip NaN/Inf — causes Ceres Iterations:-2
            
            cpp_pt = self._cpp.Point3D()
            cpp_pt.id = int(pid)
            cpp_pt.xyz = np.asarray(pt["xyz"], dtype=np.float64)
            
            cpp_pt.track = [self._cpp.TrackElement(int(iid), int(fidx)) for iid, fidx in obs]
            cpp_scene.points3d[int(pid)] = cpp_pt

        opt_choice = getattr(self.cfg, "optimizer", "ceres").lower()
        
        if opt_choice == "gtsam":
            print(f"  [BA] Dispatching to GTSAM Factor Graph (max_iter={max_nfev})")
            fg_cfg = self._cpp.FactorGraphConfig()
            fg_cfg.max_iterations = int(max_nfev)
            fg_cfg.verbose = bool(self.cfg.verbose)
            fg_cfg.pixel_sigma = 1.2
            fg_cfg.prior_rot_sigma = 1e-4
            
            report = self._cpp.optimize_full_graph(cpp_scene, fg_cfg)
            final_error = report.final_error
        else:
            print(f"  [BA] Dispatching to Ceres Solver (max_iter={max_nfev})")
            ba_cfg = self._cpp.BundleAdjustmentConfig()
            ba_cfg.max_iterations = int(max_nfev)
            ba_cfg.verbose = bool(self.cfg.verbose)
            ba_cfg.huber_loss_scale = 1.0
            ba_cfg.fix_intrinsics = True
            
            report = self._cpp.run_bundle_adjustment(cpp_scene, ba_cfg)
            final_error = report.mean_reproj_error

       
        for iid in list(registered.keys()):
            if int(iid) in cpp_scene.images:
                img_out = cpp_scene.images[int(iid)]
                registered[iid] = (img_out.pose.R.copy(), img_out.pose.translation.copy())

        for pid in range(len(pts3d)):
            if int(pid) in cpp_scene.points3d:
                pts3d[pid]["xyz"] = cpp_scene.points3d[int(pid)].xyz.copy()

        if return_cost:
            return registered, pts3d, float(final_error)
        return registered, pts3d
    # ------------------------------------------------------------------
    # Dense reconstruction
    # ------------------------------------------------------------------

    def _dense_reconstruction(self, scene: ETH3DScene, sfm: dict,
                               out: Path) -> np.ndarray:
        """Try C++ MVS; fall back to OpenCV SGBM stereo."""
        if self._cpp is not None:
            try:
                return self._dense_cpp(scene, sfm, out)
            except Exception as e:
                print(f"  [dense] C++ error ({e}) — using SGBM fallback")
        return self._dense_sgbm(scene, sfm, out)

    def _dense_cpp(self, scene: "ETH3DScene", sfm: dict, out: Path) -> np.ndarray:
        """C++ plane-sweep MVS → depth fusion → point cloud.

        Requires the pybind11 module to expose DenseStereoConfig,
        compute_all_depth_maps, FusionConfig, and fuse_scene_depth_maps.
        Raises AttributeError if the module was built without dense support.
        """
        cpp = self._cpp

        for sym in ("DenseStereoConfig", "FusionConfig",
                    "compute_all_depth_maps", "fuse_scene_depth_maps"):
            if not hasattr(cpp, sym):
                raise AttributeError(f"C++ module missing '{sym}' — rebuild with dense support")

        registered = sfm.get("poses", {})
        if len(registered) < 2:
            return np.zeros((0, 3))

        cpp_scene = cpp.Scene()

        # Cameras
        for cid, cam in scene.cameras.items():
            cpp_cam = cpp.CameraIntrinsics()
            cpp_cam.id = int(cid)
            cpp_cam.fx, cpp_cam.fy = float(cam.fx), float(cam.fy)
            cpp_cam.cx, cpp_cam.cy = float(cam.cx), float(cam.cy)
            cpp_scene.cameras[int(cid)] = cpp_cam

        # Images
        for iid, (R, t) in registered.items():
            cpp_img = cpp.Image()
            cpp_img.id = int(iid)
            cpp_img.camera_id = int(scene.images[iid].camera_id)
            cpp_img.name = scene.images[iid].name

            pose = cpp.CameraPose()
            pose.R = np.asarray(R, dtype=np.float64)
            pose.translation = np.asarray(t, dtype=np.float64)
            cpp_img.pose = pose
            cpp_img.pose_valid = True

            img_bgr = scene.get_image(iid)
            if img_bgr is not None:
                cpp_img.set_image(np.ascontiguousarray(img_bgr, dtype=np.uint8))

            cpp_scene.images[int(iid)] = cpp_img

        # Depth range from sparse cloud
        sparse_pts = sfm.get("points3d", np.zeros((0, 3)))
        min_depth, max_depth = 0.1, 100.0
        if len(sparse_pts) > 5:
            depths_all = []
            for iid, (R, t) in registered.items():
                pts_c = (R @ sparse_pts.T).T + t
                valid_z = pts_c[:, 2]
                valid_z = valid_z[valid_z > 0]
                if len(valid_z) > 0:
                    depths_all.extend(valid_z.tolist())
            if len(depths_all) > 10:
                min_depth = float(max(0.01, np.percentile(depths_all, 2)))
                max_depth = float(np.percentile(depths_all, 98) * 1.5)

        if self.cfg.verbose:
            print(f"  [dense] depth range: [{min_depth:.3f}, {max_depth:.3f}] "
                  f"({len(registered)} cameras)")

        # Dense stereo config
        dense_cfg = cpp.DenseStereoConfig()
        dense_cfg.min_depth          = float(min_depth)
        dense_cfg.max_depth          = float(max_depth)
        dense_cfg.num_depth_samples  = int(self.cfg.num_depth_samples)
        dense_cfg.num_source_images  = int(self.cfg.num_source_images)
        dense_cfg.confidence_threshold = float(self.cfg.depth_confidence_threshold)
        dense_cfg.filter_by_consistency = False

        # Compute depth maps
        cpp.compute_all_depth_maps(cpp_scene, dense_cfg)

        n_depth_maps = len(cpp_scene.depth_maps)
        if self.cfg.verbose:
            print(f"  [dense] {n_depth_maps} depth maps computed")

        if n_depth_maps == 0:
            print("  [dense] No depth maps produced — images may not be loaded")
            return np.zeros((0, 3))

        # Fuse depth maps
        fusion_cfg = cpp.FusionConfig()
        fusion_cfg.min_confidence = float(self.cfg.depth_confidence_threshold)
        fusion_cfg.depth_max      = float(max_depth)
        fusion_cfg.subsample      = 2

        pts_arr, col_arr = cpp.fuse_scene_depth_maps(cpp_scene, fusion_cfg)
        pts = np.asarray(pts_arr)
        cols = np.asarray(col_arr)

        if len(pts) == 0:
            print("  [dense] Fusion produced no points")
            return np.zeros((0, 3))

        # Cap at 2M points
        if len(pts) > 2_000_000:
            sel = np.random.default_rng(0).choice(len(pts), 2_000_000, replace=False)
            pts, cols = pts[sel], cols[sel]

        cols_u8 = np.clip(cols * 255, 0, 255).astype(np.uint8)
        _save_ply(out / "dense.ply", pts, cols_u8)
        return pts

    def _dense_sgbm(self, scene: ETH3DScene, sfm: dict, out: Path) -> np.ndarray:
        """OpenCV SGBM dense stereo: select pair, rectify, match, reproject."""
        

        registered = sfm.get("poses", {})
        if len(registered) < 2:
            return np.zeros((0, 3))

        # Select consecutive pair with baseline in [5%, 30%] of scene span.
        ids = sorted(registered.keys())
        centers = {i: -registered[i][0].T @ registered[i][1] for i in ids}

        consec = [(ids[k], ids[k+1]) for k in range(len(ids)-1)]
        pair_bl = [(a, b, float(np.linalg.norm(centers[a] - centers[b])))
                   for a, b in consec]
        pair_bl.sort(key=lambda x: x[2])

        all_bls = [x[2] for x in pair_bl]
        scene_span = max(all_bls) if all_bls else 1.0
        lo, hi = scene_span * 0.05, scene_span * 0.30
        good = [(a, b, bl) for a, b, bl in pair_bl if lo <= bl <= hi]
        if not good:
            good = pair_bl  # fallback: use all consecutive pairs
        mid = good[len(good) // 2]
        id1, id2, best_bl = mid

        img1 = scene.get_image(id1)
        img2 = scene.get_image(id2)
        if img1 is None or img2 is None:
            print("  [dense] Images unavailable for chosen stereo pair")
            return np.zeros((0, 3))

        R1, t1 = registered[id1]
        R2, t2 = registered[id2]
        K1 = scene.cameras[scene.images[id1].camera_id].K
        K2 = scene.cameras[scene.images[id2].camera_id].K

        R_rel = R2 @ R1.T
        t_rel = t2 - R_rel @ t1
        h, w = img1.shape[:2]

        try:
            R1r, R2r, P1, P2, Q, _, _ = cv2.stereoRectify(
                K1, np.zeros(5, np.float64),
                K2, np.zeros(5, np.float64),
                (w, h), R_rel, t_rel.reshape(3, 1), alpha=0, flags=0)
            m1x, m1y = cv2.initUndistortRectifyMap(
                K1, np.zeros(5), R1r, P1, (w, h), cv2.CV_32F)
            m2x, m2y = cv2.initUndistortRectifyMap(
                K2, np.zeros(5), R2r, P2, (w, h), cv2.CV_32F)
            rect1 = cv2.remap(img1, m1x, m1y, cv2.INTER_LINEAR)
            rect2 = cv2.remap(img2, m2x, m2y, cv2.INTER_LINEAR)
        except cv2.error as e:
            print(f"  [dense] Stereo rectification failed: {e}")
            return np.zeros((0, 3))

        g1 = cv2.cvtColor(rect1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(rect2, cv2.COLOR_BGR2GRAY)

        bs = 9
        sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=128, blockSize=bs,
            P1=8*3*bs*bs, P2=32*3*bs*bs,
            disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        disp = sgbm.compute(g1, g2).astype(np.float32) / 16.0
        valid = disp > 0

        pts_map = cv2.reprojectImageTo3D(disp, Q)
        pts = pts_map[valid].reshape(-1, 3)
        cols = rect1[valid].reshape(-1, 3)

        # Estimate scene depth range from sparse points visible in id1
        sparse_pts = sfm.get("points3d", np.zeros((0, 3)))
        if len(sparse_pts) > 0:
            R1, t1_ = registered[id1]
            pts_c = (R1 @ sparse_pts.T).T + t1_
            valid_z = pts_c[:, 2]
            valid_z = valid_z[valid_z > 0]
            if len(valid_z) > 5:
                min_d = float(np.percentile(valid_z, 5))
                max_d = float(np.percentile(valid_z, 95)) * 2.0
            else:
                max_d = max(best_bl * 200.0, 50.0)
                min_d = best_bl * 0.5
        else:
            max_d = max(best_bl * 200.0, 50.0)
            min_d = 0.0
        keep = (np.all(np.isfinite(pts), axis=1)
                & (pts[:, 2] > min_d) & (pts[:, 2] < max_d))
        pts, cols = pts[keep], cols[keep]

        if len(pts) == 0:
            print("  [dense] SGBM produced no valid 3D points")
            return np.zeros((0, 3))

        if len(pts) > 500_000:
            sel = np.random.default_rng(0).choice(len(pts), 500_000, replace=False)
            pts, cols = pts[sel], cols[sel]

        _save_ply(out / "dense.ply", pts, cols)
        return pts

    # ------------------------------------------------------------------
    # Depth evaluation
    # ------------------------------------------------------------------

    def _evaluate_depth(self, scene: ETH3DScene, sfm: dict) -> Optional[DepthMetrics]:
        """Evaluate monocular depth vs ETH3D GT (16-bit PNG, mm). Returns None if unavailable."""
       

        depth_dir = scene.path / "depth"
        if not depth_dir.exists() or self.depth_estimator is None:
            return None

        registered = sfm.get("poses", {})
        if not registered:
            return None

        all_m = []
        for iid in sorted(registered.keys())[:10]:
            info = scene.images.get(iid)
            if info is None:
                continue
            dp = depth_dir / (Path(info.name).stem + ".png")
            if not dp.exists():
                continue
            gt_raw = cv2.imread(str(dp), cv2.IMREAD_ANYDEPTH)
            if gt_raw is None:
                continue
            gt = gt_raw.astype(np.float32) / 1000.0   # mm → m

            img = scene.get_image(iid)
            if img is None:
                continue

            pred_inv = self.depth_estimator.estimate(img)
            vmask = pred_inv > 1e-4
            pred = np.zeros_like(pred_inv)
            pred[vmask] = 1.0 / pred_inv[vmask]

            if gt.shape != pred.shape:
                gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]),
                                interpolation=cv2.INTER_NEAREST)

            m = evaluate_depth(pred, gt)
            if m.num_valid > 0:
                all_m.append(m)

        if not all_m:
            return None

        def _avg(attr):
            return float(np.mean([getattr(m, attr) for m in all_m]))

        return DepthMetrics(
            abs_rel=_avg("abs_rel"), sq_rel=_avg("sq_rel"),
            rmse=_avg("rmse"),       rmse_log=_avg("rmse_log"),
            delta_1=_avg("delta_1"), delta_2=_avg("delta_2"),
            delta_3=_avg("delta_3"), num_valid=len(all_m),
        )

    # ------------------------------------------------------------------
    # Pose evaluation
    # ------------------------------------------------------------------

    def _evaluate_poses(self, scene: ETH3DScene, sfm: dict) -> Optional[PoseMetrics]:
        """Compare estimated camera centers to GT via Umeyama alignment."""
        poses = sfm.get("poses", {})
        est, gt = [], []
        for iid, (R, t) in poses.items():
            if iid in scene.images:
                est.append(-R.T @ t)
                gt.append(scene.images[iid].center)
        if len(est) < 3:
            return None
        return evaluate_poses(np.array(est), np.array(gt))
