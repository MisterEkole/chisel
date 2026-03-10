"""
chisel.pipeline
~~~~~~~~~~~~~~~~
End-to-end 3D reconstruction pipeline.

Orchestrates: perception → geometry → reconstruction → evaluation

Usage:
    from chisel.pipeline import ReconstructionPipeline

    pipeline = ReconstructionPipeline(config)
    results = pipeline.run("/path/to/eth3d/courtyard")
"""

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

    # Pair selection
    pair_window: int = 5                  # sequential matching window

    # Geometry
    min_match_inliers: int = 30
    ransac_threshold: float = 3.0
    ba_max_iterations: int = 100
    ba_frequency: int = 5
    optimizer: str = "ceres"             # "ceres" | "gtsam"

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
        ba_str = (f"{self.ba_mean_reproj_error:.3f} px"
                  if self.ba_mean_reproj_error is not None
                  else "n/a (C++ not built)")
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
        self.extractor = (
            SuperPointExtractor(max_keypoints=self.cfg.max_keypoints,
                                device=self.cfg.device)
            if self.cfg.feature_extractor == "superpoint"
            else SIFTExtractor(max_keypoints=self.cfg.max_keypoints)
        )
        self.matcher = (
            LightGlueMatcher(device=self.cfg.device)
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
        if len(sfm.get("poses", {})) >= 3:
            results.pose_metrics = self._evaluate_poses(scene, sfm)
            if results.pose_metrics:
                print(f"  Poses: {results.pose_metrics.summary()}")

        gt_pts, _ = scene.load_gt_points()
        if len(gt_pts) > 0 and results.num_sparse_points > 0:
            results.recon_metrics = evaluate_reconstruction(sfm["points3d"], gt_pts)
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
        """
        Window-based sequential pair selection.
        Images assumed captured in sorted ID order.
        Complexity: O(n * window) vs O(n^2) exhaustive.
        """
        pairs = []
        for i, id1 in enumerate(image_ids):
            for j in range(i + 1, min(i + 1 + window, len(image_ids))):
                pairs.append((id1, image_ids[j]))
        return pairs

    # ------------------------------------------------------------------
    # Incremental SfM
    # ------------------------------------------------------------------

    def _run_sfm(self, scene: ETH3DScene,
                 features: Dict[int, FeatureData],
                 matches: Dict[tuple, MatchResult]) -> dict:
        """
        Incremental SfM:
          - Two-view init via essential matrix + triangulation
          - PnP RANSAC registration for each new image
          - Feature track maintenance
          - New-point triangulation after each registration
          - Periodic + final bundle adjustment
        """
        import cv2

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

        # ---- Two-view initialization ------------------------------------
        # Sort candidate pairs by match count, then pick the one with the
        # best triangulation baseline. A pair with many matches but a tiny
        # baseline (near-duplicate frames) produces a near-empty track and
        # makes every subsequent PnP fail.
        sorted_pairs = sorted(matches.items(),
                              key=lambda kv: kv[1].num_matches, reverse=True)

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
            if mEc.sum() < 30:
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

            # Compute median triangulation angle to check baseline quality
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

        # ---- Incremental registration ------------------------------------
        n_reg = 2
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
                if self.cfg.verbose:
                    print(f"  Skip [{nid}] {scene.images[nid].name}: "
                          f"only {len(p2d)} 2D-3D correspondences (need ≥6)")
                continue

            K_n = scene.cameras[scene.images[nid].camera_id].K
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                np.array(p3d, dtype=np.float64),
                np.array(p2d, dtype=np.float32),
                K_n, None,
                iterationsCount=200, reprojectionError=4.0,
                confidence=0.999, flags=cv2.SOLVEPNP_EPNP,
            )
            if not ok or inliers is None or len(inliers) < 6:
                if self.cfg.verbose:
                    n_inl = len(inliers) if inliers is not None else 0
                    print(f"  Skip [{nid}] {scene.images[nid].name}: "
                          f"PnP failed (ok={ok}, inliers={n_inl})")
                continue

            R_n, _ = cv2.Rodrigues(rvec)
            t_n = tvec.ravel()
            registered[nid] = (R_n, t_n)
            n_reg += 1

            for inl in inliers.ravel():
                track[(nid, fi_list[inl])] = pi_list[inl]

            if self.cfg.verbose:
                print(f"  Registered [{nid}] {scene.images[nid].name}  "
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

                for (fn, fr), xyz in zip(new_pairs, new_xyz):
                    if (R_n @ xyz + t_n)[2] <= 0:
                        continue
                    if (R_r @ xyz + t_r)[2] <= 0:
                        continue
                    pi = len(pts3d)
                    pts3d.append({"xyz": xyz.copy()})
                    track[(nid, fn)] = pi
                    track[(rid, fr)] = pi

            # Periodic BA
            if n_reg % self.cfg.ba_frequency == 0:
                registered, pts3d = self._bundle_adjust(
                    registered, pts3d, track, features, scene,
                    max_obs=2000, max_nfev=self.cfg.ba_max_iterations)

        # Final BA
        mean_err = None
        if n_reg > 2:
            print("  Running final bundle adjustment…")
            registered, pts3d, mean_err = self._bundle_adjust(
                registered, pts3d, track, features, scene,
                max_obs=5000, max_nfev=self.cfg.ba_max_iterations,
                return_cost=True)

        arr = (np.array([p["xyz"] for p in pts3d], dtype=np.float64)
               if pts3d else np.zeros((0, 3)))

        return {"num_registered": len(registered), "num_points": len(arr),
                "points3d": arr, "poses": registered,
                "track": track, "points3d_list": pts3d,
                "mean_reproj_error": mean_err}

    # ------------------------------------------------------------------
    # Bundle Adjustment
    # ------------------------------------------------------------------

    def _bundle_adjust(self, registered, pts3d, track, features, scene,
                           max_nfev=50, return_cost= False, max_obs=3000, **kwargs):
        """Call C++ Ceres/GTSAM bundle adjustment via pybind11."""
        import numpy as np

        pt_obs: dict[int, list] = {}
        for (iid, fidx), pid in track.items():
            if iid in registered and pid < len(pts3d):
                pt_obs.setdefault(pid, []).append((iid, fidx))

        cpp_scene = self._cpp.Scene()

        # Sync Cameras (Intrinsics)
        for cid, cam in scene.cameras.items():
            cpp_cam = self._cpp.CameraIntrinsics()
            cpp_cam.id = int(cid)
            cpp_cam.fx, cpp_cam.fy = float(cam.fx), float(cam.fy)
            cpp_cam.cx, cpp_cam.cy = float(cam.cx), float(cam.cy)
            cpp_scene.cameras[int(cid)] = cpp_cam

        # Sync Images (Poses and Keypoints)
        for iid, (R, t) in registered.items():
            cpp_img = self._cpp.Image()
            cpp_img.id = int(iid)
            cpp_img.camera_id = int(scene.images[iid].camera_id)
            
            # Use the C++ CameraPose property 'R' for automatic Quaternion conversion
            pose = self._cpp.CameraPose()
            pose.R = np.asarray(R, dtype=np.float64)
            pose.translation = np.asarray(t, dtype=np.float64)
            cpp_img.pose = pose
            cpp_img.pose_valid = True
            
            # Convert keypoints to C++ list
            img_kps_arr = np.asarray(features[iid].keypoints, dtype=np.float64)
            #cpp_img.keypoints = [self._cpp.Keypoint(kp[:2]) for kp in img_kps_arr]
            kp_list = []
            for kp_xy in img_kps_arr:
                kp = self._cpp.Keypoint()
                kp.xy = kp_xy[:2].copy() # Explicitly set the Eigen vector
                kp_list.append(kp)
            cpp_img.keypoints = kp_list
            cpp_scene.images[int(iid)] = cpp_img

        # Sync 3D Points and Feature Tracks
        for pid, pt in enumerate(pts3d):
            obs = pt_obs.get(pid, [])
            if not obs: continue
            
            cpp_pt = self._cpp.Point3D()
            cpp_pt.id = int(pid)
            cpp_pt.xyz = np.asarray(pt["xyz"], dtype=np.float64)
            
            # Connect the "Elastic Bands" (Tracks)
            cpp_pt.track = [self._cpp.TrackElement(int(iid), int(fidx)) for iid, fidx in obs]
            cpp_scene.points3d[int(pid)] = cpp_pt

        # DISPATCH: Choose the Optimizer based on CLI flag
        # Defaults to ceres if not specified
        opt_choice = getattr(self.cfg, "optimizer", "ceres").lower()
        
        if opt_choice == "gtsam":
            print(f"  [BA] Dispatching to GTSAM Factor Graph (max_iter={max_nfev})")
            fg_cfg = self._cpp.FactorGraphConfig()
            fg_cfg.max_iterations = int(max_nfev)
            fg_cfg.verbose = bool(self.cfg.verbose)
            # Tuning sigmas for ViT feature noise
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

    def _dense_cpp(self, scene, sfm, out):
        raise NotImplementedError(
            "C++ dense MVS requires 'cmake --build' to complete first")

    def _dense_sgbm(self, scene: ETH3DScene, sfm: dict, out: Path) -> np.ndarray:
        """
        Dense stereo with OpenCV SGBM.
        Selects the registered pair with the largest baseline,
        rectifies, matches, reprojects, and saves dense.ply.
        """
        import cv2

        registered = sfm.get("poses", {})
        if len(registered) < 2:
            return np.zeros((0, 3))

        # Find pair with largest camera-center baseline
        ids = sorted(registered.keys())
        id1, id2, best_bl = ids[0], ids[1], 0.0
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                Ri, ti = registered[ids[i]]
                Rj, tj = registered[ids[j]]
                bl = float(np.linalg.norm((-Ri.T @ ti) - (-Rj.T @ tj)))
                if bl > best_bl:
                    best_bl, id1, id2 = bl, ids[i], ids[j]

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

        max_d = max(best_bl * 200.0, 50.0)
        keep = (np.all(np.isfinite(pts), axis=1)
                & (pts[:, 2] > 0) & (pts[:, 2] < max_d))
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
        """
        Evaluate monocular depth against ETH3D GT depth maps.
        GT depth expected at <scene>/depth/<stem>.png  (16-bit PNG, mm).
        Silently returns None if no GT depth directory exists.
        """
        import cv2

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
