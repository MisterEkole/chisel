#pragma once
// src/geometry/pose_estimation.h
#include "core/types.h"
#include <vector>

namespace chisel {
namespace geometry {

struct PnPResult {
    CameraPose pose;
    std::vector<bool> inlier_mask;
    int num_inliers = 0;
    bool success = false;
};

// Solve Perspective-n-Point (camera pose from 2D-3D correspondences)
// Uses EPnP + RANSAC via OpenCV
PnPResult solve_pnp_ransac(
    const std::vector<Vec2>& points2d,
    const std::vector<Vec3>& points3d,
    const CameraIntrinsics& camera,
    double ransac_thresh = 8.0,     // pixel reprojection threshold
    int max_iterations = 10000,
    double confidence = 0.999);

// Refine pose with iterative PnP (uses inliers only)
bool refine_pose_pnp(
    CameraPose& pose,
    const std::vector<Vec2>& points2d,
    const std::vector<Vec3>& points3d,
    const CameraIntrinsics& camera,
    const std::vector<bool>& inlier_mask);

// ─── Incremental SfM controller ─────────────
struct SfMConfig {
    int min_num_inliers = 30;
    double init_min_triangulation_angle = 5.0;  // degrees
    double reproj_error_threshold = 4.0;        // pixels
    int ba_frequency = 5;   // run BA every N registered images
    bool use_gtsam = false; // use GTSAM factor graph instead of Ceres
};

// Run incremental Structure-from-Motion on a scene
// Expects features and matches to already be computed
bool run_incremental_sfm(
    Scene& scene,
    const std::vector<struct ImagePairMatches>& pair_matches,
    const SfMConfig& cfg = SfMConfig());

}  // namespace geometry
}  // namespace chisel
