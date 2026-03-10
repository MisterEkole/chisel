#pragma once
// src/geometry/factor_graph.h  –  GTSAM-based factor graph optimization
#include "core/types.h"
#include <vector>

namespace chisel {
namespace geometry {

struct FactorGraphConfig {
    // Noise models (isotropic sigmas)
    double pixel_sigma       = 1.0;    // reprojection noise (pixels)
    double prior_rot_sigma   = 0.01;   // prior on rotation (radians)
    double prior_trans_sigma = 0.1;    // prior on translation (meters)
    double between_rot_sigma = 0.05;   // relative rotation noise
    double between_trans_sigma = 0.3;  // relative translation noise

    // Optimization
    int    max_iterations = 100;
    double rel_error_tol  = 1e-5;
    bool   use_isam2      = false;  // incremental vs batch
    bool   verbose        = true;
};

struct FactorGraphReport {
    double initial_error = 0.0;
    double final_error   = 0.0;
    int    iterations    = 0;
    bool   converged     = false;
    double time_seconds  = 0.0;
};

// Full batch optimization using GTSAM
FactorGraphReport optimize_pose_graph(
    Scene& scene,
    const FactorGraphConfig& cfg = FactorGraphConfig());

// Joint optimization: poses + landmarks via GTSAM smart factors
FactorGraphReport optimize_full_graph(
    Scene& scene,
    const FactorGraphConfig& cfg = FactorGraphConfig());

}  // namespace geometry
}  // namespace chisel
