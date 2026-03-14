#pragma once
// src/geometry/bundle_adjustment.h  –  Ceres-based bundle adjustment
#include "core/types.h"

namespace chisel {
namespace geometry {

struct BundleAdjustmentConfig {
    int    max_iterations     = 100;
    double function_tolerance = 1e-5;
    double gradient_tolerance = 1e-10;
    double parameter_tolerance = 1e-8;

    bool fix_intrinsics  = false;  // keep K constant
    bool fix_first_pose  = true;   // anchor gauge freedom

    // Robust loss
    double huber_loss_scale = 1.0;  // 0 = no robust loss

    // Solver
    int num_threads = 4;
    bool use_schur  = true;  // Schur complement (recommended for BA)
    bool verbose    = true;
};

struct BundleAdjustmentReport {
    double initial_cost = 0.0;
    double final_cost   = 0.0;
    int    iterations   = 0;
    double time_seconds = 0.0;
    bool   success      = false;
    double mean_reproj_error = 0.0;
};

// Run full bundle adjustment on the scene
BundleAdjustmentReport run_bundle_adjustment(
    Scene& scene,
    const BundleAdjustmentConfig& cfg = BundleAdjustmentConfig());

// Local BA: optimize only a subset of images and their observed points
BundleAdjustmentReport run_local_bundle_adjustment(
    Scene& scene,
    const std::vector<ImageId>& variable_images,
    const BundleAdjustmentConfig& cfg = BundleAdjustmentConfig());

}  // namespace geometry
}  // namespace chisel
