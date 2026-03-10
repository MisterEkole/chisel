// src/geometry/factor_graph.cpp  –  GTSAM factor graph optimization
#include "geometry/factor_graph.h"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/inference/Symbol.h>

#include <chrono>
#include <iostream>
#include <set>
#include <memory>

using gtsam::symbol_shorthand::X;  // Pose3 (cameras)
using gtsam::symbol_shorthand::L;  // Point3 (landmarks)

namespace chisel {
namespace geometry {

// ─── Helper: convert our types ↔ GTSAM ─────
static gtsam::Pose3 to_gtsam_pose(const CameraPose& p) {
    gtsam::Rot3 R(p.rotation.toRotationMatrix());
    gtsam::Point3 t(p.translation.x(), p.translation.y(), p.translation.z());
    return gtsam::Pose3(R, t);
}

static CameraPose from_gtsam_pose(const gtsam::Pose3& gp) {
    CameraPose p;
    Eigen::Matrix3d R = gp.rotation().matrix();
    p.rotation = Quat(R);
    auto t = gp.translation();
    p.translation = Vec3(t.x(), t.y(), t.z());
    return p;
}

// ─── Pose graph optimization ────────────────
FactorGraphReport optimize_pose_graph(
        Scene& scene,
        const FactorGraphConfig& cfg) {

    auto t0 = std::chrono::steady_clock::now();
    FactorGraphReport report;

    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;

    // Noise models
    auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << gtsam::Vector3::Constant(cfg.prior_rot_sigma),
                             gtsam::Vector3::Constant(cfg.prior_trans_sigma)).finished());

    auto pixel_noise = gtsam::noiseModel::Isotropic::Sigma(2, cfg.pixel_sigma);

    // Add camera poses as variables
    bool first = true;
    std::set<ImageId> registered_ids;
    for (auto& [id, img] : scene.images) {
        if (!img.pose_valid) continue;
        registered_ids.insert(id);

        gtsam::Pose3 gpose = to_gtsam_pose(img.pose);
        initial.insert(X(id), gpose);

        // Prior on first camera to fix gauge
        if (first) {
            graph.addPrior(X(id), gpose, prior_noise);
            first = false;
        }
    }

    // Add landmark variables and projection factors
    std::set<Point3DId> added_points;
    for (auto& [pid, pt] : scene.points3d) {
        if (pt.track.size() < 2) continue;

        gtsam::Point3 gpt(pt.xyz.x(), pt.xyz.y(), pt.xyz.z());
        initial.insert(L(pid), gpt);
        added_points.insert(pid);

        for (auto& te : pt.track) {
            if (!registered_ids.count(te.image_id)) continue;
            auto& img = scene.images[te.image_id];
            if (te.feature_idx >= img.keypoints.size()) continue;

            auto& cam = scene.cameras[img.camera_id];
            auto K = std::make_shared<gtsam::Cal3_S2>(
                cam.fx, cam.fy, 0.0, cam.cx, cam.cy);

            gtsam::Point2 obs(img.keypoints[te.feature_idx].xy.x(),
                              img.keypoints[te.feature_idx].xy.y());

            graph.emplace_shared<gtsam::GenericProjectionFactor<
                gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>>(
                obs, pixel_noise, X(te.image_id), L(pid), K);
        }
    }

    if (cfg.verbose) {
        std::cout << "[gtsam] Graph: " << graph.size() << " factors, "
                  << initial.size() << " variables ("
                  << registered_ids.size() << " cameras, "
                  << added_points.size() << " landmarks)\n";
    }

    // Optimize
    report.initial_error = graph.error(initial);

    gtsam::LevenbergMarquardtParams params;
    params.setMaxIterations(cfg.max_iterations);
    params.setRelativeErrorTol(cfg.rel_error_tol);
    if (cfg.verbose) params.setVerbosityLM("SUMMARY");

    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, params);
    gtsam::Values result = optimizer.optimize();

    report.final_error = graph.error(result);
    report.iterations  = optimizer.iterations();
    report.converged   = true;

    // Unpack results
    for (auto id : registered_ids) {
        auto gpose = result.at<gtsam::Pose3>(X(id));
        scene.images[id].pose = from_gtsam_pose(gpose);
    }

    for (auto pid : added_points) {
        auto gpt = result.at<gtsam::Point3>(L(pid));
        scene.points3d[pid].xyz = Vec3(gpt.x(), gpt.y(), gpt.z());
    }

    auto t1 = std::chrono::steady_clock::now();
    report.time_seconds = std::chrono::duration<double>(t1 - t0).count();

    if (cfg.verbose) {
        std::cout << "[gtsam] Optimization: error " << report.initial_error
                  << " → " << report.final_error
                  << " in " << report.iterations << " iterations ("
                  << report.time_seconds << " s)\n";
    }

    return report;
}

// ─── Full joint optimization ────────────────
FactorGraphReport optimize_full_graph(
        Scene& scene,
        const FactorGraphConfig& cfg) {
    // For now, same as pose graph (both optimize poses + landmarks)
    // Future: use SmartProjectionFactor for implicit triangulation
    return optimize_pose_graph(scene, cfg);
}

}  // namespace geometry
}  // namespace chisel
