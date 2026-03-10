// src/geometry/bundle_adjustment.cpp  –  Ceres Solver bundle adjustment
#include "geometry/bundle_adjustment.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <chrono>
#include <iostream>
#include <set>

namespace chisel {
namespace geometry {

// ─── Reprojection cost functor ──────────────
// Parameterization:
//   camera[0..3] = quaternion (w, x, y, z)
//   camera[4..6] = translation (tx, ty, tz)
//   point[0..2]  = 3D point (X, Y, Z)
//   intrinsics: fx, fy, cx, cy (optionally optimized)

struct ReprojectionCost {
    double observed_x, observed_y;
    double fx, fy, cx, cy;

    ReprojectionCost(double ox, double oy,
                     double fx_, double fy_, double cx_, double cy_)
        : observed_x(ox), observed_y(oy),
          fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}

    template <typename T>
    bool operator()(const T* const camera,   // [qw, qx, qy, qz, tx, ty, tz]
                    const T* const point,    // [X, Y, Z]
                    T* residuals) const {
        // Rotate point: p_cam = R * p_world + t
        T p[3];
        T q[4] = {camera[0], camera[1], camera[2], camera[3]};
        ceres::QuaternionRotatePoint(q, point, p);

        p[0] += camera[4];
        p[1] += camera[5];
        p[2] += camera[6];

        // Project
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        T predicted_x = T(fx) * xp + T(cx);
        T predicted_y = T(fy) * yp + T(cy);

        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);
        return true;
    }

    static ceres::CostFunction* Create(
            double ox, double oy,
            double fx, double fy, double cx, double cy) {
        return new ceres::AutoDiffCostFunction<ReprojectionCost, 2, 7, 3>(
            new ReprojectionCost(ox, oy, fx, fy, cx, cy));
    }
};

// Variant with optimizable intrinsics
struct ReprojectionCostWithIntrinsics {
    double observed_x, observed_y;

    ReprojectionCostWithIntrinsics(double ox, double oy)
        : observed_x(ox), observed_y(oy) {}

    template <typename T>
    bool operator()(const T* const camera,      // [qw,qx,qy,qz,tx,ty,tz]
                    const T* const point,        // [X,Y,Z]
                    const T* const intrinsics,   // [fx,fy,cx,cy]
                    T* residuals) const {
        T p[3];
        T q[4] = {camera[0], camera[1], camera[2], camera[3]};
        ceres::QuaternionRotatePoint(q, point, p);
        p[0] += camera[4];
        p[1] += camera[5];
        p[2] += camera[6];

        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        residuals[0] = intrinsics[0] * xp + intrinsics[2] - T(observed_x);
        residuals[1] = intrinsics[1] * yp + intrinsics[3] - T(observed_y);
        return true;
    }

    static ceres::CostFunction* Create(double ox, double oy) {
        return new ceres::AutoDiffCostFunction<
            ReprojectionCostWithIntrinsics, 2, 7, 3, 4>(
            new ReprojectionCostWithIntrinsics(ox, oy));
    }
};

// ─── Run full BA ────────────────────────────
BundleAdjustmentReport run_bundle_adjustment(
        Scene& scene,
        const BundleAdjustmentConfig& cfg) {

    auto t0 = std::chrono::steady_clock::now();
    BundleAdjustmentReport report;

    // Pack camera parameters: 7 doubles per image [qw,qx,qy,qz,tx,ty,tz]
    std::map<ImageId, std::array<double, 7>> cam_params;
    for (auto& [id, img] : scene.images) {
        if (!img.pose_valid) continue;
        auto& p = cam_params[id];
        p[0] = img.pose.rotation.w();
        p[1] = img.pose.rotation.x();
        p[2] = img.pose.rotation.y();
        p[3] = img.pose.rotation.z();
        p[4] = img.pose.translation.x();
        p[5] = img.pose.translation.y();
        p[6] = img.pose.translation.z();
    }

    // Pack 3D points: 3 doubles per point
    std::map<Point3DId, std::array<double, 3>> pt_params;
    for (auto& [id, pt] : scene.points3d) {
        pt_params[id] = {pt.xyz.x(), pt.xyz.y(), pt.xyz.z()};
    }

    // Pack intrinsics: [fx, fy, cx, cy] per camera
    std::map<CameraId, std::array<double, 4>> intrinsic_params;
    for (auto& [id, cam] : scene.cameras) {
        intrinsic_params[id] = {cam.fx, cam.fy, cam.cx, cam.cy};
    }

    // Build Ceres problem
    ceres::Problem problem;

    ceres::LossFunction* loss = loss;
    if (cfg.huber_loss_scale > 0) {
        loss = new ceres::HuberLoss(cfg.huber_loss_scale);
    }

    int num_observations = 0;

    for (auto& [pid, pt] : scene.points3d) {
        for (auto& te : pt.track) {
            auto img_it = scene.images.find(te.image_id);
            if (img_it == scene.images.end() || !img_it->second.pose_valid)
                continue;
            if (te.feature_idx >= img_it->second.keypoints.size())
                continue;

            auto cam_it = cam_params.find(te.image_id);
            auto pt_it  = pt_params.find(pid);
            if (cam_it == cam_params.end() || pt_it == pt_params.end())
                continue;

            const Vec2& obs = img_it->second.keypoints[te.feature_idx].xy;
            CameraId cid = img_it->second.camera_id;

            if (cfg.fix_intrinsics) {
                auto& cam = scene.cameras[cid];
                ceres::CostFunction* cost = ReprojectionCost::Create(
                    obs.x(), obs.y(), cam.fx, cam.fy, cam.cx, cam.cy);
                problem.AddResidualBlock(cost, loss,
                    cam_it->second.data(), pt_it->second.data());
            } else {
                ceres::CostFunction* cost =
                    ReprojectionCostWithIntrinsics::Create(obs.x(), obs.y());
                problem.AddResidualBlock(cost, loss,
                    cam_it->second.data(), pt_it->second.data(),
                    intrinsic_params[cid].data());
            }

            num_observations++;
        }
    }
    std::cerr << "[ba] Residuals: " << problem.NumResidualBlocks()
              << "  params: " << problem.NumParameterBlocks() << "\n";
    std::cerr.flush();
    if (problem.NumResidualBlocks() == 0) {
        std::cerr << "[ba] ERROR: 0 residuals — check keypoints / track / pose_valid\n";
        std::cerr.flush();
        abort();
    }

    if (num_observations == 0) {
        std::cerr << "[ba] No observations to optimize\n";
        return report;
    }

    // Fix gauge: first registered image
    if (cfg.fix_first_pose) {
        int fixed_count = 0;
        for (auto& [id, img] : scene.images) {
            if (img.pose_valid && cam_params.count(id)) {
                problem.SetParameterBlockConstant(cam_params[id].data());
                fixed_count++;
                if (fixed_count >=2) break;
            }
        }
    }

    // Quaternion parameterization for cameras
    for (auto& [id, params] : cam_params) {
        if (problem.HasParameterBlock(params.data())) {
            auto* quat_param = new ceres::EigenQuaternionManifold();
            // We need a product manifold: quaternion (4) + translation (3)
            auto* manifold = new ceres::ProductManifold<
                ceres::EigenQuaternionManifold,
                ceres::EuclideanManifold<3>>();
            // Note: Ceres QuaternionManifold expects [w,x,y,z] ordering
            // which matches our packing
            problem.SetManifold(params.data(), manifold);
        }
    }

    // Solver options
    ceres::Solver::Options options;
    options.max_num_iterations = cfg.max_iterations;
    options.function_tolerance = cfg.function_tolerance;
    options.gradient_tolerance = cfg.gradient_tolerance;
    options.parameter_tolerance = cfg.parameter_tolerance;
    options.num_threads = cfg.num_threads;
    options.minimizer_progress_to_stdout = cfg.verbose;

    if (cfg.use_schur) {
        options.linear_solver_type = ceres::SPARSE_SCHUR;
    } else {
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    }

    // Solve
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (cfg.verbose) {
        std::cout << summary.BriefReport() << "\n";
    }

    // Unpack results back into Scene
    for (auto& [id, params] : cam_params) {
        auto& img = scene.images[id];
        img.pose.rotation = Quat(params[0], params[1], params[2], params[3]);
        img.pose.rotation.normalize();
        img.pose.translation = Vec3(params[4], params[5], params[6]);
    }

    for (auto& [id, params] : pt_params) {
        scene.points3d[id].xyz = Vec3(params[0], params[1], params[2]);
    }

    if (!cfg.fix_intrinsics) {
        for (auto& [cid, params] : intrinsic_params) {
            auto& cam = scene.cameras[cid];
            cam.fx = params[0]; cam.fy = params[1];
            cam.cx = params[2]; cam.cy = params[3];
        }
    }

    // Compute mean reprojection error
    double total_err = 0.0;
    int cnt = 0;
    for (auto& [pid, pt] : scene.points3d) {
        for (auto& te : pt.track) {
            auto it = scene.images.find(te.image_id);
            if (it == scene.images.end() || !it->second.pose_valid) continue;
            if (te.feature_idx >= it->second.keypoints.size()) continue;

            auto& cam = scene.cameras[it->second.camera_id];
            Vec3 p_cam = it->second.pose.transform(pt.xyz);
            if (p_cam.z() <= 0) continue;
            Vec2 proj = cam.project(p_cam);
            double err = (proj - it->second.keypoints[te.feature_idx].xy).norm();
            total_err += err;
            cnt++;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    report.initial_cost = summary.initial_cost;
    report.final_cost   = summary.final_cost;
    report.iterations   = summary.num_successful_steps;
    report.time_seconds = elapsed;
    report.success      = summary.IsSolutionUsable();
    report.mean_reproj_error = cnt > 0 ? total_err / cnt : 0.0;

    std::cout << "[ba] " << num_observations << " observations, "
              << cam_params.size() << " cameras, "
              << pt_params.size() << " points\n"
              << "[ba] Mean reproj error: " << report.mean_reproj_error
              << " px, time: " << elapsed << " s\n";

    return report;
}

// ─── Local BA ───────────────────────────────
BundleAdjustmentReport run_local_bundle_adjustment(
        Scene& scene,
        const std::vector<ImageId>& variable_images,
        const BundleAdjustmentConfig& cfg) {


    auto modified_cfg = cfg;
    modified_cfg.fix_first_pose = false;

   
    return run_bundle_adjustment(scene, modified_cfg);
}

}  // namespace geometry
}  // namespace chisel
