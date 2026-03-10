// src/geometry/pose_estimation.cpp
#include "geometry/pose_estimation.h"
#include "geometry/feature_matching.h"
#include "geometry/epipolar.h"
#include "geometry/triangulation.h"
#include "geometry/bundle_adjustment.h"

#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <iostream>
#include <set>

namespace chisel {
namespace geometry {

// ─── PnP RANSAC ─────────────────────────────
PnPResult solve_pnp_ransac(
        const std::vector<Vec2>& points2d,
        const std::vector<Vec3>& points3d,
        const CameraIntrinsics& camera,
        double ransac_thresh,
        int max_iterations,
        double confidence) {

    PnPResult result;
    if (points2d.size() < 4) return result;

    // Convert to OpenCV
    std::vector<cv::Point2f> cv_pts2d(points2d.size());
    std::vector<cv::Point3f> cv_pts3d(points3d.size());
    for (size_t i = 0; i < points2d.size(); ++i) {
        cv_pts2d[i] = cv::Point2f(points2d[i].x(), points2d[i].y());
        cv_pts3d[i] = cv::Point3f(points3d[i].x(), points3d[i].y(), points3d[i].z());
    }

    cv::Mat K = (cv::Mat_<double>(3,3) <<
        camera.fx, 0, camera.cx,
        0, camera.fy, camera.cy,
        0, 0, 1);

    cv::Mat dist_coeffs;
    if (!camera.distortion.empty()) {
        dist_coeffs = cv::Mat(camera.distortion);
    }

    cv::Mat rvec, tvec, inliers_idx;
    bool ok = cv::solvePnPRansac(
        cv_pts3d, cv_pts2d, K, dist_coeffs,
        rvec, tvec, false,
        max_iterations, ransac_thresh, confidence,
        inliers_idx, cv::SOLVEPNP_EPNP);

    if (!ok) return result;

    // Convert rotation vector → matrix → quaternion
    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);

    Mat3 R;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R(i, j) = R_cv.at<double>(i, j);

    result.pose.rotation = Quat(R);
    result.pose.translation = Vec3(
        tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

    // Build inlier mask
    result.inlier_mask.assign(points2d.size(), false);
    result.num_inliers = inliers_idx.rows;
    for (int i = 0; i < inliers_idx.rows; ++i)
        result.inlier_mask[inliers_idx.at<int>(i)] = true;

    result.success = true;
    return result;
}

// ─── Iterative PnP refinement ───────────────
bool refine_pose_pnp(
        CameraPose& pose,
        const std::vector<Vec2>& points2d,
        const std::vector<Vec3>& points3d,
        const CameraIntrinsics& camera,
        const std::vector<bool>& inlier_mask) {

    std::vector<cv::Point2f> cv_pts2d;
    std::vector<cv::Point3f> cv_pts3d;
    for (size_t i = 0; i < points2d.size(); ++i) {
        if (!inlier_mask[i]) continue;
        cv_pts2d.emplace_back(points2d[i].x(), points2d[i].y());
        cv_pts3d.emplace_back(points3d[i].x(), points3d[i].y(), points3d[i].z());
    }

    if (cv_pts2d.size() < 6) return false;

    cv::Mat K = (cv::Mat_<double>(3,3) <<
        camera.fx, 0, camera.cx, 0, camera.fy, camera.cy, 0, 0, 1);

    // Initial guess from current pose
    Mat3 R = pose.rotation.toRotationMatrix();
    cv::Mat R_cv(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R_cv.at<double>(i, j) = R(i, j);

    cv::Mat rvec;
    cv::Rodrigues(R_cv, rvec);
    cv::Mat tvec = (cv::Mat_<double>(3,1) <<
        pose.translation.x(), pose.translation.y(), pose.translation.z());

    bool ok = cv::solvePnP(cv_pts3d, cv_pts2d, K, cv::Mat(),
                            rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);
    if (!ok) return false;

    cv::Rodrigues(rvec, R_cv);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R(i, j) = R_cv.at<double>(i, j);

    pose.rotation = Quat(R);
    pose.translation = Vec3(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
    return true;
}

// ─── Helper: find the best initial pair ─────
static std::pair<ImageId, ImageId> find_initial_pair(
        const Scene& scene,
        const std::vector<ImagePairMatches>& pair_matches,
        double min_angle) {

    ImageId best_i = 0, best_j = 0;
    int best_score = 0;

    for (auto& pm : pair_matches) {
        if (pm.num_inliers < 50) continue;

        // Score by number of matches (proxy for baseline quality)
        int score = pm.num_inliers;
        if (score > best_score) {
            best_score = score;
            best_i = pm.id1;
            best_j = pm.id2;
        }
    }

    std::cout << "[sfm] Initial pair: image " << best_i << " & " << best_j
              << " (" << best_score << " matches)\n";
    return {best_i, best_j};
}

// ─── Incremental SfM ────────────────────────
bool run_incremental_sfm(
        Scene& scene,
        const std::vector<ImagePairMatches>& pair_matches,
        const SfMConfig& cfg) {

    std::cout << "\n══════════════════════════════════════\n"
              << "  Incremental Structure-from-Motion\n"
              << "══════════════════════════════════════\n\n";

    // Build match lookup: (id1,id2) → matches
    std::map<std::pair<ImageId,ImageId>, const ImagePairMatches*> match_map;
    for (auto& pm : pair_matches) {
        match_map[{pm.id1, pm.id2}] = &pm;
        match_map[{pm.id2, pm.id1}] = &pm;
    }

    // ── Step 1: Initialize with best pair ────
    auto [init_i, init_j] = find_initial_pair(scene, pair_matches,
                                               cfg.init_min_triangulation_angle);
    if (init_i == 0 && init_j == 0) {
        std::cerr << "[sfm] No suitable initial pair found\n";
        return false;
    }

    auto& img1 = scene.images[init_i];
    auto& img2 = scene.images[init_j];
    auto& cam1 = scene.cameras[img1.camera_id];
    auto& cam2 = scene.cameras[img2.camera_id];

    // Image 1 is at the origin
    img1.pose = CameraPose();
    img1.pose_valid = true;

    // Estimate relative pose for image 2
    auto pm_it = match_map.find({init_i, init_j});
    if (pm_it == match_map.end()) return false;
    const auto& init_matches = pm_it->second->matches;

    std::vector<Vec2> pts1, pts2;
    for (auto& m : init_matches) {
        pts1.push_back(img1.keypoints[m.query_idx].xy);
        pts2.push_back(img2.keypoints[m.train_idx].xy);
    }

    auto epipolar_res = estimate_essential(pts1, pts2, cam1, cam2);
    if (epipolar_res.num_inliers < cfg.min_num_inliers) {
        std::cerr << "[sfm] Initial pair failed essential matrix test\n";
        return false;
    }

    img2.pose = epipolar_res.relative;
    img2.pose_valid = true;

    // Triangulate initial points
    auto tri_result = triangulate_two_view(
        img1.pose, img2.pose, cam1, cam2, pts1, pts2,
        cfg.reproj_error_threshold);

    // Add valid 3D points to scene
    for (size_t i = 0; i < tri_result.points.size(); ++i) {
        if (!tri_result.valid[i]) continue;
        Point3DId pid = scene.next_point3d_id();
        Point3D pt;
        pt.id = pid;
        pt.xyz = tri_result.points[i];
        pt.reprojection_error = tri_result.reprojection_errors[i];

        // Color from image 1
        const Vec2& px = pts1[i];
        if (!img1.image.empty()) {
            int r = std::clamp(static_cast<int>(px.y()), 0, img1.image.rows-1);
            int c = std::clamp(static_cast<int>(px.x()), 0, img1.image.cols-1);
            auto bgr = img1.image.at<cv::Vec3b>(r, c);
            pt.color = Vec3(bgr[2], bgr[1], bgr[0]);
        }

        FeatureId fi1 = init_matches[i].query_idx;
        FeatureId fi2 = init_matches[i].train_idx;
        pt.track.push_back({init_i, fi1});
        pt.track.push_back({init_j, fi2});

        scene.points3d[pid] = pt;
        img1.point3d_ids[fi1] = pid;
        img2.point3d_ids[fi2] = pid;
    }

    std::cout << "[sfm] Initialized with " << scene.points3d.size()
              << " 3D points\n";

    // ── Step 2: Incrementally register remaining images ──
    std::set<ImageId> registered = {init_i, init_j};
    int ba_counter = 0;

    while (registered.size() < scene.images.size()) {
        // Find next best image (most 2D-3D correspondences)
        ImageId best_id = 0;
        int best_num_correspondences = 0;

        for (auto& [id, img] : scene.images) {
            if (registered.count(id)) continue;

            int count = 0;
            // Count how many of this image's features match to existing 3D pts
            for (auto reg_id : registered) {
                auto it = match_map.find({id, reg_id});
                if (it == match_map.end()) continue;
                auto& reg_img = scene.images[reg_id];

                for (auto& m : it->second->matches) {
                    FeatureId fi_reg = (it->first.first == reg_id)
                        ? m.query_idx : m.train_idx;
                    if (fi_reg < reg_img.point3d_ids.size() &&
                        reg_img.point3d_ids[fi_reg] >= 0) {
                        count++;
                    }
                }
            }

            if (count > best_num_correspondences) {
                best_num_correspondences = count;
                best_id = id;
            }
        }

        if (best_id == 0 || best_num_correspondences < cfg.min_num_inliers) {
            std::cout << "[sfm] No more images can be registered\n";
            break;
        }

        // Collect 2D-3D correspondences for PnP
        auto& new_img = scene.images[best_id];
        auto& new_cam = scene.cameras[new_img.camera_id];

        std::vector<Vec2> pts_2d;
        std::vector<Vec3> pts_3d;
        std::vector<std::pair<FeatureId, Point3DId>> feature_point_map;

        for (auto reg_id : registered) {
            auto it = match_map.find({best_id, reg_id});
            if (it == match_map.end()) {
                it = match_map.find({reg_id, best_id});
                if (it == match_map.end()) continue;
            }

            auto& reg_img = scene.images[reg_id];
            for (auto& m : it->second->matches) {
                FeatureId fi_new, fi_reg;
                if (it->first.first == best_id) {
                    fi_new = m.query_idx;
                    fi_reg = m.train_idx;
                } else {
                    fi_new = m.train_idx;
                    fi_reg = m.query_idx;
                }

                if (fi_reg < reg_img.point3d_ids.size() &&
                    reg_img.point3d_ids[fi_reg] >= 0) {
                    Point3DId pid = reg_img.point3d_ids[fi_reg];
                    if (scene.points3d.count(pid)) {
                        pts_2d.push_back(new_img.keypoints[fi_new].xy);
                        pts_3d.push_back(scene.points3d[pid].xyz);
                        feature_point_map.push_back({fi_new, pid});
                    }
                }
            }
        }

        // Solve PnP
        auto pnp = solve_pnp_ransac(pts_2d, pts_3d, new_cam);
        if (!pnp.success || pnp.num_inliers < cfg.min_num_inliers) {
            std::cout << "[sfm] PnP failed for image " << best_id
                      << " (" << pnp.num_inliers << " inliers)\n";
            // Mark as un-registerable and skip
            continue;
        }

        new_img.pose = pnp.pose;
        new_img.pose_valid = true;
        registered.insert(best_id);

        // Update tracks: link inlier features to existing 3D points
        new_img.allocate_tracks();
        for (size_t i = 0; i < feature_point_map.size(); ++i) {
            if (pnp.inlier_mask[i]) {
                auto [fi, pid] = feature_point_map[i];
                new_img.point3d_ids[fi] = pid;
                scene.points3d[pid].track.push_back({best_id, fi});
            }
        }

        // Triangulate new points with all registered images
        for (auto reg_id : registered) {
            if (reg_id == best_id) continue;
            auto it = match_map.find({best_id, reg_id});
            if (it == match_map.end()) {
                it = match_map.find({reg_id, best_id});
                if (it == match_map.end()) continue;
            }

            auto& reg_img = scene.images[reg_id];
            auto& reg_cam = scene.cameras[reg_img.camera_id];

            std::vector<Vec2> new_pts1, new_pts2;
            std::vector<std::pair<FeatureId,FeatureId>> new_feature_pairs;

            for (auto& m : it->second->matches) {
                FeatureId fi_new, fi_reg;
                if (it->first.first == best_id) {
                    fi_new = m.query_idx; fi_reg = m.train_idx;
                } else {
                    fi_new = m.train_idx; fi_reg = m.query_idx;
                }

                // Only triangulate untracked features
                bool new_tracked = fi_new < new_img.point3d_ids.size() &&
                                   new_img.point3d_ids[fi_new] >= 0;
                bool reg_tracked = fi_reg < reg_img.point3d_ids.size() &&
                                   reg_img.point3d_ids[fi_reg] >= 0;
                if (new_tracked || reg_tracked) continue;

                new_pts1.push_back(new_img.keypoints[fi_new].xy);
                new_pts2.push_back(reg_img.keypoints[fi_reg].xy);
                new_feature_pairs.push_back({fi_new, fi_reg});
            }

            if (new_pts1.size() < 10) continue;

            auto tri = triangulate_two_view(
                new_img.pose, reg_img.pose, new_cam, reg_cam,
                new_pts1, new_pts2, cfg.reproj_error_threshold);

            for (size_t i = 0; i < tri.points.size(); ++i) {
                if (!tri.valid[i]) continue;
                Point3DId pid = scene.next_point3d_id();
                Point3D pt;
                pt.id = pid;
                pt.xyz = tri.points[i];
                pt.reprojection_error = tri.reprojection_errors[i];
                pt.track.push_back({best_id, new_feature_pairs[i].first});
                pt.track.push_back({reg_id, new_feature_pairs[i].second});

                scene.points3d[pid] = pt;
                new_img.point3d_ids[new_feature_pairs[i].first] = pid;
                reg_img.point3d_ids[new_feature_pairs[i].second] = pid;
            }
        }

        std::cout << "[sfm] Registered image " << best_id << " ("
                  << registered.size() << "/" << scene.images.size()
                  << "), " << scene.points3d.size() << " 3D points\n";

        // Periodic bundle adjustment
        ba_counter++;
        if (ba_counter % cfg.ba_frequency == 0) {
            std::cout << "[sfm] Running bundle adjustment...\n";
            BundleAdjustmentConfig ba_cfg;
            ba_cfg.fix_intrinsics = true;
            run_bundle_adjustment(scene, ba_cfg);
        }
    }

    // Final global BA
    std::cout << "[sfm] Final bundle adjustment...\n";
    BundleAdjustmentConfig ba_cfg;
    ba_cfg.max_iterations = 200;
    run_bundle_adjustment(scene, ba_cfg);

    std::cout << "\n[sfm] DONE: " << registered.size() << " images registered, "
              << scene.points3d.size() << " 3D points\n";

    return true;
}

}  // namespace geometry
}  // namespace chisel
