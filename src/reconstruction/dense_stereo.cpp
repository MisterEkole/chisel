// src/reconstruction/dense_stereo.cpp
#include "reconstruction/dense_stereo.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

namespace chisel {
namespace reconstruction {

// ─── Source image selection ─────────────────
std::vector<ImageId> select_source_images(
        const Scene& scene,
        ImageId ref_id,
        int num_sources) {

    auto ref_it = scene.images.find(ref_id);
    if (ref_it == scene.images.end()) return {};

    Vec3 ref_center = ref_it->second.pose.center();
    Vec3 ref_dir = ref_it->second.pose.rotation.inverse() * Vec3(0, 0, 1);

    struct Candidate {
        ImageId id;
        double score;
    };

    std::vector<Candidate> candidates;
    for (auto& [id, img] : scene.images) {
        if (id == ref_id || !img.pose_valid) continue;

        Vec3 center = img.pose.center();
        double baseline = (center - ref_center).norm();

        // View direction similarity
        Vec3 dir = img.pose.rotation.inverse() * Vec3(0, 0, 1);
        double cos_angle = ref_dir.dot(dir);

        // Score: moderate baseline, similar viewing direction
        // Penalize too small or too large baselines
        double score = cos_angle * std::min(baseline, 1.0 / (baseline + 1e-6));

        candidates.push_back({id, score});
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.score > b.score;
              });

    std::vector<ImageId> result;
    for (int i = 0; i < std::min(num_sources, (int)candidates.size()); ++i) {
        result.push_back(candidates[i].id);
    }
    return result;
}

// ─── Normalized Cross-Correlation ───────────
static float compute_ncc(const cv::Mat& patch1, const cv::Mat& patch2) {
    if (patch1.empty() || patch2.empty()) return -1.f;
    if (patch1.size() != patch2.size()) return -1.f;

    cv::Scalar mean1, stddev1, mean2, stddev2;
    cv::meanStdDev(patch1, mean1, stddev1);
    cv::meanStdDev(patch2, mean2, stddev2);

    if (stddev1[0] < 1e-5 || stddev2[0] < 1e-5) return -1.f;

    cv::Mat p1_norm, p2_norm;
    patch1.convertTo(p1_norm, CV_32F);
    patch2.convertTo(p2_norm, CV_32F);
    p1_norm = (p1_norm - mean1[0]) / stddev1[0];
    p2_norm = (p2_norm - mean2[0]) / stddev2[0];

    float ncc = static_cast<float>(p1_norm.dot(p2_norm)) / p1_norm.total();
    return ncc;
}

// ─── Homography for plane-sweep ─────────────
static cv::Mat compute_homography(
        const CameraIntrinsics& cam_ref,
        const CameraIntrinsics& cam_src,
        const CameraPose& pose_ref,
        const CameraPose& pose_src,
        double depth) {

    Mat3 K_ref = cam_ref.K();
    Mat3 K_src = cam_src.K();
    Mat3 R_ref = pose_ref.rotation.toRotationMatrix();
    Mat3 R_src = pose_src.rotation.toRotationMatrix();
    Vec3 t_ref = pose_ref.translation;
    Vec3 t_src = pose_src.translation;

    // Relative pose: src w.r.t. ref
    Mat3 R_rel = R_src * R_ref.transpose();
    Vec3 t_rel = t_src - R_rel * t_ref;

    // Plane normal in ref camera frame (fronto-parallel)
    Vec3 n(0, 0, 1);

    // Homography: H = K_src * (R_rel - t_rel * n^T / d) * K_ref^{-1}
    Mat3 H = K_src * (R_rel - t_rel * n.transpose() / depth) * K_ref.inverse();

    cv::Mat H_cv(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            H_cv.at<double>(i, j) = H(i, j);

    return H_cv;
}

// ─── Plane-sweep stereo ─────────────────────
DepthMap compute_depth_plane_sweep(
        const Scene& scene,
        ImageId ref_id,
        const std::vector<ImageId>& source_ids,
        const DenseStereoConfig& cfg) {

    auto& ref_img = scene.images.at(ref_id);
    auto& ref_cam = scene.cameras.at(ref_img.camera_id);

    DepthMap result;
    result.image_id = ref_id;

    if (ref_img.image.empty()) {
        std::cerr << "[dense] Reference image not loaded: " << ref_img.name << "\n";
        return result;
    }

    cv::Mat ref_gray;
    cv::cvtColor(ref_img.image, ref_gray, cv::COLOR_BGR2GRAY);

    int h = ref_gray.rows, w = ref_gray.cols;
    result.width  = w;
    result.height = h;
    result.depth.assign(w * h, 0.f);
    result.confidence.assign(w * h, 0.f);

    // Depth hypotheses (inverse-depth sampling)
    std::vector<float> depth_samples(cfg.num_depth_samples);
    float inv_min = 1.f / cfg.max_depth;
    float inv_max = 1.f / cfg.min_depth;
    for (int d = 0; d < cfg.num_depth_samples; ++d) {
        float alpha = static_cast<float>(d) / (cfg.num_depth_samples - 1);
        float inv_d = inv_min + alpha * (inv_max - inv_min);
        depth_samples[d] = 1.f / inv_d;
    }

    // Cost volume: [H × W × D]
    std::vector<std::vector<float>> cost_volume(
        h * w, std::vector<float>(cfg.num_depth_samples, 0.f));

    int half_patch = cfg.patch_size / 2;

    // Accumulate matching costs from each source view
    for (ImageId src_id : source_ids) {
        auto& src_img = scene.images.at(src_id);
        auto& src_cam = scene.cameras.at(src_img.camera_id);

        if (src_img.image.empty()) continue;

        cv::Mat src_gray;
        cv::cvtColor(src_img.image, src_gray, cv::COLOR_BGR2GRAY);

        for (int d = 0; d < cfg.num_depth_samples; ++d) {
            cv::Mat H = compute_homography(
                ref_cam, src_cam, ref_img.pose, src_img.pose, depth_samples[d]);

            cv::Mat warped;
            cv::warpPerspective(src_gray, warped, H, ref_gray.size());

            // Compute NCC per pixel (using patches)
            for (int r = half_patch; r < h - half_patch; r += 2) {
                for (int c = half_patch; c < w - half_patch; c += 2) {
                    cv::Rect roi(c - half_patch, r - half_patch,
                                 cfg.patch_size, cfg.patch_size);
                    float ncc = compute_ncc(ref_gray(roi), warped(roi));
                    if (ncc > -0.5f) {
                        // Convert NCC to cost (higher NCC = lower cost)
                        float cost = 1.f - ncc;
                        cost_volume[r * w + c][d] += cost;
                    }
                }
            }
        }
    }

    // Winner-takes-all: pick depth with minimum cost
    int valid_count = 0;
    for (int r = half_patch; r < h - half_patch; ++r) {
        for (int c = half_patch; c < w - half_patch; ++c) {
            int idx = r * w + c;
            auto& costs = cost_volume[idx];

            int best_d = 0;
            float best_cost = costs[0];
            for (int d = 1; d < cfg.num_depth_samples; ++d) {
                if (costs[d] < best_cost) {
                    best_cost = costs[d];
                    best_d = d;
                }
            }

            float conf = 1.f - best_cost / source_ids.size();
            if (conf > cfg.confidence_threshold) {
                result.depth[idx] = depth_samples[best_d];
                result.confidence[idx] = conf;
                valid_count++;
            }
        }
    }

    std::cout << "[dense] Plane-sweep for image " << ref_id << ": "
              << valid_count << " valid pixels\n";
    return result;
}

// ─── PatchMatch stereo (simplified) ─────────
DepthMap compute_depth_patchmatch(
        const Scene& scene,
        ImageId ref_id,
        const std::vector<ImageId>& source_ids,
        const DenseStereoConfig& cfg) {
    // PatchMatch: random initialization + propagation + refinement
    // For a full implementation this would be much more involved.
    // Fallback to plane-sweep for now.
    std::cout << "[dense] PatchMatch: delegating to plane-sweep\n";
    return compute_depth_plane_sweep(scene, ref_id, source_ids, cfg);
}

// ─── Multi-view consistency filtering ───────
void filter_depth_consistency(
        std::map<ImageId, DepthMap>& depth_maps,
        const Scene& scene,
        const DenseStereoConfig& cfg) {

    int total_filtered = 0;

    for (auto& [ref_id, ref_dm] : depth_maps) {
        auto& ref_img = scene.images.at(ref_id);
        auto& ref_cam = scene.cameras.at(ref_img.camera_id);

        for (uint32_t r = 0; r < ref_dm.height; ++r) {
            for (uint32_t c = 0; c < ref_dm.width; ++c) {
                if (!ref_dm.valid(r, c)) continue;

                float d = ref_dm.at(r, c);

                // Unproject to 3D
                Vec2 px(c, r);
                Vec3 ray = ref_cam.unproject(px);
                Vec3 pt3d = ref_img.pose.inverse().transform(ray * d);

                // Check consistency with other depth maps
                int consistent = 0;
                for (auto& [src_id, src_dm] : depth_maps) {
                    if (src_id == ref_id) continue;

                    auto& src_img = scene.images.at(src_id);
                    auto& src_cam = scene.cameras.at(src_img.camera_id);

                    Vec3 p_src = src_img.pose.transform(pt3d);
                    if (p_src.z() <= 0) continue;

                    Vec2 proj = src_cam.project(p_src);
                    int pr = static_cast<int>(proj.y());
                    int pc = static_cast<int>(proj.x());

                    if (pr < 0 || pr >= (int)src_dm.height ||
                        pc < 0 || pc >= (int)src_dm.width) continue;

                    if (!src_dm.valid(pr, pc)) continue;

                    float src_d = src_dm.at(pr, pc);
                    float reproj_d = p_src.z();
                    float diff = std::abs(src_d - reproj_d) / reproj_d;

                    if (diff < 0.1f) consistent++;
                }

                if (consistent < cfg.min_consistent_views) {
                    ref_dm.at(r, c) = 0.f;
                    ref_dm.confidence[r * ref_dm.width + c] = 0.f;
                    total_filtered++;
                }
            }
        }
    }

    std::cout << "[dense] Filtered " << total_filtered
              << " inconsistent depth values\n";
}

// ─── Compute all depth maps ─────────────────
void compute_all_depth_maps(Scene& scene, const DenseStereoConfig& cfg) {
    std::cout << "[dense] Computing depth maps for " << scene.num_registered()
              << " images...\n";

    for (auto& [id, img] : scene.images) {
        if (!img.pose_valid || img.image.empty()) continue;

        auto sources = select_source_images(scene, id, cfg.num_source_images);
        if (sources.empty()) continue;

        scene.depth_maps[id] = compute_depth_plane_sweep(
            scene, id, sources, cfg);
    }

    if (cfg.filter_by_consistency) {
        filter_depth_consistency(scene.depth_maps, scene, cfg);
    }

    std::cout << "[dense] Computed " << scene.depth_maps.size()
              << " depth maps\n";
}

}  // namespace reconstruction
}  // namespace chisel
