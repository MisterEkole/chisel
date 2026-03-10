// src/geometry/feature_matching.cpp
#include "geometry/feature_matching.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <set>

namespace chisel {
namespace geometry {

// ─── SIFT feature extraction ────────────────
void extract_sift_features(ImageData& image, int max_features, int octave_layers) {
    if (image.image.empty()) {
        std::cerr << "[match] Image not loaded: " << image.name << "\n";
        return;
    }

    cv::Mat gray;
    if (image.image.channels() == 3)
        cv::cvtColor(image.image, gray, cv::COLOR_BGR2GRAY);
    else
        gray = image.image;

    auto sift = cv::SIFT::create(max_features, octave_layers);

    std::vector<cv::KeyPoint> cv_kps;
    cv::Mat cv_desc;
    sift->detectAndCompute(gray, cv::noArray(), cv_kps, cv_desc);

    // Convert to our types
    image.keypoints.clear();
    image.descriptors.clear();
    image.keypoints.reserve(cv_kps.size());
    image.descriptors.reserve(cv_kps.size());

    for (size_t i = 0; i < cv_kps.size(); ++i) {
        Keypoint kp;
        kp.xy       = Vec2(cv_kps[i].pt.x, cv_kps[i].pt.y);
        kp.scale    = cv_kps[i].size;
        kp.angle    = cv_kps[i].angle;
        kp.response = cv_kps[i].response;
        kp.id       = static_cast<FeatureId>(i);
        image.keypoints.push_back(kp);

        Descriptor desc;
        desc.type = Descriptor::Type::FLOAT32;
        desc.float_data.resize(cv_desc.cols);
        for (int j = 0; j < cv_desc.cols; ++j)
            desc.float_data[j] = cv_desc.at<float>(i, j);
        image.descriptors.push_back(desc);
    }

    image.allocate_tracks();
    std::cout << "[match] Extracted " << image.keypoints.size()
              << " SIFT features from " << image.name << "\n";
}

// ─── Descriptor-level matching ──────────────
static cv::Mat descriptors_to_mat(const std::vector<Descriptor>& descs) {
    if (descs.empty()) return cv::Mat();
    int dim = descs[0].dim();
    cv::Mat mat(static_cast<int>(descs.size()), dim, CV_32F);
    for (size_t i = 0; i < descs.size(); ++i) {
        for (int j = 0; j < dim; ++j)
            mat.at<float>(i, j) = descs[i].float_data[j];
    }
    return mat;
}

std::vector<Match> match_features(const ImageData& img1,
                                  const ImageData& img2,
                                  const MatchingConfig& cfg) {
    cv::Mat desc1 = descriptors_to_mat(img1.descriptors);
    cv::Mat desc2 = descriptors_to_mat(img2.descriptors);

    if (desc1.empty() || desc2.empty()) return {};

    // kNN matching with k=2 for ratio test
    auto matcher = cv::BFMatcher::create(cv::NORM_L2, false);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(desc1, desc2, knn_matches, 2);

    // Ratio test
    std::vector<Match> good;
    for (auto& knn : knn_matches) {
        if (knn.size() < 2) continue;
        if (knn[0].distance < cfg.ratio_threshold * knn[1].distance) {
            if (cfg.distance_thresh > 0 && knn[0].distance > cfg.distance_thresh)
                continue;
            Match m;
            m.query_idx = knn[0].queryIdx;
            m.train_idx = knn[0].trainIdx;
            m.distance  = knn[0].distance;
            good.push_back(m);
        }
    }

    // Cross-check if requested
    if (cfg.cross_check) {
        std::vector<std::vector<cv::DMatch>> reverse_knn;
        matcher->knnMatch(desc2, desc1, reverse_knn, 2);

        std::set<int> reverse_map;  // train_idx → query_idx best
        for (auto& knn : reverse_knn) {
            if (knn.size() < 2) continue;
            if (knn[0].distance < cfg.ratio_threshold * knn[1].distance) {
                // knn[0].queryIdx is from desc2, trainIdx from desc1
                reverse_map.insert(knn[0].trainIdx * 100000 + knn[0].queryIdx);
            }
        }

        std::vector<Match> mutual;
        for (auto& m : good) {
            int key = m.query_idx * 100000 + m.train_idx;
            if (reverse_map.count(key)) mutual.push_back(m);
        }
        good = std::move(mutual);
    }

    // Geometric verification: estimate fundamental matrix
    if (cfg.verify_fundamental && good.size() >= 8) {
        std::vector<cv::Point2f> pts1, pts2;
        for (auto& m : good) {
            pts1.emplace_back(img1.keypoints[m.query_idx].xy.x(),
                              img1.keypoints[m.query_idx].xy.y());
            pts2.emplace_back(img2.keypoints[m.train_idx].xy.x(),
                              img2.keypoints[m.train_idx].xy.y());
        }

        std::vector<uchar> inlier_mask;
        cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC,
                               cfg.fundamental_thresh,
                               cfg.ransac_confidence,
                               cfg.ransac_max_iters,
                               inlier_mask);

        std::vector<Match> verified;
        for (size_t i = 0; i < good.size(); ++i) {
            if (inlier_mask[i]) verified.push_back(good[i]);
        }
        good = std::move(verified);
    }

    // Cap at max matches
    if (static_cast<int>(good.size()) > cfg.max_num_matches) {
        std::sort(good.begin(), good.end(),
                  [](const Match& a, const Match& b) {
                      return a.distance < b.distance;
                  });
        good.resize(cfg.max_num_matches);
    }

    return good;
}

// ─── Exhaustive pairwise matching ───────────
std::vector<ImagePairMatches> match_all_pairs(
        std::map<ImageId, ImageData>& images,
        const MatchingConfig& cfg,
        int min_num_inliers) {

    std::vector<ImageId> ids;
    for (auto& [id, _] : images) ids.push_back(id);
    std::sort(ids.begin(), ids.end());

    std::vector<ImagePairMatches> all_pairs;

    for (size_t i = 0; i < ids.size(); ++i) {
        for (size_t j = i + 1; j < ids.size(); ++j) {
            auto matches = match_features(images[ids[i]], images[ids[j]], cfg);

            if (static_cast<int>(matches.size()) >= min_num_inliers) {
                ImagePairMatches pm;
                pm.id1 = ids[i];
                pm.id2 = ids[j];
                pm.matches = std::move(matches);
                pm.num_inliers = static_cast<int>(pm.matches.size());

                std::cout << "[match] " << images[ids[i]].name << " <-> "
                          << images[ids[j]].name << ": "
                          << pm.num_inliers << " inlier matches\n";

                all_pairs.push_back(std::move(pm));
            }
        }
    }

    std::cout << "[match] Total valid pairs: " << all_pairs.size() << "\n";
    return all_pairs;
}

}  // namespace geometry
}  // namespace chisel
