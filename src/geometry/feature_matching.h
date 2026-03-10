#pragma once
// src/geometry/feature_matching.h
#include "core/types.h"
#include <opencv2/features2d.hpp>

namespace chisel {
namespace geometry {

struct MatchingConfig {
    float ratio_threshold  = 0.80f;  // Lowe's ratio test
    int   max_num_matches  = 8192;
    bool  cross_check      = true;
    float distance_thresh  = -1.f;   // hard distance cutoff (<=0 = off)

    // Geometric verification
    bool   verify_fundamental = true;
    double fundamental_thresh = 3.0;  // RANSAC pixel threshold
    int    ransac_max_iters   = 5000;
    double ransac_confidence  = 0.999;
};

// Extract SIFT features using OpenCV
void extract_sift_features(ImageData& image,
                           int max_features = 8192,
                           int octave_layers = 3);

// Match descriptors between two images (returns inlier matches)
std::vector<Match> match_features(
    const ImageData& img1,
    const ImageData& img2,
    const MatchingConfig& cfg = MatchingConfig());

// Exhaustive pairwise matching across all images
struct ImagePairMatches {
    ImageId id1, id2;
    std::vector<Match> matches;
    Mat3 fundamental;  // estimated F if geometric verification is on
    int num_inliers = 0;
};

std::vector<ImagePairMatches> match_all_pairs(
    std::map<ImageId, ImageData>& images,
    const MatchingConfig& cfg = MatchingConfig(),
    int min_num_inliers = 30);

}  // namespace geometry
}  // namespace chisel
