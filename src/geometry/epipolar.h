#pragma once
// src/geometry/epipolar.h
#include "core/types.h"
#include <vector>

namespace chisel {
namespace geometry {

struct EpipolarResult {
    Mat3 essential;        // essential matrix
    Mat3 fundamental;      // fundamental matrix
    CameraPose relative;   // relative pose (cam2 w.r.t. cam1)
    std::vector<bool> inlier_mask;
    int num_inliers = 0;
    double score = 0.0;
};

// Estimate essential matrix from calibrated correspondences
// Uses 5-point algorithm + RANSAC
EpipolarResult estimate_essential(
    const std::vector<Vec2>& pts1,
    const std::vector<Vec2>& pts2,
    const CameraIntrinsics& cam1,
    const CameraIntrinsics& cam2,
    double ransac_thresh = 1.0,     // pixel threshold
    int max_iterations = 10000,
    double confidence = 0.9999);

// Estimate fundamental matrix (uncalibrated)
EpipolarResult estimate_fundamental(
    const std::vector<Vec2>& pts1,
    const std::vector<Vec2>& pts2,
    double ransac_thresh = 3.0,
    int max_iterations = 10000);

// Decompose essential matrix into 4 possible [R|t] and pick
// the one with maximum points in front of both cameras
CameraPose decompose_essential(
    const Mat3& E,
    const std::vector<Vec2>& pts1,
    const std::vector<Vec2>& pts2,
    const CameraIntrinsics& cam1,
    const CameraIntrinsics& cam2,
    int& num_infront);

// Compute epipolar error for a match
double sampson_error(const Mat3& F, const Vec2& p1, const Vec2& p2);

// Compute symmetric epipolar distance
double symmetric_epipolar_distance(const Mat3& F,
                                   const Vec2& p1,
                                   const Vec2& p2);

}  // namespace geometry
}  // namespace chisel
