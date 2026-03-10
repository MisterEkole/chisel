// src/geometry/epipolar.cpp
#include "geometry/epipolar.h"
#include "geometry/triangulation.h"

#include <opencv2/calib3d.hpp>
#include <iostream>
#include <cmath>

namespace chisel {
namespace geometry {

// ─── Sampson error ──────────────────────────
double sampson_error(const Mat3& F, const Vec2& p1, const Vec2& p2) {
    Vec3 x1(p1.x(), p1.y(), 1.0);
    Vec3 x2(p2.x(), p2.y(), 1.0);

    Vec3 Fx1 = F * x1;
    Vec3 Ftx2 = F.transpose() * x2;
    double x2tFx1 = x2.dot(Fx1);

    return (x2tFx1 * x2tFx1) /
           (Fx1[0]*Fx1[0] + Fx1[1]*Fx1[1] +
            Ftx2[0]*Ftx2[0] + Ftx2[1]*Ftx2[1]);
}

double symmetric_epipolar_distance(const Mat3& F,
                                   const Vec2& p1,
                                   const Vec2& p2) {
    Vec3 x1(p1.x(), p1.y(), 1.0);
    Vec3 x2(p2.x(), p2.y(), 1.0);

    Vec3 Fx1 = F * x1;
    Vec3 Ftx2 = F.transpose() * x2;
    double x2tFx1 = x2.dot(Fx1);

    return std::abs(x2tFx1) *
           (1.0 / std::sqrt(Fx1[0]*Fx1[0] + Fx1[1]*Fx1[1]) +
            1.0 / std::sqrt(Ftx2[0]*Ftx2[0] + Ftx2[1]*Ftx2[1]));
}

// ─── Essential matrix estimation ────────────
EpipolarResult estimate_essential(
        const std::vector<Vec2>& pts1,
        const std::vector<Vec2>& pts2,
        const CameraIntrinsics& cam1,
        const CameraIntrinsics& cam2,
        double ransac_thresh,
        int max_iterations,
        double confidence) {

    EpipolarResult result;
    if (pts1.size() < 5 || pts1.size() != pts2.size()) {
        std::cerr << "[epipolar] Need >= 5 correspondences\n";
        return result;
    }

    // Convert to OpenCV points
    std::vector<cv::Point2f> cv_pts1(pts1.size()), cv_pts2(pts2.size());
    for (size_t i = 0; i < pts1.size(); ++i) {
        cv_pts1[i] = cv::Point2f(pts1[i].x(), pts1[i].y());
        cv_pts2[i] = cv::Point2f(pts2[i].x(), pts2[i].y());
    }

    cv::Mat K1 = (cv::Mat_<double>(3,3) <<
        cam1.fx, 0, cam1.cx,
        0, cam1.fy, cam1.cy,
        0, 0, 1);
    cv::Mat K2 = (cv::Mat_<double>(3,3) <<
        cam2.fx, 0, cam2.cx,
        0, cam2.fy, cam2.cy,
        0, 0, 1);

    // Threshold in normalized coordinates
    double focal = (cam1.fx + cam1.fy + cam2.fx + cam2.fy) / 4.0;
    double norm_thresh = ransac_thresh / focal;

    // Normalize points
    std::vector<cv::Point2f> norm1(pts1.size()), norm2(pts2.size());
    for (size_t i = 0; i < pts1.size(); ++i) {
        norm1[i].x = (cv_pts1[i].x - cam1.cx) / cam1.fx;
        norm1[i].y = (cv_pts1[i].y - cam1.cy) / cam1.fy;
        norm2[i].x = (cv_pts2[i].x - cam2.cx) / cam2.fx;
        norm2[i].y = (cv_pts2[i].y - cam2.cy) / cam2.fy;
    }

    cv::Mat mask;
    cv::Mat E_cv = cv::findEssentialMat(
        norm1, norm2, cv::Mat::eye(3, 3, CV_64F),
        cv::RANSAC, confidence, norm_thresh, max_iterations, mask);

    if (E_cv.empty()) {
        std::cerr << "[epipolar] Essential matrix estimation failed\n";
        return result;
    }

    // Take only first 3×3 if multiple solutions returned
    if (E_cv.rows > 3) E_cv = E_cv.rowRange(0, 3);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            result.essential(i, j) = E_cv.at<double>(i, j);

    // Fundamental: F = K2^{-T} E K1^{-1}
    Mat3 K1_inv = cam1.K().inverse();
    Mat3 K2_invt = cam2.K().inverse().transpose();
    result.fundamental = K2_invt * result.essential * K1_inv;

    // Inlier mask
    result.inlier_mask.resize(pts1.size(), false);
    result.num_inliers = 0;
    for (size_t i = 0; i < pts1.size(); ++i) {
        if (mask.at<uchar>(i)) {
            result.inlier_mask[i] = true;
            result.num_inliers++;
        }
    }

    // Decompose E → R, t
    int num_infront = 0;
    result.relative = decompose_essential(
        result.essential, pts1, pts2, cam1, cam2, num_infront);

    result.score = static_cast<double>(result.num_inliers) / pts1.size();

    std::cout << "[epipolar] E estimation: " << result.num_inliers
              << "/" << pts1.size() << " inliers, "
              << num_infront << " pts in front\n";

    return result;
}

// ─── Fundamental matrix estimation ──────────
EpipolarResult estimate_fundamental(
        const std::vector<Vec2>& pts1,
        const std::vector<Vec2>& pts2,
        double ransac_thresh,
        int max_iterations) {

    EpipolarResult result;
    if (pts1.size() < 8) return result;

    std::vector<cv::Point2f> cv_pts1(pts1.size()), cv_pts2(pts2.size());
    for (size_t i = 0; i < pts1.size(); ++i) {
        cv_pts1[i] = cv::Point2f(pts1[i].x(), pts1[i].y());
        cv_pts2[i] = cv::Point2f(pts2[i].x(), pts2[i].y());
    }

    cv::Mat mask;
    cv::Mat F_cv = cv::findFundamentalMat(
        cv_pts1, cv_pts2, cv::FM_RANSAC, ransac_thresh,
        0.999, max_iterations, mask);

    if (F_cv.empty()) return result;

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            result.fundamental(i, j) = F_cv.at<double>(i, j);

    result.inlier_mask.resize(pts1.size(), false);
    result.num_inliers = 0;
    for (size_t i = 0; i < pts1.size(); ++i) {
        if (mask.at<uchar>(i)) {
            result.inlier_mask[i] = true;
            result.num_inliers++;
        }
    }

    return result;
}

// ─── Decompose E → R, t ────────────────────
CameraPose decompose_essential(
        const Mat3& E,
        const std::vector<Vec2>& pts1,
        const std::vector<Vec2>& pts2,
        const CameraIntrinsics& cam1,
        const CameraIntrinsics& cam2,
        int& num_infront) {

    // SVD of E
    Eigen::JacobiSVD<Mat3> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat3 U = svd.matrixU();
    Mat3 V = svd.matrixV();

    // Ensure proper rotation (det = +1)
    if (U.determinant() < 0) U.col(2) *= -1;
    if (V.determinant() < 0) V.col(2) *= -1;

    Mat3 W;
    W << 0, -1, 0,
         1,  0, 0,
         0,  0, 1;

    // Four possible solutions
    Mat3 R1 = U * W * V.transpose();
    Mat3 R2 = U * W.transpose() * V.transpose();
    Vec3 t = U.col(2);

    CameraPose candidates[4];
    candidates[0].rotation = Quat(R1);  candidates[0].translation =  t;
    candidates[1].rotation = Quat(R1);  candidates[1].translation = -t;
    candidates[2].rotation = Quat(R2);  candidates[2].translation =  t;
    candidates[3].rotation = Quat(R2);  candidates[3].translation = -t;

    // Pick the one with most points in front of both cameras
    num_infront = 0;
    int best_idx = 0;

    // Identity pose for camera 1
    CameraPose pose1;

    size_t max_pts = std::min(pts1.size(), size_t(100));  // check subset for speed

    for (int c = 0; c < 4; ++c) {
        int count = 0;
        for (size_t i = 0; i < max_pts; ++i) {
            Vec3 ray1 = cam1.unproject(pts1[i]);
            Vec3 ray2 = cam2.unproject(pts2[i]);

            Vec3 pt3d = triangulate_point_linear(
                pose1, candidates[c], cam1, cam2, pts1[i], pts2[i]);

            // Check if point is in front of camera 1
            Vec3 p1_cam = pose1.transform(pt3d);
            Vec3 p2_cam = candidates[c].transform(pt3d);

            if (p1_cam.z() > 0 && p2_cam.z() > 0) count++;
        }

        if (count > num_infront) {
            num_infront = count;
            best_idx = c;
        }
    }

    return candidates[best_idx];
}

}  // namespace geometry
}  // namespace chisel
