// src/geometry/triangulation.cpp
#include "geometry/triangulation.h"

#include <Eigen/SVD>
#include <cmath>
#include <iostream>

namespace chisel {
namespace geometry {

// ─── Linear DLT triangulation (two views) ───
Vec3 triangulate_point_linear(
        const CameraPose& pose1,
        const CameraPose& pose2,
        const CameraIntrinsics& cam1,
        const CameraIntrinsics& cam2,
        const Vec2& pt1,
        const Vec2& pt2) {

    // Projection matrices P = K [R | t]
    Mat3 R1 = pose1.rotation.toRotationMatrix();
    Mat3 R2 = pose2.rotation.toRotationMatrix();

    Eigen::Matrix<double, 3, 4> P1, P2;
    P1.block<3,3>(0,0) = R1;
    P1.col(3) = pose1.translation;
    P1 = cam1.K() * P1;

    P2.block<3,3>(0,0) = R2;
    P2.col(3) = pose2.translation;
    P2 = cam2.K() * P2;

    // Build 4×4 system: each observation gives 2 equations
    Eigen::Matrix4d A;
    A.row(0) = pt1.x() * P1.row(2) - P1.row(0);
    A.row(1) = pt1.y() * P1.row(2) - P1.row(1);
    A.row(2) = pt2.x() * P2.row(2) - P2.row(0);
    A.row(3) = pt2.y() * P2.row(2) - P2.row(1);

    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
    Vec4 X = svd.matrixV().col(3);

    return X.head<3>() / X[3];
}

// ─── Multi-view triangulation ───────────────
Vec3 triangulate_point_multiview(
        const std::vector<CameraPose>& poses,
        const std::vector<CameraIntrinsics>& cameras,
        const std::vector<Vec2>& observations) {

    int n = static_cast<int>(poses.size());
    MatX A(2 * n, 4);

    for (int i = 0; i < n; ++i) {
        Mat3 R = poses[i].rotation.toRotationMatrix();
        Eigen::Matrix<double, 3, 4> P;
        P.block<3,3>(0,0) = R;
        P.col(3) = poses[i].translation;
        P = cameras[i].K() * P;

        A.row(2*i)     = observations[i].x() * P.row(2) - P.row(0);
        A.row(2*i + 1) = observations[i].y() * P.row(2) - P.row(1);
    }

    Eigen::JacobiSVD<MatX> svd(A, Eigen::ComputeFullV);
    Vec4 X = svd.matrixV().col(3);

    return X.head<3>() / X[3];
}

// ─── Triangulation angle ────────────────────
double triangulation_angle(const Vec3& point3d,
                           const Vec3& center1,
                           const Vec3& center2) {
    Vec3 ray1 = (point3d - center1).normalized();
    Vec3 ray2 = (point3d - center2).normalized();
    double cos_angle = ray1.dot(ray2);
    cos_angle = std::clamp(cos_angle, -1.0, 1.0);
    return std::acos(cos_angle) * 180.0 / M_PI;
}

// ─── Batch two-view triangulation ───────────
TriangulationResult triangulate_two_view(
        const CameraPose& pose1,
        const CameraPose& pose2,
        const CameraIntrinsics& cam1,
        const CameraIntrinsics& cam2,
        const std::vector<Vec2>& pts1,
        const std::vector<Vec2>& pts2,
        double max_reprojection_error,
        double min_triangulation_angle_deg) {

    TriangulationResult result;
    size_t n = pts1.size();
    result.points.resize(n);
    result.reprojection_errors.resize(n, 0.0);
    result.valid.resize(n, false);

    Vec3 center1 = pose1.center();
    Vec3 center2 = pose2.center();

    double total_error = 0.0;
    int valid_count = 0;

    for (size_t i = 0; i < n; ++i) {
        Vec3 pt3d = triangulate_point_linear(
            pose1, pose2, cam1, cam2, pts1[i], pts2[i]);
        result.points[i] = pt3d;

        // Check depth (must be in front of both cameras)
        Vec3 p1_cam = pose1.transform(pt3d);
        Vec3 p2_cam = pose2.transform(pt3d);
        if (p1_cam.z() <= 0 || p2_cam.z() <= 0) continue;

        // Check triangulation angle
        double angle = triangulation_angle(pt3d, center1, center2);
        if (angle < min_triangulation_angle_deg) continue;

        // Reprojection error
        Vec2 proj1 = cam1.project(p1_cam);
        Vec2 proj2 = cam2.project(p2_cam);
        double err1 = (proj1 - pts1[i]).norm();
        double err2 = (proj2 - pts2[i]).norm();
        double err = (err1 + err2) / 2.0;

        result.reprojection_errors[i] = err;

        if (err < max_reprojection_error) {
            result.valid[i] = true;
            total_error += err;
            valid_count++;
        }
    }

    result.mean_error = valid_count > 0 ? total_error / valid_count : 0.0;

    std::cout << "[tri] Triangulated " << valid_count << "/" << n
              << " points, mean reproj error: " << result.mean_error << " px\n";

    return result;
}

}  // namespace geometry
}  // namespace chisel
