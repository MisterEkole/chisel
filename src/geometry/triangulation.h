#pragma once
// src/geometry/triangulation.h
#include "core/types.h"
#include <vector>

namespace chisel {
namespace geometry {

// Linear (DLT) triangulation of a single point from two views
Vec3 triangulate_point_linear(
    const CameraPose& pose1,
    const CameraPose& pose2,
    const CameraIntrinsics& cam1,
    const CameraIntrinsics& cam2,
    const Vec2& pt1,
    const Vec2& pt2);

// Multi-view linear triangulation (>= 2 views)
Vec3 triangulate_point_multiview(
    const std::vector<CameraPose>& poses,
    const std::vector<CameraIntrinsics>& cameras,
    const std::vector<Vec2>& observations);

// Triangulate many points from two views
struct TriangulationResult {
    std::vector<Vec3> points;
    std::vector<double> reprojection_errors;
    std::vector<bool> valid;  // behind-camera or large error
    double mean_error = 0.0;
};

TriangulationResult triangulate_two_view(
    const CameraPose& pose1,
    const CameraPose& pose2,
    const CameraIntrinsics& cam1,
    const CameraIntrinsics& cam2,
    const std::vector<Vec2>& pts1,
    const std::vector<Vec2>& pts2,
    double max_reprojection_error = 4.0,
    double min_triangulation_angle_deg = 1.5);

// Check triangulation angle between two views for a 3D point
double triangulation_angle(const Vec3& point3d,
                           const Vec3& camera_center1,
                           const Vec3& camera_center2);

}  // namespace geometry
}  // namespace chisel
