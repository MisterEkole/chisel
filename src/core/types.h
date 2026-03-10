#pragma once
// src/core/types.h  –  Fundamental data types for Chisel
//
// Shared across geometry, reconstruction, and Python binding layers.

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace chisel {

// ───── Basic aliases ─────────────────────────
using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Vec4 = Eigen::Vector4d;
using Mat3 = Eigen::Matrix3d;
using Mat4 = Eigen::Matrix4d;
using MatX = Eigen::MatrixXd;
using VecX = Eigen::VectorXd;
using Quat = Eigen::Quaterniond;

using ImageId   = uint32_t;
using CameraId  = uint32_t;
using Point3DId = uint64_t;
using FeatureId  = uint32_t;

// ───── Camera intrinsics ─────────────────────
enum class CameraModel : uint8_t {
    PINHOLE = 0,
    RADIAL,
    OPENCV,       // k1,k2,p1,p2
    FULL_OPENCV,  // k1..k6,p1,p2
    SIMPLE_RADIAL
};

struct CameraIntrinsics {
    CameraId id = 0;
    CameraModel model = CameraModel::PINHOLE;
    uint32_t width  = 0;
    uint32_t height = 0;

    // Focal lengths and principal point
    double fx = 0.0, fy = 0.0;
    double cx = 0.0, cy = 0.0;

    // Distortion coefficients (up to 8)
    std::vector<double> distortion;

    // Convenience: 3×3 calibration matrix
    Mat3 K() const {
        Mat3 m = Mat3::Identity();
        m(0, 0) = fx;  m(1, 1) = fy;
        m(0, 2) = cx;  m(1, 2) = cy;
        return m;
    }

    // Project 3-D point in camera frame → pixel
    Vec2 project(const Vec3& p_cam) const {
        double u = fx * (p_cam.x() / p_cam.z()) + cx;
        double v = fy * (p_cam.y() / p_cam.z()) + cy;
        return {u, v};
    }

    // Un-project pixel → unit bearing vector
    Vec3 unproject(const Vec2& px) const {
        Vec3 ray;
        ray.x() = (px.x() - cx) / fx;
        ray.y() = (px.y() - cy) / fy;
        ray.z() = 1.0;
        return ray.normalized();
    }
};

// ───── Camera extrinsics (world → camera) ────
struct CameraPose {
    Quat rotation    = Quat::Identity();   // world → cam rotation
    Vec3 translation = Vec3::Zero();       // world → cam translation

    // Compose into 4×4 transformation
    Mat4 matrix() const {
        Mat4 T = Mat4::Identity();
        T.block<3,3>(0,0) = rotation.toRotationMatrix();
        T.block<3,1>(0,3) = translation;
        return T;
    }

    // Camera center in world coordinates
    Vec3 center() const {
        return -(rotation.inverse() * translation);
    }

    // Transform world point into camera frame
    Vec3 transform(const Vec3& pw) const {
        return rotation * pw + translation;
    }

    // Inverse pose (camera → world)
    CameraPose inverse() const {
        CameraPose inv;
        inv.rotation    = rotation.inverse();
        inv.translation = -(inv.rotation * translation);
        return inv;
    }
};

// ───── 2-D keypoint observation ──────────────
struct Keypoint {
    Vec2       xy;
    float      scale    = 0.f;
    float      angle    = 0.f;
    float      response = 0.f;
    FeatureId  id       = 0;
};

// ───── Feature descriptor (binary or float) ──
struct Descriptor {
    enum class Type : uint8_t { FLOAT32, UINT8 };
    Type type = Type::FLOAT32;
    std::vector<float>   float_data;   // used when type == FLOAT32
    std::vector<uint8_t> binary_data;  // used when type == UINT8
    int dim() const {
        return type == Type::FLOAT32
            ? static_cast<int>(float_data.size())
            : static_cast<int>(binary_data.size());
    }
};

// ───── Feature match ─────────────────────────
struct Match {
    FeatureId query_idx = 0;
    FeatureId train_idx = 0;
    float     distance  = 0.f;
};

// ───── Per-image data container ──────────────
struct ImageData {
    ImageId         id   = 0;
    CameraId        camera_id = 0;
    std::string     name;
    std::string     path;

    CameraPose      pose;
    bool            pose_valid = false;

    std::vector<Keypoint>   keypoints;
    std::vector<Descriptor> descriptors;

    // Sparse track: keypoint_idx → Point3DId (−1 = untracked)
    std::vector<int64_t> point3d_ids;

    cv::Mat image;  // lazily loaded

    void allocate_tracks() {
        point3d_ids.assign(keypoints.size(), -1);
    }
};

// ───── Sparse 3-D point ──────────────────────
struct Point3D {
    Point3DId id = 0;
    Vec3      xyz = Vec3::Zero();
    Vec3      color = Vec3::Zero();  // RGB [0,255]
    double    reprojection_error = 0.0;

    // Track: list of (image_id, keypoint_idx)
    struct TrackElement {
        ImageId   image_id;
        FeatureId feature_idx;
    };
    std::vector<TrackElement> track;

    int track_length() const { return static_cast<int>(track.size()); }
};

// ───── Dense depth map ───────────────────────
struct DepthMap {
    ImageId image_id = 0;
    uint32_t width  = 0;
    uint32_t height = 0;
    std::vector<float> depth;      // row-major, 0 = invalid
    std::vector<float> confidence; // [0,1]

    float at(int r, int c) const { return depth[r * width + c]; }
    float& at(int r, int c) { return depth[r * width + c]; }
    bool valid(int r, int c) const { return depth[r * width + c] > 0.f; }
};

// ───── Triangle mesh ─────────────────────────
struct TriangleMesh {
    std::vector<Vec3>                   vertices;
    std::vector<Vec3>                   vertex_colors;  // RGB [0,1]
    std::vector<Vec3>                   vertex_normals;
    std::vector<Eigen::Vector3i>        triangles;      // vertex indices

    size_t num_vertices()  const { return vertices.size(); }
    size_t num_triangles() const { return triangles.size(); }

    void compute_normals();  // implemented in mesh_utils.cpp
};

// ───── Scene: top-level container ────────────
struct Scene {
    std::map<CameraId,  CameraIntrinsics> cameras;
    std::map<ImageId,   ImageData>         images;
    std::map<Point3DId, Point3D>           points3d;

    // Dense outputs
    std::map<ImageId, DepthMap>  depth_maps;
    std::shared_ptr<TriangleMesh> mesh;

    size_t num_registered() const {
        size_t n = 0;
        for (auto& [id, img] : images) n += img.pose_valid;
        return n;
    }

    Point3DId next_point3d_id() const {
        Point3DId m = 0;
        for (auto& [id, _] : points3d) m = std::max(m, id);
        return m + 1;
    }
};

}  // namespace chisel
