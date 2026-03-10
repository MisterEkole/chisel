#pragma once
// src/reconstruction/depth_fusion.h  –  Depth map fusion into point cloud / TSDF
#include "core/types.h"
#include <vector>

namespace chisel {
namespace reconstruction {

struct FusionConfig {
    // TSDF volume
    double voxel_size   = 0.01;   // meters
    double trunc_margin = 0.04;   // truncation distance
    Vec3   volume_origin = Vec3(-1, -1, -1);
    Vec3   volume_size   = Vec3(2, 2, 2);     // meters

    // Point cloud fusion (simpler alternative)
    float  min_confidence = 0.3f;
    float  depth_max      = 50.0f;
    int    subsample      = 1;   // skip every N pixels
};

// Simple depth map → point cloud fusion (back-projection + merging)
void fuse_depth_to_pointcloud(
    const Scene& scene,
    const std::map<ImageId, DepthMap>& depth_maps,
    std::vector<Vec3>& points,
    std::vector<Vec3>& colors,
    const FusionConfig& cfg = FusionConfig());

// TSDF volume integration
struct TSDFVolume {
    std::vector<float> tsdf;
    std::vector<float> weight;
    std::vector<Vec3>  color;
    int nx, ny, nz;
    double voxel_size;
    Vec3 origin;

    void allocate(const Vec3& orig, const Vec3& size, double vsize);
    void integrate(const DepthMap& dm,
                   const ImageData& img,
                   const CameraIntrinsics& cam);

    // Extract surface via marching cubes
    TriangleMesh extract_mesh() const;
};

}  // namespace reconstruction
}  // namespace chisel
