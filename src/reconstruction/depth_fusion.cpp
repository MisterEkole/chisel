// src/reconstruction/depth_fusion.cpp
#include "reconstruction/depth_fusion.h"

#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace chisel {
namespace reconstruction {

// ─── Simple point cloud fusion ──────────────
void fuse_depth_to_pointcloud(
        const Scene& scene,
        const std::map<ImageId, DepthMap>& depth_maps,
        std::vector<Vec3>& points,
        std::vector<Vec3>& colors,
        const FusionConfig& cfg) {

    points.clear();
    colors.clear();

    for (auto& [id, dm] : depth_maps) {
        auto img_it = scene.images.find(id);
        if (img_it == scene.images.end()) continue;

        auto& img = img_it->second;
        auto& cam = scene.cameras.at(img.camera_id);

        CameraPose inv_pose = img.pose.inverse();

        for (uint32_t r = 0; r < dm.height; r += cfg.subsample) {
            for (uint32_t c = 0; c < dm.width; c += cfg.subsample) {
                if (!dm.valid(r, c)) continue;
                float d = dm.at(r, c);
                if (d > cfg.depth_max || dm.confidence[r * dm.width + c] < cfg.min_confidence)
                    continue;

                // Back-project to 3D
                Vec2 px(c, r);
                Vec3 ray = cam.unproject(px);
                Vec3 p_cam = ray * d;
                Vec3 p_world = inv_pose.transform(p_cam);

                points.push_back(p_world);

                // Color from image
                if (!img.image.empty()) {
                    int ir = std::clamp((int)r, 0, img.image.rows - 1);
                    int ic = std::clamp((int)c, 0, img.image.cols - 1);
                    auto bgr = img.image.at<cv::Vec3b>(ir, ic);
                    colors.push_back(Vec3(bgr[2], bgr[1], bgr[0]) / 255.0);
                } else {
                    colors.push_back(Vec3(0.5, 0.5, 0.5));
                }
            }
        }
    }

    std::cout << "[fusion] Fused " << points.size() << " points from "
              << depth_maps.size() << " depth maps\n";
}

// ─── TSDF Volume ────────────────────────────
void TSDFVolume::allocate(const Vec3& orig, const Vec3& size, double vsize) {
    origin = orig;
    voxel_size = vsize;
    nx = static_cast<int>(std::ceil(size.x() / vsize));
    ny = static_cast<int>(std::ceil(size.y() / vsize));
    nz = static_cast<int>(std::ceil(size.z() / vsize));

    size_t total = static_cast<size_t>(nx) * ny * nz;
    tsdf.assign(total, 1.0f);
    weight.assign(total, 0.0f);
    color.assign(total, Vec3::Zero());

    std::cout << "[tsdf] Allocated volume " << nx << "×" << ny << "×" << nz
              << " (" << total * 12 / (1024*1024) << " MB)\n";
}

void TSDFVolume::integrate(const DepthMap& dm,
                           const ImageData& img,
                           const CameraIntrinsics& cam) {

    Mat3 R = img.pose.rotation.toRotationMatrix();
    Vec3 t = img.pose.translation;

    for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                // Voxel center in world coordinates
                Vec3 p_world(
                    origin.x() + (ix + 0.5) * voxel_size,
                    origin.y() + (iy + 0.5) * voxel_size,
                    origin.z() + (iz + 0.5) * voxel_size);

                // Transform to camera frame
                Vec3 p_cam = R * p_world + t;
                if (p_cam.z() <= 0) continue;

                // Project to image
                Vec2 px = cam.project(p_cam);
                int u = static_cast<int>(std::round(px.x()));
                int v = static_cast<int>(std::round(px.y()));

                if (u < 0 || u >= (int)dm.width ||
                    v < 0 || v >= (int)dm.height) continue;

                float depth = dm.at(v, u);
                if (depth <= 0) continue;

                float sdf = depth - static_cast<float>(p_cam.z());
                float trunc = static_cast<float>(voxel_size * 4.0);
                if (sdf < -trunc) continue;

                float tsdf_val = std::min(1.0f, sdf / trunc);
                float w = 1.0f;

                // Running average update
                size_t idx = iz * ny * nx + iy * nx + ix;
                float old_w = weight[idx];
                float new_w = old_w + w;

                tsdf[idx] = (tsdf[idx] * old_w + tsdf_val * w) / new_w;
                weight[idx] = std::min(new_w, 100.0f);

                // Update color
                if (!img.image.empty()) {
                    int ir = std::clamp(v, 0, img.image.rows - 1);
                    int ic = std::clamp(u, 0, img.image.cols - 1);
                    auto bgr = img.image.at<cv::Vec3b>(ir, ic);
                    Vec3 c(bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0);
                    color[idx] = (color[idx] * old_w + c * w) / new_w;
                }
            }
        }
    }
}

// ─── Marching cubes (simplified) ────────────
TriangleMesh TSDFVolume::extract_mesh() const {
    TriangleMesh mesh;

    // Simplified marching cubes: extract vertices at zero-crossings
    for (int iz = 0; iz < nz - 1; ++iz) {
        for (int iy = 0; iy < ny - 1; ++iy) {
            for (int ix = 0; ix < nx - 1; ++ix) {
                // Check edges in x, y, z directions for sign changes
                auto idx = [&](int x, int y, int z) -> size_t {
                    return z * ny * nx + y * nx + x;
                };

                float v000 = tsdf[idx(ix, iy, iz)];
                float w000 = weight[idx(ix, iy, iz)];

                if (w000 < 1.0f) continue;

                // X edge
                float v100 = tsdf[idx(ix+1, iy, iz)];
                float w100 = weight[idx(ix+1, iy, iz)];
                if (w100 >= 1.0f && v000 * v100 < 0) {
                    float t = v000 / (v000 - v100);
                    Vec3 p(origin.x() + (ix + t) * voxel_size,
                           origin.y() + (iy + 0.5) * voxel_size,
                           origin.z() + (iz + 0.5) * voxel_size);
                    mesh.vertices.push_back(p);
                    Vec3 c = color[idx(ix, iy, iz)] * (1 - t) +
                             color[idx(ix+1, iy, iz)] * t;
                    mesh.vertex_colors.push_back(c);
                }

                // Y edge
                float v010 = tsdf[idx(ix, iy+1, iz)];
                float w010 = weight[idx(ix, iy+1, iz)];
                if (w010 >= 1.0f && v000 * v010 < 0) {
                    float t = v000 / (v000 - v010);
                    Vec3 p(origin.x() + (ix + 0.5) * voxel_size,
                           origin.y() + (iy + t) * voxel_size,
                           origin.z() + (iz + 0.5) * voxel_size);
                    mesh.vertices.push_back(p);
                    Vec3 c = color[idx(ix, iy, iz)] * (1 - t) +
                             color[idx(ix, iy+1, iz)] * t;
                    mesh.vertex_colors.push_back(c);
                }

                // Z edge
                float v001 = tsdf[idx(ix, iy, iz+1)];
                float w001 = weight[idx(ix, iy, iz+1)];
                if (w001 >= 1.0f && v000 * v001 < 0) {
                    float t = v000 / (v000 - v001);
                    Vec3 p(origin.x() + (ix + 0.5) * voxel_size,
                           origin.y() + (iy + 0.5) * voxel_size,
                           origin.z() + (iz + t) * voxel_size);
                    mesh.vertices.push_back(p);
                    Vec3 c = color[idx(ix, iy, iz)] * (1 - t) +
                             color[idx(ix, iy, iz+1)] * t;
                    mesh.vertex_colors.push_back(c);
                }
            }
        }
    }

    std::cout << "[tsdf] Extracted " << mesh.vertices.size()
              << " surface points\n";
    return mesh;
}

}  // namespace reconstruction
}  // namespace chisel
