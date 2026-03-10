#pragma once
// src/core/io_eth3d.h  –  ETH3D dataset loader
//
// Supports both the "multi-view" and "two-view" evaluation datasets.
// File layout expected:
//   <dataset>/images/          – source images (dslr or rig)
//   <dataset>/cameras.txt      – COLMAP-style intrinsics
//   <dataset>/images.txt       – COLMAP-style extrinsics
//   <dataset>/depthmaps/       – ground-truth depth (optional)
//   <dataset>/points3D.txt     – sparse GT cloud (optional)

#include "core/types.h"
#include <string>

namespace chisel {
namespace io {

struct ETH3DConfig {
    std::string dataset_root;
    bool load_images     = true;   // actually read pixels
    bool load_gt_depth   = false;  // load ground-truth depth maps
    bool load_gt_cloud   = false;  // load ground-truth 3-D points
    int  max_image_dim   = -1;     // resize longest edge (<=0 = no resize)
};

// Read an ETH3D scene into our Scene structure.
// Returns false on unrecoverable error.
bool load_eth3d_scene(const ETH3DConfig& cfg, Scene& scene);

// Write reconstruction results in ETH3D evaluation format.
bool write_eth3d_results(const std::string& output_dir,
                         const Scene& scene);

// ── helpers (also useful standalone) ──
bool read_colmap_cameras_txt(const std::string& path,
                             std::map<CameraId, CameraIntrinsics>& cams);

bool read_colmap_images_txt(const std::string& path,
                            std::map<ImageId, ImageData>& images);

bool read_colmap_points3d_txt(const std::string& path,
                              std::map<Point3DId, Point3D>& pts);

}  // namespace io
}  // namespace chisel
