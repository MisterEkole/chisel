// src/core/io_colmap.cpp  –  Additional COLMAP binary format support
#include "core/io_eth3d.h"
#include <fstream>
#include <iostream>

namespace chisel {
namespace io {

// COLMAP binary format readers (for .bin files)
// These complement the text readers in io_eth3d.cpp

bool read_colmap_cameras_bin(const std::string& path,
                             std::map<CameraId, CameraIntrinsics>& cams) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) return false;

    uint64_t num_cameras;
    fin.read(reinterpret_cast<char*>(&num_cameras), sizeof(uint64_t));

    for (uint64_t i = 0; i < num_cameras; ++i) {
        CameraIntrinsics cam;
        int32_t model_id;
        fin.read(reinterpret_cast<char*>(&cam.id), sizeof(uint32_t));
        fin.read(reinterpret_cast<char*>(&model_id), sizeof(int32_t));
        uint64_t w, h;
        fin.read(reinterpret_cast<char*>(&w), sizeof(uint64_t));
        fin.read(reinterpret_cast<char*>(&h), sizeof(uint64_t));
        cam.width  = static_cast<uint32_t>(w);
        cam.height = static_cast<uint32_t>(h);

        // Number of params depends on model
        int num_params = 4;  // default pinhole
        switch (model_id) {
            case 0: num_params = 4; cam.model = CameraModel::PINHOLE; break;
            case 1: num_params = 3; cam.model = CameraModel::SIMPLE_RADIAL; break;
            case 2: num_params = 4; cam.model = CameraModel::RADIAL; break;
            case 4: num_params = 8; cam.model = CameraModel::OPENCV; break;
            default: num_params = 4; cam.model = CameraModel::PINHOLE; break;
        }

        std::vector<double> params(num_params);
        fin.read(reinterpret_cast<char*>(params.data()),
                 num_params * sizeof(double));

        if (cam.model == CameraModel::PINHOLE) {
            cam.fx = params[0]; cam.fy = params[1];
            cam.cx = params[2]; cam.cy = params[3];
        } else if (cam.model == CameraModel::SIMPLE_RADIAL) {
            cam.fx = cam.fy = params[0];
            cam.cx = params[1]; cam.cy = params[2];
            cam.distortion = {params.size() > 3 ? params[3] : 0.0};
        } else {
            cam.fx = params[0]; cam.fy = params[1];
            cam.cx = params[2]; cam.cy = params[3];
            cam.distortion.assign(params.begin() + 4, params.end());
        }

        cams[cam.id] = cam;
    }

    std::cout << "[io] Loaded " << cams.size() << " cameras (binary)\n";
    return true;
}

}  // namespace io
}  // namespace chisel
