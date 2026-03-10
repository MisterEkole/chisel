// src/core/io_eth3d.cpp  –  ETH3D & COLMAP format reader/writer
#include "core/io_eth3d.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace fs = std::filesystem;

namespace chisel {
namespace io {

// ─── trim utility ────────────────────────────
static std::string trim(const std::string& s) {
    auto a = s.find_first_not_of(" \t\r\n");
    auto b = s.find_last_not_of(" \t\r\n");
    return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

// ─── Parse COLMAP cameras.txt ────────────────
bool read_colmap_cameras_txt(const std::string& path,
                             std::map<CameraId, CameraIntrinsics>& cams) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "[io] Cannot open cameras file: " << path << "\n";
        return false;
    }
    std::string line;
    while (std::getline(fin, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);

        CameraIntrinsics cam;
        std::string model_str;
        ss >> cam.id >> model_str >> cam.width >> cam.height;

        if (model_str == "PINHOLE") {
            cam.model = CameraModel::PINHOLE;
            ss >> cam.fx >> cam.fy >> cam.cx >> cam.cy;
        } else if (model_str == "SIMPLE_RADIAL") {
            cam.model = CameraModel::SIMPLE_RADIAL;
            double f, cx_, cy_, k;
            ss >> f >> cx_ >> cy_ >> k;
            cam.fx = cam.fy = f;
            cam.cx = cx_;  cam.cy = cy_;
            cam.distortion = {k};
        } else if (model_str == "RADIAL") {
            cam.model = CameraModel::RADIAL;
            double f, cx_, cy_, k1, k2;
            ss >> f >> cx_ >> cy_ >> k1 >> k2;
            cam.fx = cam.fy = f;
            cam.cx = cx_;  cam.cy = cy_;
            cam.distortion = {k1, k2};
        } else if (model_str == "OPENCV") {
            cam.model = CameraModel::OPENCV;
            double k1, k2, p1, p2;
            ss >> cam.fx >> cam.fy >> cam.cx >> cam.cy >> k1 >> k2 >> p1 >> p2;
            cam.distortion = {k1, k2, p1, p2};
        } else {
            // Fallback: treat as pinhole with params
            cam.model = CameraModel::PINHOLE;
            ss >> cam.fx >> cam.fy >> cam.cx >> cam.cy;
        }

        cams[cam.id] = cam;
    }
    std::cout << "[io] Loaded " << cams.size() << " cameras\n";
    return true;
}

// ─── Parse COLMAP images.txt ─────────────────
bool read_colmap_images_txt(const std::string& path,
                            std::map<ImageId, ImageData>& images) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "[io] Cannot open images file: " << path << "\n";
        return false;
    }
    std::string line;
    while (std::getline(fin, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;

        // Line 1: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        std::istringstream ss(line);
        ImageData img;
        double qw, qx, qy, qz;
        ss >> img.id >> qw >> qx >> qy >> qz
           >> img.pose.translation.x()
           >> img.pose.translation.y()
           >> img.pose.translation.z()
           >> img.camera_id >> img.name;

        img.pose.rotation = Quat(qw, qx, qy, qz).normalized();
        img.pose_valid = true;

        // Line 2: POINTS2D[] as (X, Y, POINT3D_ID) – skip or parse
        std::string pts_line;
        if (std::getline(fin, pts_line)) {
            pts_line = trim(pts_line);
            if (!pts_line.empty()) {
                std::istringstream ps(pts_line);
                double px, py;
                int64_t pid;
                while (ps >> px >> py >> pid) {
                    Keypoint kp;
                    kp.xy = Vec2(px, py);
                    img.keypoints.push_back(kp);
                    img.point3d_ids.push_back(pid);
                }
            }
        }

        images[img.id] = std::move(img);
    }
    std::cout << "[io] Loaded " << images.size() << " images\n";
    return true;
}

// ─── Parse COLMAP points3D.txt ───────────────
bool read_colmap_points3d_txt(const std::string& path,
                              std::map<Point3DId, Point3D>& pts) {
    std::ifstream fin(path);
    if (!fin.is_open()) return false;

    std::string line;
    while (std::getline(fin, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        Point3D pt;
        double r, g, b, err;
        ss >> pt.id >> pt.xyz.x() >> pt.xyz.y() >> pt.xyz.z()
           >> r >> g >> b >> err;
        pt.color = Vec3(r, g, b);
        pt.reprojection_error = err;

        // Track entries: IMAGE_ID POINT2D_IDX pairs
        ImageId iid;
        FeatureId fidx;
        while (ss >> iid >> fidx) {
            pt.track.push_back({iid, fidx});
        }

        pts[pt.id] = std::move(pt);
    }
    std::cout << "[io] Loaded " << pts.size() << " 3D points\n";
    return true;
}

// ─── Load a full ETH3D scene ─────────────────
bool load_eth3d_scene(const ETH3DConfig& cfg, Scene& scene) {
    fs::path root(cfg.dataset_root);
    if (!fs::exists(root)) {
        std::cerr << "[io] Dataset root does not exist: " << cfg.dataset_root << "\n";
        return false;
    }

    // Camera intrinsics
    auto cam_path = root / "cameras.txt";
    if (fs::exists(cam_path)) {
        if (!read_colmap_cameras_txt(cam_path.string(), scene.cameras))
            return false;
    } else {
        // Try dslr_calibration_undistorted sub-directory
        cam_path = root / "dslr_calibration_undistorted" / "cameras.txt";
        if (!read_colmap_cameras_txt(cam_path.string(), scene.cameras))
            return false;
    }

    // Images (extrinsics + 2D observations)
    auto img_path = root / "images.txt";
    if (!fs::exists(img_path))
        img_path = root / "dslr_calibration_undistorted" / "images.txt";
    if (!read_colmap_images_txt(img_path.string(), scene.images))
        return false;

    // Wire up image file paths and optionally load pixels
    fs::path img_dir = root / "images";
    if (!fs::exists(img_dir))
        img_dir = root / "dslr_images_undistorted";
    for (auto& [id, img] : scene.images) {
        img.path = (img_dir / img.name).string();
        if (cfg.load_images && fs::exists(img.path)) {
            img.image = cv::imread(img.path, cv::IMREAD_COLOR);
            if (cfg.max_image_dim > 0 &&
                std::max(img.image.cols, img.image.rows) > cfg.max_image_dim) {
                double s = static_cast<double>(cfg.max_image_dim) /
                           std::max(img.image.cols, img.image.rows);
                cv::resize(img.image, img.image, cv::Size(), s, s);
            }
        }
    }

    // Optional: ground-truth point cloud
    if (cfg.load_gt_cloud) {
        auto pts_path = root / "points3D.txt";
        if (!fs::exists(pts_path))
            pts_path = root / "dslr_calibration_undistorted" / "points3D.txt";
        if (fs::exists(pts_path))
            read_colmap_points3d_txt(pts_path.string(), scene.points3d);
    }

    // Optional: ground-truth depth maps
    if (cfg.load_gt_depth) {
        fs::path dm_dir = root / "depthmaps";
        if (fs::exists(dm_dir)) {
            for (auto& entry : fs::directory_iterator(dm_dir)) {
                if (entry.path().extension() == ".bin" ||
                    entry.path().extension() == ".pfm") {
                    // TODO: implement PFM / binary depth reader
                    std::cout << "[io] Found GT depth: " << entry.path() << "\n";
                }
            }
        }
    }

    std::cout << "[io] ETH3D scene loaded: "
              << scene.cameras.size() << " cams, "
              << scene.images.size()  << " images, "
              << scene.points3d.size() << " GT points\n";
    return true;
}

// ─── Write results for ETH3D evaluation ──────
bool write_eth3d_results(const std::string& output_dir,
                         const Scene& scene) {
    fs::create_directories(output_dir);

    // Write point cloud as PLY
    auto ply_path = fs::path(output_dir) / "reconstruction.ply";
    std::ofstream ply(ply_path);
    if (!ply.is_open()) return false;

    size_t n = scene.points3d.size();
    ply << "ply\nformat ascii 1.0\n"
        << "element vertex " << n << "\n"
        << "property float x\nproperty float y\nproperty float z\n"
        << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        << "end_header\n";

    for (auto& [id, pt] : scene.points3d) {
        ply << pt.xyz.x() << " " << pt.xyz.y() << " " << pt.xyz.z() << " "
            << static_cast<int>(pt.color.x()) << " "
            << static_cast<int>(pt.color.y()) << " "
            << static_cast<int>(pt.color.z()) << "\n";
    }

    std::cout << "[io] Wrote " << n << " points to " << ply_path << "\n";
    return true;
}

}  // namespace io
}  // namespace chisel
