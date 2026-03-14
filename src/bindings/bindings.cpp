// src/bindings/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include "core/types.h"
#include "core/io_eth3d.h"
#include "geometry/bundle_adjustment.h"
#include "geometry/pose_estimation.h"
#include "geometry/factor_graph.h"
#include "reconstruction/dense_stereo.h"
#include "reconstruction/depth_fusion.h"

// Opaque map types: Python gets reference view, not a dict copy.
PYBIND11_MAKE_OPAQUE(std::map<chisel::ImageId,   chisel::ImageData>);
PYBIND11_MAKE_OPAQUE(std::map<chisel::CameraId,  chisel::CameraIntrinsics>);
PYBIND11_MAKE_OPAQUE(std::map<chisel::Point3DId, chisel::Point3D>);
PYBIND11_MAKE_OPAQUE(std::map<chisel::ImageId,   chisel::DepthMap>);

namespace py = pybind11;
using namespace chisel;

PYBIND11_MODULE(_chisel_cpp, m) {
    m.doc() = "Chisel – C++ backend for 3D reconstruction";

    py::bind_map<std::map<ImageId,   ImageData>>(m,       "ImageMap");
    py::bind_map<std::map<CameraId,  CameraIntrinsics>>(m,"CameraMap");
    py::bind_map<std::map<Point3DId, Point3D>>(m,         "PointMap");
    py::bind_map<std::map<ImageId,   DepthMap>>(m,        "DepthMapMap");

    // --- Core types ---
    py::class_<CameraIntrinsics>(m, "CameraIntrinsics")
        .def(py::init<>())
        .def_readwrite("id", &CameraIntrinsics::id)
        .def_readwrite("fx", &CameraIntrinsics::fx)
        .def_readwrite("fy", &CameraIntrinsics::fy)
        .def_readwrite("cx", &CameraIntrinsics::cx)
        .def_readwrite("cy", &CameraIntrinsics::cy)
        .def("K", &CameraIntrinsics::K);

    py::class_<CameraPose>(m, "CameraPose")
        .def(py::init<>())
        .def_readwrite("rotation", &CameraPose::rotation)
        .def_readwrite("translation", &CameraPose::translation)
        .def_property("R",
            [](const CameraPose &p) { return p.rotation.toRotationMatrix(); },
            [](CameraPose &p, const Eigen::Matrix3d &R) { p.rotation = Eigen::Quaterniond(R); }
        )
        .def("matrix", &CameraPose::matrix);

    py::class_<Point3D::TrackElement>(m, "TrackElement")
        .def(py::init<>())
        .def(py::init<int, int>())
        .def_readwrite("image_id", &Point3D::TrackElement::image_id)
        .def_readwrite("feature_idx", &Point3D::TrackElement::feature_idx);

    py::class_<Keypoint>(m, "Keypoint")
        .def(py::init<>())
        .def_readwrite("xy", &Keypoint::xy);

    py::class_<chisel::ImageData>(m, "Image")
        .def(py::init<>())
        .def_readwrite("id", &chisel::ImageData::id)
        .def_readwrite("name", &chisel::ImageData::name)
        .def_readwrite("camera_id", &chisel::ImageData::camera_id)
        .def_readwrite("pose", &chisel::ImageData::pose)
        .def_readwrite("pose_valid", &chisel::ImageData::pose_valid)
        .def_readwrite("keypoints", &chisel::ImageData::keypoints)
        .def_readwrite("point3d_ids", &chisel::ImageData::point3d_ids)
        // Set pixel data from H×W×C uint8 numpy array (BGR)
        .def("set_image",
            [](chisel::ImageData& img,
               py::array_t<uint8_t, py::array::c_style | py::array::forcecast> arr) {
                if (arr.ndim() == 3) {
                    int h = arr.shape(0), w = arr.shape(1), c = arr.shape(2);
                    int type = (c == 3) ? CV_8UC3 : (c == 4 ? CV_8UC4 : CV_8UC1);
                    img.image = cv::Mat(h, w, type, (void*)arr.data()).clone();
                } else if (arr.ndim() == 2) {
                    int h = arr.shape(0), w = arr.shape(1);
                    img.image = cv::Mat(h, w, CV_8U, (void*)arr.data()).clone();
                }
            })
        .def_property_readonly("image_loaded",
            [](const chisel::ImageData& img) { return !img.image.empty(); });

    py::class_<Point3D>(m, "Point3D")
        .def(py::init<>())
        .def_readwrite("id", &chisel::Point3D::id)
        .def_readwrite("xyz", &Point3D::xyz)
        .def_readwrite("track", &Point3D::track);

    // --- Dense depth map ---
    py::class_<DepthMap>(m, "DepthMap")
        .def(py::init<>())
        .def_readwrite("image_id", &DepthMap::image_id)
        .def_readwrite("width",    &DepthMap::width)
        .def_readwrite("height",   &DepthMap::height)
        .def_property("depth",
            [](const DepthMap& dm) -> py::array_t<float> {
                if (dm.depth.empty())
                    return py::array_t<float>({0, 0});
                return py::array_t<float>(
                    {(py::ssize_t)dm.height, (py::ssize_t)dm.width},
                    dm.depth.data());
            },
            [](DepthMap& dm, py::array_t<float> arr) {
                auto r = arr.unchecked<2>();
                dm.height = r.shape(0);
                dm.width  = r.shape(1);
                dm.depth.assign(r.data(0, 0),
                                r.data(0, 0) + (size_t)r.shape(0) * r.shape(1));
            })
        .def_property("confidence",
            [](const DepthMap& dm) -> py::array_t<float> {
                if (dm.confidence.empty())
                    return py::array_t<float>({0, 0});
                return py::array_t<float>(
                    {(py::ssize_t)dm.height, (py::ssize_t)dm.width},
                    dm.confidence.data());
            },
            [](DepthMap& dm, py::array_t<float> arr) {
                auto r = arr.unchecked<2>();
                dm.confidence.assign(r.data(0, 0),
                                     r.data(0, 0) + (size_t)r.shape(0) * r.shape(1));
            })
        .def("valid_count",
            [](const DepthMap& dm) {
                return std::count_if(dm.depth.begin(), dm.depth.end(),
                                     [](float v){ return v > 0.f; });
            });

    // --- Scene ---
    py::class_<Scene>(m, "Scene")
        .def(py::init<>())
        .def_readwrite("images",     &Scene::images)
        .def_readwrite("cameras",    &Scene::cameras)
        .def_readwrite("points3d",   &Scene::points3d)
        .def_readwrite("depth_maps", &Scene::depth_maps)
        .def("num_registered", &Scene::num_registered);

    // --- Geometry & BA ---
    using namespace geometry;

    py::class_<BundleAdjustmentConfig>(m, "BundleAdjustmentConfig")
        .def(py::init<>())
        .def_readwrite("max_iterations",  &BundleAdjustmentConfig::max_iterations)
        .def_readwrite("verbose",         &BundleAdjustmentConfig::verbose)
        .def_readwrite("huber_loss_scale",&geometry::BundleAdjustmentConfig::huber_loss_scale)
        .def_readwrite("fix_intrinsics",  &geometry::BundleAdjustmentConfig::fix_intrinsics)
        .def_readwrite("fix_first_pose",  &geometry::BundleAdjustmentConfig::fix_first_pose);

    py::class_<BundleAdjustmentReport>(m, "BundleAdjustmentReport")
        .def_readonly("mean_reproj_error", &BundleAdjustmentReport::mean_reproj_error)
        .def_readonly("success",           &BundleAdjustmentReport::success);

    m.def("run_bundle_adjustment", &run_bundle_adjustment);

    // --- I/O ---
    py::class_<io::ETH3DConfig>(m, "ETH3DConfig")
        .def(py::init<>())
        .def_readwrite("dataset_root", &io::ETH3DConfig::dataset_root);

    m.def("load_eth3d_scene", &io::load_eth3d_scene);

    // --- GTSAM factor graph ---
    py::class_<geometry::FactorGraphConfig>(m, "FactorGraphConfig")
        .def(py::init<>())
        .def_readwrite("max_iterations",    &geometry::FactorGraphConfig::max_iterations)
        .def_readwrite("prior_rot_sigma",   &geometry::FactorGraphConfig::prior_rot_sigma)
        .def_readwrite("prior_trans_sigma", &geometry::FactorGraphConfig::prior_trans_sigma)
        .def_readwrite("pixel_sigma",       &geometry::FactorGraphConfig::pixel_sigma)
        .def_readwrite("verbose",           &geometry::FactorGraphConfig::verbose);

    py::class_<geometry::FactorGraphReport>(m, "FactorGraphReport")
        .def_readonly("initial_error", &geometry::FactorGraphReport::initial_error)
        .def_readonly("final_error",   &geometry::FactorGraphReport::final_error)
        .def_readonly("success",       &geometry::FactorGraphReport::converged);

    m.def("optimize_full_graph", &geometry::optimize_full_graph);

    // --- Dense stereo config ---
    using namespace reconstruction;

    py::class_<DenseStereoConfig>(m, "DenseStereoConfig")
        .def(py::init<>())
        .def_readwrite("min_depth",                   &DenseStereoConfig::min_depth)
        .def_readwrite("max_depth",                   &DenseStereoConfig::max_depth)
        .def_readwrite("num_depth_samples",           &DenseStereoConfig::num_depth_samples)
        .def_readwrite("patch_size",                  &DenseStereoConfig::patch_size)
        .def_readwrite("num_iterations",              &DenseStereoConfig::num_iterations)
        .def_readwrite("ncc_threshold",               &DenseStereoConfig::ncc_threshold)
        .def_readwrite("num_source_images",           &DenseStereoConfig::num_source_images)
        .def_readwrite("geometric_consistency_thresh",&DenseStereoConfig::geometric_consistency_thresh)
        .def_readwrite("min_consistent_views",        &DenseStereoConfig::min_consistent_views)
        .def_readwrite("confidence_threshold",        &DenseStereoConfig::confidence_threshold)
        .def_readwrite("filter_by_consistency",       &DenseStereoConfig::filter_by_consistency);

    // --- Depth fusion config ---
    py::class_<FusionConfig>(m, "FusionConfig")
        .def(py::init<>())
        .def_readwrite("min_confidence", &FusionConfig::min_confidence)
        .def_readwrite("depth_max",      &FusionConfig::depth_max)
        .def_readwrite("subsample",      &FusionConfig::subsample);

    // --- Dense stereo functions ---
    m.def("select_source_images",
          &select_source_images,
          py::arg("scene"), py::arg("ref_id"), py::arg("num_sources"));

    m.def("compute_depth_plane_sweep",
          &compute_depth_plane_sweep,
          py::arg("scene"), py::arg("ref_id"), py::arg("source_ids"),
          py::arg("cfg") = DenseStereoConfig());

    m.def("compute_all_depth_maps",
          &compute_all_depth_maps,
          py::arg("scene"), py::arg("cfg") = DenseStereoConfig());

    m.def("filter_depth_consistency",
          &filter_depth_consistency,
          py::arg("depth_maps"), py::arg("scene"),
          py::arg("cfg") = DenseStereoConfig());

    // --- Fusion: returns (points Nx3 float64, colors Nx3 float64) ---
    m.def("fuse_scene_depth_maps",
        [](const Scene& scene, const FusionConfig& cfg) {
            std::vector<Vec3> pts, cols;
            fuse_depth_to_pointcloud(scene, scene.depth_maps, pts, cols, cfg);

            const py::ssize_t n = static_cast<py::ssize_t>(pts.size());
            auto pts_arr = py::array_t<double>({n, (py::ssize_t)3});
            auto col_arr = py::array_t<double>({n, (py::ssize_t)3});

            auto p = pts_arr.mutable_unchecked<2>();
            auto c = col_arr.mutable_unchecked<2>();
            for (py::ssize_t i = 0; i < n; ++i) {
                p(i, 0) = pts[i].x();  p(i, 1) = pts[i].y();  p(i, 2) = pts[i].z();
                c(i, 0) = cols[i].x(); c(i, 1) = cols[i].y(); c(i, 2) = cols[i].z();
            }
            return py::make_tuple(pts_arr, col_arr);
        },
        py::arg("scene"), py::arg("cfg") = FusionConfig());
}
