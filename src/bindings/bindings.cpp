// src/bindings/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>

#include "core/types.h"
#include "core/io_eth3d.h"
#include "geometry/bundle_adjustment.h"
#include "geometry/pose_estimation.h"
#include "geometry/factor_graph.h"

// Make these map types opaque so Python gets a reference view, not a copy.
// Without this, def_readwrite on std::map fields returns a Python dict copy
// and writes from Python don't propagate back into the C++ object.
PYBIND11_MAKE_OPAQUE(std::map<chisel::ImageId,   chisel::ImageData>);
PYBIND11_MAKE_OPAQUE(std::map<chisel::CameraId,  chisel::CameraIntrinsics>);
PYBIND11_MAKE_OPAQUE(std::map<chisel::Point3DId, chisel::Point3D>);

namespace py = pybind11;
using namespace chisel;

PYBIND11_MODULE(_chisel_cpp, m) {
    m.doc() = "Chisel – C++ backend for 3D reconstruction";

    // Bound map types — reference semantics so Python writes go into C++ objects
    py::bind_map<std::map<ImageId,   ImageData>>(m,       "ImageMap");
    py::bind_map<std::map<CameraId,  CameraIntrinsics>>(m,"CameraMap");
    py::bind_map<std::map<Point3DId, Point3D>>(m,         "PointMap");

    // --- Types de base ---
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

    // TrackElement est imbriqué dans Point3D
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
        .def_readwrite("point3d_ids", &chisel::ImageData::point3d_ids);

    py::class_<Point3D>(m, "Point3D")
        .def(py::init<>())
        .def_readwrite("id", &chisel::Point3D::id)
        .def_readwrite("xyz", &Point3D::xyz)
        .def_readwrite("track", &Point3D::track);

    py::class_<Scene>(m, "Scene")
        .def(py::init<>())
        .def_readwrite("images", &Scene::images)
        .def_readwrite("cameras", &Scene::cameras)
        .def_readwrite("points3d", &Scene::points3d)
        .def("num_registered", &Scene::num_registered);

    // --- Géométrie & BA ---
    using namespace geometry;

    py::class_<BundleAdjustmentConfig>(m, "BundleAdjustmentConfig")
        .def(py::init<>())
        .def_readwrite("max_iterations", &BundleAdjustmentConfig::max_iterations)
        .def_readwrite("verbose", &BundleAdjustmentConfig::verbose)
        .def_readwrite("huber_loss_scale", &geometry::BundleAdjustmentConfig::huber_loss_scale);

    py::class_<BundleAdjustmentReport>(m, "BundleAdjustmentReport")
        .def_readonly("mean_reproj_error", &BundleAdjustmentReport::mean_reproj_error)
        .def_readonly("success", &BundleAdjustmentReport::success);

    m.def("run_bundle_adjustment", &run_bundle_adjustment);

    // --- I/O ---
    py::class_<io::ETH3DConfig>(m, "ETH3DConfig")
        .def(py::init<>())
        .def_readwrite("dataset_root", &io::ETH3DConfig::dataset_root);

    m.def("load_eth3d_scene", &io::load_eth3d_scene);



    py::class_<geometry::FactorGraphConfig>(m, "FactorGraphConfig")
    .def(py::init<>())
    .def_readwrite("max_iterations", &geometry::FactorGraphConfig::max_iterations)
    .def_readwrite("prior_rot_sigma", &geometry::FactorGraphConfig::prior_rot_sigma)
    .def_readwrite("prior_trans_sigma", &geometry::FactorGraphConfig::prior_trans_sigma)
    .def_readwrite("pixel_sigma", &geometry::FactorGraphConfig::pixel_sigma)
    .def_readwrite("verbose", &geometry::FactorGraphConfig::verbose);

    py::class_<geometry::FactorGraphReport>(m, "FactorGraphReport")
    .def_readonly("initial_error", &geometry::FactorGraphReport::initial_error)
    .def_readonly("final_error", &geometry::FactorGraphReport::final_error)
    .def_readonly("success", &geometry::FactorGraphReport::converged);

    m.def("optimize_full_graph", &geometry::optimize_full_graph);
}