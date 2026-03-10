add_library(chisel_geometry STATIC
    feature_matching.cpp
    epipolar.cpp
    triangulation.cpp
    pose_estimation.cpp
    bundle_adjustment.cpp
    factor_graph.cpp
)
target_include_directories(chisel_geometry PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(chisel_geometry PUBLIC
    chisel_core
    Eigen3::Eigen
    ${OpenCV_LIBS}
    Ceres::ceres
    gtsam
)
