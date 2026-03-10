add_library(chisel_core STATIC
    io_colmap.cpp
    io_eth3d.cpp
    mesh_utils.cpp
)
target_include_directories(chisel_core PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(chisel_core PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBS}
)
