add_library(chisel_reconstruction STATIC
    dense_stereo.cpp
    depth_fusion.cpp
    meshing.cpp
)
target_include_directories(chisel_reconstruction PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(chisel_reconstruction PUBLIC
    chisel_core
    chisel_geometry
    Eigen3::Eigen
    ${OpenCV_LIBS}
)
if(PCL_FOUND)
    target_include_directories(chisel_reconstruction PRIVATE ${PCL_INCLUDE_DIRS})
    target_link_libraries(chisel_reconstruction PRIVATE ${PCL_LIBRARIES})
endif()
