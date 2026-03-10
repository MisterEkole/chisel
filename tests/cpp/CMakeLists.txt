add_executable(test_geometry test_geometry.cpp)
target_link_libraries(test_geometry PRIVATE
    chisel_core
    chisel_geometry
    GTest::gtest_main
)
target_include_directories(test_geometry PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

include(GoogleTest)
gtest_discover_tests(test_geometry)
