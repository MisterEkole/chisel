cmake_minimum_required(VERSION 3.18)
project(Chisel VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build options
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_PYTHON_BINDINGS "Build pybind11 Python bindings" ON)
option(USE_CUDA "Enable CUDA support" OFF)

# ──────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────

# Eigen3 (header-only linear algebra)
find_package(Eigen3 5.0.1 REQUIRED)

# OpenCV (feature detection, image I/O)
find_package(OpenCV 4.0 REQUIRED COMPONENTS
    core imgproc imgcodecs features2d calib3d highgui)

# Ceres Solver (bundle adjustment, nonlinear optimization)
find_package(Ceres REQUIRED)

# GTSAM (factor graph optimization, SLAM backend)
find_package(GTSAM REQUIRED)

# Optional: PCL for point cloud processing
find_package(PCL QUIET COMPONENTS common io surface filters)
if(PCL_FOUND)
    message(STATUS "PCL found – enabling point cloud utilities")
    add_definitions(-DHAS_PCL)
endif()

# pybind11 for Python bindings
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 QUIET)
    if(NOT pybind11_FOUND)
        message(STATUS "pybind11 not found via find_package, trying subdirectory")
        if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/pybind11/CMakeLists.txt")
            add_subdirectory(third_party/pybind11)
        else()
            message(WARNING "pybind11 not found – Python bindings disabled")
            set(BUILD_PYTHON_BINDINGS OFF)
        endif()
    endif()
endif()

# ──────────────────────────────────────────────
# Core library
# ──────────────────────────────────────────────
add_subdirectory(src/core)
add_subdirectory(src/geometry)
add_subdirectory(src/reconstruction)

if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(src/bindings)
endif()

# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────
if(BUILD_TESTS)
    enable_testing()
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.14.0
        )
        FetchContent_MakeAvailable(googletest)
    endif()
    add_subdirectory(tests/cpp)
endif()

# ──────────────────────────────────────────────
# Install
# ──────────────────────────────────────────────
install(DIRECTORY src/core/ src/geometry/ src/reconstruction/
    DESTINATION include/chisel
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
