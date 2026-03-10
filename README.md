# Chisel — Multi-View 3D Reconstruction Pipeline

End-to-end 3D reconstruction from multi-view images, evaluated on the [ETH3D benchmark](https://www.eth3d.net/).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Python Orchestrator                        │
│              (pipeline.py — configs, I/O, evaluation)           │
├──────────────────┬──────────────────┬───────────────────────────┤
│  Perception      │   Geometry       │   Reconstruction          │
│  (Python/PyTorch)│   (C++ / Ceres / │   (C++ / OpenCV / PCL)   │
│                  │   GTSAM)         │                           │
│  • SuperPoint    │  • 5-pt E matrix │  • Plane-sweep stereo    │
│  • SIFT          │  • PnP RANSAC    │  • PatchMatch MVS        │
│  • LightGlue     │  • Triangulation │  • TSDF fusion           │
│  • Mono depth    │  • Bundle Adj.   │  • Marching cubes mesh   │
│                  │  • Factor graphs │  • PLY / OBJ export      │
└──────────────────┴──────────────────┴───────────────────────────┘
         │                   │                      │
         └──── pybind11 ─────┴──────────────────────┘
```

## Stack

| Layer | Language | Libraries |
|-------|----------|-----------|
| **Perception** | Python 3.9+ | PyTorch, OpenCV |
| **Geometry** | C++17 | Eigen3, OpenCV, Ceres Solver, GTSAM |
| **Reconstruction** | C++17 | Eigen3, OpenCV, PCL (optional) |
| **Bindings** | C++/Python | pybind11 |
| **Evaluation** | Python | NumPy, SciPy |

---

## Quick Start

### 1. Clone the repo

```bash
git clone <repo-url> chisel
cd chisel
```

### 2. Install system dependencies

#### macOS

```bash
brew install cmake eigen opencv ceres-solver gtsam pybind11
# Optional: point cloud support
brew install pcl
```

#### Linux (Ubuntu / Debian)

```bash
sudo apt install -y \
    libeigen3-dev libopencv-dev libceres-dev \
    libgtsam-dev libpcl-dev pybind11-dev \
    cmake build-essential python3-dev
```

#### Windows

- [CMake 3.18+](https://cmake.org/download/)
- [Visual Studio 2022](https://visualstudio.microsoft.com/) with "Desktop development with C++" workload
- [vcpkg](https://vcpkg.io/) for C++ dependencies

```powershell
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg install eigen3 opencv ceres gtsam pybind11
```

### 3. Install Python dependencies

```bash
pip install torch torchvision opencv-python numpy scipy pyyaml matplotlib
```

> **Using conda or an existing Python environment?**
> Activate your env first, then run the build script. It detects the active Python (including conda envs), prints it so you can confirm, and registers `chisel` as an editable package with `--no-deps` — meaning it will **not** reinstall packages already present in your env (PyTorch, OpenCV, NumPy, etc.). Only dev tools that are genuinely missing (`pytest`, `ruff`, etc.) are installed.
> ```bash
> conda activate my-env
> ./build.sh
> ```

### 4. Build

#### macOS / Linux

```bash
chmod +x build.sh
./build.sh            # builds C++ libraries + installs Python package
./build.sh --cpp      # C++ only
./build.sh --python   # Python package only
./build.sh --test     # build + run all tests
```

> **macOS + CMake path:** If CMake cannot find Homebrew packages:
> ```bash
> cmake -DCMAKE_PREFIX_PATH="$(brew --prefix)" ..
> ```

#### Windows

```powershell
mkdir build
cd build
cmake .. `
  -DCMAKE_TOOLCHAIN_FILE="<path-to-vcpkg>/scripts/buildsystems/vcpkg.cmake" `
  -DCMAKE_BUILD_TYPE=Release `
  -DBUILD_TESTS=ON `
  -DBUILD_PYTHON_BINDINGS=ON
cmake --build . --config Release
cd ..
pip install -e ".[dev]"
```

### 5. Download pretrained weights (SuperPoint / LightGlue)

Required if you use `--extractor superpoint` or `--matcher lightglue`:

```bash
python scripts/download_weights.py
# Force re-download
python scripts/download_weights.py --force
```

Weights are saved to `weights/` in the project root.

### 6. Download ETH3D data

```bash
python scripts/download_eth3d.py --output /data/eth3d --scenes courtyard delivery_area
```

---

## Run Reconstruction

```bash
# Full pipeline — SIFT features, Ceres BA (default)
python scripts/run_pipeline.py --scene /data/eth3d/training/courtyard

# SuperPoint features + LightGlue matcher
python scripts/run_pipeline.py \
    --scene /data/eth3d/training/courtyard \
    --extractor superpoint \
    --matcher lightglue \
    --optimizer ceres \
    --max-dim 1600 \
    --output ./output

# GTSAM pose graph instead of full Ceres BA
python scripts/run_pipeline.py \
    --scene /data/eth3d/training/courtyard \
    --optimizer gtsam

# Skip dense reconstruction (SfM only)
python scripts/run_pipeline.py \
    --scene /data/eth3d/training/courtyard \
    --no-dense

# Using a config file
python scripts/run_pipeline.py \
    --config configs/default.yaml \
    --scene /data/eth3d/training/courtyard
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--scene` | — | Path to scene directory |
| `--dataset` + `--scene-name` | — | Alternative: ETH3D root + scene name |
| `--extractor` | `sift` | Feature extractor: `sift`, `superpoint` |
| `--matcher` | `nn` | Feature matcher: `nn`, `lightglue` |
| `--optimizer` | `ceres` | BA optimizer: `ceres`, `gtsam` |
| `--max-dim` | `1600` | Max image dimension (px) |
| `--max-keypoints` | `4096` | Max keypoints per image |
| `--no-dense` | off | Skip dense reconstruction |
| `--output` | `./output` | Output directory |
| `--device` | `auto` | Compute device: `auto`, `cpu`, `cuda` |

---

## Evaluate

```bash
# All scenes in output directory
python scripts/run_eval.py --results ./output --dataset /data/eth3d

# Single scene
python scripts/run_eval.py \
    --recon-ply ./output/courtyard/reconstruction.ply \
    --gt-ply /data/eth3d/training/courtyard/points3D.txt
```

---

## Run Tests

```bash
# All tests (C++ + Python)
./build.sh --test

# Python tests only
python -m pytest tests/python/ -v --tb=short

# C++ tests only (macOS/Linux)
./build/tests/cpp/test_geometry --gtest_color=yes

# C++ tests (Windows)
.\build\tests\cpp\Release\test_geometry.exe
```

---

## Project Structure

```
chisel/
├── CMakeLists.txt              # C++ build system
├── build.sh                    # Build script (macOS / Linux)
├── pyproject.toml              # Python package config
├── configs/
│   └── default.yaml            # Pipeline configuration
├── weights/                    # Pretrained model weights (downloaded)
├── src/
│   ├── core/                   # Core types, I/O, mesh utils
│   │   ├── types.h             # CameraIntrinsics, CameraPose, Scene, etc.
│   │   ├── io_eth3d.{h,cpp}    # ETH3D + COLMAP format reader/writer
│   │   └── mesh_utils.cpp      # Triangle mesh normals
│   ├── geometry/               # Geometric reconstruction (C++)
│   │   ├── feature_matching    # SIFT extraction + ratio test + F-matrix
│   │   ├── epipolar            # Essential/fundamental matrix estimation
│   │   ├── triangulation       # DLT + multi-view triangulation
│   │   ├── pose_estimation     # PnP RANSAC + incremental SfM
│   │   ├── bundle_adjustment   # Ceres Solver (Schur complement BA)
│   │   └── factor_graph        # GTSAM pose graph + landmark optimization
│   ├── reconstruction/         # Dense reconstruction (C++)
│   │   ├── dense_stereo        # Plane-sweep + PatchMatch MVS
│   │   ├── depth_fusion        # TSDF volume integration
│   │   └── meshing             # Surface extraction + PLY/OBJ export
│   └── bindings/               # pybind11 Python bridge
│       └── bindings.cpp
├── python/chisel/              # Python package
│   ├── perception/             # PyTorch perception layer
│   │   ├── feature_extractor   # SuperPoint + SIFT
│   │   ├── feature_matcher     # NN + LightGlue attention matcher
│   │   └── depth_estimator     # Monocular depth (DPT-style)
│   ├── data/                   # Dataset loaders
│   │   └── eth3d_dataset       # ETH3D loader + downloader
│   ├── eval/                   # Evaluation
│   │   └── metrics             # ETH3D metrics (accuracy, completeness, F1)
│   ├── utils/                  # Visualization + helpers
│   └── pipeline.py             # Main orchestrator
├── scripts/
│   ├── run_pipeline.py         # CLI entry point
│   ├── run_eval.py             # Evaluation runner
│   ├── download_eth3d.py       # Dataset downloader
│   └── download_weights.py     # Pretrained weight downloader
└── tests/
    ├── cpp/test_geometry.cpp   # GTest C++ unit tests
    └── python/test_pipeline.py # pytest Python unit tests
```

---

## Pipeline Stages

| Phase | What happens |
|-------|-------------|
| **1 — Feature extraction** | SIFT or SuperPoint keypoints + descriptors for every image |
| **2 — Feature matching** | Sequential window matching (each image matched against ±N neighbours). Ratio test + geometric verification (F-matrix RANSAC) |
| **3 — Structure from Motion** | Init pair selected by triangulation angle. Incremental PnP registration. Triangulation of new points. Periodic + final Ceres/GTSAM bundle adjustment |
| **4 — Dense reconstruction** | C++ PatchMatch MVS (if built), otherwise OpenCV SGBM stereo fallback. Outputs depth maps |
| **5 — Meshing & export** | TSDF fusion → marching cubes → PLY point cloud / OBJ mesh |
| **6 — Evaluation** | ETH3D accuracy / completeness / F1 against ground truth. ATE and RPE for camera trajectories |

---

## Design Notes

### Ceres Solver vs GTSAM — why both?

They both solve nonlinear optimization problems and overlap in capability, but are designed for different levels of abstraction.

**Ceres Solver** is a general-purpose nonlinear least-squares library. You give it variables (a flat array of numbers) and cost functions that produce residuals, and it finds the values that minimize the sum of squared residuals. It has no concept of "poses" or "cameras" — it just sees numbers. In Chisel it is used for **bundle adjustment (BA)**: jointly optimizing all camera poses (6-DoF each) and all 3D point positions (3-DoF each) at once, by minimizing reprojection errors across every observation. Because BA has a very specific sparse block structure (each 3D point only connects to the cameras that see it), Ceres can exploit this with the Schur complement trick to make it fast — this is what `bundle_adjustment.cpp` does.

**GTSAM** operates one level higher. Instead of variables and raw residuals, you work with a **factor graph**: typed variable nodes (e.g. `Pose3`, `Point3`) connected by factor nodes that encode probabilistic measurements (relative transforms, landmark observations, IMU readings). GTSAM understands the geometry of poses — you add a "between factor" for a relative transform and it handles the Lie group math for you. It also supports **incremental updates** via iSAM2: when a new camera is added, it updates only the affected part of the graph without re-solving everything from scratch. In Chisel it is used for **pose graph optimization** in `factor_graph.cpp`: only camera poses are nodes (3D points are marginalized out), and edges are relative pose measurements between cameras. This makes it faster than full BA for large scenes, and the natural choice if you later add loop-closure constraints.

| | Ceres | GTSAM |
|---|---|---|
| **Abstraction** | Raw variables + residuals | Typed pose/point nodes + factors |
| **Geometry awareness** | None — you write it | Built-in Lie group / SE(3) |
| **What Chisel uses it for** | Full bundle adjustment (cameras + points) | Pose graph (cameras only, points marginalized) |
| **When to prefer it** | Maximum accuracy, fine control over cost functions | Large-scale, incremental, loop closure, SLAM |
| **Incremental updates** | No — re-solves from scratch | Yes — iSAM2 updates only changed nodes |

In short: they are **complementary, not redundant**. Ceres is the precise optimizer used after SfM; GTSAM is the scalable, pose-aware alternative suited for real-time or large-scale scenarios. Switch between them with `--optimizer ceres | gtsam`.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | % of reconstructed points within τ cm of ground truth |
| **Completeness** | % of ground truth points within τ cm of reconstruction |
| **F1 Score** | Harmonic mean of accuracy and completeness |
| **ATE RMSE** | Absolute Trajectory Error for camera poses |
| **RPE** | Relative Pose Error (rotation + translation) |

---

## License

MIT
