# Chisel — Multi-View 3D Reconstruction

End-to-end pipeline: feature extraction → matching → SfM → dense reconstruction, evaluated on [ETH3D](https://www.eth3d.net/).

```
Perception (Python/PyTorch)  →  Geometry & Reconstruction (C++/Ceres/GTSAM)
         └──────────────── pybind11 bridge ─────────────────┘
```

---

## Requirements

| | |
|---|---|
| **Python** | 3.9+ |
| **C++** | C++17 compiler |
| **Build** | CMake 3.18+ |
| **C++ libs** | Eigen3, OpenCV, Ceres Solver, GTSAM, pybind11 |
| **Python libs** | PyTorch, OpenCV, NumPy, SciPy |

---

## Setup

### 1. Clone

```bash
git clone <repo-url> chisel && cd chisel
```

### 2. System dependencies

**macOS**
```bash
brew install cmake eigen opencv ceres-solver gtsam pybind11
```

**Linux (Ubuntu/Debian)**
```bash
sudo apt install -y cmake build-essential python3-dev \
    libeigen3-dev libopencv-dev libceres-dev libgtsam-dev pybind11-dev
```

**Windows**
Install [CMake](https://cmake.org/download/) and [Visual Studio 2022](https://visualstudio.microsoft.com/) (Desktop C++ workload), then use [vcpkg](https://vcpkg.io/):
```powershell
git clone https://github.com/microsoft/vcpkg && cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg install eigen3 opencv ceres gtsam pybind11
```

### 3. Python dependencies

```bash
pip install torch torchvision opencv-python numpy scipy pyyaml
```

### 4. Build

**macOS / Linux**
```bash
./build.sh            # C++ + Python package
./build.sh --test     # build + run tests
```

**Windows**
```powershell
mkdir build && cd build
cmake .. `
  -DCMAKE_TOOLCHAIN_FILE="<vcpkg-root>/scripts/buildsystems/vcpkg.cmake" `
  -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON
cmake --build . --config Release
cd .. && pip install -e ".[dev]"
```

> **macOS CMake tip:** If packages aren't found, add `-DCMAKE_PREFIX_PATH="$(brew --prefix)"` to the cmake command.

### 5. Download weights

Required for `--extractor superpoint` or `--matcher lightglue`:

```bash
python scripts/download_weights.py
```

Weights are saved to `weights/`.

### 6. Download ETH3D data

```bash
python scripts/download_eth3d.py --output /data/eth3d --scenes courtyard delivery_area
```

---

## Run

```bash
# Default: SIFT + NN matcher
python scripts/run_pipeline.py --scene /data/eth3d/training/courtyard

# SuperPoint + LightGlue
python scripts/run_pipeline.py \
    --scene /data/eth3d/training/courtyard \
    --extractor superpoint \
    --matcher lightglue

# GTSAM pose graph instead of Ceres BA
python scripts/run_pipeline.py --scene /data/eth3d/training/courtyard --optimizer gtsam

# SfM only (skip dense)
python scripts/run_pipeline.py --scene /data/eth3d/training/courtyard --no-dense
```

**All CLI options**

| Flag | Default | Description |
|------|---------|-------------|
| `--scene` | — | Path to scene directory |
| `--dataset` + `--scene-name` | — | ETH3D root + scene name (alternative to `--scene`) |
| `--extractor` | `sift` | `sift` or `superpoint` |
| `--matcher` | `nn` | `nn` or `lightglue` |
| `--optimizer` | `ceres` | `ceres` or `gtsam` |
| `--weights-dir` | `weights/` | Directory with pretrained weights |
| `--max-dim` | `1600` | Max image dimension (px) |
| `--max-keypoints` | `4096` | Max keypoints per image |
| `--no-dense` | off | Skip dense reconstruction |
| `--output` | `./output` | Output directory |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |

---

## Evaluate

```bash
python scripts/run_eval.py --results ./output --dataset /data/eth3d
```

---

## Tests

```bash
./build.sh --test                                  # all tests
python -m pytest tests/python/ -v                  # Python only
./build/tests/cpp/test_geometry                    # C++ only (macOS/Linux)
.\build\tests\cpp\Release\test_geometry.exe        # C++ only (Windows)
```

---

## Project structure

```
chisel/
├── src/                        # C++ core
│   ├── core/                   # Types, I/O, mesh utils
│   ├── geometry/               # Epipolar, triangulation, PnP, BA, factor graph
│   ├── reconstruction/         # Dense stereo, TSDF fusion, meshing
│   └── bindings/               # pybind11 bridge
├── python/chisel/              # Python package
│   ├── perception/             # SuperPoint, SIFT, LightGlue, depth estimation
│   ├── data/                   # ETH3D dataset loader
│   ├── eval/                   # Accuracy / completeness / F1 metrics
│   └── pipeline.py             # Main orchestrator
├── scripts/                    # CLI entry points
├── weights/                    # Pretrained weights (after download)
└── tests/                      # C++ (GTest) + Python (pytest) tests
```

---

## Pipeline stages

| Phase | Description |
|-------|-------------|
| **1 — Extraction** | Keypoints + descriptors per image (SIFT or SuperPoint) |
| **2 — Matching** | Sequential window matching with ratio test + F-matrix RANSAC |
| **3 — SfM** | Two-view init → incremental PnP registration → triangulation → bundle adjustment |
| **4 — Dense** | PatchMatch MVS (C++) or SGBM stereo fallback (OpenCV) |
| **5 — Evaluation** | ETH3D F1 score, ATE/RPE for camera poses |

---

## Optimizer choice: Ceres vs GTSAM

Both solve the same nonlinear optimization problems but at different levels of abstraction.

- **Ceres** (`--optimizer ceres`) — used for **full bundle adjustment**: jointly optimizes all camera poses and 3D points by minimizing reprojection error. Exploits the sparse block structure of BA via the Schur complement. Best accuracy.
- **GTSAM** (`--optimizer gtsam`) — used for **pose graph optimization**: only camera poses are optimized (3D points marginalized). Supports incremental updates via iSAM2. Better suited for large scenes or if you add loop-closure constraints later.

They are complementary. Use Ceres for maximum accuracy on small-to-medium scenes; GTSAM for scalability or SLAM-style workflows.

---

## License

MIT
