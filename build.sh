#!/usr/bin/env bash
# build.sh  –  Build Chisel (C++ libraries + Python package)
#
# Usage:
#   ./build.sh              # full build
#   ./build.sh --python     # Python package only
#   ./build.sh --cpp        # C++ only
#   ./build.sh --test       # build + run tests

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Detect number of CPU cores cross-platform
if command -v nproc &>/dev/null; then
    NUM_CORES=$(nproc)
elif command -v sysctl &>/dev/null; then
    NUM_CORES=$(sysctl -n hw.logicalcpu)
else
    NUM_CORES=4
fi

echo "══════════════════════════════════════════"
echo "  Chisel Build System"
echo "══════════════════════════════════════════"
echo ""

MODE="${1:-all}"

build_cpp() {
    echo "── Building C++ libraries ──"
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"

    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTS=ON \
        -DBUILD_PYTHON_BINDINGS=ON \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

    cmake --build . --parallel "${NUM_CORES}"

    echo "✓ C++ build complete"
    echo ""
}

build_python() {
    echo "── Setting up Python package ──"
    cd "${SCRIPT_DIR}"

    # Show active Python so the user can confirm the right env is active
    PYTHON_BIN="$(command -v python3 || command -v python)"
    echo "  Python : ${PYTHON_BIN}"
    echo "  Version: $(${PYTHON_BIN} --version 2>&1)"
    if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
        echo "  Conda  : ${CONDA_DEFAULT_ENV}"
    fi
    echo ""

    # Install only missing dependencies (--no-deps skips re-resolving packages
    # that are already present; editable install registers the chisel package).
    # pip's dependency resolver will skip packages whose version constraints
    # are already satisfied in the active environment.
    pip install -e "." --no-deps --quiet 2>/dev/null || \
    pip install -e "." --no-deps --break-system-packages --quiet

    # Install dev extras only if any are missing (pytest, ruff, etc.)
    MISSING_DEV=""
    for pkg in pytest black mypy ruff; do
        if ! ${PYTHON_BIN} -c "import ${pkg}" 2>/dev/null; then
            MISSING_DEV="${MISSING_DEV} ${pkg}"
        fi
    done
    if [ -n "${MISSING_DEV}" ]; then
        echo "  Installing missing dev tools:${MISSING_DEV}"
        pip install ${MISSING_DEV} --quiet 2>/dev/null || \
        pip install ${MISSING_DEV} --break-system-packages --quiet
    else
        echo "  Dev tools already present, skipping"
    fi

    echo "✓ Python package installed"
    echo ""
}

run_tests() {
    echo "── Running tests ──"

    # C++ tests
    if [ -f "${BUILD_DIR}/tests/cpp/test_geometry" ]; then
        echo "Running C++ tests..."
        "${BUILD_DIR}/tests/cpp/test_geometry" --gtest_color=yes
    fi

    # Python tests
    echo "Running Python tests..."
    cd "${SCRIPT_DIR}"
    python -m pytest tests/python/ -v --tb=short

    echo "✓ All tests passed"
}

case "${MODE}" in
    --cpp)
        build_cpp
        ;;
    --python)
        build_python
        ;;
    --test)
        build_cpp
        build_python
        run_tests
        ;;
    all|*)
        build_cpp
        build_python
        echo "══════════════════════════════════════════"
        echo "  Build complete!"
        echo ""
        echo "  Run tests:    ./build.sh --test"
        echo "  Run pipeline: python scripts/run_pipeline.py --scene /path/to/scene"
        echo "══════════════════════════════════════════"
        ;;
esac
