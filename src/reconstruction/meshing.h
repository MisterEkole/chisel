#pragma once
// src/reconstruction/meshing.h  –  Surface reconstruction
#include "core/types.h"

namespace chisel {
namespace reconstruction {

struct MeshingConfig {
    enum class Method {
        DELAUNAY,     // Delaunay-based visibility filtering
        POISSON,      // Screened Poisson (requires normals)
        BALL_PIVOT    // Ball pivoting algorithm
    };

    Method method = Method::DELAUNAY;

    // Delaunay parameters
    double max_edge_length = 0.1;     // meters
    double visibility_threshold = 2.0; // reprojection threshold

    // Poisson parameters
    int poisson_depth = 8;
    float poisson_trim = 7.0f;

    // General
    bool  compute_normals = true;
    float normal_radius   = 0.05f;
    int   normal_knn      = 30;
};

// Generate mesh from dense point cloud
TriangleMesh generate_mesh(
    const std::vector<Vec3>& points,
    const std::vector<Vec3>& colors,
    const std::vector<Vec3>& normals,
    const MeshingConfig& cfg = MeshingConfig());

// Estimate normals from point cloud
std::vector<Vec3> estimate_normals(
    const std::vector<Vec3>& points,
    int knn = 30,
    float radius = 0.05f);

// Mesh cleaning / post-processing
void remove_small_components(TriangleMesh& mesh, int min_triangles = 100);
void smooth_mesh(TriangleMesh& mesh, int iterations = 3, double lambda = 0.5);

// Export mesh to PLY
bool write_mesh_ply(const std::string& path, const TriangleMesh& mesh);

// Export mesh to OBJ
bool write_mesh_obj(const std::string& path, const TriangleMesh& mesh);

}  // namespace reconstruction
}  // namespace chisel
