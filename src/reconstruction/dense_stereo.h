#pragma once
// src/reconstruction/dense_stereo.h  –  Dense multi-view stereo
#include "core/types.h"
#include <vector>

namespace chisel {
namespace reconstruction {

struct DenseStereoConfig {
    // Depth range
    float min_depth = 0.1f;
    float max_depth = 100.0f;
    int   num_depth_samples = 128;  // for plane sweep

    // PatchMatch parameters
    int   patch_size = 7;
    int   num_iterations = 5;
    float ncc_threshold = 0.3f;     // normalized cross-correlation

    // Multi-view consistency
    int   num_source_images = 5;    // number of neighbor views
    float geometric_consistency_thresh = 2.0f;  // pixels
    int   min_consistent_views = 2;

    // Filtering
    float confidence_threshold = 0.5f;
    bool  filter_by_consistency = true;
};

// Select best source images for a reference view (by baseline & overlap)
std::vector<ImageId> select_source_images(
    const Scene& scene,
    ImageId ref_id,
    int num_sources);

// Compute dense depth map for a single reference view using plane-sweep
DepthMap compute_depth_plane_sweep(
    const Scene& scene,
    ImageId ref_id,
    const std::vector<ImageId>& source_ids,
    const DenseStereoConfig& cfg = DenseStereoConfig());

// Compute dense depth map using PatchMatch stereo
DepthMap compute_depth_patchmatch(
    const Scene& scene,
    ImageId ref_id,
    const std::vector<ImageId>& source_ids,
    const DenseStereoConfig& cfg = DenseStereoConfig());

// Multi-view geometric consistency filtering
void filter_depth_consistency(
    std::map<ImageId, DepthMap>& depth_maps,
    const Scene& scene,
    const DenseStereoConfig& cfg = DenseStereoConfig());

// Compute all depth maps for the scene
void compute_all_depth_maps(
    Scene& scene,
    const DenseStereoConfig& cfg = DenseStereoConfig());

}  // namespace reconstruction
}  // namespace chisel
