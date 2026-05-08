// Portions of the following code and their modifications are originally from
// https://github.com/MIT-SPARK/Hydra/tree/main and are licensed under the following
// license:
/* -----------------------------------------------------------------------------
 * Copyright 2022 Massachusetts Institute of Technology.
 * All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Research was sponsored by the United States Air Force Research Laboratory and
 * the United States Air Force Artificial Intelligence Accelerator and was
 * accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views
 * and conclusions contained in this document are those of the authors and should
 * not be interpreted as representing the official policies, either expressed or
 * implied, of the United States Air Force or the U.S. Government. The U.S.
 * Government is authorized to reproduce and distribute reprints for Government
 * purposes notwithstanding any copyright notation herein.
 * -------------------------------------------------------------------------- */
#pragma once
#include <config_utilities/virtual_config.h>
#include <hydra/utils/nearest_neighbor_utilities.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <spatial_hash/types.h>

#include <utility>
#include <vector>

#include "hydra/common/dsg_types.h"
#include "hydra/frontend/frontier_places_interface.h"
#include "hydra/reconstruction/reconstruction_output.h"

namespace hydra {

struct Frontier {
 public:
  Frontier() {}
  Frontier(Eigen::Vector3d c,
           Eigen::Vector3d s,
           Eigen::Quaterniond o,
           size_t n,
           spatial_hash::BlockIndex b)
      : center(c),
        scale(s),
        orientation(o),
        num_frontier_voxels(n),
        block_index(b),
        has_shape_information(true) {}
  Frontier(Eigen::Vector3d c, size_t n, spatial_hash::BlockIndex b)
      : center(c),
        num_frontier_voxels(n),
        block_index(b),
        has_shape_information(false) {}

 public:
  Eigen::Vector3d center;
  Eigen::Vector3d scale;
  Eigen::Quaterniond orientation;
  size_t num_frontier_voxels = 0;
  spatial_hash::BlockIndex block_index;
  bool has_shape_information = false;
};

class FrontierExtractor : public FrontierPlacesInterface {
 public:
  struct Config {
    char prefix = 'f';
    double cluster_tolerance = .3;
    size_t min_cluster_size = 10;
    size_t max_cluster_size = 100000;
    double max_place_radius = 5;
    bool dense_frontiers = false;
    double frontier_splitting_threshold = 0.2;
    size_t point_threshold = 10;
    size_t culling_point_threshold = 10;
    double recent_block_distance = 25;
    double minimum_relative_z = -0.2;
    double maximum_relative_z = 1;
    bool compute_frontier_shape = false;
  } const config;

  explicit FrontierExtractor(const Config& config);

  void updateRecentBlocks(Eigen::Vector3d current_position, double block_size) override;
  void detectFrontiers(const ReconstructionOutput& input,
                       DynamicSceneGraph& graph,
                       NearestNodeFinder& finder) override;
  void addFrontiers(uint64_t timestamp_ns,
                    DynamicSceneGraph& graph,
                    NearestNodeFinder& finder) override;

 private:
  NodeSymbol next_node_id_;
  std::vector<std::pair<NodeId, BlockIndex>> nodes_to_remove_;

  BlockIndices recently_archived_blocks_;

  std::vector<Frontier> frontiers_;
  std::vector<Frontier> archived_frontiers_;

  void populateDenseFrontiers(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                              const pcl::PointCloud<pcl::PointXYZ>::Ptr archived_cloud,
                              const double voxel_scale,
                              const TsdfLayer& layer);

  inline static const auto registration_ =
      config::RegistrationWithConfig<FrontierPlacesInterface,
                                     FrontierExtractor,
                                     Config>("voxel_clustering");

  // Helper functions.
  void computeSparseFrontiers(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                              const bool compute_frontier_shape,
                              const TsdfLayer& layer,
                              std::vector<Frontier>& frontiers) const;
};

Eigen::Vector3d frontiersToCenters(const std::vector<Eigen::Vector3f>& positions);

void declare_config(FrontierExtractor::Config& conf);

}  // namespace hydra
