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
#include <pcl/common/centroid.h>
#include <pcl/pcl_base.h>
#include <pcl/point_types.h>

#include <utility>
#include <vector>

#include "hydra/common/dsg_types.h"
#include "spark_dsg/dynamic_scene_graph.h"
#include "spark_dsg/node_attributes.h"

namespace hydra {

struct Place2d {
  // using PointT = pcl::PointXYZRGBA;
  using PointT = spark_dsg::Mesh::Pos;
  using CloudT = spark_dsg::Mesh::Positions;
  using CentroidT = pcl::CentroidPoint<pcl::PointXYZ>;
  using Index = size_t;
  CentroidT centroid;
  std::vector<Index> indices;
  size_t min_mesh_index;
  size_t max_mesh_index;
  std::vector<Index> boundary_indices;
  std::vector<Eigen::Vector3d> boundary;
  Eigen::Matrix2d ellipse_matrix_compress;
  Eigen::Matrix2d ellipse_matrix_expand;
  Eigen::Vector2d ellipse_centroid;
  Eigen::Vector2d cut_plane;
  bool can_split;
};

void addRectInfo(const Place2d::CloudT& points,
                 const double connection_ellipse_scale_factor,
                 Place2dNodeAttributes& attrs);

void addRectInfo(const Place2d::CloudT& points,
                 const double connection_ellipse_scale_factor,
                 Place2d& place);

void addBoundaryInfo(const Place2d::CloudT& points, Place2d& place);

void addBoundaryInfo(const Place2d::CloudT& points, Place2dNodeAttributes& attrs);

std::pair<Place2d, Place2d> splitPlace(const Place2d::CloudT& points,
                                       const Place2d& place,
                                       const double connection_ellipse_scale_factor);

std::vector<Place2d> decomposePlaces(const Place2d::CloudT& cloud,
                                     const std::vector<Place2d>& initial_places,
                                     double min_size,
                                     size_t min_points,
                                     const double connection_ellipse_scale_factor);

std::vector<Place2d> decomposePlace(const Place2d::CloudT& cloud_pts,
                                    const Place2d& place,
                                    const double min_size,
                                    const size_t min_points,
                                    const double connection_ellipse_scale_factor);

bool shouldAddPlaceConnection(const Place2dNodeAttributes& attrs1,
                              const Place2dNodeAttributes& attrs2,
                              const double place_overlap_threshold,
                              const double place_max_neighbor_z_diff,
                              EdgeAttributes& edge_attrs);

bool shouldAddPlaceConnection(const Place2d& p1,
                              const Place2d& p2,
                              const double place_overlap_threshold,
                              const double place_max_neighbor_z_diff,
                              double& weight);

}  // namespace hydra
