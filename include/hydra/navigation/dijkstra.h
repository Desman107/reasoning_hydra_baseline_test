// Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
// Technology All rights reserved.

// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <spark_dsg/edge_container.h>
#include <spark_dsg/scene_graph_node.h>
#include <spark_dsg/scene_graph_types.h>

#include <Eigen/Dense>
#include <functional>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <utility>
#include <vector>

namespace hydra {
using namespace spark_dsg;
typedef std::priority_queue<std::pair<double, NodeId>,
                            std::vector<std::pair<double, NodeId>>,
                            std::greater<>>
    MinHeap;

void dijkstra(const std::map<NodeId, SceneGraphNode::Ptr>& nodes,
              const std::set<EdgeKey>& edges,
              const NodeId& start,
              const NodeId& goal,
              std::vector<NodeId>& path,
              std::vector<Eigen::Vector3d>& path_points);

}  // namespace hydra
