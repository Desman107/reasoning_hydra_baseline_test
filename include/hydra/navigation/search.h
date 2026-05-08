// Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
// Technology All rights reserved.

// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace hydra {
class Search {
 public:
  using Ptr = std::unique_ptr<Search>;
  Search() {}
  virtual ~Search() = default;

  virtual bool searchRoom(
      const Eigen::VectorXf& text_room_embedding,
      const std::vector<std::vector<Eigen::VectorXf>>& room_embeddings,
      const float& prob_threshold,
      size_t& result,
      std::vector<float>& probs) const = 0;
  virtual bool searchRooms(
      const std::vector<Eigen::VectorXf>& text_room_embeddings,
      const std::vector<std::vector<Eigen::VectorXf>>& room_embeddings,
      const float& prob_threshold,
      std::vector<size_t>& results,
      std::vector<float>& probs) const = 0;
  virtual bool searchObject(const Eigen::VectorXf& text_object_embedding,
                            const std::vector<Eigen::VectorXf>& object_embeddings,
                            const float& prob_threshold,
                            std::vector<size_t>& result,
                            std::vector<float>& probs) const = 0;
};

}  // namespace hydra
