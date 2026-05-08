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

// Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
// Technology All rights reserved.

// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Geometry>
#include <limits>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hydra/common/common_types.h"
#include "hydra/input/sensor.h"
#include "hydra/utils/pair_hash.h"

namespace hydra {

struct InputData {
  using Ptr = std::shared_ptr<InputData>;

  // Types of the stored image data.
  using ColorType = cv::Vec3b;
  using RangeType = float;
  using VertexType = cv::Vec3f;
  using LabelType = int;
  using RelationFeature = Eigen::MatrixXf;

  explicit InputData(Sensor::ConstPtr sensor) : sensor_(std::move(sensor)) {}
  virtual ~InputData() = default;

  // Time stamp this input data was captured.
  TimeStamp timestamp_ns;

  // Pose of the robot body in the world frame.
  Eigen::Isometry3d world_T_body;

  // Color image as RGB.
  cv::Mat color_image;

  // Depth image as planar depth in metres.
  cv::Mat depth_image;

  // Ray lengths in meters.
  cv::Mat range_image;

  // Label image for semantic input data.
  cv::Mat label_image;

  // RGB label pointcloud.
  pcl::PointCloud<pcl::PointXYZRGBL>::Ptr pointcloud;

  // Valid pixels mask.
  std::vector<std::vector<bool>> valid;

  // Semantic features mask.
  std::optional<cv::Mat> features_mask;

  // Semantic features.
  std::optional<std::unordered_map<uint16_t, Eigen::VectorXf>> semantic_features;

  // Image feature vector.
  std::optional<Eigen::VectorXf> image_feature;

  // Semantic relations features
  std::optional<PairHashMap> relations;

  // Sensor to sensor isometry transform.
  std::optional<Eigen::Isometry3d> sensor1_T_sensor2;

  // 3D points of the range image in sensor or world frame.
  cv::Mat vertex_map;
  bool points_in_world_frame = false;

  // Min and max range observed in the range image.
  float min_range = 0.0f;
  float max_range = std::numeric_limits<float>::infinity();

  /**
   * @brief Get the sensor that captured this data.
   */
  const Sensor& getSensor() const { return *sensor_; }

  /**
   * @brief Get the pose of the sensor in world frame when this data was captured.
   */
  Eigen::Isometry3d getSensorPose() const {
    return world_T_body * sensor_->body_T_sensor();
  }

 private:
  Sensor::ConstPtr sensor_;
};

};  // namespace hydra
