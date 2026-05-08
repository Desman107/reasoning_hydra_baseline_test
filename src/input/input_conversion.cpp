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
#include "hydra/input/input_conversion.h"

#include <glog/logging.h>

#include <opencv2/imgproc.hpp>

#include "hydra/common/global_info.h"
#include "hydra/common/semantic_color_map.h"
#include "hydra/input/input_packet.h"
#include "hydra/input/sensor.h"

namespace hydra::conversions {

std::string showTypeInfo(const cv::Mat& mat) {
  std::stringstream ss;
  ss << "{depth: " << mat.depth() << ", channels: " << mat.channels() << "}";
  return ss.str();
}

std::unique_ptr<InputData> parseInputPacket(const InputPacket& input_packet,
                                            const bool vertices_in_world_frame) {
  if (!input_packet.sensor_input) {
    LOG(ERROR) << "[Input Conversion] Input packet has no sensor input.";
    return nullptr;
  }

  if (input_packet.sensor_input->sensor_id >= GlobalInfo::instance().numSensors()) {
    LOG(ERROR) << "[Input Conversion] Input sensor ID "
               << input_packet.sensor_input->sensor_id
               << " is out of range. Existing sensors: "
               << GlobalInfo::instance().numSensors() << ".";
    return nullptr;
  }
  auto data = std::make_unique<InputData>(
      GlobalInfo::instance().getSensor(input_packet.sensor_input->sensor_id));

  if (!input_packet.fillInputData(*data)) {
    LOG(ERROR) << "[Input Conversion] Unable to fill input data from input packet.";
    return nullptr;
  }

  if (!normalizeData(*data)) {
    LOG(ERROR) << "[Input Conversion] Unable to normalize data.";
    return nullptr;
  }

  if (!data->getSensor().finalizeRepresentations(*data)) {
    LOG(ERROR) << "[Input Conversion] Unable to compute inputs for integration";
    return nullptr;
  }

  convertVertexMap(*data, vertices_in_world_frame);
  return data;
}

bool hasSufficientData(const InputData& data) {
  // we accept data as either a pointcloud (points) or an rgbd image (depth_image)
  // labels or color can encode the labels for a depth image or pointcloud. For the
  // former case, color_image and/or label_image will share the same resolution as
  // depth image and for the latter it will share the same resolution as the
  // pointcloud...
  return (!data.color_image.empty() || !data.label_image.empty()) &&
         (!data.depth_image.empty() || !data.vertex_map.empty());
}

bool normalizeDepth(InputData& data) { return convertDepth(data); }

bool normalizeData(InputData& data, bool normalize_labels) {
  if (!convertDepth(data)) {
    return false;
  }

  if (!convertColor(data)) {
    return false;
  }

  // must come after convertColor as it uses color image
  if (normalize_labels && !convertLabels(data)) {
    return false;
  }

  if (!data.vertex_map.empty() && data.vertex_map.type() != CV_32FC3) {
    LOG(ERROR) << "pointcloud must be of type CV_32FC3, not "
               << showTypeInfo(data.vertex_map);
    return false;
  }

  return true;
}

bool colorToLabels(cv::Mat& label_image, const cv::Mat& colors) {
  if (colors.empty() || colors.channels() != 3) {
    LOG(ERROR) << "color image required to decode semantic labels";
    return false;
  }

  CHECK_EQ(colors.type(), CV_8UC3);

  const auto colormap_ptr = GlobalInfo::instance().getSemanticColorMap();
  if (!colormap_ptr || !colormap_ptr->isValid()) {
    LOG(ERROR)
        << "label colormap not valid, but required for converting colors to labels!";
    return false;
  }

  cv::Mat new_label_image(colors.size(), CV_32SC1);
  for (int r = 0; r < colors.rows; ++r) {
    for (int c = 0; c < colors.cols; ++c) {
      const auto& pixel = colors.at<cv::Vec3b>(r, c);
      Color color(pixel[0], pixel[1], pixel[2]);
      // this is lazy, but works out to the same invalid label we normally use
      new_label_image.at<int32_t>(r, c) =
          colormap_ptr->getLabelFromColor(color).value_or(-1);
    }
  }

  label_image = new_label_image;
  return true;
}

bool convertLabels(InputData& data) {
  if (data.label_image.empty()) {
    return colorToLabels(data.label_image, data.color_image);
  }

  if (data.label_image.channels() != 1) {
    return colorToLabels(data.label_image, data.label_image);
  }

  // Enforcing requirement for int32_t at this point
  if (data.label_image.type() != CV_32SC1) {
    cv::Mat new_label_image(data.label_image.size(), CV_32SC1);
    data.label_image.convertTo(new_label_image, CV_32SC1);
    data.label_image = new_label_image;
  }

  LabelRemapper label_remapper = GlobalInfo::instance().getLabelRemapper();
  if (!label_remapper.empty()) {
    for (int r = 0; r < data.label_image.rows; ++r) {
      for (int c = 0; c < data.label_image.cols; ++c) {
        // TODO(marcus): any reason to cache image and reassign with a new one?
        const auto& pixel = data.label_image.at<int32_t>(r, c);
        data.label_image.at<int32_t>(r, c) =
            label_remapper.remapLabel(pixel).value_or(-1);
      }
    }
  }

  const auto label_type = data.label_image.type();
  if (label_type == CV_32SC1) {
    return true;
  }

  if (label_type != CV_8UC1 && label_type != CV_16UC1 && label_type != CV_8SC1 &&
      label_type != CV_16SC1) {
    LOG(ERROR) << "label image must be integer type, not "
               << showTypeInfo(data.label_image);
    return false;
  }

  if (label_type == CV_16SC1 || label_type == CV_8SC1) {
    LOG_FIRST_N(WARNING, 5)
        << "signed to unsigned conversion of labels may not do what you want!";
  }

  cv::Mat label_converted;
  data.label_image.convertTo(label_converted, CV_32SC1);
  data.label_image = label_converted;
  return true;
}

bool convertDepth(InputData& data) {
  if (data.depth_image.empty()) {
    return true;
  }

  if (data.depth_image.channels() != 1) {
    LOG(ERROR) << "depth image must be single-channel";
    return false;
  }

  if (data.depth_image.type() == CV_32FC1) {
    return true;  // nothing else to do
  }

  if (data.depth_image.type() != CV_16UC1) {
    LOG(ERROR) << "only CV_32FC1 or CV_16UC1 formats supported, not "
               << showTypeInfo(data.depth_image);
    return false;
  }

  cv::Mat depth_converted;
  data.depth_image.convertTo(depth_converted, CV_32FC1, 1.0e-3);
  data.depth_image = depth_converted;
  return true;
}

bool convertColor(InputData& data) {
  if (data.color_image.empty()) {
    return true;
  }

  if (data.color_image.type() != CV_8UC3) {
    LOG(ERROR) << "only 3-channel rgb images supported";
    return false;
  }
  return true;
}

void convertVertexMap(InputData& data, bool in_world_frame) {
  if (data.points_in_world_frame == in_world_frame) {
    return;
  }
  Eigen::Isometry3f transform = data.getSensorPose().cast<float>();  // world_T_sensor
  if (!in_world_frame) {
    transform = transform.inverse();  // Instead get sensor_T_world
  }
  for (int r = 0; r < data.vertex_map.rows; ++r) {
    for (int c = 0; c < data.vertex_map.cols; ++c) {
      cv::Vec3f& point = data.vertex_map.at<cv::Vec3f>(r, c);
      Eigen::Vector3f point_eigen(point[0], point[1], point[2]);
      point_eigen = transform * point_eigen;
      point[0] = point_eigen.x();
      point[1] = point_eigen.y();
      point[2] = point_eigen.z();
    }
  }
  data.points_in_world_frame = in_world_frame;
}

RvlCodec::RvlCodec() {}

void RvlCodec::EncodeVLE(int value) {
  do {
    int nibble = value & 0x7;        // lower 3 bits
    if (value >>= 3) nibble |= 0x8;  // more to come
    word_ <<= 4;
    word_ |= nibble;
    if (++nibblesWritten_ == 8) {
      *pBuffer_++ = word_;
      nibblesWritten_ = 0;
      word_ = 0;
    }
  } while (value);
}

int RvlCodec::DecodeVLE() {
  unsigned int nibble;
  int value = 0, bits = 29;
  do {
    if (!nibblesWritten_) {
      word_ = *pBuffer_++;  // load word
      nibblesWritten_ = 8;
    }
    nibble = word_ & 0xf0000000;
    value |= (nibble << 1) >> bits;
    word_ <<= 4;
    nibblesWritten_--;
    bits -= 3;
  } while (nibble & 0x80000000);
  return value;
}

int RvlCodec::CompressRVL(const unsigned short* input,
                          unsigned char* output,
                          int numPixels) {
  buffer_ = pBuffer_ = (int*)output;
  nibblesWritten_ = 0;
  const unsigned short* end = input + numPixels;
  unsigned short previous = 0;
  while (input != end) {
    int zeros = 0, nonzeros = 0;
    for (; (input != end) && !*input; input++, zeros++)
      ;
    EncodeVLE(zeros);  // number of zeros
    for (const unsigned short* p = input; (p != end) && *p++; nonzeros++)
      ;
    EncodeVLE(nonzeros);  // number of nonzeros
    for (int i = 0; i < nonzeros; i++) {
      unsigned short current = *input++;
      int delta = current - previous;
      int positive = (delta << 1) ^ (delta >> 31);
      EncodeVLE(positive);  // nonzero value
      previous = current;
    }
  }
  if (nibblesWritten_)  // last few values
    *pBuffer_++ = word_ << 4 * (8 - nibblesWritten_);
  return int((unsigned char*)pBuffer_ - (unsigned char*)buffer_);  // num bytes
}

void RvlCodec::DecompressRVL(const unsigned char* input,
                             unsigned short* output,
                             int numPixels) {
  buffer_ = pBuffer_ = const_cast<int*>(reinterpret_cast<const int*>(input));
  nibblesWritten_ = 0;
  unsigned short current, previous = 0;
  int numPixelsToDecode = numPixels;
  while (numPixelsToDecode) {
    int zeros = DecodeVLE();  // number of zeros
    numPixelsToDecode -= zeros;
    for (; zeros; zeros--) *output++ = 0;
    int nonzeros = DecodeVLE();  // number of nonzeros
    numPixelsToDecode -= nonzeros;
    for (; nonzeros; nonzeros--) {
      int positive = DecodeVLE();  // nonzero value
      int delta = (positive >> 1) ^ -(positive & 1);
      current = previous + delta;
      *output++ = current;
      previous = current;
    }
  }
}

}  // namespace hydra::conversions
