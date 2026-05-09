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

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "hydra/common/batch_pipeline.h"
#include "hydra/common/config_utilities.h"
#include "hydra/common/global_info.h"
#include "hydra/frontend/place_2d_segmenter.h"
#include "hydra/input/camera.h"
#include "hydra/input/input_data.h"
#include "hydra/input/sensor.h"
#include "hydra/reconstruction/projective_integrator.h"
#include "hydra/reconstruction/semantic_integrator.h"
#include "hydra/reconstruction/volumetric_map.h"

namespace fs = std::filesystem;

DEFINE_string(data_path, "", "Path to Clio scene directory");
DEFINE_string(output_dir, "/tmp/clio_output", "Output directory");
DEFINE_string(config_path, "", "Path to hydra config directory");

// Camera intrinsics (default: Kinect-like 640x480)
DEFINE_int32(img_width, 640, "Image width");
DEFINE_int32(img_height, 480, "Image height");
DEFINE_double(fx, 525.0, "Camera focal length x");
DEFINE_double(fy, 525.0, "Camera focal length y");
DEFINE_double(cx, 319.5, "Camera principal point x");
DEFINE_double(cy, 239.5, "Camera principal point y");

// Map config
DEFINE_double(voxel_size, 0.05, "TSDF voxel size (meters)");
DEFINE_int32(voxels_per_side, 16, "Voxels per block side");
DEFINE_double(truncation_distance, 0.3, "TSDF truncation distance (meters)");

// Frame skipping for faster processing
DEFINE_int32(frame_skip, 1, "Process every Nth frame only");

namespace hydra {

struct ClioFrame {
  uint64_t timestamp_ns;
  Eigen::Isometry3d world_T_body;
  std::string color_path;
  std::string depth_path;
};

std::vector<ClioFrame> loadClioData(const std::string& data_path,
                                    const std::string& traj_file,
                                    int frame_skip) {
  std::vector<ClioFrame> frames;

  // Read trajectory
  std::string traj_path = data_path + "/" + traj_file;
  std::ifstream traj_in(traj_path);
  if (!traj_in.is_open()) {
    LOG(FATAL) << "Cannot open trajectory file: " << traj_path;
  }

  std::vector<Eigen::Matrix4d> poses;
  std::string line;
  while (std::getline(traj_in, line)) {
    if (line.empty()) continue;
    std::istringstream iss(line);
    std::vector<double> vals(16);
    for (int i = 0; i < 16; ++i) {
      iss >> vals[i];
    }
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    // Column-major: row r, col c = vals[r + 4*c]
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        T(r, c) = vals[r + 4 * c];
      }
    }
    poses.push_back(T);
  }
  traj_in.close();

  LOG(INFO) << "Loaded " << poses.size() << " poses from " << traj_path;

  // List color images
  std::string img_dir = data_path + "/images";
  std::string depth_dir = data_path + "/depth";

  std::vector<std::string> color_files;
  for (const auto& entry : fs::directory_iterator(img_dir)) {
    if (entry.path().extension() == ".jpg" || entry.path().extension() == ".jpeg") {
      color_files.push_back(entry.path().filename().string());
    }
  }
  std::sort(color_files.begin(), color_files.end());

  LOG(INFO) << "Found " << color_files.size() << " color images";

  size_t num_frames = std::min(poses.size(), color_files.size());
  uint64_t base_timestamp = 1710362415000000000ULL;  // ~2024-03-13

  for (size_t i = 0; i < num_frames; i += frame_skip) {
    if (i >= poses.size() || i >= color_files.size()) break;

    ClioFrame frame;
    frame.timestamp_ns = base_timestamp + static_cast<uint64_t>(i) * 142857142ULL;  // ~7fps
    frame.world_T_body = Eigen::Isometry3d(poses[i]);

    // Derive depth filename from color filename
    std::string color_name = color_files[i];
    std::string base_name = color_name.substr(0, color_name.find_last_of('.'));
    // color: rgb_NNN.jpg, depth: depth_NNN.png
    std::string num_str = base_name;
    if (num_str.find("rgb_") == 0) {
      num_str = num_str.substr(4);  // remove "rgb_"
    }

    frame.color_path = img_dir + "/" + color_name;
    frame.depth_path = depth_dir + "/depth_" + num_str + ".png";

    frames.push_back(frame);
  }

  LOG(INFO) << "Prepared " << frames.size() << " frames for processing";
  return frames;
}

void runClioPipeline(const std::string& data_path, const std::string& output_dir) {
  // 1. Create output directory
  fs::create_directories(output_dir);

  // 2. Setup GlobalInfo with PipelineConfig
  PipelineConfig pipeline_config;
  pipeline_config.default_verbosity = 1;
  pipeline_config.default_num_threads = -1;
  pipeline_config.enable_reconstruction = true;
  pipeline_config.enable_lcd = false;
  pipeline_config.logs.log_dir = output_dir + "/hydra_logs";

  // Configure label space (minimal: 2 labels)
  pipeline_config.label_space.invalid_labels = {0};
  pipeline_config.label_space.surface_places_labels = {1};
  pipeline_config.label_space.object_labels = {};
  pipeline_config.label_space.dynamic_labels = {};
  pipeline_config.label_names[0] = "unknown";
  pipeline_config.label_names[1] = "building";

  // Map config
  pipeline_config.map.voxels_per_side = FLAGS_voxels_per_side;
  pipeline_config.map.voxel_size = FLAGS_voxel_size;
  pipeline_config.map.truncation_distance = FLAGS_truncation_distance;

  GlobalInfo::init(pipeline_config, 0, true);
  LOG(INFO) << "GlobalInfo initialized";

  // 3. Create Camera
  Camera::Config camera_config;
  camera_config.width = FLAGS_img_width;
  camera_config.height = FLAGS_img_height;
  camera_config.fx = FLAGS_fx;
  camera_config.fy = FLAGS_fy;
  camera_config.cx = FLAGS_cx;
  camera_config.cy = FLAGS_cy;
  camera_config.min_range = 0.1;
  camera_config.max_range = 10.0;
  camera_config.extrinsics = IdentitySensorExtrinsics::Config{};
  camera_config.horizontal_fov = -1;  // computed from intrinsics

  auto camera = std::make_shared<Camera>(camera_config);

  // 4. Create ProjectiveIntegrator and VolumetricMap
  ProjectiveIntegratorConfig integrator_config;
  integrator_config.num_threads = -1;
  integrator_config.interp_method = "bilinear";
  integrator_config.semantic_integrator = MLESemanticIntegrator::Config{};

  ProjectiveIntegrator integrator(integrator_config);

  VolumetricMap::Config map_config = pipeline_config.map;
  VolumetricMap map(map_config, true);  // with_semantics = true (required for BatchPipeline)

  // 5. Load data and integrate
  auto frames = loadClioData(data_path, "traj_odom.txt", FLAGS_frame_skip);
  if (frames.empty()) {
    LOG(FATAL) << "No frames loaded from " << data_path;
  }

  LOG(INFO) << "Starting TSDF integration for " << frames.size() << " frames...";

  for (size_t idx = 0; idx < frames.size(); ++idx) {
    const auto& frame = frames[idx];

    // Read images
    cv::Mat color = cv::imread(frame.color_path, cv::IMREAD_COLOR);
    if (color.empty()) {
      LOG(WARNING) << "Cannot read color image: " << frame.color_path;
      continue;
    }
    // OpenCV reads as BGR, convert to RGB
    cv::cvtColor(color, color, cv::COLOR_BGR2RGB);

    cv::Mat depth = cv::imread(frame.depth_path, cv::IMREAD_UNCHANGED);
    if (depth.empty()) {
      LOG(WARNING) << "Cannot read depth image: " << frame.depth_path;
      continue;
    }

    // Convert 16-bit depth (mm) to float (meters)
    cv::Mat depth_float;
    depth.convertTo(depth_float, CV_32FC1, 1.0 / 1000.0);

    // Create empty label image (all zeros = unknown)
    cv::Mat labels = cv::Mat::zeros(color.size(), CV_32SC1);

    // Create InputData
    InputData input_data(camera);
    input_data.timestamp_ns = frame.timestamp_ns;
    input_data.world_T_body = frame.world_T_body;
    input_data.color_image = color;
    input_data.depth_image = depth_float;
    input_data.label_image = labels;

    // Let camera finalize representations (compute range image + vertex map)
    camera->finalizeRepresentations(input_data);

    // Integrate into TSDF map
    auto updated_blocks = integrator.updateMap(input_data, map, true);

    if ((idx + 1) % 100 == 0 || idx == frames.size() - 1) {
      LOG(INFO) << "Processed " << (idx + 1) << "/" << frames.size()
                << " frames, map has " << map.getTsdfLayer().numBlocks() << " blocks";
    }
  }

  LOG(INFO) << "TSDF integration complete. Map has " << map.getTsdfLayer().numBlocks() << " blocks";

  // 6. Save VolumetricMap
  std::string map_path = output_dir + "/tsdf_map";
  map.save(map_path);
  LOG(INFO) << "Saved TSDF map to " << map_path;

  // 7. Generate mesh and build DSG via BatchPipeline
  LOG(INFO) << "Starting scene graph construction...";

  // Create FrontendModule config from our Clio config file
  // If config_path is specified, load from YAML; otherwise use defaults
  FrontendModule::Config frontend_config;

  if (!FLAGS_config_path.empty() &&
      fs::exists(FLAGS_config_path + "/clio/frontend_config.yaml")) {
    frontend_config = config::fromYamlFile<FrontendModule::Config>(
        FLAGS_config_path + "/clio/frontend_config.yaml");
    LOG(INFO) << "Loaded frontend config from YAML";
  } else {
    // Use default frontend config
    frontend_config.pgmo.mesh_resolution = 0.05;
    frontend_config.pgmo.d_graph_resolution = 2.5;
    frontend_config.pgmo.time_horizon = 15.0;
    frontend_config.min_object_vertices = 200;
    LOG(INFO) << "Using default frontend config";
  }

  // Set up surface places (required for room detection)
  if (!frontend_config.surface_places.isValid()) {
    Place2dSegmenter::Config places_config;
    places_config.cluster_tolerance = 0.3;
    places_config.min_cluster_size = 50;
    places_config.max_cluster_size = 100000;
    frontend_config.surface_places = places_config;
  }

  BatchPipeline batch_pipeline(pipeline_config);

  config::VirtualConfig<FrontendModule> vf_config(frontend_config);
  auto dsg = batch_pipeline.construct(vf_config, map);

  if (!dsg) {
    LOG(FATAL) << "Failed to construct scene graph!";
  }

  // 8. Save DSG
  std::string dsg_path = output_dir + "/dsg.json";
  dsg->save(dsg_path);
  LOG(INFO) << "Saved DSG to " << dsg_path;
  LOG(INFO) << "DSG stats: " << dsg->numLayers() << " layers, "
            << dsg->numNodes() << " nodes, " << dsg->numEdges() << " edges";
}

}  // namespace hydra

int main(int argc, char* argv[]) {
  FLAGS_minloglevel = 0;
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  google::SetUsageMessage("Run Hydra scene graph pipeline on Clio data");
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  if (FLAGS_data_path.empty()) {
    LOG(FATAL) << "Must specify --data_path!";
  }

  hydra::runClioPipeline(FLAGS_data_path, FLAGS_output_dir);
  hydra::GlobalInfo::exit();

  LOG(INFO) << "Clio pipeline completed successfully.";
  return 0;
}
