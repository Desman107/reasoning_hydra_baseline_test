# dataset_debug.md — Clio 数据集接入问题与修复记录

> 最后更新: 2026-05-12 | 关联: CLAUDE.md 阶段 C

本文档记录将 Clio 数据集接入 reasoning_hydra 场景图构建流程时遇到的所有问题及解决方案。

---

## 问题 1: 共享库缺失 (libzmq.so.5, libKimeraRPGO.so)

**症状**: 编译后的二进制无法运行，提示 `libzmq.so.5: cannot open shared object file` 或 `libKimeraRPGO.so: cannot open shared object file`

**原因**: conda 环境中的动态库和 catkin devel 库不在 `LD_LIBRARY_PATH` 中

**解决方案**:
```bash
export LD_LIBRARY_PATH="/home/DazhiHuang/anaconda3/envs/ros-noetic/lib:/home/DazhiHuang/catkin_ws/devel/lib:$LD_LIBRARY_PATH"
```

---

## 问题 2: num_threads=-1 导致运行时错误

**症状**: `ProjectiveIntegratorConfig` 和 `MeshIntegratorConfig` 抛出 "num_threads must be positive" 异常

**原因**: 配置文件中的 `num_threads: -1` 表示使用所有可用核心，但实际代码要求正值

**解决方案**: 在 `run_clio_pipeline.cpp` 中显式设置:
```cpp
pipeline_config.default_num_threads = 2;
integrator_config.num_threads = 2;
```

---

## 问题 3: 图像高度不匹配 (360 vs 480)

**症状**: 图像处理输出异常

**原因**: 相机内参来自 RealSense D455 的 `realsense_pipeline_intrinsics.yaml`（分辨率 640x360），但 Clio 数据集实际图像为 640x480

**解决方案**: 将 `img_height` 默认值从 360 改为 480:
```cpp
DEFINE_int32(img_height, 480, "Image height");
```
相机内参 fx/fy/cx/cy 保持不变（这些值对 640x480 仍然正确）。

---

## 问题 4: 轨迹矩阵排列方式错误 (行优先 vs 列优先)

**症状**: TSDF 积分极其缓慢（40 分钟未完成），所有位姿错误

**原因**: `traj_odom.txt` 每行 16 个 float 是行优先 (row-major) 排列的 4x4 变换矩阵，但代码按列优先 (column-major) 读取: `T(r, c) = vals[r + 4*c]`

**解决方案**: 改为行优先读取:
```cpp
// 行优先: 行 r, 列 c = vals[r*4 + c]
for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
        T(r, c) = vals[r * 4 + c];
    }
}
```

---

## 问题 5: GlobalInfo 已冻结 (freeze=true)

**症状**: 在 `BatchPipeline::construct()` 中 `GlobalInfo::init()` 抛出 "GlobalInfo already frozen" 异常

**原因**: `BatchPipeline` 构造函数调用 `GlobalInfo::init(config, robot_id, true)` 设置 freeze=true，而我们之前调用 `GlobalInfo::init(pipeline_config, 0, true)` 也设置了 freeze=true，导致重复冻结

**解决方案**: 在 `run_clio_pipeline.cpp` 中使用 `freeze=false`:
```cpp
GlobalInfo::init(pipeline_config, 0, false);  // 不要冻结，BatchPipeline 会重新初始化
```

---

## 问题 6: tsdf_interpolator 导致 GVD 段错误

**症状**: 在 `GvdPlaceExtractor::detect()` 中调用 `tsdf_interpolator_->interpolate()` 时发生段错误

**原因**: `TsdfInterpolator` 在处理从单帧构建的 TSDF map 时崩溃，可能是因为 map 数据不足以进行下采样插值

**解决方案**: 从 `config/clio/frontend_config.yaml` 中移除 `tsdf_interpolator` 配置段

---

## 问题 7: freespace_places (GVD) 导致段错误

**症状**: 在 `GvdIntegrator::updateFromTsdf()` 中发生段错误

**原因**: GVD (Generalized Voronoi Diagram) 积分器在处理 sparse TSDF map 时崩溃

**解决方案**: 从 `config/clio/frontend_config.yaml` 中移除整个 `freespace_places` 段（同时移除 `frontier_places` 和 `use_frontiers`），仅保留 `surface_places` (Place2dSegmenter) 进行地点提取

---

## 问题 8: ReconstructionOutput::sensor_data 空指针

**症状**: 在 `FrontendModule::updateObjects()` 中访问 `input.sensor_data->relations` 时发生段错误

**原因**: `BatchPipeline::construct()` 创建 `ReconstructionOutput` 时未设置 `sensor_data` 字段（该字段需要 `InputData` 对象，而创建 `InputData` 需要 `Sensor` 指针），导致 `sensor_data` 为空指针

**解决方案**: 在 `frontend_module.cpp` 中添加空指针检查:
- `updateObjects()`: `input.sensor_data ? input.sensor_data->relations : std::nullopt`
- `spinOnce()`: 用 `if (msg->sensor_data)` 和 `if (msg->getMapPointer())` 包裹相关访问

---

## 问题 9: state_->backend_graph 空指针 (mutex lock 失败)

**症状**: `std::system_error: Operation not permitted` 在 `std::unique_lock<std::mutex>::lock()` 中，位置为 `state_->backend_graph->mutex`

**原因**: `BatchPipeline::construct()` 创建 `SharedModuleState` 后未初始化 `state_->backend_graph`。在正常 ROS pipeline 中，`HydraPipeline` 构造函数会调用 `shared_state_->backend_graph = config.createSharedDsg()`，但 batch pipeline 跳过了这一步

**解决方案**: 在 `batch_pipeline.cpp` 的 `construct()` 中添加:
```cpp
state->backend_graph = GlobalInfo::instance().createSharedDsg();
```

---

## 问题 10: label_space.total_labels = 0 导致所有标签被拒绝

**症状**: 日志大量输出 "Encountered invalid label: 1"，场景图中没有 place/object 节点

**原因**: `LabelSpaceConfig::total_labels` 默认值为 0，`MLESemanticIntegrator::isValidLabel()` 检查 `label >= total_labels` 时所有标签（包括标签 1）都被拒绝

**解决方案**: 在 `run_clio_pipeline.cpp` 中设置:
```cpp
pipeline_config.label_space.total_labels = 2;
```

---

## 问题 11: 标签图像全为零 (未知/无效标签)

**症状**: 即使修复了 `total_labels`，场景图中仍然缺乏 surface places 和 objects

**原因**: 所有像素标签设为 0 (unknown)，而 label 0 在 `invalid_labels` 集合中，导致语义积分器跳过所有像素

**解决方案**: 将所有像素标签设为 1 (building):
```cpp
cv::Mat labels = cv::Mat::ones(color.size(), CV_32SC1);
```

---

## 经验总结

1. **batch pipeline 缺少完整的初始化流程**: `BatchPipeline::construct()` 相比于正式的 `HydraPipeline`，缺少 `SharedModuleState` 的完整初始化（backend_graph, lcd_queue 等）
2. **配置参数需要显式设置**: 很多配置参数（如 `total_labels`, `num_threads`）有默认值 0 或 -1，在离线 pipeline 中需要显式设置
3. **标签空间配置至关重要**: 语义标签体系需要正确配置，包括 `total_labels`, `invalid_labels`, `surface_places_labels` 等，否则整个语义重建和地点提取流程都会失败
4. **freespace_places (GVD) 对 sparse TSDF 不稳定**: 单帧或少帧构建的 TSDF map 可能不足以支持 GVD 地点提取，可以暂时禁用以使用更稳定的 surface_places (Place2dSegmenter)
