# CLAUDE.md — reasoning_hydra + Clio 数据集 部署与测试指南

## 1. 项目概述

**reasoning_hydra** 是一个 ROS Noetic + catkin 构建的 C++ 分层三维场景图 (Hierarchical 3D Scene Graph) 系统。基于 MIT SPARK Lab 的 Hydra 项目增强，由 NTNU Autonomous Robots Lab 开发 (ICRA 2026)。

核心流程：RGB-D 输入 → TSDF 三维重建 → 前端地点/物体提取 → 后端场景图更新 → 房间分割 → 任务推理 (LLM + VLM)

技术栈：C++17, CMake, catkin_tools, ROS Noetic, GTSAM, OpenCV, PCL, spark_dsg

## 2. 部署步骤

### 2.1 系统要求
- Ubuntu 20.04 + ROS Noetic (`ros-noetic-desktop-full`)
- Python 3 + pip, vcstool, catkin_tools

```bash
sudo apt install python3-rosdep python3-catkin-tools python3-vcstool
```

### 2.2 构建项目

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin init
catkin config -DCMAKE_BUILD_TYPE=Release

cd src
# 当前仓库已在 /home/DazhiHuang/project/reasoning_hydra，可软链接或拷贝
ln -s /home/DazhiHuang/project/reasoning_hydra .

# 导入依赖
vcs import . < reasoning_hydra/install/packages.repos
rosdep install --from-paths . --ignore-src -r -y

cd ..
catkin build
```

### 2.3 语义推理环境 (Python)

语义推理在独立仓库 [semantic_inference_ros](https://github.com/ntnu-arl/semantic_inference_ros)，需要：
- `OPENAI_API_KEY` — 用于 LLM 推理 (OpenAI API)
- `FASTAPI_API_KEY` — 用于 VLM 推理服务端
- 安装对应的 Python 依赖

### 2.4 CMake 关键选项
| 选项 | 默认值 | 说明 |
|------|--------|------|
| `HYDRA_ENABLE_EVAL` | ON | 构建评估模块 |
| `HYDRA_ENABLE_GNN` | OFF | GNN 闭环检测 (需 ONNX Runtime) |
| `HYDRA_ENABLE_TESTS` | OFF | 单元测试 |

## 3. Clio 数据集特征

路径: `/data/YueChang/Clio`，总大小约 14 GB。

### 3.1 四个场景

| 场景 | 帧数 | 时长(秒) | 数据大小 | 说明 |
|------|------|---------|---------|------|
| apartment | 1,168 | ~155 | 2.4 GB | 公寓 |
| building | 3,843 | ~825 | 6.7 GB | 建筑楼 |
| cubicle | 640 | ~91 | 1.2 GB | 工位/隔间 |
| office | 1,460 | ~195 | 2.8 GB | 办公室 |

### 3.2 每场景数据文件

```
<scene>/
├── traj_odom.txt          # 相机轨迹，每行 16 个浮点数 (4x4 变换矩阵按行展开)
├── images/                # RGB 图像 (640x480 JPEG)，命名 rgb_<序号>.jpg
├── depth/                 # 深度图 (640x480 16-bit PNG)，命名 depth_<序号>.png
├── rooms_<scene>.yaml     # 房间语义分割标注
├── tasks_<scene>.yaml     # 单任务标注（本地副本）
├── region_tasks_<scene>.yaml  # 区域自然语言描述
├── dense/                 # 稠密重建 (fused.ply, meshed-poisson.ply)，权限受限
└── sparse/                # 稀疏重建 (points3D.h5)，权限受限
```

### 3.3 轨迹格式 (traj_odom.txt)

每行 16 个空格分隔的 double 值，代表 4x4 变换矩阵（行优先，列优先？需验证前几行）。
帧数与图像一一对应。

### 3.4 标注体系 (`annotations/`)

- **single_task/**: 单任务标注，每个任务对应目标物体的 3D 边界框 (center + extents + rotation quaternion)
- **multi_task/**: 多任务时间线，包含 inject/query 事件和帧号（30 FPS 基准）。cubicle 有 9 个任务，apartment/office 仅有框架
- **complex_single_task_v2/**: 复杂任务，含 explicit_object + implicit_objects (最多3个物体，含 interaction_role)
- **complicate_single_task/**: 旧版复杂任务 + 隐式任务描述 (Markdown)

### 3.5 数据来源

从 ROS bag 提取（话题: `/dominic/forward/color/image_raw`, `/dominic/forward/depth/image_rect_raw`, `/dominic/forward/colmap_odom`）。原始 bag 不在仓库中。

## 4. 关键差距分析 (Clio → reasoning_hydra)

要将 Clio 数据接入 reasoning_hydra，需解决以下问题:

### 4.1 数据格式不匹配
- **问题**: Clio 是原始图片+轨迹文件，不包含语义分割标签。reasoning_hydra 通过 ROS 话题接收 RGB+深度+语义标签+位姿。
- **方案**: 需要编写 ROS 数据发布节点，读取 Clio 图片和轨迹，发布到对应 topic；同时需要语义推理模块生成语义标签。

### 4.2 缺少 Clio 配置
- **问题**: 没有 `config/clio/` 目录，需要创建 frontend/backend/reconstruction 等 yaml 配置。
- **方案**: 以 `config/replica/` 为模板（Replica 同为室内场景，参数相似），创建 Clio 专用配置。

### 4.3 缺少标签空间定义
- **问题**: 需要知道 Clio 场景中图像标注或推理使用的语义类别体系。
- **方案**: 检查 semantic_inference 使用的开放词汇标签体系，或使用 ADE20K 标签空间。

### 4.4 缺少相机内参
- **问题**: Clio 图像 640x480，但未确认 fx/fy/cx/cy。
- **方案**: 检查 ROS bag 原始数据或提取记录的 CameraInfo，或使用默认估计值。

### 4.5 外部依赖仓库
- **问题**: `reasoning_hydra_ros` (ROS 封装层) 和 `semantic_inference_ros` (语义推理) 在独立仓库，需要先 clone 和构建。
- **方案**: 确保 `install/packages.repos` 中的仓库可访问。

## 5. 工作步骤

### 第一步: 构建项目
1. 创建 catkin workspace
2. 导入所有外部依赖仓库并安装依赖
3. `catkin build` 确保编译通过
4. 构建并安装 `semantic_inference_ros` 的 Python 依赖

### 第二步: 创建 Clio 配置
1. 创建 `config/clio/` 目录
2. 以 `config/replica/` 为模板创建配置文件:
   - `frontend_config.yaml`
   - `backend_config.yaml`
   - `backend_subscriber_config.yaml`
   - `reconstruction_config.yaml`
   - `lcd_config.yaml` (可选)
   - `object_search_config.yaml` (可选)
3. 根据 Clio 数据特点调整参数（voxel_size 建议 0.03-0.05，房间尺寸等）

### 第三步: 确定标签体系
1. 确认 semantic_inference 使用的标签空间
2. 创建 `config/label_spaces/clio_label_space.yaml`
3. 在配置中设置 `enable_rooms: true` 和合适的 `building_semantic_label`

### 第四步: 数据接入
1. 编写或修改 ROS 数据发布节点，将 Clio 的 RGB/深度/轨迹数据发布为 ROS topics:
   - RGB: `sensor_msgs/Image`
   - Depth: `sensor_msgs/Image`
   - Pose: `nav_msgs/Odometry` 或 TF
2. 确认语义推理模块能实时处理图像生成语义标签
3. 建立 launch 文件启动完整流程

### 第五步: 运行测试
1. 先测最小场景 cubicle (640 帧, ~91 秒) 验证流程
2. 扩展到 apartment (1,168 帧) 和 office (1,460 帧)
3. 最后测最大的 building (3,843 帧)

### 第六步: 任务推理评估
1. 将 Clio 的 multi_task 时间线转换为评估格式
2. 运行 scene graph 构建
3. 使用 object_search 模块进行任务推理
4. 计算命中率等评估指标

## 6. 修改/新增的文件清单

### 需要创建的文件:
- `config/clio/frontend_config.yaml`
- `config/clio/backend_config.yaml`
- `config/clio/backend_subscriber_config.yaml`
- `config/clio/reconstruction_config.yaml`
- `config/clio/lcd_config.yaml`
- `config/clio/object_search_config.yaml`
- `config/label_spaces/clio_label_space.yaml`
- `config/clio/pipeline.yaml` (如果不使用 default 的 pipeline)

### 需要外部创建 (在 reasoning_hydra_ros 仓库):
- ROS 数据发布节点 (Clio 图片+轨迹 → ROS topics)
- Launch 文件 (clio.launch)

### 可能需要修改的文件:
- `install/packages.repos` — 确认所有依赖仓库 URL 可访问
