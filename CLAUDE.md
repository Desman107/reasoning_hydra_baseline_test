# CLAUDE.md — reasoning_hydra + Clio 数据集 工作指南

> 最后更新: 2026-05-08 | 状态: 规划阶段

---

## 1. 核心目标

**跑通场景图构建流程，对 Clio 数据集计算 IoU 等评估指标。**

具体交付物：
- 4 个 Clio 场景 (apartment / building / cubicle / office) 的场景图 (DSG) 输出
- 物体检测 IoU、房间分割准确率等评估指标
- （不要求任务推理，不需要 LLM/VLM API Key）

---

## 2. 项目概述

**reasoning_hydra** — ROS Noetic 的 C++ 分层三维场景图 (Hierarchical 3D Scene Graph) 系统。

- 来源: NTNU Autonomous Robots Lab, ICRA 2026
- 基于: MIT SPARK Lab 的 Hydra 项目增强
- 技术栈: C++17 + CMake + catkin_tools + ROS Noetic + GTSAM + OpenCV + PCL + spark_dsg

**场景图构建流程:**

```
RGB-D 输入 → TSDF 三维重建 → 前端地点/物体提取 → 后端场景图更新 → 房间分割
                                                                        ↓
                                                               Dynamic Scene Graph (DSG)
```

流程中不涉及任何外部 API 调用，全部在 C++ 侧完成。

---

## 3. 环境与部署方案

### 3.1 当前系统状态

| 项目 | 状态 |
|------|------|
| OS | Ubuntu 22.04 (Jammy) |
| 编译器 | g++ 11.4.0 |
| CMake | 3.22.1 |
| sudo | **不可用** |
| Docker | 已安装但**无权限** |
| SSH (GitHub) | 已配置 (Desman107) |
| conda | 已安装 (/home/DazhiHuang/anaconda3) |
| catkin-tools | 已通过 pip 安装 |
| 磁盘 /home | 15T 总量, 206G 可用 |
| 磁盘 /data | 3.5T 总量, **已满** |
| 内存 | 1TB 总量, 889G 可用 |

### 3.2 部署方案: conda (robostack) + 源码编译

由于无 sudo 权限且 Docker 不可用，采用 **conda + robostack channel** 安装 ROS Noetic，其余依赖库通过 conda-forge 安装或源码编译。

```bash
# 步骤 1: 创建 ROS Noetic conda 环境
conda create -n ros-noetic python=3.9
conda activate ros-noetic

# 步骤 2: 安装 ROS Noetic (robostack channel 已验证可用)
conda install -c robostack -c conda-forge ros-noetic-ros-base

# 步骤 3: 安装 C++ 依赖库 (尽量用 conda，避免系统包缺失)
conda install -c conda-forge eigen opencv pcl gtsam glog

# 步骤 4: 创建 catkin workspace
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin init
catkin config -DCMAKE_BUILD_TYPE=Release

# 步骤 5: 软链接项目 + 导入外部依赖
cd src
ln -s /home/DazhiHuang/project/reasoning_hydra .
vcs import . < reasoning_hydra/install/packages.repos

# 步骤 6: 安装 ROS 依赖
rosdep install --from-paths . --ignore-src -r -y

# 步骤 7: 构建
catkin build
```

### 3.3 外部依赖仓库

通过 `install/packages.repos` 导入（均需 SSH 访问 GitHub）：

| 仓库 | 用途 |
|------|------|
| `catkin_simple` | 简化 catkin CMake |
| `config_utilities` | 配置管理 (NTNU 定制分支 `reasoning_hydra`) |
| `hydra_ros` | ROS 封装层 (`reasoning_hydra_ros`) |
| `kimera_pgmo` | 网格和位姿图优化 |
| `kimera_rpgo` | 鲁棒位姿图优化 |
| `mesh_rviz_plugins` | rviz 可视化插件 |
| `pose_graph_tools` | 位姿图工具 |
| `semantic_inference` | 语义推理 ROS 节点 (`semantic_inference_ros`) |
| `spark_dsg` | 动态场景图库 (NTNU 定制分支 `reasoning_Spark-DSG`) |
| `spatial_hash` | 空间哈希 |
| `teaser_plusplus` | 鲁棒配准 |

### 3.4 conda 方案的已知风险

- ROS Noetic 的 conda 版本是 Python 3.8/3.9 构建，功能可能不完全等同于 apt 安装版
- PCL、GTSAM 的 conda 版本可能与项目期望的 Ubuntu 20.04 apt 版本有 ABI 差异
- 部分依赖（如 `kimera_pgmo`）可能没有 conda 包，需源码编译
- rosdep 在 conda 环境下可能无法正确解析依赖，需手动处理

---

## 4. Clio 数据集

### 4.1 概览

路径: `/data/YueChang/Clio`，总大小约 14 GB。4 个真实室内场景的 RGB-D 序列 + 轨迹 + 3D 标注。

| 场景 | 帧数 | 时长 | 数据大小 | 优先度 |
|------|------|------|---------|--------|
| cubicle | 640 | ~91s | 1.2 GB | ★★★ 先测（最小） |
| apartment | 1,168 | ~155s | 2.4 GB | ★★ |
| office | 1,460 | ~195s | 2.8 GB | ★★ |
| building | 3,843 | ~825s | 6.7 GB | ★ 最后（最大） |

### 4.2 每场景数据结构

```
<scene>/
├── traj_odom.txt           # 相机轨迹 (每行 16 个 double, 4x4 变换矩阵)
├── images/                 # RGB 图像 640x480 JPEG, 命名 rgb_<序号>.jpg
├── depth/                  # 深度图 640x480 16-bit PNG, 命名 depth_<序号>.png
├── rooms_<scene>.yaml      # 房间语义分割标注 (3D 边界框)
├── tasks_<scene>.yaml      # 单任务标注 (3D 物体边界框, 本地副本)
├── region_tasks_<scene>.yaml  # 区域自然语言描述
├── extract_summary.yaml    # 提取元信息
├── dense/                  # 稠密重建 (fused.ply, meshed-poisson.ply)
└── sparse/                 # 稀疏重建 (points3D.h5)
```

### 4.3 轨迹格式

`traj_odom.txt` 每行 16 个空格分隔的浮点数，代表 4x4 变换矩阵（世界坐标系下相机位姿），需验证行优先/列优先排列方式。

### 4.4 标注体系 (annotations/)

| 标注层级 | 目录 | 内容 |
|----------|------|------|
| 单任务 | `annotations/single_task/` | 任务描述 + 目标物体 3D bbox (center + extents + quaternion) |
| 多任务时间线 | `annotations/multi_task/` | 在线流式评估: inject/query 事件 + 帧号 (30 FPS) |
| 复杂单任务 v2 | `annotations/complex_single_task_v2/` | explicit_object + implicit_objects (含 interaction_role) |
| 复杂单任务 v1 | `annotations/complicate_single_task/` | 旧版复杂任务 + 隐式交互任务描述 (Markdown) |

### 4.5 数据来源

从 ROS bag 提取，原始话题：
- `/dominic/forward/color/image_raw` → RGB 图像
- `/dominic/forward/depth/image_rect_raw` → 深度图
- `/dominic/forward/colmap_odom` → 里程计/位姿

原始 bag 文件不在仓库中。

---

## 5. 关键差距 (Clio ↔ reasoning_hydra)

要将 Clio 数据接入 reasoning_hydra 场景图构建流程，需解决以下差距：

### 5.1 数据格式不匹配

| 问题 | Clio 现状 | reasoning_hydra 期望 |
|------|----------|---------------------|
| 数据载体 | 原始图片 + 轨迹 txt | ROS topics (sensor_msgs/Image, nav_msgs/Odometry) |
| 语义标签 | 无 (RGB-D only) | cv::Mat 整数标签图像 |
| 深度格式 | 16-bit PNG (毫米) | cv::Mat float (米) |
| 位姿 | traj_odom.txt (4x4 矩阵) | Eigen::Isometry3d (通过 ROS tf/odometry) |

**方案**: 编写 ROS 数据发布节点，读取 Clio 图片和轨迹文件，发布到对应 topics。

### 5.2 语义标签缺失

Clio 只有 RGB-D 图像，没有语义分割标签。reasoning_hydra 的 TSDF 积分器需要 label_image。

**方案**: 几种可能路径：
- A) 使用 `semantic_inference` 模块提取开放词汇特征（需要 VLM 服务器）
- B) 使用预训练语义分割模型 (如 DeepLabv3+ ADE20K) 生成伪标签
- C) 先用空标签跑通流程，验证重建和几何提取是否正常

### 5.3 缺少 Clio 配置

项目没有 `config/clio/` 目录。需要创建专用配置。

**方案**: 以 `config/replica/` 为模板（Replica 同为室内场景），创建 Clio 配置。

### 5.4 缺少相机内参

Clio 图像 640x480，需要 fx, fy, cx, cy。目前未知。

**方案**: 检查 `extract_summary.yaml` 或原始 bag 的 CameraInfo，或用 640x480 默认估计值。

### 5.5 缺少标签空间

**方案**: 确定使用的语义标签体系后，创建 `config/label_spaces/clio_label_space.yaml`。

---

## 6. 工作步骤与进度

### 阶段 A: 环境部署

| 步骤 | 任务 | 状态 | 备注 |
|------|------|------|------|
| A1 | 创建 conda 环境 + 安装 ROS Noetic | ⬜ 待开始 | |
| A2 | 安装 C++ 依赖库 (Eigen, OpenCV, PCL, GTSAM, glog) | ⬜ 待开始 | |
| A3 | 创建 catkin workspace | ⬜ 待开始 | |
| A4 | clone 外部依赖仓库 (vcs import) | ⬜ 待开始 | |
| A5 | rosdep install 解析剩余依赖 | ⬜ 待开始 | |
| A6 | catkin build 编译 | ⬜ 待开始 | |
| A7 | 解决编译错误（预计有） | ⬜ 待开始 | |

### 阶段 B: Clio 配置创建

| 步骤 | 任务 | 状态 | 备注 |
|------|------|------|------|
| B1 | 确定语义标签方案 (A/B/C) | ⬜ 待开始 | |
| B2 | 获取相机内参 | ⬜ 待开始 | |
| B3 | 创建 `config/clio/` 配置文件 | ⬜ 待开始 | 基于 replica 模板 |
| B4 | 创建 `config/label_spaces/clio_label_space.yaml` | ⬜ 待开始 | |
| B5 | 验证配置被正确加载 | ⬜ 待开始 | |

### 阶段 C: 数据接入

| 步骤 | 任务 | 状态 | 备注 |
|------|------|------|------|
| C1 | 编写/修改 ROS 数据发布节点 | ⬜ 待开始 | 读取 Clio 图片+轨迹, 发布 ROS topics |
| C2 | 创建 Clio launch 文件 | ⬜ 待开始 | 在 reasoning_hydra_ros 中 |
| C3 | 验证数据流 (ROS topic echo) | ⬜ 待开始 | |

### 阶段 D: 场景图构建

| 步骤 | 任务 | 状态 | 备注 |
|------|------|------|------|
| D1 | 测试 cubicle (640 帧) | ⬜ 待开始 | 最小场景，快速验证 |
| D2 | 测试 apartment (1,168 帧) | ⬜ 待开始 | |
| D3 | 测试 office (1,460 帧) | ⬜ 待开始 | |
| D4 | 测试 building (3,843 帧) | ⬜ 待开始 | |

### 阶段 E: 评估

| 步骤 | 任务 | 状态 | 备注 |
|------|------|------|------|
| E1 | 编写 IoU 计算脚本 (物体 bbox) | ⬜ 待开始 | 利用 Clio 标注的 3D bbox |
| E2 | 编写房间分割评估脚本 | ⬜ 待开始 | 利用 rooms_*.yaml |
| E3 | 汇总评估结果 | ⬜ 待开始 | |

---

## 7. 关键决策记录

| 日期 | 决策 | 原因 |
|------|------|------|
| 2026-05-08 | 采用 conda (robostack) 路线安装 ROS Noetic | 无 sudo 权限，Docker 无权限 |
| 2026-05-08 | 先不配置 API Key，聚焦场景图构建 + IoU 评估 | 核心目标不需要 LLM/VLM 推理 |
| 2026-05-08 | 从 cubicle (640帧) 开始测试 | 最小场景，快速迭代 |
| 2026-05-08 | 以 replica 配置为模板创建 clio 配置 | 两者均为室内场景，参数最接近 |
| 2026-05-08 | CLAUDE.md 作为唯一工作指南和进度文件 | 统一信息来源，避免状态分散 |

---

## 8. 需要创建/修改的文件清单

### 本项目内 (reasoning_hydra):

```
config/clio/
├── pipeline.yaml                    # (可选，默认 default/pipeline.yaml 即可)
├── frontend_config.yaml             # 基于 replica 修改
├── backend_config.yaml              # 基于 replica 修改
├── backend_subscriber_config.yaml   # 直接复制 replica
├── reconstruction_config.yaml       # 基于 replica 修改
├── lcd_config.yaml                  # 基于 replica 修改
└── object_search_config.yaml        # 基于 replica 修改

config/label_spaces/
└── clio_label_space.yaml            # 新建标签空间
```

### 外部仓库 (reasoning_hydra_ros):

```
launch/
└── clio.launch                      # Clio 启动文件

src/
└── clio_data_publisher.cpp          # Clio 数据发布节点 (读取文件 → 发布 ROS topics)
```

---

## 9. 下一步行动

1. **立即**: 开始阶段 A — 创建 conda 环境，安装 ROS Noetic
2. 同时进行: 阶段 B 的前期调研 — 确定语义标签方案 (A/B/C)，查找相机内参
