# AReaL GPU Monitor

这个工具用于监控AReaL训练过程中各个actor的GPU使用情况，包括GPU利用率和显存占用。

## 功能特性

- 实时监控AReaL各个worker进程的GPU使用情况
- 记录GPU利用率和显存占用数据
- 支持按进程和按GPU设备分别统计
- 训练结束后自动生成可视化图表
- 数据保存为CSV和JSON格式，便于后续分析

## 安装依赖

```bash
pip install -r gpu_monitor_requirements.txt
```

## 使用方法

### 1. 简化版本（推荐）

使用基于进程检测的简化版本，不需要Ray API：

```bash
python areal_gpu_monitor_simple.py --experiment_name <实验名称> --trial_name <试验名称>
```

### 2. 完整版本

使用Ray API获取actor信息的完整版本：

```bash
python areal_gpu_monitor.py --experiment_name <实验名称> --trial_name <试验名称>
```

### 参数说明

- `--experiment_name`: AReaL实验名称（必需）
- `--trial_name`: AReaL试验名称（必需）
- `--output_dir`: 输出目录，默认为 `gpu_monitor_data`
- `--interval`: 监控间隔（秒），默认为5秒

### 使用示例

```bash
# 基本使用
python areal_gpu_monitor_simple.py --experiment_name my_experiment --trial_name trial_001

# 自定义输出目录和监控间隔
python areal_gpu_monitor_simple.py \
    --experiment_name my_experiment \
    --trial_name trial_001 \
    --output_dir ./my_gpu_data \
    --interval 10
```

## 输出文件

监控结束后，会在输出目录中生成以下文件：

1. **CSV数据文件**: `gpu_usage_YYYYMMDD_HHMMSS.csv`
   - 包含所有时间点的GPU使用数据
   - 按时间戳、进程名、GPU索引组织

2. **JSON数据文件**: `gpu_usage_YYYYMMDD_HHMMSS.json`
   - 包含实验元数据和所有监控数据
   - 便于程序化处理

3. **可视化图表**: `gpu_usage_plot_YYYYMMDD_HHMMSS.png`
   - 4个子图显示不同维度的GPU使用情况
   - 包括按进程的GPU利用率、显存使用量
   - 按设备的总体GPU利用率和显存利用率

## 数据字段说明

CSV文件包含以下字段：

- `timestamp`: 时间戳（ISO格式）
- `process_name`: 进程名称（如 `rollout_worker/0`, `model_worker/1`）
- `pid`: 进程ID
- `gpu_index`: GPU设备索引
- `gpu_utilization`: GPU利用率（百分比）
- `memory_utilization`: 显存利用率（百分比）
- `memory_used_mb`: 已使用显存（MB）
- `memory_total_mb`: 总显存（MB）
- `memory_free_mb`: 空闲显存（MB）
- `process_count`: 使用该GPU的进程数量（仅overall记录）

## 监控的进程类型

脚本会自动识别以下AReaL相关的进程：

- `master_worker`: 主控进程
- `rollout_worker`: 数据收集进程
- `model_worker`: 模型推理进程
- `gserver_manager`: 生成服务器管理进程

## 注意事项

1. **权限要求**: 需要能够访问GPU设备信息，通常需要适当的权限
2. **Ray集群**: 如果使用完整版本，需要确保Ray集群正在运行
3. **进程识别**: 简化版本通过命令行参数识别AReaL进程，可能无法识别所有进程
4. **数据量**: 长时间监控会产生大量数据，注意磁盘空间
5. **性能影响**: 监控脚本本身会消耗少量CPU和内存资源

## 故障排除

### 无法获取GPU信息
- 检查NVIDIA驱动是否正确安装
- 确认有访问GPU设备的权限
- 验证pynvml库是否正确安装

### 无法找到AReaL进程
- 确认AReaL训练任务正在运行
- 检查进程命令行是否包含预期的关键词
- 尝试使用完整版本（需要Ray）

### 可视化图表显示异常
- 检查matplotlib是否正确安装
- 确认有图形界面或设置适当的后端
- 验证数据文件是否完整

## 扩展功能

可以根据需要扩展以下功能：

1. **实时Web界面**: 添加Web服务器提供实时监控界面
2. **告警功能**: 当GPU使用率过高或显存不足时发送告警
3. **历史数据对比**: 支持多个实验的数据对比分析
4. **自定义指标**: 添加更多GPU相关指标的监控
5. **分布式监控**: 支持多节点集群的GPU监控 