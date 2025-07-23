# AReaL GPU Monitor - 完整工具包

## 概述

我为你创建了一个完整的AReaL GPU监控工具包，可以记录AReaL运行过程中各个actor对GPU的使用量，包括利用率和显存占用。这个工具包包含多个组件，提供了灵活的使用方式。

## 创建的文件

### 1. 核心监控脚本

#### `areal_gpu_monitor_simple.py` (推荐使用)
- **功能**: 简化版本的GPU监控脚本
- **特点**: 
  - 不依赖Ray API，通过进程检测识别AReaL进程
  - 更稳定可靠，兼容性更好
  - 支持实时监控和数据收集
- **使用方法**: 
  ```bash
  python areal_gpu_monitor_simple.py --experiment_name <exp_name> --trial_name <trial_name>
  ```

#### `areal_gpu_monitor.py`
- **功能**: 完整版本的GPU监控脚本
- **特点**: 
  - 使用Ray API获取actor信息
  - 更精确的actor识别
  - 需要Ray集群运行
- **使用方法**: 
  ```bash
  python areal_gpu_monitor.py --experiment_name <exp_name> --trial_name <trial_name>
  ```

### 2. 依赖和配置

#### `gpu_monitor_requirements.txt`
- 包含所有必需的Python包
- 安装命令: `pip install -r gpu_monitor_requirements.txt`

#### `README_gpu_monitor.md`
- 详细的使用说明文档
- 包含安装、使用、故障排除等信息

### 3. 辅助工具

#### `start_gpu_monitor.sh`
- **功能**: 快速启动脚本
- **特点**: 
  - 自动检查依赖
  - 自动测试GPU访问
  - 简化启动过程
- **使用方法**: 
  ```bash
  ./start_gpu_monitor.sh <experiment_name> <trial_name> [output_dir] [interval]
  ```

#### `test_gpu_monitor.py`
- **功能**: 测试脚本
- **特点**: 
  - 验证GPU访问功能
  - 测试进程检测
  - 测试数据收集
- **使用方法**: 
  ```bash
  python test_gpu_monitor.py
  ```

#### `example_usage.py`
- **功能**: 使用示例脚本
- **特点**: 
  - 展示程序化使用方式
  - 包含数据分析功能
  - 提供多种使用模式

## 主要功能

### 1. 进程识别
- 自动识别AReaL相关进程:
  - `master_worker`: 主控进程
  - `rollout_worker`: 数据收集进程  
  - `model_worker`: 模型推理进程
  - `gserver_manager`: 生成服务器管理进程

### 2. GPU监控
- 实时监控GPU利用率
- 监控显存使用情况
- 按进程和按设备分别统计
- 支持多GPU环境

### 3. 数据记录
- 保存为CSV格式，便于分析
- 保存为JSON格式，便于程序处理
- 包含时间戳、进程信息、GPU指标等

### 4. 可视化
- 自动生成4个子图的可视化报告
- 显示GPU利用率、显存使用量趋势
- 按进程和按设备分别展示

## 数据字段

监控脚本会记录以下数据字段：

| 字段名 | 说明 |
|--------|------|
| timestamp | 时间戳（ISO格式） |
| process_name | 进程名称 |
| pid | 进程ID |
| gpu_index | GPU设备索引 |
| gpu_utilization | GPU利用率（%） |
| memory_utilization | 显存利用率（%） |
| memory_used_mb | 已使用显存（MB） |
| memory_total_mb | 总显存（MB） |
| memory_free_mb | 空闲显存（MB） |
| process_count | 使用该GPU的进程数量 |

## 使用流程

### 1. 安装依赖
```bash
pip install -r gpu_monitor_requirements.txt
```

### 2. 测试功能
```bash
python test_gpu_monitor.py
```

### 3. 启动监控
```bash
# 方法1: 使用快速启动脚本
./start_gpu_monitor.sh my_experiment trial_001

# 方法2: 直接使用Python脚本
python areal_gpu_monitor_simple.py --experiment_name my_experiment --trial_name trial_001
```

### 4. 查看结果
监控结束后，会在输出目录生成：
- `gpu_usage_YYYYMMDD_HHMMSS.csv`: 数据文件
- `gpu_usage_YYYYMMDD_HHMMSS.json`: 元数据文件  
- `gpu_usage_plot_YYYYMMDD_HHMMSS.png`: 可视化图表

## 技术特点

### 1. 稳定性
- 使用pynvml库直接访问GPU信息
- 进程检测基于psutil，兼容性好
- 异常处理完善，不会因单个错误中断监控

### 2. 灵活性
- 支持自定义监控间隔
- 支持自定义输出目录
- 支持程序化调用

### 3. 可扩展性
- 模块化设计，易于扩展新功能
- 支持添加新的监控指标
- 支持自定义可视化

## 注意事项

1. **权限要求**: 需要访问GPU设备的权限
2. **依赖环境**: 需要NVIDIA驱动和pynvml库
3. **性能影响**: 监控脚本本身消耗少量资源
4. **数据量**: 长时间监控会产生大量数据文件

## 扩展建议

1. **实时Web界面**: 可以添加Flask/Django服务器提供Web监控界面
2. **告警功能**: 当GPU使用率过高时发送邮件或消息通知
3. **历史对比**: 支持多个实验的数据对比分析
4. **分布式监控**: 支持多节点集群的统一监控
5. **自定义指标**: 添加更多GPU相关指标的监控

这个工具包提供了完整的AReaL GPU监控解决方案，可以满足你的需求。建议先使用简化版本进行测试，确认功能正常后再用于生产环境。 