#!/usr/bin/env python3
"""
简化的AReaL Worker甘特图生成器

此工具直接监控worker状态并生成甘特图，展示训练和推理的异步并行特性。
"""

import argparse
import json
import time
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import realhf.base.constants as constants
import realhf.base.name_resolve as name_resolve
import realhf.base.names as names


@dataclass
class WorkerSnapshot:
    """Worker状态快照"""
    worker_name: str
    worker_type: str
    worker_index: int
    status: str
    timestamp: float


class SimpleWorkerMonitor:
    """简化的Worker监控器"""
    
    def __init__(self, experiment_name: str, trial_name: str):
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        
        # 存储worker状态历史
        self.worker_snapshots: Dict[str, List[WorkerSnapshot]] = defaultdict(list)
        self.discovered_workers: Dict[str, dict] = {}
        
        # 颜色配置
        self.status_colors = {
            'READY': '#90EE90',      # 浅绿色
            'RUNNING': '#32CD32',    # 绿色  
            'PAUSED': '#FFD700',     # 金色
            'COMPLETED': '#4169E1',  # 蓝色
            'ERROR': '#DC143C',      # 红色
            'INTERRUPTED': '#FF6347', # 番茄色
            'UNKNOWN': '#808080',    # 灰色
        }
        
        # Worker类型配置
        self.worker_type_info = {
            'generation_server': {'color': '#FF6B9D', 'category': 'inference'},
            'gserver_manager': {'color': '#4ECDC4', 'category': 'inference'},
            'rollout_worker': {'color': '#FFEAA7', 'category': 'inference'},
            'model_worker': {'color': '#45B7D1', 'category': 'training'},
            'master_worker': {'color': '#96CEB4', 'category': 'training'},
        }

    def discover_workers(self) -> List[str]:
        """发现活跃的worker"""
        workers = []
        worker_types = ['generation_server', 'gserver_manager', 'model_worker', 
                       'master_worker', 'rollout_worker']
        
        for worker_type in worker_types:
            for i in range(50):  # 检查每种类型最多50个worker
                worker_name = f"{worker_type}/{i}"
                try:
                    status = self._get_worker_status(worker_name)
                    if status is not None:
                        workers.append(worker_name)
                        self.discovered_workers[worker_name] = {
                            'type': worker_type,
                            'index': i,
                            'category': self.worker_type_info.get(worker_type, {}).get('category', 'other')
                        }
                        print(f"发现worker: {worker_name} - {status}")
                except:
                    # 连续失败3次就跳过
                    if i > 2:
                        break
        
        return workers

    def _get_worker_status(self, worker_name: str) -> Optional[str]:
        """获取worker当前状态"""
        try:
            status_key = names.worker_status(
                experiment_name=self.experiment_name,
                trial_name=self.trial_name,
                worker_name=worker_name
            )
            return name_resolve.wait(status_key, timeout=2)
        except:
            return None

    def take_snapshot(self):
        """拍摄当前所有worker状态的快照"""
        current_time = time.time()
        
        for worker_name, info in self.discovered_workers.items():
            status = self._get_worker_status(worker_name)
            if status:
                snapshot = WorkerSnapshot(
                    worker_name=worker_name,
                    worker_type=info['type'],
                    worker_index=info['index'],
                    status=status,
                    timestamp=current_time
                )
                self.worker_snapshots[worker_name].append(snapshot)

    def monitor_for_duration(self, duration_seconds: float, interval: float = 1.0):
        """监控指定时间"""
        print(f"监控 {duration_seconds} 秒...")
        
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            self.take_snapshot()
            time.sleep(interval)
            
            # 打印进度
            elapsed = time.time() - start_time
            progress = elapsed / duration_seconds * 100
            print(f"\r进度: {progress:.1f}%", end='', flush=True)
        
        print("\n监控完成")

    def generate_gantt_chart(self, output_file: str = "areal_gantt.png"):
        """生成甘特图"""
        if not self.worker_snapshots:
            print("没有监控数据")
            return
        
        # 获取时间范围
        all_timestamps = []
        for snapshots in self.worker_snapshots.values():
            all_timestamps.extend([s.timestamp for s in snapshots])
        
        if not all_timestamps:
            print("没有有效的时间数据")
            return
            
        min_time = min(all_timestamps)
        max_time = max(all_timestamps)
        
        # 按类型和索引排序worker
        sorted_workers = self._sort_workers_for_display()
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(16, max(6, len(sorted_workers) * 0.6)))
        
        # 绘制每个worker的时间线
        y_pos = 0
        y_labels = []
        y_positions = []
        
        # 添加类型分隔线的位置
        training_start = None
        inference_start = None
        
        for worker_name in sorted_workers:
            worker_info = self.discovered_workers[worker_name]
            snapshots = self.worker_snapshots[worker_name]
            
            # 记录类型分组的开始位置
            if worker_info['category'] == 'training' and training_start is None:
                training_start = y_pos
            elif worker_info['category'] == 'inference' and inference_start is None:
                inference_start = y_pos
            
            # 绘制worker时间线
            self._draw_worker_timeline(ax, worker_name, snapshots, y_pos, min_time, max_time)
            
            # 设置标签
            label = f"{worker_info['type']}-{worker_info['index']}"
            y_labels.append(label)
            y_positions.append(y_pos)
            y_pos += 1
        
        # 添加类型分隔线
        if training_start is not None and inference_start is not None:
            separator_y = (training_start + inference_start - 1) / 2 + 0.5
            ax.axhline(y=separator_y, color='gray', linestyle='--', alpha=0.5, linewidth=2)
            
            # 添加类型标签
            ax.text(min_time, training_start - 0.3, 'Training Workers', 
                   fontweight='bold', fontsize=10, color='blue')
            ax.text(min_time, inference_start - 0.3, 'Inference Workers', 
                   fontweight='bold', fontsize=10, color='red')
        
        # 设置轴
        ax.set_ylim(-0.5, y_pos - 0.5)
        ax.set_xlim(min_time, max_time)
        
        # 时间轴格式化
        time_range = max_time - min_time
        if time_range < 60:  # 小于1分钟，显示秒
            time_format = '%H:%M:%S'
            num_ticks = 10
        elif time_range < 3600:  # 小于1小时，显示分钟
            time_format = '%H:%M'
            num_ticks = 12
        else:  # 显示小时
            time_format = '%H:%M'
            num_ticks = 8
            
        time_ticks = np.linspace(min_time, max_time, num_ticks)
        time_labels = [datetime.fromtimestamp(t).strftime(time_format) for t in time_ticks]
        ax.set_xticks(time_ticks)
        ax.set_xticklabels(time_labels, rotation=45)
        
        # Y轴
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=9)
        
        # 标题和标签
        duration_str = f"{time_range:.1f}秒" if time_range < 60 else f"{time_range/60:.1f}分钟"
        ax.set_title(f'AReaL Worker 甘特图 - 训练推理异步并行\n'
                    f'实验: {self.experiment_name}/{self.trial_name} | 持续时间: {duration_str}', 
                    fontsize=12, pad=20)
        ax.set_xlabel('时间', fontsize=10)
        ax.set_ylabel('Worker', fontsize=10)
        
        # 添加图例
        self._add_legend(ax)
        
        # 网格
        ax.grid(True, alpha=0.3, axis='x')
        
        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"甘特图已保存至: {output_file}")
        
        # 显示统计信息
        self._print_statistics(time_range)

    def _sort_workers_for_display(self) -> List[str]:
        """按类型和索引排序worker用于显示"""
        workers_by_category = defaultdict(list)
        
        for worker_name, info in self.discovered_workers.items():
            workers_by_category[info['category']].append((worker_name, info))
        
        sorted_workers = []
        
        # 先显示训练worker，再显示推理worker
        for category in ['training', 'inference', 'other']:
            if category in workers_by_category:
                # 按worker类型和索引排序
                category_workers = workers_by_category[category]
                category_workers.sort(key=lambda x: (x[1]['type'], x[1]['index']))
                sorted_workers.extend([w[0] for w in category_workers])
        
        return sorted_workers

    def _draw_worker_timeline(self, ax, worker_name: str, snapshots: List[WorkerSnapshot], 
                            y_pos: int, min_time: float, max_time: float):
        """绘制单个worker的时间线"""
        if not snapshots:
            return
        
        # 构建状态段
        current_status = 'READY'
        last_time = min_time
        
        for snapshot in snapshots:
            if snapshot.timestamp < min_time or snapshot.timestamp > max_time:
                continue
                
            if snapshot.status != current_status:
                # 绘制前一个状态段
                if snapshot.timestamp > last_time:
                    duration = snapshot.timestamp - last_time
                    self._draw_status_bar(ax, last_time, duration, y_pos, current_status)
                
                # 更新状态
                current_status = snapshot.status
                last_time = snapshot.timestamp
        
        # 绘制最后一个状态段
        if last_time < max_time:
            duration = max_time - last_time
            self._draw_status_bar(ax, last_time, duration, y_pos, current_status)

    def _draw_status_bar(self, ax, start_time: float, duration: float, y_pos: int, status: str):
        """绘制状态条"""
        if duration <= 0:
            return
        
        color = self.status_colors.get(status, self.status_colors['UNKNOWN'])
        
        bar = patches.Rectangle(
            (start_time, y_pos - 0.35), duration, 0.7,
            facecolor=color, alpha=0.8, edgecolor='black', linewidth=0.5
        )
        ax.add_patch(bar)
        
        # 在状态条上添加文本（如果足够长）
        time_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        if duration > time_range * 0.03:  # 如果状态条长度超过时间范围的3%
            ax.text(start_time + duration/2, y_pos, status, 
                   ha='center', va='center', fontsize=8, fontweight='bold',
                   color='white' if status in ['ERROR', 'INTERRUPTED'] else 'black')

    def _add_legend(self, ax):
        """添加图例"""
        legend_elements = []
        
        # 状态图例
        for status, color in self.status_colors.items():
            if status != 'UNKNOWN':
                legend_elements.append(
                    patches.Patch(color=color, label=status)
                )
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.12, 1))

    def _print_statistics(self, duration: float):
        """打印统计信息"""
        print(f"\n=== 监控统计 ===")
        print(f"监控时长: {duration:.1f}秒")
        print(f"发现的Worker: {len(self.discovered_workers)}")
        
        # 按类型统计
        type_counts = defaultdict(int)
        category_counts = defaultdict(int)
        
        for info in self.discovered_workers.values():
            type_counts[info['type']] += 1
            category_counts[info['category']] += 1
        
        print("\n按类型分布:")
        for worker_type, count in sorted(type_counts.items()):
            print(f"  {worker_type}: {count}")
        
        print("\n按功能分组:")
        for category, count in sorted(category_counts.items()):
            print(f"  {category}: {count}")
        
        # 状态变化统计
        total_snapshots = sum(len(snapshots) for snapshots in self.worker_snapshots.values())
        print(f"\n总状态快照数: {total_snapshots}")

    def save_data(self, filename: str = "areal_monitor_data.json"):
        """保存监控数据到文件"""
        data = {
            'experiment_name': self.experiment_name,
            'trial_name': self.trial_name,
            'workers': self.discovered_workers,
            'snapshots': {}
        }
        
        # 转换快照数据为可序列化格式
        for worker_name, snapshots in self.worker_snapshots.items():
            data['snapshots'][worker_name] = [
                {
                    'worker_name': s.worker_name,
                    'worker_type': s.worker_type,
                    'worker_index': s.worker_index,
                    'status': s.status,
                    'timestamp': s.timestamp
                }
                for s in snapshots
            ]
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"监控数据已保存至: {filename}")


def main():
    parser = argparse.ArgumentParser(description='AReaL Worker甘特图生成器')
    parser.add_argument('--experiment-name', required=True, help='实验名称')
    parser.add_argument('--trial-name', required=True, help='试验名称')
    parser.add_argument('--duration', type=float, default=60, help='监控持续时间(秒)')
    parser.add_argument('--interval', type=float, default=1.0, help='监控间隔(秒)')
    parser.add_argument('--output', default='areal_gantt.png', help='输出图片文件名')
    parser.add_argument('--save-data', help='保存监控数据的JSON文件名')
    
    args = parser.parse_args()
    
    # 设置实验环境
    constants.set_experiment_trial_names(args.experiment_name, args.trial_name)
    
    # 创建监控器
    monitor = SimpleWorkerMonitor(args.experiment_name, args.trial_name)
    
    try:
        # 发现worker
        print("发现worker...")
        workers = monitor.discover_workers()
        
        if not workers:
            print("未发现任何worker，请确保AReaL实验正在运行")
            print("提示：检查实验名称和试验名称是否正确")
            return
        
        print(f"发现 {len(workers)} 个worker")
        
        # 开始监控
        monitor.monitor_for_duration(args.duration, args.interval)
        
        # 生成甘特图
        print("生成甘特图...")
        monitor.generate_gantt_chart(args.output)
        
        # 保存数据（如果指定）
        if args.save_data:
            monitor.save_data(args.save_data)
        
        print("完成！")
        
    except KeyboardInterrupt:
        print("\n\n监控被用户中断")
        if monitor.worker_snapshots:
            print("生成当前数据的甘特图...")
            monitor.generate_gantt_chart(args.output)
            if args.save_data:
                monitor.save_data(args.save_data)
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 