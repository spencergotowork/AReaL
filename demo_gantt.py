#!/usr/bin/env python3
"""
AReaL Worker甘特图演示脚本

这个脚本生成模拟的worker状态数据，用于演示甘特图功能
"""

import random
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


@dataclass
class MockWorkerSnapshot:
    """模拟Worker状态快照"""
    worker_name: str
    worker_type: str
    worker_index: int
    status: str
    timestamp: float


class MockWorkerMonitor:
    """模拟Worker监控器"""
    
    def __init__(self):
        # 模拟worker配置
        self.workers = {
            'model_worker/0': {'type': 'model_worker', 'index': 0, 'category': 'training'},
            'model_worker/1': {'type': 'model_worker', 'index': 1, 'category': 'training'},
            'master_worker/0': {'type': 'master_worker', 'index': 0, 'category': 'training'},
            'rollout_worker/0': {'type': 'rollout_worker', 'index': 0, 'category': 'inference'},
            'rollout_worker/1': {'type': 'rollout_worker', 'index': 1, 'category': 'inference'},
            'rollout_worker/2': {'type': 'rollout_worker', 'index': 2, 'category': 'inference'},
            'generation_server/0': {'type': 'generation_server', 'index': 0, 'category': 'inference'},
            'gserver_manager/0': {'type': 'gserver_manager', 'index': 0, 'category': 'inference'},
        }
        
        self.worker_snapshots: Dict[str, List[MockWorkerSnapshot]] = defaultdict(list)
        
        # 状态配置
        self.status_colors = {
            'READY': '#90EE90',      # 浅绿色
            'RUNNING': '#32CD32',    # 绿色  
            'PAUSED': '#FFD700',     # 金色
            'COMPLETED': '#4169E1',  # 蓝色
            'ERROR': '#DC143C',      # 红色
            'INTERRUPTED': '#FF6347', # 番茄色
        }

    def generate_mock_data(self, duration_seconds: int = 120):
        """生成模拟的worker状态数据"""
        print(f"生成 {duration_seconds} 秒的模拟数据...")
        
        start_time = time.time()
        
        # 为每个worker生成状态变化序列
        for worker_name, info in self.workers.items():
            current_time = start_time
            current_status = 'READY'
            
            # 初始状态
            self.worker_snapshots[worker_name].append(
                MockWorkerSnapshot(
                    worker_name=worker_name,
                    worker_type=info['type'],
                    worker_index=info['index'],
                    status=current_status,
                    timestamp=current_time
                )
            )
            
            # 生成状态变化
            while current_time < start_time + duration_seconds:
                # 根据worker类型模拟不同的行为模式
                if info['category'] == 'training':
                    # 训练worker: 更多的RUNNING和PAUSED状态
                    next_status = self._get_next_training_status(current_status)
                    time_delta = random.uniform(5, 20)  # 5-20秒的状态持续时间
                else:
                    # 推理worker: 更频繁的状态变化
                    next_status = self._get_next_inference_status(current_status)
                    time_delta = random.uniform(2, 15)  # 2-15秒的状态持续时间
                
                current_time += time_delta
                if current_time >= start_time + duration_seconds:
                    break
                    
                self.worker_snapshots[worker_name].append(
                    MockWorkerSnapshot(
                        worker_name=worker_name,
                        worker_type=info['type'],
                        worker_index=info['index'],
                        status=next_status,
                        timestamp=current_time
                    )
                )
                current_status = next_status
        
        print("模拟数据生成完成")

    def _get_next_training_status(self, current_status: str) -> str:
        """为训练worker生成下一个状态"""
        transitions = {
            'READY': ['RUNNING', 'PAUSED'],
            'RUNNING': ['RUNNING', 'PAUSED', 'COMPLETED'],
            'PAUSED': ['RUNNING', 'READY'],
            'COMPLETED': ['READY'],
            'ERROR': ['READY']
        }
        
        # 训练worker更倾向于保持RUNNING状态
        if current_status == 'RUNNING':
            return random.choices(['RUNNING', 'PAUSED'], weights=[0.8, 0.2])[0]
        elif current_status == 'READY':
            return random.choices(['RUNNING', 'PAUSED'], weights=[0.9, 0.1])[0]
        else:
            return random.choice(transitions.get(current_status, ['RUNNING']))

    def _get_next_inference_status(self, current_status: str) -> str:
        """为推理worker生成下一个状态"""
        transitions = {
            'READY': ['RUNNING'],
            'RUNNING': ['RUNNING', 'PAUSED', 'READY'],
            'PAUSED': ['RUNNING', 'READY'],
            'COMPLETED': ['READY'],
            'ERROR': ['READY']
        }
        
        # 推理worker状态变化更频繁
        if current_status == 'RUNNING':
            return random.choices(['RUNNING', 'PAUSED', 'READY'], weights=[0.7, 0.2, 0.1])[0]
        else:
            return random.choice(transitions.get(current_status, ['RUNNING']))

    def generate_gantt_chart(self, output_file: str = "demo_areal_gantt.png"):
        """生成甘特图"""
        if not self.worker_snapshots:
            print("没有数据")
            return
        
        # 获取时间范围
        all_timestamps = []
        for snapshots in self.worker_snapshots.values():
            all_timestamps.extend([s.timestamp for s in snapshots])
        
        min_time = min(all_timestamps)
        max_time = max(all_timestamps)
        
        # 按类型排序worker
        sorted_workers = self._sort_workers_for_display()
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(16, max(6, len(sorted_workers) * 0.6)))
        
        # 绘制每个worker的时间线
        y_pos = 0
        y_labels = []
        y_positions = []
        
        training_start = None
        inference_start = None
        
        for worker_name in sorted_workers:
            worker_info = self.workers[worker_name]
            snapshots = self.worker_snapshots[worker_name]
            
            # 记录类型分组位置
            if worker_info['category'] == 'training' and training_start is None:
                training_start = y_pos
            elif worker_info['category'] == 'inference' and inference_start is None:
                inference_start = y_pos
            
            # 绘制时间线
            self._draw_worker_timeline(ax, worker_name, snapshots, y_pos, min_time, max_time)
            
            label = f"{worker_info['type']}-{worker_info['index']}"
            y_labels.append(label)
            y_positions.append(y_pos)
            y_pos += 1
        
        # 添加分隔线
        if training_start is not None and inference_start is not None:
            separator_y = (training_start + inference_start - 1) / 2 + 0.5
            ax.axhline(y=separator_y, color='gray', linestyle='--', alpha=0.5, linewidth=2)
            
            ax.text(min_time, training_start - 0.3, 'Training Workers (异步训练)', 
                   fontweight='bold', fontsize=10, color='blue')
            ax.text(min_time, inference_start - 0.3, 'Inference Workers (异步推理)', 
                   fontweight='bold', fontsize=10, color='red')
        
        # 设置轴
        ax.set_ylim(-0.5, y_pos - 0.5)
        ax.set_xlim(min_time, max_time)
        
        # 时间轴
        time_range = max_time - min_time
        num_ticks = 12
        time_ticks = np.linspace(min_time, max_time, num_ticks)
        time_labels = [datetime.fromtimestamp(t).strftime('%H:%M:%S') for t in time_ticks]
        ax.set_xticks(time_ticks)
        ax.set_xticklabels(time_labels, rotation=45)
        
        # Y轴
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=9)
        
        # 标题
        ax.set_title('AReaL Worker 甘特图演示 - 训练推理异步并行\n'
                    f'持续时间: {time_range:.1f}秒 | 展示训练和推理的完全解耦', 
                    fontsize=12, pad=20)
        ax.set_xlabel('时间', fontsize=10)
        ax.set_ylabel('Worker', fontsize=10)
        
        # 图例
        self._add_legend(ax)
        
        # 网格
        ax.grid(True, alpha=0.3, axis='x')
        
        # 添加异步并行说明
        ax.text(0.02, 0.98, 
               '异步并行特性:\n'
               '• 训练和推理完全解耦\n'
               '• 不同worker独立运行\n'
               '• 无需等待最长序列完成\n'
               '• 显著提升GPU利用率',
               transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"演示甘特图已保存至: {output_file}")
        
        self._print_analysis()

    def _sort_workers_for_display(self) -> List[str]:
        """排序worker用于显示"""
        workers_by_category = defaultdict(list)
        
        for worker_name, info in self.workers.items():
            workers_by_category[info['category']].append((worker_name, info))
        
        sorted_workers = []
        for category in ['training', 'inference']:
            if category in workers_by_category:
                category_workers = workers_by_category[category]
                category_workers.sort(key=lambda x: (x[1]['type'], x[1]['index']))
                sorted_workers.extend([w[0] for w in category_workers])
        
        return sorted_workers

    def _draw_worker_timeline(self, ax, worker_name: str, snapshots: List[MockWorkerSnapshot], 
                            y_pos: int, min_time: float, max_time: float):
        """绘制worker时间线"""
        if not snapshots:
            return
        
        current_status = 'READY'
        last_time = min_time
        
        for snapshot in snapshots:
            if snapshot.timestamp < min_time or snapshot.timestamp > max_time:
                continue
                
            if snapshot.status != current_status:
                if snapshot.timestamp > last_time:
                    duration = snapshot.timestamp - last_time
                    self._draw_status_bar(ax, last_time, duration, y_pos, current_status)
                
                current_status = snapshot.status
                last_time = snapshot.timestamp
        
        # 最后一段
        if last_time < max_time:
            duration = max_time - last_time
            self._draw_status_bar(ax, last_time, duration, y_pos, current_status)

    def _draw_status_bar(self, ax, start_time: float, duration: float, y_pos: int, status: str):
        """绘制状态条"""
        if duration <= 0:
            return
        
        color = self.status_colors.get(status, '#808080')
        
        bar = patches.Rectangle(
            (start_time, y_pos - 0.35), duration, 0.7,
            facecolor=color, alpha=0.8, edgecolor='black', linewidth=0.5
        )
        ax.add_patch(bar)
        
        # 添加状态文本
        time_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        if duration > time_range * 0.03:
            ax.text(start_time + duration/2, y_pos, status, 
                   ha='center', va='center', fontsize=8, fontweight='bold',
                   color='white' if status in ['ERROR', 'INTERRUPTED'] else 'black')

    def _add_legend(self, ax):
        """添加图例"""
        legend_elements = []
        for status, color in self.status_colors.items():
            legend_elements.append(patches.Patch(color=color, label=status))
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.12, 1))

    def _print_analysis(self):
        """打印分析结果"""
        print(f"\n=== 演示数据分析 ===")
        print(f"模拟的Worker数量: {len(self.workers)}")
        
        # 按类型统计
        type_counts = defaultdict(int)
        category_counts = defaultdict(int)
        
        for info in self.workers.values():
            type_counts[info['type']] += 1
            category_counts[info['category']] += 1
        
        print("\n按类型分布:")
        for worker_type, count in sorted(type_counts.items()):
            print(f"  {worker_type}: {count}")
        
        print("\n按功能分组:")
        for category, count in sorted(category_counts.items()):
            print(f"  {category}: {count}")
        
        # 状态统计
        total_snapshots = sum(len(snapshots) for snapshots in self.worker_snapshots.values())
        print(f"\n总状态变化数: {total_snapshots}")
        
        print("\n这个甘特图展示了AReaL的核心优势:")
        print("✓ 训练worker和推理worker完全异步运行")
        print("✓ 不需要等待批次中最长序列完成")
        print("✓ 显著减少GPU空闲时间")
        print("✓ 支持大规模分布式并行")


def main():
    print("AReaL Worker甘特图演示")
    print("="*50)
    
    # 创建监控器
    monitor = MockWorkerMonitor()
    
    # 生成模拟数据
    monitor.generate_mock_data(duration_seconds=120)
    
    # 生成甘特图
    monitor.generate_gantt_chart()
    
    print("\n演示完成! 请查看生成的 demo_areal_gantt.png 文件")
    print("这个甘特图展示了AReaL训练推理异步并行的特性")


if __name__ == "__main__":
    main() 