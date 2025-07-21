#!/usr/bin/env python3
"""
AReaL Worker Monitor and Gantt Chart Generator

这个工具用于监控AReaL系统中各个worker的运行时间和状态，并生成甘特图。

支持的worker类型：
- generation_server: 生成服务器
- gserver_manager: 生成服务器管理器  
- model_worker: 模型worker
- master_worker: 主worker
- rollout_worker: 回滚worker

支持的状态：
- READY: 就绪
- RUNNING: 运行中
- PAUSED: 暂停
- COMPLETED: 完成
- ERROR: 错误
- INTERRUPTED: 中断
"""

import argparse
import asyncio
import json
import os
import re
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import numpy as np

import realhf.base.constants as constants
import realhf.base.name_resolve as name_resolve
import realhf.base.names as names
from realhf.system.worker_base import WorkerServerStatus


@dataclass
class WorkerEvent:
    """Worker状态事件"""
    worker_name: str
    worker_type: str
    worker_index: int
    status: str
    timestamp: float
    duration: Optional[float] = None
    metadata: Optional[Dict] = None


@dataclass
class WorkerTimeline:
    """Worker时间线"""
    worker_name: str
    worker_type: str
    worker_index: int
    events: List[WorkerEvent]
    start_time: float
    end_time: float


class AReaLWorkerMonitor:
    """AReaL Worker监控器"""
    
    def __init__(self, experiment_name: str, trial_name: str, poll_interval: float = 1.0):
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.poll_interval = poll_interval
        
        # 存储worker状态历史
        self.worker_timelines: Dict[str, WorkerTimeline] = {}
        self.worker_info: Dict[str, Dict] = {}
        
        # 监控线程控制
        self.monitoring = False
        self.monitor_thread = None
        
        # 颜色映射
        self.status_colors = {
            'READY': '#90EE90',      # 浅绿色
            'RUNNING': '#32CD32',    # 绿色
            'PAUSED': '#FFD700',     # 金色
            'COMPLETED': '#4169E1',  # 蓝色
            'ERROR': '#DC143C',      # 红色
            'INTERRUPTED': '#FF6347', # 番茄色
            'UNKNOWN': '#808080',    # 灰色
        }
        
        # Worker类型颜色
        self.worker_type_colors = {
            'generation_server': '#FF6B9D',
            'gserver_manager': '#4ECDC4',
            'model_worker': '#45B7D1',
            'master_worker': '#96CEB4',
            'rollout_worker': '#FFEAA7'
        }

    def discover_workers(self) -> List[str]:
        """发现所有可用的worker"""
        workers = []
        
        # 尝试发现各种类型的worker
        worker_types = ['generation_server', 'gserver_manager', 'model_worker', 
                       'master_worker', 'rollout_worker']
        
        for worker_type in worker_types:
            # 尝试不同的索引
            for i in range(20):  # 最多检查20个worker
                worker_name = f"{worker_type}/{i}"
                try:
                    # 检查worker状态
                    status_key = names.worker_status(
                        experiment_name=self.experiment_name,
                        trial_name=self.trial_name,
                        worker_name=worker_name
                    )
                    status = name_resolve.wait(status_key, timeout=1)
                    if status is not None:
                        workers.append(worker_name)
                        # 解析worker信息
                        self.worker_info[worker_name] = {
                            'type': worker_type,
                            'index': i,
                            'status': status,
                        }
                        print(f"发现worker: {worker_name} - {status}")
                except:
                    # 如果连续失败，可能已经到达该类型的worker数量上限
                    if i > 3 and worker_name not in workers:
                        break
        
        return workers

    def collect_worker_events(self, worker_name: str) -> List[WorkerEvent]:
        """收集单个worker的事件历史"""
        events = []
        
        try:
            # 获取历史事件
            history_pattern = f"time_monitor/{worker_name}/history/"
            
            # 尝试通过name_resolve查找历史事件
            # 这里我们使用一个模式匹配的方法
            for i in range(1000):  # 限制查找范围
                try:
                    timestamp = time.time() - i * 60  # 假设每分钟检查一次
                    history_key = names.worker_key(
                        experiment_name=self.experiment_name,
                        trial_name=self.trial_name,
                        key=f"time_monitor/{worker_name}/history/{timestamp}"
                    )
                    
                    event_data = name_resolve.wait(history_key, timeout=0.1)
                    if event_data:
                        event_dict = json.loads(event_data)
                        event = WorkerEvent(**event_dict)
                        events.append(event)
                        
                except:
                    continue
                    
        except Exception as e:
            print(f"收集worker {worker_name}事件失败: {e}")
        
        return sorted(events, key=lambda x: x.timestamp)

    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("开始监控worker状态...")

    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("停止监控")

    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                # 发现新的worker
                workers = self.discover_workers()
                
                # 收集每个worker的事件
                for worker_name in workers:
                    if worker_name not in self.worker_timelines:
                        events = self.collect_worker_events(worker_name)
                        if events:
                            info = self.worker_info[worker_name]
                            timeline = WorkerTimeline(
                                worker_name=worker_name,
                                worker_type=info['type'],
                                worker_index=info['index'],
                                events=events,
                                start_time=events[0].timestamp if events else time.time(),
                                end_time=events[-1].timestamp if events else time.time()
                            )
                            self.worker_timelines[worker_name] = timeline
                
                time.sleep(self.poll_interval)
                
            except Exception as e:
                print(f"监控循环出错: {e}")
                time.sleep(1)

    def generate_gantt_chart(self, output_file: str = "areal_workers_gantt.png", 
                           duration_hours: float = 1.0):
        """生成甘特图"""
        if not self.worker_timelines:
            print("没有找到worker时间线数据")
            return
            
        # 设置图形
        fig, ax = plt.subplots(figsize=(15, max(8, len(self.worker_timelines) * 0.8)))
        
        # 获取时间范围
        all_events = []
        for timeline in self.worker_timelines.values():
            all_events.extend(timeline.events)
            
        if not all_events:
            print("没有找到任何事件")
            return
            
        min_time = min(event.timestamp for event in all_events)
        max_time = max(event.timestamp for event in all_events)
        
        # 如果指定了持续时间，使用指定的时间窗口
        if duration_hours > 0:
            max_time = min_time + duration_hours * 3600
            
        # 按worker类型分组
        worker_groups = defaultdict(list)
        for timeline in self.worker_timelines.values():
            worker_groups[timeline.worker_type].append(timeline)
            
        # 排序worker
        y_pos = 0
        y_labels = []
        y_positions = []
        
        for worker_type, timelines in worker_groups.items():
            # 按索引排序
            timelines.sort(key=lambda x: x.worker_index)
            
            for timeline in timelines:
                # 绘制worker时间线
                self._draw_worker_timeline(ax, timeline, y_pos, min_time, max_time)
                
                y_labels.append(f"{timeline.worker_type}-{timeline.worker_index}")
                y_positions.append(y_pos)
                y_pos += 1
        
        # 设置图形属性
        ax.set_ylim(-0.5, y_pos - 0.5)
        ax.set_xlim(min_time, max_time)
        
        # 设置时间轴
        time_ticks = np.linspace(min_time, max_time, 10)
        time_labels = [datetime.fromtimestamp(t).strftime('%H:%M:%S') for t in time_ticks]
        ax.set_xticks(time_ticks)
        ax.set_xticklabels(time_labels, rotation=45)
        
        # 设置Y轴
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)
        
        # 标题和标签
        ax.set_title(f'AReaL Workers 甘特图 - {self.experiment_name}/{self.trial_name}')
        ax.set_xlabel('时间')
        ax.set_ylabel('Worker')
        
        # 添加图例
        self._add_legend(ax)
        
        # 网格
        ax.grid(True, alpha=0.3)
        
        # 保存图形
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"甘特图已保存至: {output_file}")
        
        return fig

    def _draw_worker_timeline(self, ax, timeline: WorkerTimeline, y_pos: int, 
                            min_time: float, max_time: float):
        """绘制单个worker的时间线"""
        events = [e for e in timeline.events if min_time <= e.timestamp <= max_time]
        
        if not events:
            return
            
        # 构建状态段
        current_status = 'READY'
        last_time = max(min_time, timeline.start_time)
        
        for event in events:
            if event.status != current_status:
                # 绘制上一个状态段
                if event.timestamp > last_time:
                    duration = event.timestamp - last_time
                    self._draw_status_bar(ax, last_time, duration, y_pos, current_status, 
                                        timeline.worker_type)
                
                # 更新状态
                current_status = event.status
                last_time = event.timestamp
        
        # 绘制最后一个状态段到结束时间
        if last_time < max_time:
            duration = max_time - last_time
            self._draw_status_bar(ax, last_time, duration, y_pos, current_status, 
                                timeline.worker_type)

    def _draw_status_bar(self, ax, start_time: float, duration: float, y_pos: int, 
                        status: str, worker_type: str):
        """绘制状态条"""
        if duration <= 0:
            return
            
        color = self.status_colors.get(status, self.status_colors['UNKNOWN'])
        
        # 添加worker类型的色调变化
        if worker_type in self.worker_type_colors:
            # 混合状态颜色和worker类型颜色
            import matplotlib.colors as mcolors
            type_color = self.worker_type_colors[worker_type]
            # 简单的颜色混合
            color = color  # 保持状态颜色主导
            
        bar = patches.Rectangle(
            (start_time, y_pos - 0.4), duration, 0.8,
            facecolor=color, alpha=0.8, edgecolor='black', linewidth=0.5
        )
        ax.add_patch(bar)
        
        # 如果状态段足够长，添加状态文本
        if duration > (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.05:
            ax.text(start_time + duration/2, y_pos, status, 
                   ha='center', va='center', fontsize=8, 
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
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

    def print_summary(self):
        """打印摘要信息"""
        if not self.worker_timelines:
            print("没有找到worker数据")
            return
            
        print(f"\n=== AReaL Worker 摘要 ===")
        print(f"实验: {self.experiment_name}/{self.trial_name}")
        print(f"发现的Worker数量: {len(self.worker_timelines)}")
        
        # 按类型分组统计
        type_counts = defaultdict(int)
        for timeline in self.worker_timelines.values():
            type_counts[timeline.worker_type] += 1
            
        print("\nWorker类型分布:")
        for worker_type, count in type_counts.items():
            print(f"  {worker_type}: {count}")
            
        print(f"\n总事件数: {sum(len(t.events) for t in self.worker_timelines.values())}")


def main():
    parser = argparse.ArgumentParser(description='AReaL Worker甘特图监控工具')
    parser.add_argument('--experiment-name', required=True, help='实验名称')
    parser.add_argument('--trial-name', required=True, help='试验名称')
    parser.add_argument('--output', default='areal_workers_gantt.png', help='输出文件名')
    parser.add_argument('--duration', type=float, default=1.0, help='监控持续时间(小时)')
    parser.add_argument('--poll-interval', type=float, default=1.0, help='轮询间隔(秒)')
    parser.add_argument('--monitor-only', action='store_true', help='只监控不生成图表')
    
    args = parser.parse_args()
    
    # 设置实验和试验名称
    constants.set_experiment_trial_names(args.experiment_name, args.trial_name)
    
    # 创建监控器
    monitor = AReaLWorkerMonitor(
        experiment_name=args.experiment_name,
        trial_name=args.trial_name,
        poll_interval=args.poll_interval
    )
    
    try:
        if args.monitor_only:
            # 只监控模式
            monitor.start_monitoring()
            print(f"监控中... 按Ctrl+C停止")
            while True:
                time.sleep(1)
        else:
            # 监控并生成图表
            print("发现worker...")
            workers = monitor.discover_workers()
            
            if not workers:
                print("未发现任何worker，请确保实验正在运行")
                return
                
            monitor.start_monitoring()
            
            # 监控指定时间
            print(f"监控 {args.duration} 小时...")
            time.sleep(args.duration * 3600)
            
            monitor.stop_monitoring()
            
            # 生成甘特图
            print("生成甘特图...")
            monitor.generate_gantt_chart(args.output, args.duration)
            
            # 打印摘要
            monitor.print_summary()
            
    except KeyboardInterrupt:
        print("\n监控被用户中断")
        monitor.stop_monitoring()
        
        # 如果有数据，仍然生成图表
        if monitor.worker_timelines and not args.monitor_only:
            print("生成当前数据的甘特图...")
            monitor.generate_gantt_chart(args.output)
            monitor.print_summary()


if __name__ == "__main__":
    main()