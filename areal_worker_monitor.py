#!/usr/bin/env python3eaL Worker Monitor and Gantt Chart Generator

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
- LOST: 丢失
- UNKNOWN: 未知
"""

import asyncio
import json
import time
import argparse
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.dates import DateFormatter, MinuteLocator
import numpy as np
import realhf.base.name_resolve as name_resolve
import realhf.base.names as names
import realhf.base.constants as constants
from realhf.system.worker_base import WorkerServerStatus


@dataclass
class WorkerEvent:
   Worker状态事件"  worker_name: str
    worker_type: str
    worker_index: int
    status: str
    timestamp: float
    duration: Optional[float] = None


@dataclass
class WorkerTimeline:
  r时间线"  worker_name: str
    worker_type: str
    worker_index: int
    events: List[WorkerEvent]
    start_time: float
    end_time: float


class AReaLWorkerMonitor:
    ReaL Worker监控器"   
    def __init__(self, experiment_name: str, trial_name: str, poll_interval: float = 1):
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.poll_interval = poll_interval
        
        # 存储worker状态历史
        self.worker_status_history: Dict[str, List[WorkerEvent]] = defaultdict(list)
        self.worker_info: Dict[str, Dict] = {}
        
        # 监控线程控制
        self.monitoring = False
        self.monitor_thread = None
        
        # 颜色映射
        self.status_colors = {
           READY: #90EE90',      # 浅绿色
           RUNNING: #32CD32',    # 绿色
            PAUSED':#FFD700',     # 金色
            COMPLETED':#4169E1  # 蓝色
           ERROR: #DC143C',      # 红色
           INTERRUPTED:#FF6347 # 番茄色
            LOST: #800 # 深红色
          UNKNOWN: #808080,   # 灰色
        }
        
        # Worker类型颜色
        self.worker_type_colors = {
            generation_server': '#FF66B',
            gserver_manager': '#4ECDC4            model_worker': '#45B7D1',
          master_worker': '#96CEB4',
           rollout_worker': #FFEAA7        }

    def get_worker_status(self, worker_name: str) -> Optional[str]:
       orker状态"""
        try:
            status = name_resolve.wait(
                names.worker_status(
                    experiment_name=self.experiment_name,
                    trial_name=self.trial_name,
                    worker_name=worker_name
                ),
                timeout=5
            )
            return status
        except Exception as e:
            print(f"获取worker {worker_name}状态失败: {e}")
            return None

    def discover_workers(self) -> List[str]:
       有worker"
        workers = []
        
        # 尝试发现各种类型的worker
        worker_types = [generation_server', gserver_manager,model_worker                   master_worker,rollout_worker']
        
        for worker_type in worker_types:
            # 尝试不同的索引
            for i in range(100):  # 最多检查100个worker
                worker_name = f"{worker_type}/{i}"
                try:
                    status = self.get_worker_status(worker_name)
                    if status is not None:
                        workers.append(worker_name)
                        # 解析worker信息
                        self.worker_info[worker_name] = {
                        type': worker_type,
                      index                  status                   }
                        print(f发现worker: {worker_name} - {status})            except:
                    # 如果连续失败，可能已经到达该类型的worker数量上限
                    if i >0nd worker_name not in workers:
                        break
        
        return workers

    def monitor_workers(self):
     ker状态变化"        print(f"开始监控实验 {self.experiment_name}/{self.trial_name}")
        
        # 初始发现worker
        workers = self.discover_workers()
        print(f发现 {len(workers)} 个worker")
        
        # 记录初始状态
        for worker_name in workers:
            status = self.get_worker_status(worker_name)
            if status:
                event = WorkerEvent(
                    worker_name=worker_name,
                    worker_type=self.worker_info[worker_name]['type'],
                    worker_index=self.worker_info[worker_name]['index'],
                    status=status,
                    timestamp=time.time()
                )
                self.worker_status_history[worker_name].append(event)
        
        last_status = {worker: self.get_worker_status(worker) for worker in workers}
        
        while self.monitoring:
            try:
                # 检查现有worker状态变化
                for worker_name in workers[:]:  # 复制列表避免修改
                    current_status = self.get_worker_status(worker_name)
                    
                    if current_status is None:
                        # worker可能已经退出
                        if worker_name in last_status and last_status[worker_name] is not None:
                            event = WorkerEvent(
                                worker_name=worker_name,
                                worker_type=self.worker_info[worker_name]['type'],
                                worker_index=self.worker_info[worker_name]['index'],
                                status='LOST',
                                timestamp=time.time()
                            )
                            self.worker_status_history[worker_name].append(event)
                            last_status[worker_name] = None
                        continue
                    
                    if current_status != last_status.get(worker_name):
                        # 状态发生变化
                        event = WorkerEvent(
                            worker_name=worker_name,
                            worker_type=self.worker_info[worker_name]['type'],
                            worker_index=self.worker_info[worker_name]['index'],
                            status=current_status,
                            timestamp=time.time()
                        )
                        self.worker_status_history[worker_name].append(event)
                        last_status[worker_name] = current_status
                        print(f"{datetime.now().strftime('%H:%M:%S)} {worker_name}: {current_status}")
                
                # 尝试发现新的worker
                new_workers = self.discover_workers()
                for worker_name in new_workers:
                    if worker_name not in workers:
                        workers.append(worker_name)
                        status = self.get_worker_status(worker_name)
                        if status:
                            last_status[worker_name] = status
                            print(f"发现新worker: {worker_name} - {status}")
                
                time.sleep(self.poll_interval)
                
            except KeyboardInterrupt:
                print("监控被用户中断)             break
            except Exception as e:
                print(f监控过程中出现错误: {e})              time.sleep(self.poll_interval)
        
        print("监控结束")

    def start_monitoring(self):
        开始监控"
        if self.monitoring:
            print("监控已经在运行")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_workers)
        self.monitor_thread.start()
        print("监控已启动")

    def stop_monitoring(self):
        监控   self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("监控已停止")

    def generate_gantt_chart(self, output_file: str = "areal_workers_gantt.png", 
                           figsize: Tuple[int, int] = (16,10)):
  成甘特图       if not self.worker_status_history:
            print(没有监控数据，无法生成甘特图")
            return
        
        # 准备数据
        timelines = []
        all_times = []
        
        for worker_name, events in self.worker_status_history.items():
            if not events:
                continue
                
            # 计算事件持续时间
            for i in range(len(events)):
                if i < len(events) - 1:
                    events[i].duration = events[i + 1].timestamp - events[i].timestamp
                else:
                    # 最后一个事件，假设持续到当前时间
                    events[i].duration = time.time() - events[i].timestamp
            
            # 创建时间线
            timeline = WorkerTimeline(
                worker_name=worker_name,
                worker_type=events[0].worker_type,
                worker_index=events0x,
                events=events,
                start_time=events[0].timestamp,
                end_time=events[-1].timestamp + events[-1].duration
            )
            timelines.append(timeline)
            all_times.extend([e.timestamp for e in events])
            all_times.extend([e.timestamp + e.duration for e in events if e.duration])
        
        if not timelines:
            print("没有有效的时间线数据")
            return
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        # 设置时间范围
        min_time = min(all_times)
        max_time = max(all_times)
        
        # 按worker类型分组
        worker_types = {}
        for timeline in timelines:
            if timeline.worker_type not in worker_types:
                worker_types[timeline.worker_type] = []
            worker_types[timeline.worker_type].append(timeline)
        
        # 绘制甘特图
        y_pos = 0        y_labels = 
        y_ticks = []
        
        for worker_type, type_timelines in worker_types.items():
            # 按worker索引排序
            type_timelines.sort(key=lambda x: x.worker_index)
            
            for timeline in type_timelines:
                y_labels.append(f"{timeline.worker_type}/{timeline.worker_index})
                y_ticks.append(y_pos)
                
                # 绘制每个状态段
                for event in timeline.events:
                    if event.duration is None or event.duration <= 0:
                        continue
                    
                    start_dt = datetime.fromtimestamp(event.timestamp)
                    end_dt = datetime.fromtimestamp(event.timestamp + event.duration)
                    
                    # 创建矩形
                    rect = patches.Rectangle(
                        (start_dt, y_pos - 0.3),
                        end_dt - start_dt,
           0.6                   facecolor=self.status_colors.get(event.status, '#808080'),
                        edgecolor='black',
                        linewidth=0.5,
                        alpha=0.8
                    )
                    ax.add_patch(rect)
                    
                    # 添加状态标签
                    if event.duration > 30  # 只对持续时间超过30秒的事件添加标签
                        ax.text(
                            start_dt + (end_dt - start_dt) / 2,
                            y_pos,
                            event.status,
                            ha='center',
                            va='center',
                            fontsize=8,
                            fontweight='bold',
                            color='black'
                        )
                
                y_pos += 1        
        # 设置图表属性
        ax.set_ylim(-05, len(y_labels) - 00.5)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        
        # 设置时间轴
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(MinuteLocator(interval=5))
        
        # 添加图例
        legend_elements = []
        for status, color in self.status_colors.items():
            legend_elements.append(patches.Patch(color=color, label=status))
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(10.15)
        
        # 设置标题和标签
        ax.set_title(fAReaL Worker 运行状态甘特图\n{self.experiment_name}/{self.trial_name}', 
                    fontsize=16, fontweight=bold')
        ax.set_xlabel('时间', fontsize=12     ax.set_ylabel(Worker', fontsize=12)
        
        # 旋转x轴标签
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(output_file, dpi=300bbox_inches='tight)
        print(f"甘特图已保存到: {output_file}")
        
        # 显示图表
        plt.show()

    def save_monitoring_data(self, output_file: str = areal_worker_data.json"):
   保存监控数据
        data = {
            experiment_name: self.experiment_name,
            trial_name: self.trial_name,
        worker_info': self.worker_info,
        worker_status_history':[object Object]            worker: [asdict(event) for event in events]
                for worker, events in self.worker_status_history.items()
            }
        }
        
        with open(output_file, w, encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f监控数据已保存到: {output_file}")

    def load_monitoring_data(self, input_file: str):
   加载监控数据"""
        with open(input_file, r, encoding='utf-8') as f:
            data = json.load(f)
        
        self.experiment_name = data['experiment_name']
        self.trial_name = data['trial_name']
        self.worker_info = data['worker_info']
        
        # 重建事件对象
        self.worker_status_history.clear()
        for worker, events_data in data['worker_status_history'].items():
            events =          for event_data in events_data:
                event = WorkerEvent(
                    worker_name=event_data['worker_name'],
                    worker_type=event_data['worker_type'],
                    worker_index=event_dataworker_index                   status=event_data['status'],
                    timestamp=event_data['timestamp'],
                    duration=event_data.get('duration)                )
                events.append(event)
            self.worker_status_history[worker] = events
        
        print(f"监控数据已从 {input_file} 加载")


def main():
    parser = argparse.ArgumentParser(description=AReaL Worker监控和甘特图生成工具')
    parser.add_argument(--experiment', '-e', required=True, help=实验名称')
    parser.add_argument('--trial', '-t', required=True, help=试验名称')
    parser.add_argument(--poll-interval',-pype=float, default=1, help='轮询间隔(秒)')
    parser.add_argument('--output,-o, default='areal_workers_gantt.png,help='输出文件路径')
    parser.add_argument('--monitor', '-m', action=store_true,help='启动实时监控')
    parser.add_argument('--duration, d', type=int, help='监控持续时间(秒)')
    parser.add_argument('--load-data',-l, help='从JSON文件加载监控数据')
    parser.add_argument('--save-data', -s', help='保存监控数据到JSON文件)
    args = parser.parse_args()
    
    # 创建监控器
    monitor = AReaLWorkerMonitor(args.experiment, args.trial, args.poll_interval)
    
    if args.load_data:
        # 从文件加载数据
        monitor.load_monitoring_data(args.load_data)
        monitor.generate_gantt_chart(args.output)
        if args.save_data:
            monitor.save_monitoring_data(args.save_data)
    elif args.monitor:
        # 实时监控
        try:
            monitor.start_monitoring()
            
            if args.duration:
                print(f"监控将持续 {args.duration} 秒...)              time.sleep(args.duration)
            else:
                print("按 Ctrl+C 停止监控...)             while True:
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            print("\n收到中断信号，停止监控...)
        finally:
            monitor.stop_monitoring()
            
            # 生成甘特图
            monitor.generate_gantt_chart(args.output)
            
            # 保存数据
            if args.save_data:
                monitor.save_monitoring_data(args.save_data)
    else:
        # 只生成当前状态的甘特图
        print("发现当前worker状态...)
        workers = monitor.discover_workers()
        if workers:
            monitor.generate_gantt_chart(args.output)
        else:
            print("未发现任何worker")


if __name__ == "__main__":
    main() 