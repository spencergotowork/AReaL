#!/usr/bin/env python3
"""
简单的甘特图生成脚本，用于可视化AReaL框架中worker的运行时间线
"""

import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import sys
import argparse

def load_timeline_data(filepath):
    """加载时间线数据"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"错误：找不到文件 {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"错误：无法解析JSON文件 {filepath}")
        return None

def process_events(data):
    """处理事件数据，计算每个worker的运行时间段"""
    workers = {}
    
    for event in data:
        worker_name = event['worker_name']
        worker_type = event['worker_type']
        event_type = event['event']
        timestamp = event['timestamp']
        
        if worker_name not in workers:
            workers[worker_name] = {
                'type': worker_type,
                'events': {},
                'phases': []
            }
        
        workers[worker_name]['events'][event_type] = timestamp
    
    # 计算时间段
    for worker_name, worker_info in workers.items():
        events = worker_info['events']
        
        # 配置阶段
        if 'configure_start' in events and 'configure_done' in events:
            worker_info['phases'].append({
                'name': '配置',
                'start': events['configure_start'],
                'end': events['configure_done'],
                'color': 'orange'
            })
        
        # 运行阶段
        if 'run_start' in events:
            # 如果有init_done，分为初始化和运行两个阶段
            if 'init_done' in events:
                worker_info['phases'].append({
                    'name': '初始化',
                    'start': events['run_start'],
                    'end': events['init_done'],
                    'color': 'lightblue'
                })
                
                # 运行阶段从init_done开始
                run_end = events.get('run_end', max(events.values()))
                if events['init_done'] < run_end:
                    worker_info['phases'].append({
                        'name': '运行',
                        'start': events['init_done'],
                        'end': run_end,
                        'color': 'green'
                    })
            else:
                # 没有init_done，整个作为运行阶段
                run_end = events.get('run_end', max(events.values()))
                worker_info['phases'].append({
                    'name': '运行',
                    'start': events['run_start'],
                    'end': run_end,
                    'color': 'green'
                })
    
    return workers

def create_gantt_chart(workers, output_file='worker_gantt.png'):
    """创建甘特图"""
    if not workers:
        print("没有worker数据可以绘制")
        return
    
    # 按worker类型分组
    worker_types = {}
    for worker_name, worker_info in workers.items():
        worker_type = worker_info['type']
        if worker_type not in worker_types:
            worker_types[worker_type] = []
        worker_types[worker_type].append((worker_name, worker_info))
    
    # 计算全局时间范围
    all_times = []
    for worker_info in workers.values():
        for phase in worker_info['phases']:
            all_times.extend([phase['start'], phase['end']])
    
    if not all_times:
        print("没有有效的时间数据")
        return
    
    min_time = min(all_times)
    max_time = max(all_times)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, max(6, len(workers) * 0.5)))
    
    y_pos = 0
    y_labels = []
    y_positions = []
    
    # 为每种worker类型使用不同的颜色组
    type_colors = {
        'generation_server': 'red',
        'gserver_manager': 'blue', 
        'model_worker': 'green',
        'master_worker': 'purple',
        'rollout_worker': 'orange'
    }
    
    for worker_type, worker_list in worker_types.items():
        # 在不同类型之间添加间距
        if y_pos > 0:
            y_pos += 0.5
            
        for worker_name, worker_info in sorted(worker_list):
            y_labels.append(f"{worker_name}")
            y_positions.append(y_pos)
            
            # 绘制每个阶段
            for phase in worker_info['phases']:
                duration = phase['end'] - phase['start']
                start_time = phase['start'] - min_time
                
                # 根据worker类型调整颜色
                base_color = type_colors.get(worker_type, 'gray')
                if phase['name'] == '配置':
                    color = 'orange'
                elif phase['name'] == '初始化':
                    color = 'lightblue'
                else:
                    color = base_color
                
                ax.barh(y_pos, duration, left=start_time, height=0.8, 
                       color=color, alpha=0.7, label=phase['name'])
                
                # 在条形图上添加文本
                if duration > (max_time - min_time) * 0.05:  # 只在足够长的条形图上添加文本
                    ax.text(start_time + duration/2, y_pos, phase['name'], 
                           ha='center', va='center', fontsize=8)
            
            y_pos += 1
    
    # 设置y轴
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()
    
    # 设置x轴
    ax.set_xlabel('run time (s)')
    ax.set_title('AReaL Worker Runtime ')
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 添加图例（去重）
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"甘特图已保存到: {output_file}")
    
    # 显示图片
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='生成AReaL Worker运行时间甘特图')
    parser.add_argument('--input', '-i', default='worker_timeline.json', 
                       help='输入的时间线JSON文件 (默认: worker_timeline.json)')
    parser.add_argument('--output', '-o', default='worker_gantt.png',
                       help='输出的甘特图文件 (默认: worker_gantt.png)')
    
    args = parser.parse_args()
    
    # 加载数据
    data = load_timeline_data(args.input)
    if data is None:
        return 1
    
    # 处理事件
    workers = process_events(data)
    
    # 创建甘特图
    create_gantt_chart(workers, args.output)
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 