#!/usr/bin/env python3
"""
测试脚本：生成示例worker时间线数据并生成甘特图
"""

import json
import time
import subprocess
import os

def generate_sample_data():
    """生成示例worker时间线数据"""
    
    # 模拟时间戳
    base_time = time.time()
    
    sample_data = [
        # master_worker
        {"worker_name": "master_worker/0", "worker_type": "master_worker", "event": "configure_start", "timestamp": base_time, "human_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(base_time))},
        {"worker_name": "master_worker/0", "worker_type": "master_worker", "event": "configure_done", "timestamp": base_time + 2, "human_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(base_time + 2))},
        {"worker_name": "master_worker/0", "worker_type": "master_worker", "event": "run_start", "timestamp": base_time + 5, "human_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(base_time + 5))},
        {"worker_name": "master_worker/0", "worker_type": "master_worker", "event": "init_done", "timestamp": base_time + 8, "human_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(base_time + 8))},
        {"worker_name": "master_worker/0", "worker_type": "master_worker", "event": "run_end", "timestamp": base_time + 50, "human_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(base_time + 50))},
        
        # model_worker
        {"worker_name": "model_worker/0", "worker_type": "model_worker", "event": "configure_start", "timestamp": base_time + 1, "human_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(base_time + 1))},
        {"worker_name": "model_worker/0", "worker_type": "model_worker", "event": "configure_done", "timestamp": base_time + 4, "human_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(base_time + 4))},
        {"worker_name": "model_worker/0", "worker_type": "model_worker", "event": "run_start", "timestamp": base_time + 6, "human_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(base_time + 6))},
        {"worker_name": "model_worker/0", "worker_type": "model_worker", "event": "init_done", "timestamp": base_time + 10, "human_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(base_time + 10))},
        {"worker_name": "model_worker/0", "worker_type": "model_worker", "event": "run_end", "timestamp": base_time + 45, "human_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(base_time + 45))},
        
        {"worker_name": "model_worker/1", "worker_type": "model_worker", "event": "configure_start", "timestamp": base_time + 1.5, "human_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(base_time + 1.5))},
        {"worker_name": "model_worker/1", "worker_type": "model_worker", "event": "configure_done", "timestamp": base_time + 4.5, "human_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(base_time + 4.5))},
        {"worker_name": "model_worker/1", "worker_type": "model_worker", "event": "run_start", "timestamp": base_time + 6.5, "human_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(base_time + 6.5))},
        {"worker_name": "model_worker/1", "worker_type": "model_worker", "event": "init_done", "timestamp": base_time + 11, "human_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(base_time + 11))},
        {"worker_name": "model_worker/1", "worker_type": "model_worker", "event": "run_end", "timestamp": base_time + 47, "human_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(base_time + 47))},
        
        # rollout_worker
        {"worker_name": "rollout_worker/0", "worker_type": "rollout_worker", "event": "configure_start", "timestamp": base_time + 2, "human_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(base_time + 2))},
        {"worker_name": "rollout_worker/0", "worker_type": "rollout_worker", "event": "configure_done", "timestamp": base_time + 5, "human_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(base_time + 5))},
        {"worker_name": "rollout_worker/0", "worker_type": "rollout_worker", "event": "run_start", "timestamp": base_time + 8, "human_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(base_time + 8))},
        {"worker_name": "rollout_worker/0", "worker_type": "rollout_worker", "event": "init_done", "timestamp": base_time + 12, "human_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(base_time + 12))},
        {"worker_name": "rollout_worker/0", "worker_type": "rollout_worker", "event": "run_end", "timestamp": base_time + 48, "human_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(base_time + 48))},
        
        # generation_server
        {"worker_name": "generation_server/0", "worker_type": "generation_server", "event": "configure_start", "timestamp": base_time + 0.5, "human_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(base_time + 0.5))},
        {"worker_name": "generation_server/0", "worker_type": "generation_server", "event": "configure_done", "timestamp": base_time + 3, "human_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(base_time + 3))},
        {"worker_name": "generation_server/0", "worker_type": "generation_server", "event": "run_start", "timestamp": base_time + 4, "human_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(base_time + 4))},
        {"worker_name": "generation_server/0", "worker_type": "generation_server", "event": "init_done", "timestamp": base_time + 7, "human_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(base_time + 7))},
        {"worker_name": "generation_server/0", "worker_type": "generation_server", "event": "run_end", "timestamp": base_time + 49, "human_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(base_time + 49))},
    ]
    
    return sample_data

def main():
    print("生成示例worker时间线数据...")
    
    # 生成示例数据
    sample_data = generate_sample_data()
    
    # 保存到JSON文件
    with open('worker_timeline.json', 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print("示例数据已保存到 worker_timeline.json")
    
    # 打印数据概要
    print("\n数据概要:")
    worker_types = set()
    worker_names = set()
    for event in sample_data:
        worker_types.add(event['worker_type'])
        worker_names.add(event['worker_name'])
    
    print(f"Worker类型: {sorted(worker_types)}")
    print(f"Worker数量: {len(worker_names)}")
    print(f"事件总数: {len(sample_data)}")
    
    # 检查是否有matplotlib
    try:
        import matplotlib
        print("\n生成甘特图...")
        
        # 运行甘特图生成脚本
        result = subprocess.run(['python3', 'generate_gantt.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("甘特图生成成功！")
            print("输出文件:")
            print("- worker_timeline.json (时间线数据)")
            print("- worker_gantt.png (甘特图)")
        else:
            print(f"甘特图生成失败: {result.stderr}")
            
    except ImportError:
        print("\n注意: 没有安装matplotlib，无法生成甘特图")
        print("请运行以下命令安装matplotlib:")
        print("pip install matplotlib")

if __name__ == '__main__':
    main() 