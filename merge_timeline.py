#!/usr/bin/env python3
"""
合并多个worker的时间线文件
"""

import json
import os
import glob
import argparse

def merge_worker_files(base_filename="worker_timeline.json"):
    """合并所有worker的时间线文件"""
    
    # 查找所有worker文件
    pattern = f"{base_filename}.*"
    worker_files = glob.glob(pattern)
    
    # 排除主文件本身
    worker_files = [f for f in worker_files if not f.endswith(base_filename)]
    
    if not worker_files:
        print(f"没有找到worker文件: {pattern}")
        return
    
    print(f"找到 {len(worker_files)} 个worker文件:")
    for f in worker_files:
        print(f"  - {f}")
    
    # 合并所有数据
    all_records = []
    
    for worker_file in worker_files:
        try:
            with open(worker_file, 'r') as f:
                worker_data = json.load(f)
                all_records.extend(worker_data)
                print(f"从 {worker_file} 读取了 {len(worker_data)} 条记录")
        except Exception as e:
            print(f"读取 {worker_file} 失败: {e}")
    
    # 按时间戳排序
    all_records.sort(key=lambda x: x['timestamp'])
    
    # 写入合并后的文件
    with open(base_filename, 'w') as f:
        json.dump(all_records, f, indent=2)
    
    print(f"\n合并完成！")
    print(f"总记录数: {len(all_records)}")
    print(f"输出文件: {base_filename}")
    
    # 统计信息
    worker_types = set()
    worker_names = set()
    for record in all_records:
        worker_types.add(record['worker_type'])
        worker_names.add(record['worker_name'])
    
    print(f"Worker类型: {sorted(worker_types)}")
    print(f"Worker数量: {len(worker_names)}")

def cleanup_worker_files(base_filename="worker_timeline.json"):
    """清理worker文件（可选）"""
    pattern = f"{base_filename}.*"
    worker_files = glob.glob(pattern)
    worker_files = [f for f in worker_files if not f.endswith(base_filename)]
    
    if worker_files:
        print(f"\n清理 {len(worker_files)} 个worker文件...")
        for f in worker_files:
            try:
                os.remove(f)
                print(f"删除: {f}")
            except Exception as e:
                print(f"删除 {f} 失败: {e}")
    else:
        print("没有worker文件需要清理")

def main():
    parser = argparse.ArgumentParser(description='合并worker时间线文件')
    parser.add_argument('--input', '-i', default='worker_timeline.json',
                       help='基础文件名 (默认: worker_timeline.json)')
    parser.add_argument('--cleanup', '-c', action='store_true',
                       help='合并后清理worker文件')
    
    args = parser.parse_args()
    
    # 合并文件
    merge_worker_files(args.input)
    
    # 清理文件（可选）
    if args.cleanup:
        cleanup_worker_files(args.input)

if __name__ == '__main__':
    main() 