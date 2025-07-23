#!/usr/bin/env python3
"""
AReaL GPU Monitor (Simplified Version)

This script monitors GPU usage of AReaL actors during training.
It uses a simpler approach to find AReaL processes and monitor their GPU usage.

Usage:
    python areal_gpu_monitor_simple.py --experiment_name <exp_name> --trial_name <trial_name>
"""

import argparse
import csv
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import psutil
import pynvml
from matplotlib.dates import DateFormatter
import ray


class AReaLGPUMonitorSimple:
    """Simplified monitor for GPU usage of AReaL actors."""
    
    def __init__(self, experiment_name: str, trial_name: str, output_dir: str = "gpu_monitor_data"):
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize NVML
        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()
        
        # Data storage
        self.gpu_data = []
        self.process_pid_map = {}
        self.running = True
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print(f"Initialized GPU monitor for {experiment_name}/{trial_name}")
        print(f"Found {self.device_count} GPU devices")
        print(f"Output directory: {self.output_dir}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        self.running = False
    
    def find_areal_processes(self) -> Dict[str, int]:
        """Find AReaL-related processes by looking for specific patterns."""
        areal_processes = {}
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    # Look for Python processes that might be AReaL workers
                    if proc.info['name'] == 'python' or proc.info['name'] == 'python3':
                        cmdline = proc.info['cmdline']
                        if cmdline:
                            cmdline_str = ' '.join(cmdline)
                            
                            # Look for AReaL-related patterns
                            if any(pattern in cmdline_str.lower() for pattern in [
                                'realhf', 'areal', 'rollout_worker', 'model_worker', 
                                'master_worker', 'gserver_manager'
                            ]):
                                # Try to extract worker type from command line
                                worker_type = "unknown"
                                if 'rollout_worker' in cmdline_str:
                                    worker_type = "rollout_worker"
                                elif 'model_worker' in cmdline_str:
                                    worker_type = "model_worker"
                                elif 'master_worker' in cmdline_str:
                                    worker_type = "master_worker"
                                elif 'gserver_manager' in cmdline_str:
                                    worker_type = "gserver_manager"
                                
                                # Try to extract worker index
                                worker_index = "0"
                                for i, arg in enumerate(cmdline):
                                    if arg in ['-w', '--worker_type'] and i + 1 < len(cmdline):
                                        worker_type = cmdline[i + 1]
                                    elif arg in ['-i', '--worker_index'] and i + 1 < len(cmdline):
                                        worker_index = cmdline[i + 1]
                                
                                process_name = f"{worker_type}/{worker_index}"
                                areal_processes[process_name] = proc.info['pid']
                                
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
                    
        except Exception as e:
            print(f"Error finding AReaL processes: {e}")
        
        return areal_processes
    
    def get_gpu_usage_by_pid(self, pid: int) -> List[Dict]:
        """Get GPU usage for a specific PID."""
        usage_data = []
        
        for gpu_idx in range(self.device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
                
                # Get GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                # Get memory info
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Get processes using this GPU
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                
                # Check if our PID is using this GPU
                for proc in procs:
                    if proc.pid == pid:
                        usage_data.append({
                            'gpu_index': gpu_idx,
                            'gpu_utilization': util.gpu,
                            'memory_utilization': util.memory,
                            'memory_used_mb': proc.usedGpuMemory / 1024 / 1024,
                            'memory_total_mb': meminfo.total / 1024 / 1024,
                            'memory_free_mb': meminfo.free / 1024 / 1024,
                        })
                        break
                else:
                    # PID not found on this GPU, but record GPU info
                    usage_data.append({
                        'gpu_index': gpu_idx,
                        'gpu_utilization': util.gpu,
                        'memory_utilization': util.memory,
                        'memory_used_mb': 0,
                        'memory_total_mb': meminfo.total / 1024 / 1024,
                        'memory_free_mb': meminfo.free / 1024 / 1024,
                    })
                    
            except Exception as e:
                print(f"Error getting GPU {gpu_idx} info: {e}")
        
        return usage_data
    
    def get_all_gpu_usage(self) -> List[Dict]:
        """Get GPU usage for all GPUs without specific PID filtering."""
        all_usage = []
        
        for gpu_idx in range(self.device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
                
                # Get GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                # Get memory info
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Get processes using this GPU
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                
                # Calculate total memory used by all processes
                total_used_memory = sum(proc.usedGpuMemory for proc in procs)
                
                all_usage.append({
                    'gpu_index': gpu_idx,
                    'gpu_utilization': util.gpu,
                    'memory_utilization': util.memory,
                    'memory_used_mb': total_used_memory / 1024 / 1024,
                    'memory_total_mb': meminfo.total / 1024 / 1024,
                    'memory_free_mb': meminfo.free / 1024 / 1024,
                    'process_count': len(procs)
                })
                    
            except Exception as e:
                print(f"Error getting GPU {gpu_idx} info: {e}")
        
        return all_usage
    
    def get_ray_actor_pid_map(self):
        """通过Ray API获取所有actor的名字和PID。"""
        actor_pid_map = {}
        try:
            # 获取所有命名actor（如 rollout_worker/0, model_worker/1 ...）
            actor_names = ray.util.list_named_actors(all_namespaces=True)
            for name in actor_names:
                try:
                    actor_handle = ray.get_actor(name)
                    pid = ray.get(actor_handle.get_pid.remote())
                    actor_pid_map[name] = pid
                except Exception as e:
                    print(f"Warning: Could not get PID for actor {name}: {e}")
        except Exception as e:
            print(f"Error getting Ray actors: {e}")
        return actor_pid_map

    def collect_gpu_data(self):
        """采集GPU使用数据，优先采集Ray actor的GPU占用。"""
        timestamp = datetime.now()
        # 1. 通过Ray API获取actor->PID
        ray_actor_pid_map = self.get_ray_actor_pid_map()
        # 2. 用PID查找GPU占用
        for actor_name, pid in ray_actor_pid_map.items():
            try:
                gpu_usage = self.get_gpu_usage_by_pid(pid)
                for gpu_data in gpu_usage:
                    record = {
                        'timestamp': timestamp.isoformat(),
                        'actor_name': actor_name,
                        'pid': pid,
                        **gpu_data
                    }
                    self.gpu_data.append(record)
            except Exception as e:
                print(f"Error collecting data for actor {actor_name} (PID {pid}): {e}")
        # 3. 兜底：用原有进程名逻辑采集未被Ray识别的进程
        # 避免重复采集已通过Ray采集的PID
        ray_pids = set(ray_actor_pid_map.values())
        current_processes = self.find_areal_processes()
        for process_name, pid in current_processes.items():
            if pid in ray_pids:
                continue  # 已采集过
            try:
                gpu_usage = self.get_gpu_usage_by_pid(pid)
                for gpu_data in gpu_usage:
                    record = {
                        'timestamp': timestamp.isoformat(),
                        'actor_name': process_name,  # 这里用进程名兜底
                        'pid': pid,
                        **gpu_data
                    }
                    self.gpu_data.append(record)
            except Exception as e:
                print(f"Error collecting data for process {process_name} (PID {pid}): {e}")
        # 4. 也可采集overall（可选）
        overall_usage = self.get_all_gpu_usage()
        for gpu_data in overall_usage:
            record = {
                'timestamp': timestamp.isoformat(),
                'actor_name': 'overall',
                'pid': -1,
                **gpu_data
            }
            self.gpu_data.append(record)
    
    def save_data(self):
        """Save collected data to CSV and JSON files."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as CSV
        csv_file = self.output_dir / f"gpu_usage_{timestamp_str}.csv"
        if self.gpu_data:
            fieldnames = self.gpu_data[0].keys()
            with open(csv_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.gpu_data)
            print(f"Saved {len(self.gpu_data)} records to {csv_file}")
        
        # Save as JSON
        json_file = self.output_dir / f"gpu_usage_{timestamp_str}.json"
        with open(json_file, 'w') as f:
            json.dump({
                'experiment_name': self.experiment_name,
                'trial_name': self.trial_name,
                'device_count': self.device_count,
                'data': self.gpu_data
            }, f, indent=2)
        print(f"Saved data to {json_file}")
        
        return csv_file, json_file
    
    def create_visualizations(self, csv_file: Path):
        """Create visualization plots from the collected data."""
        if not self.gpu_data:
            print("No data to visualize")
            return
        
        # Read data
        timestamps = []
        actor_names = set()
        gpu_indices = set()
        
        for record in self.gpu_data:
            timestamps.append(datetime.fromisoformat(record['timestamp']))
            actor_names.add(record['actor_name'])
            gpu_indices.add(record['gpu_index'])
        
        timestamps = sorted(set(timestamps))
        actor_names = sorted(actor_names)
        gpu_indices = sorted(gpu_indices)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'GPU Usage Monitor - {self.experiment_name}/{self.trial_name}', fontsize=16)
        
        # 1. GPU Utilization by Process
        ax1 = axes[0, 0]
        for actor_name in actor_names:
            if actor_name != 'overall':  # Skip overall for this plot
                actor_data = [r for r in self.gpu_data if r['actor_name'] == actor_name]
                if actor_data:
                    times = [datetime.fromisoformat(r['timestamp']) for r in actor_data]
                    utils = [r['gpu_utilization'] for r in actor_data]
                    ax1.plot(times, utils, label=actor_name, marker='o', markersize=2)
        
        ax1.set_title('GPU Utilization by Process')
        ax1.set_ylabel('GPU Utilization (%)')
        ax1.set_xlabel('Time')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Memory Usage by Process
        ax2 = axes[0, 1]
        for actor_name in actor_names:
            if actor_name != 'overall':  # Skip overall for this plot
                actor_data = [r for r in self.gpu_data if r['actor_name'] == actor_name]
                if actor_data:
                    times = [datetime.fromisoformat(r['timestamp']) for r in actor_data]
                    mem_used = [r['memory_used_mb'] for r in actor_data]
                    ax2.plot(times, mem_used, label=actor_name, marker='s', markersize=2)
        
        ax2.set_title('GPU Memory Usage by Process')
        ax2.set_ylabel('Memory Used (MB)')
        ax2.set_xlabel('Time')
        ax2.legend()
        ax2.grid(True)
        
        # 3. Overall GPU Utilization by Device
        ax3 = axes[1, 0]
        overall_data = [r for r in self.gpu_data if r['actor_name'] == 'overall']
        for gpu_idx in gpu_indices:
            gpu_data = [r for r in overall_data if r['gpu_index'] == gpu_idx]
            if gpu_data:
                times = [datetime.fromisoformat(r['timestamp']) for r in gpu_data]
                utils = [r['gpu_utilization'] for r in gpu_data]
                ax3.plot(times, utils, label=f'GPU {gpu_idx}', marker='^', markersize=2)
        
        ax3.set_title('Overall GPU Utilization by Device')
        ax3.set_ylabel('GPU Utilization (%)')
        ax3.set_xlabel('Time')
        ax3.legend()
        ax3.grid(True)
        
        # 4. Overall Memory Utilization by Device
        ax4 = axes[1, 1]
        for gpu_idx in gpu_indices:
            gpu_data = [r for r in overall_data if r['gpu_index'] == gpu_idx]
            if gpu_data:
                times = [datetime.fromisoformat(r['timestamp']) for r in gpu_data]
                mem_utils = [r['memory_utilization'] for r in gpu_data]
                ax4.plot(times, mem_utils, label=f'GPU {gpu_idx}', marker='d', markersize=2)
        
        ax4.set_title('Overall Memory Utilization by Device')
        ax4.set_ylabel('Memory Utilization (%)')
        ax4.set_xlabel('Time')
        ax4.legend()
        ax4.grid(True)
        
        # Format x-axis
        for ax in axes.flat:
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / f"gpu_usage_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {plot_file}")
        
        plt.show()
    
    def run(self, interval: float = 5.0):
        """Run the GPU monitor."""
        print(f"Starting GPU monitor with {interval}s interval...")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while self.running:
                self.collect_gpu_data()
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nStopping GPU monitor...")
        finally:
            # Save data
            csv_file, json_file = self.save_data()
            
            # Create visualizations
            self.create_visualizations(csv_file)
            
            # Cleanup
            pynvml.nvmlShutdown()
            print("GPU monitor stopped.")


def main():
    parser = argparse.ArgumentParser(description="Monitor GPU usage of AReaL processes")
    parser.add_argument("--experiment_name", required=True, help="Experiment name")
    parser.add_argument("--trial_name", required=True, help="Trial name")
    parser.add_argument("--output_dir", default="gpu_monitor_data", help="Output directory for data files")
    parser.add_argument("--interval", type=float, default=5.0, help="Monitoring interval in seconds")
    
    args = parser.parse_args()
    
    # Create and run monitor
    monitor = AReaLGPUMonitorSimple(
        experiment_name=args.experiment_name,
        trial_name=args.trial_name,
        output_dir=args.output_dir
    )
    
    monitor.run(interval=args.interval)


if __name__ == "__main__":
    main() 