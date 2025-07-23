#!/usr/bin/env python3
"""
AReaL GPU Monitor

This script monitors GPU usage of AReaL actors during training.
It collects GPU utilization and memory usage for each actor and saves the data
for later analysis and visualization.

Usage:
    python areal_gpu_monitor.py --experiment_name <exp_name> --trial_name <trial_name>
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
import ray
from matplotlib.dates import DateFormatter


class AReaLGPUMonitor:
    """Monitor GPU usage of AReaL actors."""
    
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
        self.actor_pid_map = {}
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
    
    def get_ray_actors(self) -> Dict[str, int]:
        """Get all Ray actors and their PIDs."""
        try:
            # Get runtime context to access actor information
            runtime_context = ray.get_runtime_context()
            
            # Get all actors
            actors = runtime_context.list_actors()
            
            actor_pid_map = {}
            for actor in actors:
                if actor.state == "ALIVE":
                    try:
                        # Get actor handle
                        actor_handle = ray.get_actor(actor.name)
                        
                        # Get actor info including PID
                        actor_info = ray.get(actor_handle._get_actor_info.remote())
                        if hasattr(actor_info, 'pid') and actor_info.pid:
                            actor_pid_map[actor.name] = actor_info.pid
                        else:
                            # Fallback: try to get PID from actor metadata
                            try:
                                pid = ray.get(actor_handle._get_pid.remote())
                                actor_pid_map[actor.name] = pid
                            except:
                                print(f"Warning: Could not get PID for actor {actor.name}")
                    except Exception as e:
                        print(f"Warning: Error getting info for actor {actor.name}: {e}")
            
            return actor_pid_map
            
        except Exception as e:
            print(f"Error getting Ray actors: {e}")
            return {}
    
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
    
    def collect_gpu_data(self):
        """Collect GPU usage data for all actors."""
        timestamp = datetime.now()
        
        # Get current actor PIDs
        current_actor_pid_map = self.get_ray_actors()
        
        # Update our actor PID map
        self.actor_pid_map.update(current_actor_pid_map)
        
        # Collect data for each actor
        for actor_name, pid in self.actor_pid_map.items():
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
        
        # 1. GPU Utilization by Actor
        ax1 = axes[0, 0]
        for actor_name in actor_names:
            actor_data = [r for r in self.gpu_data if r['actor_name'] == actor_name]
            if actor_data:
                times = [datetime.fromisoformat(r['timestamp']) for r in actor_data]
                utils = [r['gpu_utilization'] for r in actor_data]
                ax1.plot(times, utils, label=actor_name, marker='o', markersize=2)
        
        ax1.set_title('GPU Utilization by Actor')
        ax1.set_ylabel('GPU Utilization (%)')
        ax1.set_xlabel('Time')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Memory Usage by Actor
        ax2 = axes[0, 1]
        for actor_name in actor_names:
            actor_data = [r for r in self.gpu_data if r['actor_name'] == actor_name]
            if actor_data:
                times = [datetime.fromisoformat(r['timestamp']) for r in actor_data]
                mem_used = [r['memory_used_mb'] for r in actor_data]
                ax2.plot(times, mem_used, label=actor_name, marker='s', markersize=2)
        
        ax2.set_title('GPU Memory Usage by Actor')
        ax2.set_ylabel('Memory Used (MB)')
        ax2.set_xlabel('Time')
        ax2.legend()
        ax2.grid(True)
        
        # 3. GPU Utilization by Device
        ax3 = axes[1, 0]
        for gpu_idx in gpu_indices:
            gpu_data = [r for r in self.gpu_data if r['gpu_index'] == gpu_idx]
            if gpu_data:
                times = [datetime.fromisoformat(r['timestamp']) for r in gpu_data]
                utils = [r['gpu_utilization'] for r in gpu_data]
                ax3.plot(times, utils, label=f'GPU {gpu_idx}', marker='^', markersize=2)
        
        ax3.set_title('GPU Utilization by Device')
        ax3.set_ylabel('GPU Utilization (%)')
        ax3.set_xlabel('Time')
        ax3.legend()
        ax3.grid(True)
        
        # 4. Memory Utilization by Device
        ax4 = axes[1, 1]
        for gpu_idx in gpu_indices:
            gpu_data = [r for r in self.gpu_data if r['gpu_index'] == gpu_idx]
            if gpu_data:
                times = [datetime.fromisoformat(r['timestamp']) for r in gpu_data]
                mem_utils = [r['memory_utilization'] for r in gpu_data]
                ax4.plot(times, mem_utils, label=f'GPU {gpu_idx}', marker='d', markersize=2)
        
        ax4.set_title('Memory Utilization by Device')
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
    parser = argparse.ArgumentParser(description="Monitor GPU usage of AReaL actors")
    parser.add_argument("--experiment_name", required=True, help="Experiment name")
    parser.add_argument("--trial_name", required=True, help="Trial name")
    parser.add_argument("--output_dir", default="gpu_monitor_data", help="Output directory for data files")
    parser.add_argument("--interval", type=float, default=5.0, help="Monitoring interval in seconds")
    
    args = parser.parse_args()
    
    # Initialize Ray if not already done
    if not ray.is_initialized():
        try:
            ray.init(ignore_reinit_error=True)
            print("Ray initialized")
        except Exception as e:
            print(f"Error initializing Ray: {e}")
            sys.exit(1)
    
    # Create and run monitor
    monitor = AReaLGPUMonitor(
        experiment_name=args.experiment_name,
        trial_name=args.trial_name,
        output_dir=args.output_dir
    )
    
    monitor.run(interval=args.interval)


if __name__ == "__main__":
    main() 