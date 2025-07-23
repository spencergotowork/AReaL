#!/usr/bin/env python3
"""
Example usage of AReaL GPU Monitor

This script demonstrates how to use the GPU monitor programmatically.
"""

import time
import threading
from areal_gpu_monitor_simple import AReaLGPUMonitorSimple


def monitor_gpu_usage(experiment_name, trial_name, duration=60, interval=5):
    """
    Monitor GPU usage for a specified duration.
    
    Args:
        experiment_name: Name of the experiment
        trial_name: Name of the trial
        duration: Monitoring duration in seconds
        interval: Monitoring interval in seconds
    """
    print(f"Starting GPU monitoring for {experiment_name}/{trial_name}")
    print(f"Duration: {duration}s, Interval: {interval}s")
    
    # Create monitor
    monitor = AReaLGPUMonitorSimple(
        experiment_name=experiment_name,
        trial_name=trial_name,
        output_dir="example_output"
    )
    
    # Start monitoring in a separate thread
    def monitor_thread():
        start_time = time.time()
        while time.time() - start_time < duration:
            monitor.collect_gpu_data()
            time.sleep(interval)
        
        # Save data and create visualizations
        csv_file, json_file = monitor.save_data()
        monitor.create_visualizations(csv_file)
        
        print(f"Monitoring completed. Data saved to {csv_file}")
    
    # Run monitoring
    thread = threading.Thread(target=monitor_thread)
    thread.start()
    
    # Wait for completion
    thread.join()
    
    # Cleanup
    pynvml.nvmlShutdown()


def continuous_monitoring(experiment_name, trial_name, interval=5):
    """
    Start continuous monitoring until interrupted.
    
    Args:
        experiment_name: Name of the experiment
        trial_name: Name of the trial
        interval: Monitoring interval in seconds
    """
    print(f"Starting continuous GPU monitoring for {experiment_name}/{trial_name}")
    print("Press Ctrl+C to stop")
    
    try:
        monitor = AReaLGPUMonitorSimple(
            experiment_name=experiment_name,
            trial_name=trial_name,
            output_dir="continuous_output"
        )
        
        monitor.run(interval=interval)
        
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")


def analyze_existing_data(data_file):
    """
    Analyze existing GPU monitoring data.
    
    Args:
        data_file: Path to the CSV data file
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    
    print(f"Analyzing data from {data_file}")
    
    # Load data
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(f"Total records: {len(df)}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Processes monitored: {df['process_name'].unique()}")
    print(f"GPUs monitored: {df['gpu_index'].unique()}")
    
    # Average GPU utilization by process
    print("\nAverage GPU Utilization by Process:")
    avg_util = df.groupby('process_name')['gpu_utilization'].mean()
    for process, util in avg_util.items():
        print(f"  {process}: {util:.1f}%")
    
    # Peak memory usage
    print("\nPeak Memory Usage by Process:")
    peak_mem = df.groupby('process_name')['memory_used_mb'].max()
    for process, mem in peak_mem.items():
        print(f"  {process}: {mem:.1f} MB")
    
    # Create summary plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # GPU utilization over time
    for process in df['process_name'].unique():
        if process != 'overall':
            process_data = df[df['process_name'] == process]
            axes[0].plot(process_data['timestamp'], process_data['gpu_utilization'], 
                        label=process, marker='o', markersize=2)
    
    axes[0].set_title('GPU Utilization Over Time')
    axes[0].set_ylabel('GPU Utilization (%)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Memory usage over time
    for process in df['process_name'].unique():
        if process != 'overall':
            process_data = df[df['process_name'] == process]
            axes[1].plot(process_data['timestamp'], process_data['memory_used_mb'], 
                        label=process, marker='s', markersize=2)
    
    axes[1].set_title('Memory Usage Over Time')
    axes[1].set_ylabel('Memory Used (MB)')
    axes[1].set_xlabel('Time')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
    print("\nAnalysis plot saved to data_analysis.png")
    plt.show()


def main():
    """Main function demonstrating different usage patterns."""
    print("AReaL GPU Monitor - Usage Examples")
    print("=" * 40)
    
    # Example 1: Monitor for a fixed duration
    print("\nExample 1: Fixed duration monitoring")
    print("-" * 30)
    try:
        monitor_gpu_usage("example_exp", "example_trial", duration=30, interval=5)
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    # Example 2: Continuous monitoring
    print("\nExample 2: Continuous monitoring")
    print("-" * 30)
    print("This will run until you press Ctrl+C")
    try:
        continuous_monitoring("example_exp", "example_trial", interval=5)
    except Exception as e:
        print(f"Example 2 failed: {e}")
    
    # Example 3: Analyze existing data
    print("\nExample 3: Analyze existing data")
    print("-" * 30)
    print("This example requires an existing CSV data file")
    print("You can run this after collecting some data:")
    print("  analyze_existing_data('gpu_monitor_data/gpu_usage_YYYYMMDD_HHMMSS.csv')")


if __name__ == "__main__":
    main() 