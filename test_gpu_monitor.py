#!/usr/bin/env python3
"""
Test script for AReaL GPU Monitor

This script tests the basic functionality of the GPU monitor without requiring
AReaL to be running.
"""

import time
import pynvml
import psutil
from areal_gpu_monitor_simple import AReaLGPUMonitorSimple


def test_gpu_access():
    """Test basic GPU access functionality."""
    print("Testing GPU access...")
    
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        print(f"Found {device_count} GPU devices")
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            print(f"GPU {i}: {name.decode('utf-8')}")
            
            # Get utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            print(f"  GPU Utilization: {util.gpu}%")
            print(f"  Memory Utilization: {util.memory}%")
            
            # Get memory info
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"  Total Memory: {meminfo.total / 1024 / 1024:.1f} MB")
            print(f"  Used Memory: {meminfo.used / 1024 / 1024:.1f} MB")
            print(f"  Free Memory: {meminfo.free / 1024 / 1024:.1f} MB")
            
            # Get processes
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            print(f"  Processes using GPU: {len(procs)}")
            for proc in procs:
                print(f"    PID {proc.pid}: {proc.usedGpuMemory / 1024 / 1024:.1f} MB")
        
        pynvml.nvmlShutdown()
        print("GPU access test passed!")
        return True
        
    except Exception as e:
        print(f"GPU access test failed: {e}")
        return False


def test_process_detection():
    """Test process detection functionality."""
    print("\nTesting process detection...")
    
    try:
        # Create a test monitor
        monitor = AReaLGPUMonitorSimple("test_exp", "test_trial", "test_output")
        
        # Find processes
        processes = monitor.find_areal_processes()
        print(f"Found {len(processes)} potential AReaL processes:")
        
        for process_name, pid in processes.items():
            print(f"  {process_name}: PID {pid}")
            
            # Try to get process info
            try:
                proc = psutil.Process(pid)
                print(f"    Command: {' '.join(proc.cmdline())}")
                print(f"    Status: {proc.status()}")
            except psutil.NoSuchProcess:
                print(f"    Process {pid} no longer exists")
            except psutil.AccessDenied:
                print(f"    Cannot access process {pid}")
        
        # Cleanup
        pynvml.nvmlShutdown()
        print("Process detection test completed!")
        return True
        
    except Exception as e:
        print(f"Process detection test failed: {e}")
        return False


def test_data_collection():
    """Test data collection functionality."""
    print("\nTesting data collection...")
    
    try:
        # Create a test monitor
        monitor = AReaLGPUMonitorSimple("test_exp", "test_trial", "test_output")
        
        # Collect data for a few iterations
        for i in range(3):
            print(f"Collecting data iteration {i+1}/3...")
            monitor.collect_gpu_data()
            time.sleep(2)
        
        print(f"Collected {len(monitor.gpu_data)} data records")
        
        # Show sample data
        if monitor.gpu_data:
            print("Sample data record:")
            sample = monitor.gpu_data[0]
            for key, value in sample.items():
                print(f"  {key}: {value}")
        
        # Save data
        csv_file, json_file = monitor.save_data()
        print(f"Data saved to {csv_file} and {json_file}")
        
        # Cleanup
        pynvml.nvmlShutdown()
        print("Data collection test completed!")
        return True
        
    except Exception as e:
        print(f"Data collection test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("AReaL GPU Monitor Test Suite")
    print("=" * 40)
    
    tests = [
        ("GPU Access", test_gpu_access),
        ("Process Detection", test_process_detection),
        ("Data Collection", test_data_collection),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("Test Results Summary:")
    print("=" * 40)
    
    passed = 0
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("All tests passed! GPU monitor should work correctly.")
    else:
        print("Some tests failed. Please check the error messages above.")


if __name__ == "__main__":
    main() 