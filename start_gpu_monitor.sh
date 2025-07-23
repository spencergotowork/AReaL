#!/bin/bash

# AReaL GPU Monitor Quick Start Script

set -e

echo "AReaL GPU Monitor Quick Start"
echo "=============================="

# Check if experiment name and trial name are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <experiment_name> <trial_name> [output_dir] [interval]"
    echo ""
    echo "Arguments:"
    echo "  experiment_name: Name of the AReaL experiment"
    echo "  trial_name: Name of the AReaL trial"
    echo "  output_dir: Output directory (optional, default: gpu_monitor_data)"
    echo "  interval: Monitoring interval in seconds (optional, default: 5)"
    echo ""
    echo "Example:"
    echo "  $0 my_experiment trial_001"
    echo "  $0 my_experiment trial_001 ./my_data 10"
    exit 1
fi

EXPERIMENT_NAME=$1
TRIAL_NAME=$2
OUTPUT_DIR=${3:-gpu_monitor_data}
INTERVAL=${4:-5}

echo "Configuration:"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Trial: $TRIAL_NAME"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Monitoring Interval: ${INTERVAL}s"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import pynvml, psutil, matplotlib, numpy" 2>/dev/null || {
    echo "Installing required packages..."
    pip install -r gpu_monitor_requirements.txt
}

# Test GPU access
echo "Testing GPU access..."
python3 -c "
import pynvml
try:
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    print(f'Found {device_count} GPU device(s)')
    pynvml.nvmlShutdown()
except Exception as e:
    print(f'Error accessing GPU: {e}')
    exit(1)
"

# Start monitoring
echo ""
echo "Starting GPU monitor..."
echo "Press Ctrl+C to stop monitoring"
echo ""

python3 areal_gpu_monitor_simple.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --trial_name "$TRIAL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --interval "$INTERVAL" 