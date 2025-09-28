"""
Configuration settings for GPU-accelerated operations.

This file centralizes hardware-dependent parameters, such as batch sizes,
allowing for easy tuning without modifying the core algorithm logic.
Each parameter is tailored for a specific GPU-accelerated function.
"""

# Batch size for the main fitness calculation function.
# This determines how many solutions (individuals) are processed simultaneously on the GPU.
# A larger batch size can lead to better GPU utilization but requires more VRAM.
# This value should be tuned based on the available GPU memory.
CALCULATE_METRICS_BATCH_SIZE = 512
