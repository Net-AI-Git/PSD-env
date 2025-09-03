"""
GPU Utility Module

This module provides a unified interface for numerical operations, automatically
selecting between CuPy for GPU acceleration and NumPy for CPU execution.

It checks for the availability of a compatible GPU and the CuPy library. If
both are present and the global configuration 'USE_GPU' is True, it sets the
'xp' alias to 'cupy'. Otherwise, it defaults to 'numpy'.

This allows the rest of the application to use a single API (e.g., xp.array,
xp.log10) for numerical computations, making the code cleaner and agnostic to
the underlying hardware.
"""

import numpy as np
from . import config

# Default to numpy
xp = np
IS_GPU_AVAILABLE = False

if config.USE_GPU:
    try:
        import cupy as cp
        # Check if a GPU device is actually available
        if cp.cuda.runtime.getDeviceCount() > 0:
            xp = cp
            IS_GPU_AVAILABLE = True
            print("✅ GPU acceleration enabled (CuPy backend selected).")
        else:
            print("⚠️ CuPy is installed, but no active GPU device was found. Falling back to CPU (NumPy).")
    except ImportError:
        print("ℹ️ CuPy is not installed. Falling back to CPU (NumPy). To enable GPU acceleration, run: pip install cupy-cudaXX")
    except Exception as e:
        print(f"An unexpected error occurred while initializing CuPy: {e}")
        print("Falling back to CPU (NumPy).")

def to_gpu(data):
    """
    Moves a NumPy array to the GPU if GPU is available.
    """
    if IS_GPU_AVAILABLE:
        return xp.asarray(data)
    return data # Returns a numpy array if not on GPU

def to_cpu(data):
    """
    Moves a CuPy array back to the CPU if it's on the GPU.
    """
    if IS_GPU_AVAILABLE and isinstance(data, xp.ndarray):
        return cp.asnumpy(data)
    return data # Returns the numpy array as is
