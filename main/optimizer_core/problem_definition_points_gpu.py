import cupy as cp
import random
import time
# Use a relative import to access the config file within the same package
from . import config
from . import config_gpu  # Import the new GPU config file
from .problem_definition_base import is_segment_valid, build_valid_jumps_graph, create_random_solution, prune_dead_end_nodes


# ===================================================================
#
#           Problem Definition for PSD Envelope Optimization (GPU Batch Version)
#
# ===================================================================


def calculate_metrics_batch(paths, simplified_points, original_psd_freqs, original_psd_values,
                        target_area_ratio, target_points, X_AXIS_MODE, **kwargs):
    """
    Calculates the cost and fitness for a batch of solution paths using the GPU.

    This function processes multiple solutions (an entire batch) in a single,
    highly parallelized GPU operation. It handles paths of varying lengths by
    padding them to a uniform size and using a mask to ignore padded values
    during calculations. All operations, including interpolation and area
    calculation, are vectorized to maximize GPU throughput.

    Args:
        paths (list[list[int]]): A list of paths, where each path is a list of
                                 point indices. This is the batch of solutions.
        simplified_points (cp.ndarray): A CuPy array of all candidate points.
        original_psd_freqs (cp.ndarray): The frequency values of the original signal.
        original_psd_values (cp.ndarray): The PSD values of the original signal.
        target_points (int): The desired number of points for the envelope.
        target_area_ratio (float): The target for the linear area ratio.
        X_AXIS_MODE (str): The X-axis scale for area integration ('Log' or 'Linear').
        **kwargs: Catches any other arguments.

    Returns:
        tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]: A tuple of CuPy arrays:
            - total_cost: Vector of combined costs for each solution in the batch.
            - fitness: Vector of fitness scores.
            - path_lengths: Vector of actual point counts for each solution.
            - linear_area_ratio: Vector of calculated linear area ratios.
    """
    batch_size = len(paths)
    if batch_size == 0:
        return cp.array([]), cp.array([]), cp.array([]), cp.array([])

    # --- 1. Pad paths to uniform length and create a mask ---
    path_lengths = cp.array([len(p) for p in paths], dtype=cp.int32)
    max_len = cp.max(path_lengths).item()

    # Padded paths matrix, initialized with -1 (or any invalid index)
    padded_paths = cp.full((batch_size, max_len), -1, dtype=cp.int32)
    # Mask to identify valid (non-padded) points
    mask = cp.arange(max_len) < path_lengths[:, None]

    for i, path in enumerate(paths):
        padded_paths[i, :len(path)] = cp.array(path, dtype=cp.int32)

    # --- 2. Decode paths into coordinates ---
    # Advanced indexing to get coordinates for all paths in the batch
    # Shape: (batch_size, max_len, 2)
    decoded_points = simplified_points[padded_paths]
    
    # Mask out the coordinates of padded points to avoid affecting calculations
    decoded_points[~mask] = 0
    
    x_paths = decoded_points[:, :, 0]  # Shape: (batch_size, max_len)
    y_paths = decoded_points[:, :, 1]  # Shape: (batch_size, max_len)

    # --- 3. Vectorized Interpolation ---
    # This is a complex operation performed without loops.
    x_target = original_psd_freqs  # Shape: (n_freqs,)

    # Prepare shapes for broadcasting
    x_paths_b = x_paths[:, :, None]      # (batch, max_len, 1)
    x_target_b = x_target[None, None, :] # (1, 1, n_freqs)

    # Find the index of the first point in each path that is > the target frequency
    # Use a large value for padded points to ensure they don't interfere
    x_paths_masked = cp.where(mask, x_paths, cp.inf)[:, :, None]
    
    comparison = x_paths_masked > x_target_b # (batch, max_len, n_freqs)
    right_indices = cp.argmax(comparison, axis=1) # (batch, n_freqs)
    left_indices = cp.maximum(0, right_indices - 1)

    # Gather the x,y coordinates of the bracketing points for every target frequency
    x_left = cp.take_along_axis(x_paths, left_indices, axis=1)
    y_left = cp.take_along_axis(y_paths, left_indices, axis=1)
    x_right = cp.take_along_axis(x_paths, right_indices, axis=1)
    y_right = cp.take_along_axis(y_paths, right_indices, axis=1)

    # Calculate slope, avoiding division by zero
    dx = x_right - x_left
    dy = y_right - y_left
    slope = dy / (dx + 1e-12)

    # Perform the final interpolation for the entire batch
    # Result shape: (batch_size, n_freqs)
    interp_envelope_values = y_left + slope * (x_target[None, :] - x_left)

    # --- 4. Vectorized Area Cost Calculation ---
    epsilon = 1e-12
    log_envelope_y = cp.log10(interp_envelope_values + epsilon)
    log_original_y = cp.log10(original_psd_values + epsilon) # Shape: (n_freqs,)

    # Shift graphs to be non-negative for area ratio calculation
    min_log_y = cp.min(log_original_y)
    shifted_envelope_y = log_envelope_y + (-min_log_y)
    shifted_original_y = log_original_y[None, :] + (-min_log_y) # Broadcast original

    x_full = original_psd_freqs
    if X_AXIS_MODE == 'Log':
        x_full_log = cp.log10(original_psd_freqs + epsilon)
    else:
        x_full_log = x_full

    # Calculate area for all solutions in the batch using trapezoid rule along axis=1
    envelope_area_log_y = cp.trapz(shifted_envelope_y, x=x_full_log, axis=1)
    original_area_log_y = cp.trapz(shifted_original_y, x=x_full_log, axis=1)
    area_cost = envelope_area_log_y / (original_area_log_y + epsilon)

    # --- 5. Vectorized Points Penalty ---
    # path_lengths is already a vector of shape (batch_size,)
    points_error = ((path_lengths - target_points) / target_points) ** 2

    # --- 6. Vectorized Final Cost and Fitness ---
    area_error = cp.abs(area_cost - 1)

    # Calculate linear area ratio for reporting and as a secondary cost term
    linear_envelope_area = cp.trapz(interp_envelope_values, x=original_psd_freqs, axis=1)
    linear_original_area = cp.trapz(original_psd_values, x=original_psd_freqs) # This is a scalar
    linear_area_ratio = linear_envelope_area / (linear_original_area + epsilon)
    
    linear_area_error = cp.abs(linear_area_ratio - target_area_ratio)

    total_cost = (config.AREA_WEIGHT * area_error +
                  config.AREA_WEIGHT_LINEAR * linear_area_error +
                  config.POINTS_WEIGHT * points_error)

    fitness = 1.0 / (1.0 + total_cost)

    return total_cost, fitness, path_lengths, linear_area_ratio
