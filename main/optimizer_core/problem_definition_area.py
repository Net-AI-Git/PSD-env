import numpy as np
import random
import time
# Use a relative import to access the config file within the same package
from . import config
from .problem_definition_base import is_segment_valid, build_valid_jumps_graph, create_random_solution, prune_dead_end_nodes

# ===================================================================
#
#           Problem Definition for PSD Envelope Optimization
#
# This module defines the specifics of the optimization problem. It
# contains the fitness function (how to evaluate a solution), constraint
# validation (what makes a solution valid), and methods for creating
# and representing solutions. If the GA were to be adapted for a
# different problem, this would be the primary file to modify.
#
# ===================================================================

def calculate_metrics_linear(path, simplified_points, original_psd_freqs, original_psd_values, target_points, **kwargs):
    """
    Calculates the cost and fitness using LINEAR X-axis integration.
    For area optimization mode: minimizes number of points while targeting specific area ratio.
    """
    if not path or len(path) < 2:
        return float('inf'), 0, 0, float('inf')

    # Decode the path (indices) into actual coordinates
    decoded_points = simplified_points[path]
    # Interpolate the envelope to match the original frequency points for comparison
    interp_envelope_values = np.interp(original_psd_freqs, decoded_points[:, 0], decoded_points[:, 1])

    # Calculate the current area ratio for penalty calculation
    envelope_area = np.trapezoid(interp_envelope_values, x=original_psd_freqs)
    original_area = np.trapezoid(original_psd_values, x=original_psd_freqs)
    current_area_ratio = envelope_area / original_area if original_area > 0 else float('inf')

    # 1. Main cost: Number of Points (what we want to minimize)
    num_points = len(path)

    # 2. Calculate Area Ratio Penalty
    target_area_ratio = config.TARGET_AREA_RATIO
    penalty_factor = 1.0 + ((current_area_ratio - target_area_ratio) / target_area_ratio) ** 2

    # 3. Calculate the final cost.
    # The area_error is multiplied by a large weight to make it the primary optimization goal.
    # The number of points acts as a secondary goal or a "tie-breaker".
    area_error = penalty_factor - 1.0
    total_cost = (config.AREA_LOG_AWEIGHT * area_error) + num_points

    # Convert cost to fitness (higher is better)
    fitness = 1.0 / (1.0 + total_cost) if total_cost >= 0 else 1.0 + abs(total_cost)

    return total_cost, fitness, len(path), current_area_ratio


def calculate_metrics_log(path, simplified_points, original_psd_freqs, original_psd_values, target_points, **kwargs):
    """
    Calculates cost and fitness using a multi-objective approach for LOG-scale optimization.
    The goal is to find an envelope that is visually tight in a log-log plot,
    has a low number of points, and respects the target LINEAR area ratio.
    """
    if not path or len(path) < 2:
        return float('inf'), 0, 0, float('inf')

    # Decode path into coordinates
    decoded_points = simplified_points[path]
    interp_envelope_values = np.interp(original_psd_freqs, decoded_points[:, 0], decoded_points[:, 1])
    num_points = len(path)
    epsilon = 1e-12  # To avoid log(0)

    # 1. Calculate LINEAR area ratio for the penalty factor
    # This ensures we converge to the correct numerical target (e.g., 1.2)
    linear_envelope_area = np.trapezoid(interp_envelope_values, x=original_psd_freqs)
    linear_original_area = np.trapezoid(original_psd_values, x=original_psd_freqs)
    linear_area_ratio = linear_envelope_area / linear_original_area if linear_original_area > 0 else float('inf')
    
    target_area_ratio = config.TARGET_AREA_RATIO
    penalty_factor = 1.0 + ((linear_area_ratio - target_area_ratio) / target_area_ratio) ** 2

    # 2. Calculate the final cost.
    # The area_error is multiplied by a large weight to make it the primary optimization goal.
    # The number of points acts as a secondary goal or a "tie-breaker".
    area_error = penalty_factor - 1.0
    total_cost = (config.AREA_LOG_AWEIGHT * area_error) + num_points

    # Convert cost to fitness (higher is better)
    # The cost should now always be positive.
    fitness = 1.0 / (1.0 + total_cost)

    # IMPORTANT: Return the LINEAR area ratio for display purposes
    return total_cost, fitness, num_points, linear_area_ratio


def calculate_metrics(path, simplified_points, original_psd_freqs, original_psd_values, target_points, **kwargs):
    """
    Main function that calls the appropriate calculation method based on configuration.
    """
    # Determine which calculation method to use based on configuration
    if getattr(config, 'AREA_X_AXIS_MODE', 'Linear').lower() == 'log':
        return calculate_metrics_log(path, simplified_points, original_psd_freqs, original_psd_values, target_points, **kwargs)
    else:
        return calculate_metrics_linear(path, simplified_points, original_psd_freqs, original_psd_values, target_points, **kwargs)
