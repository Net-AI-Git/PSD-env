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


def calculate_metrics(path, simplified_points, original_psd_freqs, original_psd_values,
                        target_area_ratio, target_points, X_AXIS_MODE, **kwargs):
    """
    Calculates the cost and fitness of a given solution path.

    The function computes a multi-objective cost, prioritizing the tightness of the
    envelope (measured by area ratio) while also penalizing solutions that deviate
    from the target number of points. It supports both Linear and Log scaling for
    the X-axis during area calculation, making it flexible for different analysis
    scenarios.

    Args:
        path (list[int]): A list of indices representing the points in the solution
                          from the `simplified_points` array.
        simplified_points (np.ndarray): An array of all candidate points for the envelope.
        original_psd_freqs (np.ndarray): The frequency values of the original signal.
        original_psd_values (np.ndarray): The PSD values of the original signal.
        target_points (int, optional): The desired number of points for the envelope.
                                       Defaults to 45.
        target_area_ratio (float, optional): The target for the linear area ratio,
                                             used in the cost calculation.
                                             Defaults to 1.25.
        X_AXIS_MODE (str, optional): Determines the X-axis scale for area integration.
                                     Can be 'Log' or 'Linear'. Defaults to 'Log'.
        **kwargs: Catches any other arguments that might be passed.

    Returns:
        tuple[float, float, int, float]: A tuple containing:
            - total_cost (float): The combined cost of the solution.
            - fitness (float): The fitness score (higher is better).
            - len(path) (int): The number of points in the solution.
            - linear_area_ratio (float): The calculated area ratio in linear space.
    """
    if not path or len(path) < 2:
        return float('inf'), 0, 0, float('inf')

    # Decode the path (indices) into actual coordinates
    decoded_points = simplified_points[path]
    # Interpolate the envelope to match the original frequency points for comparison
    interp_envelope_values = np.interp(original_psd_freqs, decoded_points[:, 0], decoded_points[:, 1])

    # 1. Calculate Area Cost as a ratio of areas (Log Y-axis, Linear X-axis)
    epsilon = 1e-12
    log_envelope_y = np.log10(interp_envelope_values + epsilon)
    log_original_y = np.log10(original_psd_values + epsilon)

    # Shift graphs to be non-negative for correct area ratio calculation
    min_log_y = np.min(log_original_y)
    shift_value = -min_log_y  # Ensures the lowest point is at y=0
    shifted_envelope_y = log_envelope_y + shift_value
    shifted_original_y = log_original_y + shift_value

    # Use linear frequencies for X-axis
    x_full = original_psd_freqs
    if X_AXIS_MODE == 'Log':
        x_full_log = np.log10(original_psd_freqs + epsilon)
    else:
        x_full_log = x_full

    # If enabled, apply a weight to the low-frequency area
    if config.ENRICH_LOW_FREQUENCIES and config.LOW_FREQ_AREA_WEIGHT > 1.0:
        low_freq_mask = original_psd_freqs <= config.LOW_FREQUENCY_THRESHOLD

        # Calculate weighted areas for both envelope and original PSD using SHIFTED values
        # Low-frequency part
        low_freq_envelope_area = np.trapezoid(shifted_envelope_y[low_freq_mask], x=x_full_log[low_freq_mask])
        low_freq_original_area = np.trapezoid(shifted_original_y[low_freq_mask], x=x_full_log[low_freq_mask])

        # High-frequency part
        high_freq_envelope_area = np.trapezoid(shifted_envelope_y[~low_freq_mask], x=x_full_log[~low_freq_mask])
        high_freq_original_area = np.trapezoid(shifted_original_y[~low_freq_mask], x=x_full_log[~low_freq_mask])

        # Combine with weight
        weighted_envelope_area = (low_freq_envelope_area * config.LOW_FREQ_AREA_WEIGHT) + high_freq_envelope_area
        weighted_original_area = (low_freq_original_area * config.LOW_FREQ_AREA_WEIGHT) + high_freq_original_area

        area_cost = weighted_envelope_area / weighted_original_area if weighted_original_area > 0 else float('inf')
    else:
        # Default behavior: calculate area ratio over the entire range using SHIFTED values
        envelope_area_log_y = np.trapezoid(shifted_envelope_y, x=x_full_log)
        original_area_log_y = np.trapezoid(shifted_original_y, x=x_full_log)
        area_cost = envelope_area_log_y / original_area_log_y if original_area_log_y > 0 else float('inf')

    # 2. Calculate Points Penalty
    num_points = len(path)
    points_error = ((num_points - target_points) / target_points) ** 2
    
    # 3. Combine into total cost with heavy weight on area error
    # This prioritizes getting a tight envelope over minimizing points.
    
    # The primary, log-based area error is the absolute distance from the target ratio.
    area_error = abs(area_cost - 1)

    # For reporting and for an additional linear error term, calculate the simple area ratio in linear space
    linear_envelope_area = np.trapezoid(interp_envelope_values, x=original_psd_freqs)
    linear_original_area = np.trapezoid(original_psd_values, x=original_psd_freqs)
    linear_area_ratio = linear_envelope_area / linear_original_area if linear_original_area > 0 else float('inf')
    
    # The secondary, linear-based area error provides an additional constraint.
    linear_area_error = abs(linear_area_ratio - target_area_ratio)

    # Use a large weight to make both area errors the primary optimization goal
    total_cost = config.AREA_WEIGHT * area_error + config.AREA_WEIGHT_LINEA * linear_area_error + config.POINTS_WEIGHT * points_error

    # Convert cost to fitness (higher is better)
    fitness = 1.0 / (1.0 + total_cost) if total_cost >= 0 else 1.0 + abs(total_cost)

    return total_cost, fitness, len(path), linear_area_ratio
