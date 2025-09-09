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

    # If enabled, apply a weight to the low-frequency area
    if config.ENRICH_LOW_FREQUENCIES and config.LOW_FREQ_AREA_WEIGHT > 1.0:
        low_freq_mask = original_psd_freqs <= config.LOW_FREQUENCY_THRESHOLD

        # Calculate weighted areas for both envelope and original PSD using SHIFTED values
        # Low-frequency part
        low_freq_envelope_area = np.trapezoid(shifted_envelope_y[low_freq_mask], x=x_full[low_freq_mask])
        low_freq_original_area = np.trapezoid(shifted_original_y[low_freq_mask], x=x_full[low_freq_mask])

        # High-frequency part
        high_freq_envelope_area = np.trapezoid(shifted_envelope_y[~low_freq_mask], x=x_full[~low_freq_mask])
        high_freq_original_area = np.trapezoid(shifted_original_y[~low_freq_mask], x=x_full[~low_freq_mask])

        # Combine with weight
        weighted_envelope_area = (low_freq_envelope_area * config.LOW_FREQ_AREA_WEIGHT) + high_freq_envelope_area
        weighted_original_area = (low_freq_original_area * config.LOW_FREQ_AREA_WEIGHT) + high_freq_original_area

        area_cost = weighted_envelope_area / weighted_original_area if weighted_original_area > 0 else float('inf')
    else:
        # Default behavior: calculate area ratio over the entire range using SHIFTED values
        envelope_area_log_y = np.trapezoid(shifted_envelope_y, x=x_full)
        original_area_log_y = np.trapezoid(shifted_original_y, x=x_full)
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
    linear_area_error = abs(linear_area_ratio - config.TARGET_AREA_RATIO)

    # Use a large weight to make both area errors the primary optimization goal
    total_cost = (config.POINTS_LOG_WEIGHT * (area_error + linear_area_error)) + points_error

    # Convert cost to fitness (higher is better)
    fitness = 1.0 / (1.0 + total_cost) if total_cost >= 0 else 1.0 + abs(total_cost)

    return total_cost, fitness, len(path), linear_area_ratio


def calculate_metrics_log(path, simplified_points, original_psd_freqs, original_psd_values, target_points, **kwargs):
    """
    Calculates the cost and fitness using LOGARITHMIC X-axis integration.
    """
    if not path or len(path) < 2:
        return float('inf'), 0, 0, float('inf')

    # Decode the path (indices) into actual coordinates
    decoded_points = simplified_points[path]
    # Interpolate the envelope to match the original frequency points for comparison
    interp_envelope_values = np.interp(original_psd_freqs, decoded_points[:, 0], decoded_points[:, 1])

    # 1. Calculate Area Cost in LOG space
    epsilon = 1e-12
    log_y_diff = np.log10(interp_envelope_values + epsilon) - np.log10(original_psd_values + epsilon)

    # Use log frequencies for X-axis
    x_full = np.log10(original_psd_freqs + epsilon)

    # If enabled, apply a weight to the low-frequency area
    if config.ENRICH_LOW_FREQUENCIES and config.LOW_FREQ_AREA_WEIGHT > 1.0:
        low_freq_mask = original_psd_freqs <= config.LOW_FREQUENCY_THRESHOLD

        # Calculate area for low-frequency part
        low_freq_area = np.trapezoid(log_y_diff[low_freq_mask], x=x_full[low_freq_mask])

        # Calculate area for high-frequency part
        high_freq_area = np.trapezoid(log_y_diff[~low_freq_mask], x=x_full[~low_freq_mask])

        # Combine with weight
        area_cost = (low_freq_area * config.LOW_FREQ_AREA_WEIGHT) + high_freq_area
    else:
        # Default behavior: calculate area over the entire range
        area_cost = np.trapezoid(log_y_diff, x=x_full)

    # 2. Calculate Points Penalty
    num_points = len(path)
    penalty_factor = 1.0 + ((num_points - target_points) / target_points) ** 2

    # Combine into total cost
    total_cost = area_cost * penalty_factor

    # Convert cost to fitness (higher is better)
    fitness = 1.0 / (1.0 + total_cost) if total_cost >= 0 else 1.0 + abs(total_cost)

    # For reporting purposes, calculate the simple area ratio in linear space
    envelope_area = np.trapezoid(interp_envelope_values, x=original_psd_freqs)
    original_area = np.trapezoid(original_psd_values, x=original_psd_freqs)
    area_ratio = envelope_area / original_area

    return total_cost, fitness, len(path), area_ratio


def calculate_metrics(path, simplified_points, original_psd_freqs, original_psd_values, target_points, **kwargs):
    """
    Main function that calls the appropriate calculation method based on configuration.
    """
    # Determine which calculation method to use based on configuration
    if getattr(config, 'AREA_X_AXIS_MODE', 'Linear').lower() == 'log':
        return calculate_metrics_log(path, simplified_points, original_psd_freqs, original_psd_values, target_points, **kwargs)
    else:
        return calculate_metrics_linear(path, simplified_points, original_psd_freqs, original_psd_values, target_points, **kwargs)
