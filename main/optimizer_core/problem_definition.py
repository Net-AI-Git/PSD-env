import numpy as np
import random
import time
# Use a relative import to access the config file within the same package
from . import config


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

def is_segment_valid(p1, p2, original_psd_freqs, original_psd_values):
    """
    Robustly checks if a line segment between two points (p1, p2) is valid.

    A segment is considered valid if its line representation remains entirely
    above the original PSD signal at all critical points. The check is
    performed in semi-log space (linear X-axis for frequency, logarithmic
    Y-axis for PSD values) to perfectly match the visual representation of
    the plot and guarantee that no visual intersections are missed. This was
    a critical fix for a subtle bug where linear checks missed log-space intersections.

    Args:
        p1 (tuple): The (x, y) coordinates of the starting point.
        p2 (tuple): The (x, y) coordinates of the ending point.
        original_psd_freqs (np.array): The frequency points of the original PSD.
        original_psd_values (np.array): The amplitude values of the original PSD.

    Returns:
        bool: True if the segment is valid (does not intersect), False otherwise.
    """
    x1, y1 = p1
    x2, y2 = p2
    min_freq, max_freq = min(x1, x2), max(x1, x2)

    # Use a small epsilon to avoid log(0) errors
    epsilon = 1e-12
    log_y1 = np.log10(y1 + epsilon)
    log_y2 = np.log10(y2 + epsilon)
    log_original_psd_values = np.log10(original_psd_values + epsilon)

    # Get all original PSD points that fall between the segment's endpoints
    mask = (original_psd_freqs >= min_freq) & (original_psd_freqs <= max_freq)
    freqs_in_range = original_psd_freqs[mask]

    # The points we need to check against are the original PSD points in range,
    # plus the segment's own endpoints to be thorough.
    check_freqs = np.union1d(freqs_in_range, [x1, x2])

    if len(check_freqs) < 2:
        return True  # Not enough points to form a line

    # Perform linear interpolation in log-space
    envelope_line_log_values = np.interp(check_freqs, [x1, x2], [log_y1, log_y2])
    original_psd_log_values = np.interp(check_freqs, original_psd_freqs, log_original_psd_values)

    # Check if any point on the envelope line is below the original PSD line
    tolerance = 1e-9  # Use a small tolerance for floating point comparisons
    if np.any(envelope_line_log_values < original_psd_log_values - tolerance):
        return False

    return True


def build_valid_jumps_graph(simplified_points, original_psd_freqs, original_psd_values):
    """
    Pre-computes a directed acyclic graph of all valid "jumps" between candidate points.

    This is a critical pre-computation step that significantly speeds up the
    evolutionary process. The result is an adjacency list where graph[i]
    contains all indices j > i such that the segment from point i to point j
    is valid (i.e., does not intersect the PSD). A "smart break" optimization
    is used to avoid unnecessary checks.

    Args:
        simplified_points (np.array): The pool of candidate points for the envelope.
        original_psd_freqs (np.array): The frequency points of the original PSD.
        original_psd_values (np.array): The amplitude values of the original PSD.

    Returns:
        list[list[int]]: The adjacency list representation of the valid jumps graph.
    """
    N = len(simplified_points)
    graph = [[] for _ in range(N)]
    print("\nBuilding valid jumps graph with smart optimization...")
    start_time = time.time()

    BREAK_THRESHOLD = 50  # Stop checking if we find this many consecutive invalid jumps

    for i in range(N - 1):
        consecutive_invalid_count = 0
        for j in range(i + 1, N):
            if is_segment_valid(simplified_points[i], simplified_points[j], original_psd_freqs, original_psd_values):
                graph[i].append(j)
                consecutive_invalid_count = 0  # Reset counter on valid jump
            else:
                consecutive_invalid_count += 1
                # Optimization: if many consecutive jumps are invalid, later ones are likely invalid too
                if consecutive_invalid_count >= BREAK_THRESHOLD:
                    break

    end_time = time.time()
    print(f"Graph built in {end_time - start_time:.2f} seconds.")
    return graph


def create_random_solution(graph, target_points):
    """
    Creates a single, valid random solution (path) through the pre-computed graph.

    The choice of the next node is weighted to prefer shorter jumps, encouraging
    initial solutions to follow the PSD shape more closely rather than making
    large, unrealistic jumps across the frequency spectrum.

    Args:
        graph (list[list[int]]): The valid jumps graph (adjacency list).
        target_points (int): The target number of points (used for initial guidance).

    Returns:
        list[int]: A list of node indices representing a valid path from the
                   first to the last point.
    """
    path = [0]  # All paths start at the first point
    current_node = 0
    last_node = len(graph) - 1
    while current_node < last_node:
        valid_options = [node for node in graph[current_node] if node > current_node]
        if not valid_options:
            # If no valid forward jumps, force a jump to the end
            next_node = last_node
        else:
            # Weight choices to prefer closer points
            weights = 1 / np.array([node - current_node for node in valid_options]) ** 2
            weights /= np.sum(weights)  # Normalize to a probability distribution
            next_node = np.random.choice(valid_options, p=weights)
        path.append(next_node)
        current_node = next_node

    # Clean up the path to ensure it's strictly increasing and ends correctly
    final_path = [path[0]]
    for node in path[1:]:
        if node > final_path[-1]:
            final_path.append(node)
    if final_path[-1] != last_node:
        final_path.append(last_node)

    return final_path


def calculate_metrics(path, simplified_points, original_psd_freqs, original_psd_values, target_points, **kwargs):
    """
    Calculates the cost and fitness of a given solution (path).

    This is the core of the multi-objective optimization. The total cost is a
    combination of two objectives:
    1. Area Cost: The integrated area between the envelope and the PSD in log-space.
       This cost can be weighted to give more importance to low-frequency areas.
    2. Points Penalty: A penalty for deviating from the `target_points`.

    Args:
        path (list[int]): The solution to evaluate.
        simplified_points (np.array): The pool of candidate points.
        original_psd_freqs (np.array): The frequency points of the original PSD.
        original_psd_values (np.array): The amplitude values of the original PSD.
        target_points (int): The ideal number of points for the envelope.

    Returns:
        tuple: A tuple containing (total_cost, fitness, num_points, area_ratio).
    """
    if not path or len(path) < 2:
        return float('inf'), 0, 0, float('inf')

    # Decode the path (indices) into actual coordinates
    decoded_points = simplified_points[path]
    # Interpolate the envelope to match the original frequency points for comparison
    interp_envelope_values = np.interp(original_psd_freqs, decoded_points[:, 0], decoded_points[:, 1])

    # 1. Calculate Area Cost in log-space
    epsilon = 1e-12
    log_y_diff = np.log10(interp_envelope_values + epsilon) - np.log10(original_psd_values + epsilon)

    # Determine X-axis domain for integration per configuration
    if getattr(config, 'AREA_X_AXIS_MODE', 'Linear').lower() == 'log':
        x_full = np.log10(original_psd_freqs + epsilon)
    else:
        x_full = original_psd_freqs

    # If enabled, apply a weight to the low-frequency area to prioritize a tighter fit there.
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
    # The penalty grows quadratically as the number of points deviates from the target
    penalty_factor = 1.0 + ((num_points - target_points) / target_points) ** 2

    # Combine into total cost
    total_cost = area_cost * penalty_factor

    # Convert cost to fitness (higher is better)
    fitness = 1.0 / (1.0 + total_cost) if total_cost >= 0 else 1.0 + abs(total_cost)

    # For reporting purposes, calculate the simple area ratio in linear space
    envelope_area = np.trapezoid(interp_envelope_values, x=original_psd_freqs)
    original_area = np.trapezoid(original_psd_values, x=original_psd_freqs)
    area_ratio = envelope_area / original_area if original_area > 0 else float('inf')

    return total_cost, fitness, len(path), area_ratio
