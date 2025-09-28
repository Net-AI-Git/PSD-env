import numpy as np
import random
import time
# Use a relative import to access the config file within the same package
from . import config


# ===================================================================
#
#           Problem Definition for PSD Envelope Optimization (Base)
#
# This module contains the common, shared logic for problem definitions,
# including constraint validation (is_segment_valid), graph creation
# (build_valid_jumps_graph), and base solution generation
# (create_random_solution). Specific fitness functions are implemented
# in separate files that import from this base module.
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

    for i in range(N - 1):
        consecutive_invalid_count = 0
        for j in range(i + 1, N):
            if is_segment_valid(simplified_points[i], simplified_points[j], original_psd_freqs, original_psd_values):
                graph[i].append(j)
                consecutive_invalid_count = 0  # Reset counter on valid jump
            else:
                consecutive_invalid_count += 1
                # Optimization: if many consecutive jumps are invalid, later ones are likely invalid too
                if consecutive_invalid_count >= config.BREAK_THRESHOLD:
                    break

    end_time = time.time()
    print(f"Graph built in {end_time - start_time:.2f} seconds.")
    return graph


def prune_dead_end_nodes(graph):
    """
    Prunes the graph by iteratively removing nodes that have no outgoing connections.

    This function identifies "dead-end" nodes (nodes that are not the final
    node but have no valid forward jumps) and removes the connections leading
    to them. This process repeats iteratively, as removing a connection might
    create a new dead-end node. The result is a cleaner, more efficient graph
    for the GA to traverse, ensuring all paths can reach the final node.

    Args:
        graph (list[list[int]]): The adjacency list of the graph.

    Returns:
        list[list[int]]: The pruned graph.
    """
    print("\n--- Starting graph pruning process ---")
    start_time = time.time()

    num_nodes = len(graph)
    last_node = num_nodes - 1

    # Step 0: Create an inverted graph to quickly find parent nodes.
    in_edges = [[] for _ in range(num_nodes)]
    for i, connections in enumerate(graph):
        for j in connections:
            in_edges[j].append(i)

    # Step 1: Find the initial set of dead-end nodes.
    # A dead-end is a node that is not the last node and has no outgoing edges.
    dead_ends_to_process = [
        i for i in range(num_nodes - 1) if not graph[i]
    ]
    
    print(f"Found {len(dead_ends_to_process)} initial dead-end nodes.")
    
    removed_connections_count = 0

    # Step 2: Iteratively process the dead-ends list.
    # The list might grow as we remove connections.
    i = 0
    while i < len(dead_ends_to_process):
        dead_node = dead_ends_to_process[i]
        i += 1 # Move pointer forward

        # Find all "parent" nodes that connect to this dead-end node.
        parent_nodes = in_edges[dead_node]
        
        for parent in parent_nodes:
            # Check if the connection still exists before trying to remove.
            if dead_node in graph[parent]:
                graph[parent].remove(dead_node)
                removed_connections_count += 1
                
                # If the parent has now become a dead-end itself, add it for processing.
                if not graph[parent] and parent != last_node:
                    if parent not in dead_ends_to_process:
                        dead_ends_to_process.append(parent)

    end_time = time.time()
    print(f"Removed a total of {removed_connections_count} inbound connections to dead-end nodes.")
    print(f"Graph pruning complete in {end_time - start_time:.2f} seconds.")
    
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
        list[int] or None: A list of node indices representing a valid path,
                         or None if a valid path to the end could not be found.
    """
    path = [0]  # All paths start at the first point
    current_node = 0
    last_node = len(graph) - 1
    while current_node < last_node:
        valid_options = [node for node in graph[current_node] if node > current_node]
        if not valid_options:
            # If no valid forward jumps, this is an invalid solution.
            return None
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
