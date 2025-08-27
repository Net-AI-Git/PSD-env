import random
import numpy as np


# ===================================================================
#
#           Genetic Operators for the Path-Based GA
#
# This module contains the core evolutionary operators: selection,
# crossover, and a toolkit of mutations. These functions are responsible
# for evolving the population of solutions from one generation to the next.
#
# ===================================================================

def selection(population, fitness_scores, tournament_size=3):
    """
    Selects a single parent from the population using tournament selection.

    In tournament selection, a few individuals are chosen at random from the
    population, and the one with the best fitness is selected as a parent.

    Args:
        population (list[list[int]]): The current population of solutions (paths).
        fitness_scores (list[float]): The fitness score of each solution.
        tournament_size (int): The number of individuals to compete in the tournament.

    Returns:
        list[int]: The winning solution (chromosome) to be used as a parent.
    """
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitness)]
    return population[winner_index]


def crossover_multipoint_paths(parent1, parent2):
    """
    Creates a child solution by combining segments from two parent solutions.

    This method identifies common points (nodes) between the two parents and
    uses them as potential crossover points. It then splices segments from
    alternating parents to create a new, valid child path.

    Args:
        parent1 (list[int]): The first parent solution (a list of point indices).
        parent2 (list[int]): The second parent solution (a list of point indices).

    Returns:
        list[int]: The resulting child solution.
    """
    common_nodes = sorted(list(set(parent1) & set(parent2)))
    start_node, end_node = parent1[0], parent1[-1]
    valid_cut_points = [n for n in common_nodes if n != start_node and n != end_node]

    if len(valid_cut_points) < 2:
        # Fallback to single-point crossover if not enough common points
        crossover_point = random.choice(valid_cut_points) if valid_cut_points else start_node
        if crossover_point == start_node: return random.choice([parent1, parent2])
        idx1 = parent1.index(crossover_point)
        idx2 = parent2.index(crossover_point)
        return parent1[:idx1] + parent2[idx2:]

    # Multi-point crossover
    num_cuts = random.randint(1, min(5, len(valid_cut_points)))
    cut_points = sorted(random.sample(valid_cut_points, num_cuts))
    boundaries = [start_node] + cut_points + [end_node]
    child_path = []
    for i in range(len(boundaries) - 1):
        seg_start, seg_end = boundaries[i], boundaries[i + 1]
        donor = random.choice([parent1, parent2])
        try:
            start_idx, end_idx = donor.index(seg_start), donor.index(seg_end)
            child_path.extend(donor[start_idx:end_idx])
        except ValueError:  # If segment not in chosen donor, use the other one
            fallback_donor = parent2 if donor is parent1 else parent1
            start_idx, end_idx = fallback_donor.index(seg_start), fallback_donor.index(seg_end)
            child_path.extend(fallback_donor[start_idx:end_idx])
    child_path.append(end_node)

    # Remove duplicates while preserving order
    seen = set()
    return [x for x in child_path if not (x in seen or seen.add(x))]


# --- Mutation Toolkit ---

def calculate_segment_area_cost(segment_path, simplified_points, original_psd_freqs, original_psd_values):
    """
    Helper function to calculate the area cost for a small segment of a path.
    This is used by the pruning mutation to make intelligent decisions.

    Args:
        segment_path (list[int]): A short list of point indices (e.g., [prev, current, next]).
        simplified_points (np.array): The main pool of candidate points.
        original_psd_freqs (np.array): The frequency points of the original PSD.
        original_psd_values (np.array): The amplitude values of the original PSD.

    Returns:
        float: The calculated area cost for this specific segment.
    """
    if not segment_path or len(segment_path) < 2:
        return float('inf')

    decoded_segment = simplified_points[segment_path]
    min_freq, max_freq = decoded_segment[0, 0], decoded_segment[-1, 0]

    mask = (original_psd_freqs >= min_freq) & (original_psd_freqs <= max_freq)
    freqs_in_range = original_psd_freqs[mask]
    psd_in_range = original_psd_values[mask]

    if len(freqs_in_range) < 2:
        return 0

    interp_env = np.interp(freqs_in_range, decoded_segment[:, 0], decoded_segment[:, 1])
    epsilon = 1e-12
    log_y_diff = np.log10(interp_env + epsilon) - np.log10(psd_in_range + epsilon)
    return np.trapezoid(log_y_diff, x=freqs_in_range)


def mutate_prune_useless_points(path, **all_ga_params):
    """
    Intelligently removes points from a path that have a negligible impact on area cost.
    This is a "refinement" mutation that helps reduce the number of points in a solution.

    Args:
        path (list[int]): The solution path to be pruned.
        **all_ga_params (dict): A dictionary of all GA parameters.

    Returns:
        list[int]: The new, potentially shorter, path.
    """
    graph = all_ga_params['graph']
    simplified_points = all_ga_params['simplified_points']
    original_psd_freqs = all_ga_params['original_psd_freqs']
    original_psd_values = all_ga_params['original_psd_values']
    prune_threshold = all_ga_params['prune_threshold']

    new_path = list(path)
    i = 1
    while i < len(new_path) - 1:
        prev_node, current_node, next_node = new_path[i - 1], new_path[i], new_path[i + 1]

        # First, check if a direct jump is even possible (doesn't intersect PSD)
        if next_node in graph[prev_node]:
            # If possible, check if the point is redundant in terms of area
            original_segment = [prev_node, current_node, next_node]
            original_area = calculate_segment_area_cost(original_segment, simplified_points, original_psd_freqs,
                                                        original_psd_values)
            simplified_segment = [prev_node, next_node]
            simplified_area = calculate_segment_area_cost(simplified_segment, simplified_points, original_psd_freqs,
                                                          original_psd_values)

            # If the change in area is below the threshold, remove the point
            if abs(original_area) > 1e-9 and abs((simplified_area - original_area) / original_area) < prune_threshold:
                new_path.pop(i)
                continue  # Re-evaluate the new segment at the same index i
        i += 1
    return new_path


def mutate_remove_point(path, graph):
    """A simple mutation that removes a random point from the path, if valid."""
    if len(path) <= 2: return path
    idx_to_remove = random.randint(1, len(path) - 2)
    prev_node, next_node = path[idx_to_remove - 1], path[idx_to_remove + 1]
    if next_node in graph[prev_node]:
        return path[:idx_to_remove] + path[idx_to_remove + 1:]
    return path


def mutate_replace_point(path, graph, candidate_points):
    """A simple mutation that replaces a random point with another valid point."""
    if len(path) <= 2: return path
    idx_to_replace = random.randint(1, len(path) - 2)
    prev_node_idx, next_node_idx = path[idx_to_replace - 1], path[idx_to_replace + 1]
    min_freq, max_freq = candidate_points[prev_node_idx][0], candidate_points[next_node_idx][0]

    # Find all possible replacement points between the neighbors
    potential_replacements = [i for i, p in enumerate(candidate_points) if min_freq < p[0] < max_freq and i not in path]
    random.shuffle(potential_replacements)

    # Find the first valid replacement
    for new_node_idx in potential_replacements:
        if new_node_idx in graph[prev_node_idx] and next_node_idx in graph[new_node_idx]:
            new_path = list(path)
            new_path[idx_to_replace] = new_node_idx
            return new_path
    return path


def apply_mutations(path, all_ga_params, best_solution_len, adaptive_mutation_threshold):
    """
    Applies an adaptive number of mutations based on the current best solution's length.
    This implements the "adaptive mutation rate" strategy ("turbo mode"). When the
    solutions get close to the target point count, the mutation rate increases
    to encourage exploration and escape local optima.

    Args:
        path (list[int]): The child solution to mutate.
        all_ga_params (dict): Dictionary of all GA parameters.
        best_solution_len (int): The number of points in the current best solution.
        adaptive_mutation_threshold (int): The threshold to activate "turbo mode".

    Returns:
        list[int]: The mutated path.
    """
    if best_solution_len < adaptive_mutation_threshold:
        num_mutations = random.randint(3, 6)  # "Turbo mode"
    else:
        num_mutations = random.randint(1, 2)  # Normal mode

    mutated_path = list(path)
    for _ in range(num_mutations):
        mutation_choice = random.random()
        graph = all_ga_params['graph']
        candidate_points = all_ga_params['simplified_points']

        if mutation_choice < 0.5:
            mutated_path = mutate_remove_point(mutated_path, graph)
        else:
            mutated_path = mutate_replace_point(mutated_path, graph, candidate_points)

    return mutated_path
