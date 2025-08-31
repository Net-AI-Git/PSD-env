import time
import numpy as np
import random
import os
import sys

# Import the genetic algorithm modules from the 'optimizer_core' package
from optimizer_core import config
from optimizer_core import psd_utils
from optimizer_core import problem_definition as problem
from optimizer_core import ga_operators as operators


# ===================================================================
#
#           Main Execution Script for PSD Envelope Optimization
#
# This script orchestrates the entire optimization process by:
# 1. Managing input/output directories.
# 2. Looping through all valid input files.
# 3. Running the full optimization process for each file.
# 4. Saving the results as images.
#
# ===================================================================

def process_psd_file(filepath):
    """
    Runs the complete genetic algorithm optimization for a single PSD file.

    Args:
        filepath (str): The full path to the input PSD data file.
    """
    overall_start_time = time.time()

    # --- Data Preprocessing ---
    frequencies, psd_values = psd_utils.read_psd_data(filepath)
    if frequencies is None or len(frequencies) == 0:
        print(f"Could not read data from {filepath}. Skipping.")
        return

    candidate_points = psd_utils.create_multi_scale_envelope(
        frequencies, psd_values, config.WINDOW_SIZES
    )

    # --- Enforce the correct starting point ---
    first_point = np.array([[frequencies[0], psd_values[0]]])
    other_points = candidate_points[candidate_points[:, 0] > frequencies[0]]
    candidate_points = np.vstack((first_point, other_points))

    # --- Graph Construction ---
    valid_jumps_graph = problem.build_valid_jumps_graph(
        candidate_points, frequencies, psd_values
    )

    # Pack shared parameters into a dictionary for cleaner passing to functions
    ga_params = {
        'graph': valid_jumps_graph,
        'simplified_points': candidate_points,
        'original_psd_freqs': frequencies,
        'original_psd_values': psd_values,
        'prune_threshold': config.PRUNE_THRESHOLD
    }

    # --- Initial Population Creation ---
    print(f"\n--- Creating Initial Population of {config.POPULATION_SIZE} solutions ---")
    pop_creation_start = time.time()
    population = [
        problem.create_random_solution(valid_jumps_graph, config.TARGET_POINTS)
        for _ in range(config.POPULATION_SIZE)
    ]
    pop_creation_end = time.time()
    print(f"Initial population created in {pop_creation_end - pop_creation_start:.2f} seconds.")

    best_solution_so_far, best_cost_so_far = None, float('inf')

    # --- Main Evolution Loop ---
    print("\n--- Starting Evolution ---")
    evolution_start_time = time.time()

    # Initialize variables for early stopping
    generations_without_improvement = 0
    last_best_cost = float('inf')
    generation = 0

    while generation < config.MAX_GENERATIONS:
        population = [p for p in population if p and len(p) > 1]
        if not population:
            print("Population became empty. Exiting evolution.")
            break

        all_metrics = [
            problem.calculate_metrics(path, **ga_params, target_points=config.TARGET_POINTS)
            for path in population
        ]
        costs = [m[0] for m in all_metrics]
        fitness_scores = [m[1] for m in all_metrics]

        current_min_cost = np.min(costs)
        if current_min_cost < best_cost_so_far:
            best_cost_so_far = current_min_cost
            best_solution_so_far = population[np.argmin(costs)]

        current_best_len = len(best_solution_so_far)

        # Print progress update every 10 generations
        if (generation + 1) % 10 == 0:
            best_cost_report, best_fitness_report, best_len_report, best_ratio = problem.calculate_metrics(
                best_solution_so_far, **ga_params, target_points=config.TARGET_POINTS
            )
            print(
                f"Gen {generation + 1}/{config.MAX_GENERATIONS} | "
                f"Area Ratio: {best_ratio:.4f} | "
                f"Points: {best_len_report} | "
                f"Fitness: {best_fitness_report:.4f} | "
                f"Cost: {best_cost_report:.4f}"
            )

        # --- Early Stopping Logic ---
        if config.USE_CONVERGENCE_TERMINATION:
            improvement = last_best_cost - best_cost_so_far
            if improvement < config.CONVERGENCE_THRESHOLD:
                generations_without_improvement += 1
            else:
                generations_without_improvement = 0  # Reset counter on significant improvement

            last_best_cost = best_cost_so_far

            if generations_without_improvement >= config.CONVERGENCE_PATIENCE:
                print(f"\n--- Terminating early at generation {generation + 1} due to convergence ---")
                break

        # --- Generate the next generation ---
        new_population = []
        if config.ELITISM_SIZE > 0:
            elite_indices = np.argsort(fitness_scores)[-config.ELITISM_SIZE:]
            elite_solutions = [population[i] for i in elite_indices]
            new_population.extend(elite_solutions)

            for elite_sol in elite_solutions:
                if len(elite_sol) < config.ADAPTIVE_MUTATION_THRESHOLD:
                    pruned_version = operators.mutate_prune_useless_points(elite_sol, **ga_params)
                    if pruned_version != elite_sol:
                        new_population.append(pruned_version)

        while len(new_population) < config.POPULATION_SIZE:
            parent1 = operators.selection(population, fitness_scores)
            parent2 = operators.selection(population, fitness_scores)
            child = operators.crossover_multipoint_paths(parent1, parent2)

            if random.random() < config.MUTATION_RATE:
                mutated_child = operators.apply_mutations(
                    child, ga_params, current_best_len, config.ADAPTIVE_MUTATION_THRESHOLD
                )
            else:
                mutated_child = child
            new_population.append(mutated_child)
        population = new_population
        generation += 1

    end_time = time.time()

    # --- Final Results ---
    if best_solution_so_far:
        final_cost, _, final_len, final_ratio = problem.calculate_metrics(
            best_solution_so_far, **ga_params, target_points=config.TARGET_POINTS
        )
        print("\n--- Optimization Finished ---")
        print(f"Evolution process time: {end_time - evolution_start_time:.2f} seconds")
        print(f"Total process time: {end_time - overall_start_time:.2f} seconds")
        print(f"Best solution has {final_len} points.")
        print(f"Final Internal Cost: {final_cost:.4f}")
        print(f"Final Area Ratio: {final_ratio:.6f}")

        final_points_coords = ga_params['simplified_points'][best_solution_so_far]

        # Save the final solution plot instead of showing it
        psd_utils.plot_final_solution(
            frequencies, psd_values, final_points_coords, final_ratio, filepath
        )
    else:
        print("\n--- No valid solution found ---")


def main():
    """
    Main batch processing function. Manages directories and loops through input files.
    """
    # --- Directory Management ---
    if not os.path.exists(config.INPUT_DIR):
        os.makedirs(config.INPUT_DIR)
        print(f"Input directory '{config.INPUT_DIR}' was not found.")
        print("It has been created for you.")
        print(f"Please place your PSD data files (e.g., *.txt) inside '{config.INPUT_DIR}' and run the script again.")
        sys.exit(0)  # Exit gracefully after creating the directory

    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
        print(f"Output directory '{config.OUTPUT_DIR}' created.")

    # --- File Discovery and Processing Loop ---
    files_to_process = [
        f for f in os.listdir(config.INPUT_DIR)
        if f.endswith(config.INPUT_FILE_EXTENSION)
    ]

    if not files_to_process:
        print(f"No files with extension '{config.INPUT_FILE_EXTENSION}' found in '{config.INPUT_DIR}'.")
        return

    print(f"Found {len(files_to_process)} files to process: {files_to_process}")

    for filename in files_to_process:
        print(f"\n{'=' * 60}")
        print(f"Processing file: {filename}")
        print(f"{'=' * 60}")

        filepath = os.path.join(config.INPUT_DIR, filename)
        process_psd_file(filepath)

    print(f"\n{'=' * 60}")
    print("Batch processing complete.")
    print(f"All results have been saved in the '{config.OUTPUT_DIR}' directory.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
