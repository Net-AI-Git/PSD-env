import time
import numpy as np
import random
import os
import sys

# Import the genetic algorithm modules from the 'optimizer_core' package
# This import will trigger the GPU availability check
from optimizer_core import gpu_utils
from optimizer_core import config
from optimizer_core import psd_utils
from optimizer_core import problem_definition as problem
from optimizer_core import ga_operators as operators
from optimizer_core import data_loader


# ===================================================================
#
#           Main Execution Script for PSD Envelope Optimization
#
# This script orchestrates the entire optimization process by:
# 1. Loading and sorting all measurements from the input directory.
# 2. Looping through each measurement ("job").
# 3. Running the full optimization process for each job.
# 4. Saving the results as images with detailed names.
#
# ===================================================================

def process_psd_job(job):
    """
    Runs the complete genetic algorithm optimization for a single measurement job.

    Args:
        job (dict): A dictionary containing the measurement data and metadata.
    """
    overall_start_time = time.time()

    # --- Data Preprocessing ---
    # Move data to GPU if available, otherwise it stays as NumPy arrays
    frequencies = gpu_utils.xp.asarray(job['frequencies'])
    psd_values = gpu_utils.xp.asarray(job['psd_values'])
    output_filename_base = job['output_filename_base']

    if len(frequencies) == 0:
        print(f"Job '{output_filename_base}' has no data. Skipping.")
        return

<<<<<<< Updated upstream
    # NOTE: create_multi_scale_envelope is not yet vectorized, so it needs CPU data.
    candidate_points_cpu = psd_utils.create_multi_scale_envelope(
        gpu_utils.to_cpu(frequencies), gpu_utils.to_cpu(psd_values), config.WINDOW_SIZES
=======
    # --- Pre-computation ---
    # Calculate the area of the original PSD signal once. This is a constant
    # used in area ratio calculations, so computing it upfront saves redundant
    # calculations inside the main evolution loop.
    original_area = np.trapezoid(psd_values, x=frequencies)

    candidate_points = psd_utils.create_multi_scale_envelope(
        frequencies, psd_values, config.WINDOW_SIZES
>>>>>>> Stashed changes
    )
    # Move result back to the active device (GPU or CPU)
    candidate_points = gpu_utils.xp.asarray(candidate_points_cpu)


    # --- Enforce the correct starting point ---
    if len(frequencies) > 0:
        first_point = gpu_utils.xp.array([[frequencies[0], psd_values[0]]])
        other_points_mask = candidate_points[:, 0] > frequencies[0]
        other_points = candidate_points[other_points_mask]
        candidate_points = gpu_utils.xp.vstack((first_point, other_points))
    else:  # Handle case with no valid data points after filtering
        print(f"No data points left for '{output_filename_base}' after pre-processing. Skipping.")
        return

    # --- Graph Construction ---
    if gpu_utils.IS_GPU_AVAILABLE:
        valid_jumps_graph = problem.build_valid_jumps_graph_vectorized(
            candidate_points, frequencies, psd_values
        )
    else:
        # The original function expects NumPy arrays
        valid_jumps_graph = problem.build_valid_jumps_graph(
            gpu_utils.to_cpu(candidate_points),
            gpu_utils.to_cpu(frequencies),
            gpu_utils.to_cpu(psd_values)
        )

    # Pack shared parameters. For now, the rest of the GA runs on the CPU,
    # so we ensure all data passed to it is on the CPU.
    ga_params = {
        'graph': valid_jumps_graph,
<<<<<<< Updated upstream
        'simplified_points': gpu_utils.to_cpu(candidate_points),
        'original_psd_freqs': gpu_utils.to_cpu(frequencies),
        'original_psd_values': gpu_utils.to_cpu(psd_values),
        'prune_threshold': config.PRUNE_THRESHOLD
=======
        'simplified_points': candidate_points,
        'original_psd_freqs': frequencies,
        'original_psd_values': psd_values,
        'prune_threshold': config.PRUNE_THRESHOLD,
        'original_area': original_area
>>>>>>> Stashed changes
    }

    # --- Initial Population Creation ---
    if config.USE_CPU_PARALLELISM:
        # Use the new parallel function
        population = problem.create_initial_population_parallel(
            valid_jumps_graph, config.TARGET_POINTS, config.POPULATION_SIZE
        )
    else:
        # Fallback to the original sequential method
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

        # NOTE: The metric calculation is not yet vectorized. It runs on the CPU.
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

        # Display results based on the optimization mode
        if config.OPTIMIZATION_MODE == 'TARGET_AREA_RATIO':
            print(f"Target Area Ratio: {config.TARGET_AREA_RATIO:.4f}")
            print(f"Achieved Area Ratio: {final_ratio:.6f}")
            print(f"Final Point Count: {final_len}")
        else: # Original TARGET_POINTS mode
            print(f"Best solution has {final_len} points.")
            print(f"Final Area Ratio: {final_ratio:.6f}")

        print(f"Final Internal Cost: {final_cost:.4f}")

        final_points_coords = ga_params['simplified_points'][best_solution_so_far]

        # Save the final solution plot using the new comprehensive name
        psd_utils.plot_final_solution(
            ga_params['original_psd_freqs'],
            ga_params['original_psd_values'],
            final_points_coords,
            final_ratio,
            output_filename_base  # <-- PASS THE NEW FILENAME BASE
        )
    else:
        print("\n--- No valid solution found ---")


def main():
    """
    Main batch processing function. Manages directories and loops through input files.
    """
    # --- Directory Management ---
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
        print(f"Output directory '{config.OUTPUT_DIR}' created.")

    # --- Data Loading and Sorting Phase ---
    # The data_loader now handles loading from all files and sorting the results.
    jobs_to_process = data_loader.load_all_data_from_input_dir()

    if not jobs_to_process:
        print(f"\nNo valid measurements found in '{config.INPUT_DIR}'. Exiting.")
        return

    print("\n--- Starting Batch Processing ---")
    print("The following measurements were loaded and will be processed in order:")
    for job in jobs_to_process:
        print(f"  - {job['output_filename_base']}")

    # --- Processing Loop ---
    for job in jobs_to_process:
        print(f"\n{'=' * 60}")
        print(f"Processing measurement: {job['output_filename_base']}")
        print(f"{'=' * 60}")
        process_psd_job(job)  # <-- PROCESS JOB INSTEAD OF FILEPATH

    print(f"\n{'=' * 60}")
    print("Batch processing complete.")
    print(f"All results have been saved in the '{config.OUTPUT_DIR}' directory.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

