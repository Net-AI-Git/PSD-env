import time
import numpy as np
import random
import os
import sys
from typing import Literal

# Import the genetic algorithm modules from the 'optimizer_core' package
from optimizer_core import config
from optimizer_core import psd_utils
from optimizer_core import ga_operators as operators
from optimizer_core import data_loader

# Dynamic import based on optimization mode
# THIS BLOCK WILL BE MOVED.
# if config.OPTIMIZATION_MODE == "area":
#     from optimizer_core import problem_definition_area as problem
# else:
#     from optimizer_core import problem_definition_points as problem


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
    # This dynamic import is moved here to ensure that the correct problem
    # definition is loaded every time the function is called, based on the
    # current configuration set by the calling function.
    if config.OPTIMIZATION_MODE == "area":
        from optimizer_core import problem_definition_area as problem
    else:
        from optimizer_core import problem_definition_points as problem

    overall_start_time = time.time()

    # --- Data Preprocessing ---
    # Data is now passed directly in the 'job' dictionary
    frequencies = job['frequencies']
    psd_values = job['psd_values']
    output_filename_base = job['output_filename_base']

    if frequencies is None or len(frequencies) == 0:
        print(f"Job '{output_filename_base}' has no data. Skipping.")
        return

    candidate_points = psd_utils.create_multi_scale_envelope(
        frequencies, psd_values, config.WINDOW_SIZES
    )

    # --- Enforce the correct starting point ---
    if len(frequencies) > 0:
        first_point = np.array([[frequencies[0], psd_values[0]]])
        # Ensure other_points only contains points with frequency greater than the first point
        other_points_mask = candidate_points[:, 0] > frequencies[0]
        other_points = candidate_points[other_points_mask]
        candidate_points = np.vstack((first_point, other_points))
    else:  # Handle case with no valid data points after filtering
        print(f"No data points left for '{output_filename_base}' after pre-processing. Skipping.")
        return

    # --- Graph Construction ---
    valid_jumps_graph = problem.build_valid_jumps_graph(
        candidate_points, frequencies, psd_values
    )
    # --- Prune the graph to remove dead-end paths ---
    valid_jumps_graph = problem.prune_dead_end_nodes(valid_jumps_graph)

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
    population = []
    attempts = 0
    # Set a high but finite number of attempts to find solutions
    max_attempts = config.POPULATION_SIZE * 20

    while len(population) < config.POPULATION_SIZE and attempts < max_attempts:
        solution = problem.create_random_solution(valid_jumps_graph, config.TARGET_POINTS)
        if solution is not None:
            population.append(solution)
        attempts += 1

    pop_creation_end = time.time()
    print(f"Initial population created in {pop_creation_end - pop_creation_start:.2f} seconds ({attempts} attempts).")

    # Critical check: if no solutions could be found, we cannot proceed.
    if not population:
        print(f"\nFATAL: Could not generate any valid solutions after {max_attempts} attempts.")
        print("This may happen if the input data or parameters result in a graph with no possible paths.")
        print("Aborting optimization for this job.")
        return

    # Warning if the population is smaller than desired
    if len(population) < config.POPULATION_SIZE:
        print(f"Warning: Could only create {len(population)}/{config.POPULATION_SIZE} valid initial solutions.")

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
                f"RMS Ratio: {np.sqrt(best_ratio):.4f} | "
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

        # --- Prune a percentage of the general (non-elite) population ---
        # Identify non-elite solutions to avoid reprocessing elites
        non_elite_indices = [i for i in range(len(population)) if i not in elite_indices]
        non_elite_population = [population[i] for i in non_elite_indices]

        # Determine how many individuals to select for pruning
        num_to_prune = int(len(non_elite_population) * config.PRUNE_PERCENTAGE_OF_POPULATION)

        if num_to_prune > 0 and len(non_elite_population) > num_to_prune:
            # Randomly select a subset of the non-elite population
            solutions_to_prune = random.sample(non_elite_population, num_to_prune)

            # Apply the pruning mutation to the selected individuals
            for sol in solutions_to_prune:
                pruned_version = operators.mutate_prune_useless_points(sol, **ga_params)
                # If the solution was improved, add it to the next generation
                if pruned_version != sol:
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
        print(f"Final RMS Ratio: {np.sqrt(final_ratio):.6f}")

        final_points_coords = ga_params['simplified_points'][best_solution_so_far]

        # Save the final solution plot using the new comprehensive name
        psd_utils.plot_final_solution(
            frequencies,
            psd_values,
            final_points_coords,
            np.sqrt(final_ratio),
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


def run_optimization_process(
        min_frequency_hz: int,
    max_frequency_hz: int,
    optimization_mode: Literal["points", "area"],
    target: float,  # Can be integer for points or float for area ratio
        stab_wide: Literal["narrow", "wide"],
    area_x_axis_mode: Literal["Log", "Linear"]
) -> None:
    """
    Sets up the configuration and runs the entire PSD optimization process.

    This function serves as the main entry point for running the optimization
    with a specific set of parameters, allowing for programmatic execution
    without manually editing configuration files.

    Args:
        optimization_mode: The strategy to use ("points" or "area").
        min_frequency_hz: The minimum frequency for data filtering.
        max_frequency_hz: The maximum frequency for data filtering.
        target: The target value. If mode is "points", this is the desired
                number of points. If mode is "area", this is the desired
                area ratio (e.g., 1.2).
        stab_wide: Defines the stability analysis mode, affecting parameters
                   like WINDOW_SIZES.
        area_x_axis_mode: The X-axis domain for area integration ("Log" or "Linear").

    Returns:
        None
    """
    # --- 1. Update Configuration from Arguments ---
    print("--- Configuring optimization run ---")
    config.OPTIMIZATION_MODE = optimization_mode
    config.MIN_FREQUENCY_HZ = min_frequency_hz
    config.MAX_FREQUENCY_HZ = max_frequency_hz
    config.AREA_X_AXIS_MODE = area_x_axis_mode

    # Set the area weight based on the X-axis mode
    if area_x_axis_mode == "Linear":
        config.LOW_FREQ_AREA_WEIGHT = 1
    else:  # "Log"
        config.LOW_FREQ_AREA_WEIGHT = 1

    # --- 2. Set Target based on Optimization Mode ---
    if optimization_mode == "points":
        target_points = int(target)
        config.TARGET_P = target_points
        config.TARGET_POINTS = target_points * 0.9
        print(f"Mode: 'points', Target Points: {config.TARGET_POINTS} (from {target_points})")
    elif optimization_mode == "area":
        target_area = float(target)
        config.TARGET_A = target_area
        config.TARGET_AREA_RATIO = (target_area ** 2) * 0.98
        print(f"Mode: 'area', Target Area Ratio: {config.TARGET_AREA_RATIO} (from {target_area})")

    # --- 3. Set Stability-related Parameters ---
    if stab_wide == "narrow":
        config.WINDOW_SIZES = [10, 20, 30]
        config.ENRICH_LOW_FREQUENCIES = True
        print("Using 'narrow' stability settings (more detailed scan).")
    else:  # "wide"
        config.WINDOW_SIZES = [20,30, 40, 50]
        config.ENRICH_LOW_FREQUENCIES = False
        print("Using 'wide' stability settings (broader scan).")

    print("--- Running Optimization with the following parameters ---")
    print(f"  - Optimization Mode : {config.OPTIMIZATION_MODE}")
    if config.OPTIMIZATION_MODE == "points":
        print(f"  - Target Points       : {config.TARGET_POINTS}")
    else:  # "area"
        print(f"  - Target Area Ratio   : {config.TARGET_AREA_RATIO}")
    print(f"  - Frequency Range     : {config.MIN_FREQUENCY_HZ}Hz - {config.MAX_FREQUENCY_HZ}Hz")
    print(f"  - Window Sizes        : {config.WINDOW_SIZES}")
    print(f"  - Enrich Low Freqs    : {config.ENRICH_LOW_FREQUENCIES}")
    print(f"  - Area X-Axis Mode    : {config.AREA_X_AXIS_MODE}")
    print(f"  - Low Freq Weight     : {config.LOW_FREQ_AREA_WEIGHT}")
    print("---------------------------------------------------------")

    # --- 4. Execute the Main Process ---
    main()


if __name__ == "__main__":
    # The main execution block now calls the control function with a set
    # of default parameters. This preserves the script's runnability while
    # using the new, more flexible mechanism.
    run_optimization_process(
        min_frequency_hz=5,
        max_frequency_hz=2000,
        optimization_mode="area",
        target=1.25,
        stab_wide="narrow",
        area_x_axis_mode="Log"
    )



