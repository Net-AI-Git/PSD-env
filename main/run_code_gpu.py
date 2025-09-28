import time
import numpy as np
import cupy as cp
import random
import os
import sys
from typing import Literal

# Import the genetic algorithm modules from the 'optimizer_core' package
from optimizer_core import config
from optimizer_core import config_gpu  # Import GPU configuration
from optimizer_core import psd_utils_gpu as psd_utils
from optimizer_core import ga_operators_gpu as operators
from optimizer_core import problem_definition_points_gpu as problem
from optimizer_core import data_loader
# from custom_point_generator import generate_custom_candidate_points


# ===================================================================
#
#           Main Execution Script for PSD Envelope Optimization (GPU Version)
#
# ===================================================================

def process_psd_job(job, output_directory):
    """
    Runs the complete genetic algorithm optimization for a single measurement job.
    """
    overall_start_time = time.time()

    # --- Data Preprocessing (on CPU with NumPy) ---
    frequencies = job['frequencies']
    psd_values = job['psd_values']
    output_filename_base = job['output_filename_base']

    if frequencies is None or len(frequencies) == 0:
        print(f"Job '{output_filename_base}' has no data. Skipping.")
        return

    candidate_points_np = psd_utils.create_multi_scale_envelope(
        frequencies, psd_values, config.WINDOW_SIZES
    )

    if len(frequencies) > 0:
        first_point = np.array([[frequencies[0], psd_values[0]]])
        other_points_mask = candidate_points_np[:, 0] > frequencies[0]
        other_points = candidate_points_np[other_points_mask]
        candidate_points_np = np.vstack((first_point, other_points))
    else:
        print(f"No data points left for '{output_filename_base}' after pre-processing. Skipping.")
        return

    # --- Data Transfer to GPU ---
    print("--- Transferring data to GPU ---")
    frequencies_gpu = cp.asarray(frequencies)
    psd_values_gpu = cp.asarray(psd_values)
    candidate_points_gpu = cp.asarray(candidate_points_np)

    # --- Graph Construction (on CPU, as it's logic-heavy) ---
    valid_jumps_graph = problem.build_valid_jumps_graph(
        candidate_points_np, frequencies, psd_values
    )
    valid_jumps_graph = problem.prune_dead_end_nodes(valid_jumps_graph)

    # Pack shared GPU parameters for the fitness function
    ga_params = {
        'simplified_points': candidate_points_gpu,
        'original_psd_freqs': frequencies_gpu,
        'original_psd_values': psd_values_gpu,
        'prune_threshold': config.PRUNE_THRESHOLD
    }

    # --- Initial Population Creation (on CPU) ---
    print(f"\n--- Creating Initial Population of {config.POPULATION_SIZE} solutions ---")
    pop_creation_start = time.time()
    population = []
    attempts = 0
    max_attempts = config.POPULATION_SIZE * 20

    while len(population) < config.POPULATION_SIZE and attempts < max_attempts:
        solution = problem.create_random_solution(valid_jumps_graph, config.TARGET_POINTS)
        if solution is not None:
            population.append(solution)
        attempts += 1

    pop_creation_end = time.time()
    print(f"Initial population created in {pop_creation_end - pop_creation_start:.2f} seconds ({attempts} attempts).")

    if not population:
        print(f"\nFATAL: Could not generate any valid solutions after {max_attempts} attempts.")
        return

    if len(population) < config.POPULATION_SIZE:
        print(f"Warning: Could only create {len(population)}/{config.POPULATION_SIZE} valid initial solutions.")

    best_solution_so_far, best_cost_so_far = None, float('inf')

    # --- Main Evolution Loop ---
    print("\n--- Starting Evolution on GPU ---")
    evolution_start_time = time.time()

    generations_without_improvement = 0
    last_best_cost = float('inf')
    generation = 0

    while generation < config.MAX_GENERATIONS:
        population = [p for p in population if p and len(p) > 1]
        if not population:
            print("Population became empty. Exiting evolution.")
            break
            
        # --- Batch Fitness Calculation on GPU ---
        all_costs, all_fitness, all_lengths, all_ratios = [], [], [], []
        batch_size = config_gpu.CALCULATE_METRICS_BATCH_SIZE
        
        for i in range(0, len(population), batch_size):
            batch_paths = population[i:i + batch_size]
            
            # Run the batch calculation on the GPU
            batch_costs, batch_fitness, _, _ = problem.calculate_metrics_batch(
                batch_paths, **ga_params,
                target_points=config.TARGET_POINTS,
                target_area_ratio=config.TARGET_AREA_RATIO,
                X_AXIS_MODE=config.AREA_X_AXIS_MODE
            )
            all_costs.append(batch_costs)
            all_fitness.append(batch_fitness)

        # Concatenate results from all batches into single CuPy arrays
        costs = cp.concatenate(all_costs)
        fitness_scores = cp.concatenate(all_fitness)
        
        # --- Find best solution (operations on GPU) ---
        current_min_cost = cp.min(costs)
        if current_min_cost < best_cost_so_far:
            best_cost_so_far = current_min_cost
            # Argmin returns the index, population is on CPU, so this is fine
            best_solution_so_far = population[cp.argmin(costs).item()]

        current_best_len = len(best_solution_so_far)
        
        # Print progress update every 10 generations
        if (generation + 1) % 10 == 0:
            # For reporting, we can run a single calculation on the best solution
            # or retrieve the values from the batch calculation.
            # Here we just report the best cost found.
            print(
                f"Gen {generation + 1}/{config.MAX_GENERATIONS} | "
                f"Best Cost: {best_cost_so_far.item():.4f}"
            )

        # --- Early Stopping Logic (on CPU) ---
        if config.USE_CONVERGENCE_TERMINATION:
            improvement = last_best_cost - best_cost_so_far.item()
            if improvement < config.CONVERGENCE_THRESHOLD:
                generations_without_improvement += 1
            else:
                generations_without_improvement = 0

            last_best_cost = best_cost_so_far.item()

            if generations_without_improvement >= config.CONVERGENCE_PATIENCE:
                print(f"\n--- Terminating early at generation {generation + 1} due to convergence ---")
                break
        
        # --- Generate the next generation (on CPU) ---
        # Note: Selection and mutation operators are logic-heavy and remain on CPU
        new_population = []
        # Elitism - need fitness scores on CPU
        fitness_scores_cpu = cp.asnumpy(fitness_scores)
        
        if config.ELITISM_SIZE > 0:
            elite_indices = np.argsort(fitness_scores_cpu)[-config.ELITISM_SIZE:]
            elite_solutions = [population[i] for i in elite_indices]
            new_population.extend(elite_solutions)

        while len(new_population) < config.POPULATION_SIZE:
            parent1 = operators.selection(population, fitness_scores_cpu)
            parent2 = operators.selection(population, fitness_scores_cpu)
            child = operators.crossover_multipoint_paths(parent1, parent2)

            if random.random() < config.MUTATION_RATE:
                # Mutation uses the CPU graph and params
                mutated_child = operators.apply_mutations(
                    child,
                    {'graph': valid_jumps_graph, 'simplified_points': candidate_points_np},
                    current_best_len, config.ADAPTIVE_MUTATION_THRESHOLD
                )
            else:
                mutated_child = child
            new_population.append(mutated_child)
            
        population = new_population
        generation += 1

    end_time = time.time()

    # --- Final Results ---
    if best_solution_so_far:
        print("\n--- Optimization Finished ---")
        print(f"Evolution process time: {end_time - evolution_start_time:.2f} seconds")
        print(f"Total process time: {end_time - overall_start_time:.2f} seconds")
        
        # To get final metrics, run one last calculation for the best solution
        final_cost, _, final_len, final_ratio = problem.calculate_metrics_batch(
            [best_solution_so_far], **ga_params,
            target_points=config.TARGET_POINTS,
            target_area_ratio=config.TARGET_AREA_RATIO,
            X_AXIS_MODE=config.AREA_X_AXIS_MODE
        )
        
        print(f"Best solution has {final_len[0].item()} points.")
        print(f"Final Internal Cost: {final_cost[0].item():.4f}")
        print(f"Final RMS Ratio: {cp.sqrt(final_ratio[0]).item():.6f}")

        # --- Data Transfer back to CPU for Plotting ---
        final_points_coords_gpu = candidate_points_gpu[best_solution_so_far]
        
        psd_utils.plot_final_solution(
            cp.asnumpy(frequencies_gpu),
            cp.asnumpy(psd_values_gpu),
            cp.asnumpy(final_points_coords_gpu),
            cp.sqrt(final_ratio[0]).item(),
            output_filename_base,
            output_directory
        )
    else:
        print("\n--- No valid solution found ---")


def main():
    """
    Main batch processing function. Manages directories and loops through input files.
    """
    # --- Directory Management ---
    # The main output directory is created here. Sub-directories for each file
    # will be created within the processing loop.
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
        print(f"Output directory '{config.OUTPUT_DIR}' created.")

    if not os.path.exists(config.INPUT_DIR):
        print(f"Error: Input directory '{config.INPUT_DIR}' not found. Exiting.")
        return

    print("\n--- Starting File-by-File Batch Processing ---")

    # --- Processing Loop for each file in the input directory ---
    for filename in sorted(os.listdir(config.INPUT_DIR)):
        filepath = os.path.join(config.INPUT_DIR, filename)
        
        # Load all jobs from the current file
        jobs_from_file = data_loader.load_data_from_file(filepath)

        if not jobs_from_file:
            # The loader function will print a warning, so we just continue
            continue

        # Create a dedicated output directory for this source file
        source_filename_no_ext = os.path.splitext(filename)[0]
        output_dir_for_file = os.path.join(config.OUTPUT_DIR, source_filename_no_ext)
        if not os.path.exists(output_dir_for_file):
            os.makedirs(output_dir_for_file)
            print(f"\nCreated output sub-directory: {output_dir_for_file}")

        # Sort the jobs from this file naturally
        jobs_from_file.sort(key=data_loader.natural_sort_key)
        
        print(f"\nProcessing file '{filename}' ({len(jobs_from_file)} measurements):")

        # --- Loop through each measurement (job) from the current file ---
        for job in jobs_from_file:
            print(f"\n{'=' * 60}")
            print(f"  Processing measurement: {job['output_filename_base']}")
            print(f"  Results will be saved in: {output_dir_for_file}")
            print(f"{'-' * 60}")
            process_psd_job(job, output_dir_for_file)

    print(f"\n{'=' * 60}")
    print("Batch processing complete.")
    print(f"All results have been saved in sub-directories within '{config.OUTPUT_DIR}'.")
    print(f"{'=' * 60}")


def run_optimization_process(
        min_frequency_hz: int = 5,
        max_frequency_hz: int = 2000,
        target_points: int = 45,
        target_area_ratio: float = 1.25,
        stab_wide: Literal["narrow", "wide"] = "narrow",
        area_x_axis_mode: Literal["Log", "Linear"] = "Log",
        input_dir: str = None
        ) -> None:
    """
    Sets up the configuration and runs the entire PSD optimization process.

    This function serves as the main entry point for running the optimization
    with a specific set of parameters, allowing for programmatic execution
    without manually editing configuration files.

    Args:
        min_frequency_hz: The minimum frequency for data filtering.
        max_frequency_hz: The maximum frequency for data filtering.
        target_points: The target number of points for the envelope.
        stab_wide: Defines the stability analysis mode, affecting parameters
                   like WINDOW_SIZES.
        area_x_axis_mode: The X-axis domain for area integration ("Log" or "Linear").
        input_dir (str, optional): Overrides the default input directory
                                   from the config file. Defaults to None.

    Returns:
        None
    """
    # --- 1. Update Configuration from Arguments ---
    print("--- Configuring optimization run ---")
    # If an input_dir is provided, override the config file setting
    if input_dir:
        config.INPUT_DIR = input_dir

    config.TARGET_AREA_RATIO = target_area_ratio**2
    config.TARGET_POINTS = target_points
    config.MIN_FREQUENCY_HZ = min_frequency_hz
    config.MAX_FREQUENCY_HZ = max_frequency_hz
    config.AREA_X_AXIS_MODE = area_x_axis_mode

    # --- 2. Calculate Derived Configuration Values ---
    # These values depend on the arguments passed to the function, so they
    # are calculated here instead of being static in the config file.
    
    #config.TARGET_POINTS = config.TARGET_P * 0.9
    #config.TARGET_AREA_RATIO = (config.TARGET_A ** 2) * 0.98

    # Set the area weight based on the X-axis mode
    if area_x_axis_mode == "Linear":
        config.LOW_FREQ_AREA_WEIGHT = 1
    else:  # "Log"
        config.LOW_FREQ_AREA_WEIGHT = 1


    # print(f"Target Points: {config.TARGET_P}, Target RMS Ratio: {config.TARGET_A}")
    print(f"Target Points: {config.TARGET_POINTS}, Target Area Ratio: {config.TARGET_AREA_RATIO}")

    # --- 3. Set Stability-related Parameters ---
    if stab_wide == "narrow":
        config.WINDOW_SIZES = [10, 20, 30]
        config.ENRICH_LOW_FREQUENCIES = True
        print("Using 'narrow' stability settings (more detailed scan).")
    else:  # "wide"
        config.WINDOW_SIZES = [20, 30, 40, 50]
        config.ENRICH_LOW_FREQUENCIES = False
        print("Using 'wide' stability settings (broader scan).")

    print("--- Running Optimization with the following parameters ---")
    print(f"  - Optimization Mode : {config.OPTIMIZATION_MODE}")
    print(f"  - Target Points       : {config.TARGET_POINTS}")
    print(f"  - Target Area Ratio  : {config.TARGET_AREA_RATIO}")
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
        target_area_ratio=1.4,
        target_points=20,
        stab_wide="narrow",
        area_x_axis_mode="Log"
    )



