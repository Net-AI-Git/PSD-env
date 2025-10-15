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
from optimizer_core.data_loader import FileType
from optimizer_core import problem_definition_points as problem
# from custom_point_generator import generate_custom_candidate_points
from utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)


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

def process_psd_job(job, output_directory, stop_event=None):
    """
    Runs the complete genetic algorithm optimization for a single measurement job.

    Args:
        job (dict): A dictionary containing the measurement data and metadata.
        output_directory (str): The directory to save the results in.
        stop_event (threading.Event, optional): Event to signal stop request. If set, optimization will terminate.
    """
    # This dynamic import is moved here to ensure that the correct problem
    # definition is loaded every time the function is called, based on the
    # current configuration set by the calling function.


    overall_start_time = time.time()

    # --- Data Preprocessing ---
    # Data is now passed directly in the 'job' dictionary
    frequencies = job['frequencies']
    psd_values = job['psd_values']
    output_filename_base = job['output_filename_base']

    if frequencies is None or len(frequencies) == 0:
        logger.warning(f"Job '{output_filename_base}' has no data. Skipping.")
        return

    # --- Original candidate points generator ---
    candidate_points = psd_utils.create_multi_scale_envelope(
        frequencies, psd_values, config.WINDOW_SIZES
    )
    
    # --- Custom candidate points generator (commented out for now) ---
    # candidate_points = generate_custom_candidate_points(
    #     frequencies, psd_values, config.LIFT_FACTOR
    # )

    # --- Enforce the correct starting point ---
    if len(frequencies) > 0:
        first_point = np.array([[frequencies[0], psd_values[0]]])
        # Ensure other_points only contains points with frequency greater than the first point
        other_points_mask = candidate_points[:, 0] > frequencies[0]
        other_points = candidate_points[other_points_mask]
        candidate_points = np.vstack((first_point, other_points))
    else:  # Handle case with no valid data points after filtering
        logger.warning(f"No data points left for '{output_filename_base}' after pre-processing. Skipping.")
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
    logger.info(f"Creating Initial Population of {config.POPULATION_SIZE} solutions")
    pop_creation_start = time.time()
    population = []
    attempts = 0
    # Set a high but finite number of attempts to find solutions
    max_attempts = config.POPULATION_SIZE * 20

    while len(population) < config.POPULATION_SIZE and attempts < max_attempts:
        # Check if stop was requested
        if stop_event and stop_event.is_set():
            logger.warning("Optimization stopped by user during population creation")
            return
        
        solution = problem.create_random_solution(valid_jumps_graph, config.TARGET_POINTS)
        if solution is not None:
            population.append(solution)
        attempts += 1

    pop_creation_end = time.time()
    logger.info(f"Initial population created in {pop_creation_end - pop_creation_start:.2f} seconds ({attempts} attempts).")

    # Critical check: if no solutions could be found, we cannot proceed.
    if not population:
        logger.error(f"FATAL: Could not generate any valid solutions after {max_attempts} attempts.")
        logger.error("This may happen if the input data or parameters result in a graph with no possible paths.")
        logger.error("Aborting optimization for this job.")
        return

    # Warning if the population is smaller than desired
    if len(population) < config.POPULATION_SIZE:
        logger.warning(f"Could only create {len(population)}/{config.POPULATION_SIZE} valid initial solutions.")

    best_solution_so_far, best_cost_so_far = None, float('inf')

    # --- Main Evolution Loop ---
    logger.info("Starting Evolution")
    evolution_start_time = time.time()

    # Initialize variables for early stopping
    generations_without_improvement = 0
    last_best_cost = float('inf')
    generation = 0

    while generation < config.MAX_GENERATIONS:
        # Check if stop was requested
        if stop_event and stop_event.is_set():
            logger.warning("Optimization stopped by user at generation {}".format(generation + 1))
            return
        
        population = [p for p in population if p and len(p) > 1]
        if not population:
            logger.warning("Population became empty. Exiting evolution.")
            break

        all_metrics = [
            problem.calculate_metrics(path, **ga_params,
                                      target_points=config.TARGET_POINTS,
                                      target_area_ratio=config.TARGET_AREA_RATIO,
                                      X_AXIS_MODE=config.AREA_X_AXIS_MODE)
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
                best_solution_so_far, **ga_params,
                target_points=config.TARGET_POINTS,
                target_area_ratio=config.TARGET_AREA_RATIO,
                X_AXIS_MODE=config.AREA_X_AXIS_MODE
            )
            logger.info(
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
                logger.info(f"Terminating early at generation {generation + 1} due to convergence")
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
            best_solution_so_far, **ga_params,
            target_points=config.TARGET_POINTS,
            target_area_ratio=config.TARGET_AREA_RATIO,
            X_AXIS_MODE=config.AREA_X_AXIS_MODE
        )
        logger.info("Optimization Finished")
        logger.info(f"Evolution process time: {end_time - evolution_start_time:.2f} seconds")
        logger.info(f"Total process time: {end_time - overall_start_time:.2f} seconds")
        logger.info(f"Best solution has {final_len} points.")
        logger.info(f"Final Internal Cost: {final_cost:.4f}")
        logger.info(f"Final RMS Ratio: {np.sqrt(final_ratio):.6f}")

        final_points_coords = ga_params['simplified_points'][best_solution_so_far]

        # Save the final solution plot using the new comprehensive name
        psd_utils.plot_final_solution(
            frequencies,
            psd_values,
            final_points_coords,
            np.sqrt(final_ratio),
            output_filename_base,
            output_directory
        )
    else:
        logger.warning("No valid solution found")


def main(file_type=None, stop_event=None):
    """
    Main batch processing function. Manages directories and loops through input files.
    
    Args:
        file_type (FileType, optional): The type of file to process. If None,
                                       will attempt to determine from extension.
        stop_event (threading.Event, optional): Event to signal stop request. If set, processing will terminate.
    """
    # --- Directory Management ---
    # The main output directory is created here. Sub-directories for each file
    # will be created within the processing loop.
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
        logger.info(f"Output directory '{config.OUTPUT_DIR}' created.")

    if not os.path.exists(config.INPUT_DIR):
        logger.error(f"Input directory '{config.INPUT_DIR}' not found. Exiting.")
        return

    if config.FULL_ENVELOPE:
        logger.info("Starting Full Envelope Processing")
        
        # Load all jobs using the full envelope function
        envelope_jobs, channel_groups = data_loader.load_full_envelope_data(config.INPUT_DIR, file_type)
        
        if not envelope_jobs:
            logger.warning("No valid jobs found for full envelope processing. Exiting.")
            return
        
        # Create output directory for envelope results
        envelope_output_dir = os.path.join(config.OUTPUT_DIR, "full_envelope")
        if not os.path.exists(envelope_output_dir):
            os.makedirs(envelope_output_dir)
            logger.info(f"Created output directory: {envelope_output_dir}")
        
        # Create envelop subdirectory for comparison plots
        envelop_plots_dir = os.path.join(envelope_output_dir, "envelop")
        if not os.path.exists(envelop_plots_dir):
            os.makedirs(envelop_plots_dir)
            logger.info(f"Created envelop plots directory: {envelop_plots_dir}")
        
        # Create comparison plots for all channels BEFORE optimization
        logger.info("Creating Envelope Comparison Plots")
        for channel_name, original_jobs in channel_groups.items():
            if len(original_jobs) > 1:  # Only create plots for channels with multiple measurements
                # Find the corresponding envelope job
                envelope_job = None
                for job in envelope_jobs:
                    if job['output_filename_base'] == f"{channel_name}_envelope":
                        envelope_job = job
                        break
                
                if envelope_job is not None:
                    plot_path = os.path.join(envelop_plots_dir, f"{channel_name}_envelope.png")
                    logger.info(f"Creating comparison plot for channel: {channel_name}")
                    data_loader.plot_envelope_comparison(
                        original_jobs, 
                        envelope_job, 
                        channel_name, 
                        plot_path
                    )
        
        # Sort the envelope jobs naturally
        envelope_jobs.sort(key=data_loader.natural_sort_key)
        
        logger.info(f"Processing {len(envelope_jobs)} envelope measurements:")
        
        # --- Loop through each envelope measurement ---
        for envelope_job in envelope_jobs:
            # Check if stop was requested
            if stop_event and stop_event.is_set():
                logger.warning("Envelope processing stopped by user")
                return
            
            logger.info(f"{'=' * 60}")
            logger.info(f"Processing envelope measurement: {envelope_job['output_filename_base']}")
            logger.info(f"Results will be saved in: {envelope_output_dir}")
            logger.info(f"{'-' * 60}")
            process_psd_job(envelope_job, envelope_output_dir, stop_event)
    
    else:
        logger.info("Starting File-by-File Batch Processing")

        # --- Processing Loop for each file in the input directory ---
        for filename in sorted(os.listdir(config.INPUT_DIR)):
            filepath = os.path.join(config.INPUT_DIR, filename)
            
            # Load all jobs from the current file
            jobs_from_file = data_loader.load_data_from_file(filepath, file_type)

            if not jobs_from_file:
                # The loader function will print a warning, so we just continue
                continue

            # Create a dedicated output directory for this source file
            source_filename_no_ext = os.path.splitext(filename)[0]
            output_dir_for_file = os.path.join(config.OUTPUT_DIR, source_filename_no_ext)
            if not os.path.exists(output_dir_for_file):
                os.makedirs(output_dir_for_file)
                logger.info(f"Created output sub-directory: {output_dir_for_file}")

            # Sort the jobs from this file naturally
            jobs_from_file.sort(key=data_loader.natural_sort_key)
            
            logger.info(f"Processing file '{filename}' ({len(jobs_from_file)} measurements):")

            # --- Loop through each measurement (job) from the current file ---
            for job in jobs_from_file:
                # Check if stop was requested
                if stop_event and stop_event.is_set():
                    logger.warning("Batch processing stopped by user")
                    return
                
                logger.info(f"{'=' * 60}")
                logger.info(f"Processing measurement: {job['output_filename_base']}")
                logger.info(f"Results will be saved in: {output_dir_for_file}")
                logger.info(f"{'-' * 60}")
                process_psd_job(job, output_dir_for_file, stop_event)

    logger.info(f"{'=' * 60}")
    logger.info("Batch processing complete.")
    logger.info(f"All results have been saved in sub-directories within '{config.OUTPUT_DIR}'.")
    logger.info(f"{'=' * 60}")


def run_optimization_process(
        min_frequency_hz: int = 5,
        max_frequency_hz: int = 2000,
        target_points: int = 45,
        target_area_ratio: float = 1.25,
        stab_wide: Literal["narrow", "wide"] = "narrow",
        area_x_axis_mode: Literal["Log", "Linear"] = "Log",
        input_dir: str = None,
        full_envelope: bool = False,
        file_type: FileType = None,
        stop_event = None
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
        full_envelope (bool): If True, loads all files and creates envelope
                              from maximum PSD values across matching channels.
                              Defaults to False.
        file_type (FileType, optional): The type of file to process. If None,
                                       will attempt to determine from extension.
                                       Defaults to None.
        stop_event (threading.Event, optional): Event to signal stop request. If set, optimization will terminate.

    Returns:
        None
    """
    # --- 1. Update Configuration from Arguments ---
    logger.info("Configuring optimization run")
    # If an input_dir is provided, override the config file setting
    if input_dir:
        config.INPUT_DIR = input_dir

    config.TARGET_AREA_RATIO = target_area_ratio**2
    config.TARGET_POINTS = target_points
    config.MIN_FREQUENCY_HZ = min_frequency_hz
    config.MAX_FREQUENCY_HZ = max_frequency_hz
    config.AREA_X_AXIS_MODE = area_x_axis_mode
    config.FULL_ENVELOPE = full_envelope

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
    logger.info(f"Target Points: {config.TARGET_POINTS}, Target Area Ratio: {config.TARGET_AREA_RATIO}")

    # --- 3. Set Stability-related Parameters ---
    if stab_wide == "narrow":
        config.WINDOW_SIZES = [10, 20, 30]
        config.ENRICH_LOW_FREQUENCIES = True
        logger.info("Using 'narrow' stability settings (more detailed scan).")
    else:  # "wide"
        config.WINDOW_SIZES = [20, 30, 40, 50]
        config.ENRICH_LOW_FREQUENCIES = False
        logger.info("Using 'wide' stability settings (broader scan).")

    logger.info("Running Optimization with the following parameters")
    logger.info(f"  - Optimization Mode : {config.OPTIMIZATION_MODE}")
    logger.info(f"  - Target Points       : {config.TARGET_POINTS}")
    logger.info(f"  - Target Area Ratio  : {config.TARGET_AREA_RATIO}")
    logger.info(f"  - Frequency Range     : {config.MIN_FREQUENCY_HZ}Hz - {config.MAX_FREQUENCY_HZ}Hz")
    logger.info(f"  - Window Sizes        : {config.WINDOW_SIZES}")
    logger.info(f"  - Enrich Low Freqs    : {config.ENRICH_LOW_FREQUENCIES}")
    logger.info(f"  - Area X-Axis Mode    : {config.AREA_X_AXIS_MODE}")
    logger.info(f"  - Low Freq Weight     : {config.LOW_FREQ_AREA_WEIGHT}")
    logger.info("---------------------------------------------------------")

    # --- 4. Execute the Main Process ---
    main(file_type, stop_event)


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
        area_x_axis_mode="Log",
        full_envelope=True,  # Set to True to enable full envelope mode
        file_type=FileType.TESTLAB  # Specify file type explicitly
    )



