import os
import time
import numpy as np
import random

from optimizer_core import config
from optimizer_core import data_loader
from optimizer_core import psd_utils
from optimizer_core import ga_operators as operators
# We need to import both problem definitions and switch between them dynamically
from optimizer_core import problem_definition_area
from optimizer_core import problem_definition_points

def run_qa_scenarios():
    """
    Runs a predefined set of QA scenarios to test the PSD optimization
    with different configurations.
    """
    # --- Group scenarios by their WINDOW_SIZES configuration ---
    scenario_groups = [
        {
            "WINDOW_SIZES": [10, 20, 30],  # Default configuration
            "scenarios": [
                {'name': 'Linear_40Points', 'AREA_X_AXIS_MODE': 'Linear', 'OPTIMIZATION_MODE': 'TARGET_P', 'TARGET_P': 45},
                {'name': 'Linear_1.2Ratio', 'AREA_X_AXIS_MODE': 'Linear', 'OPTIMIZATION_MODE': 'TARGET_A', 'TARGET_A': 1.2},
                {'name': 'Log_40Points', 'AREA_X_AXIS_MODE': 'Log', 'OPTIMIZATION_MODE': 'TARGET_P', 'TARGET_P': 45},
                {'name': 'Log_1.2Ratio', 'AREA_X_AXIS_MODE': 'Log', 'OPTIMIZATION_MODE': 'TARGET_A', 'TARGET_A': 1.2},
                {'name': 'Log_1.4Ratio', 'AREA_X_AXIS_MODE': 'Log', 'OPTIMIZATION_MODE': 'TARGET_A', 'TARGET_A': 1.4},
                {'name': 'Log_1.25Ratio', 'AREA_X_AXIS_MODE': 'Log', 'OPTIMIZATION_MODE': 'TARGET_A', 'TARGET_A': 1.25},
                {'name': 'Log_1.1Ratio', 'AREA_X_AXIS_MODE': 'Log', 'OPTIMIZATION_MODE': 'TARGET_A', 'TARGET_A': 1.1},
                {'name': 'Log_6Points', 'AREA_X_AXIS_MODE': 'Log', 'OPTIMIZATION_MODE': 'TARGET_P', 'TARGET_P': 6},
                {'name': 'Log_20Points', 'AREA_X_AXIS_MODE': 'Log', 'OPTIMIZATION_MODE': 'TARGET_P', 'TARGET_P': 20},
                {'name': 'Log_60Points', 'AREA_X_AXIS_MODE': 'Log', 'OPTIMIZATION_MODE': 'TARGET_P', 'TARGET_P': 60},
                {'name': 'Log_90Points', 'AREA_X_AXIS_MODE': 'Log', 'OPTIMIZATION_MODE': 'TARGET_P', 'TARGET_P': 90},
            ]
        },
        {
            "WINDOW_SIZES": [20, 30],
            "scenarios": [
                {'name': 'Log_40Points_Win20_30', 'AREA_X_AXIS_MODE': 'Log', 'OPTIMIZATION_MODE': 'TARGET_P', 'TARGET_P': 40},
            ]
        },
        {
            "WINDOW_SIZES": [30],
            "scenarios": [
                {'name': 'Log_40Points_Win30', 'AREA_X_AXIS_MODE': 'Log', 'OPTIMIZATION_MODE': 'TARGET_P', 'TARGET_P': 40},
            ]
        },
        {
            "WINDOW_SIZES": [10, 30],
            "scenarios": [
                {'name': 'Log_40Points_Win10_30', 'AREA_X_AXIS_MODE': 'Log', 'OPTIMIZATION_MODE': 'TARGET_P', 'TARGET_P': 40},
            ]
        }
    ]

    target_gauge = 'A1X'

    print("--- Loading all data to find the target gauge ---")
    all_jobs = data_loader.load_all_data_from_input_dir()
    
    target_job_template = None
    for job in all_jobs:
        if job['output_filename_base'].strip().startswith(target_gauge):
            target_job_template = job
            break

    if not target_job_template:
        print(f"Error: Could not find data for the target gauge '{target_gauge}'. Exiting.")
        return

    print(f"Successfully found data for '{target_gauge}'. Starting scenarios...")

    # --- Create a dedicated output directory for the QA run ---
    qa_output_dir = os.path.join(config.OUTPUT_DIR, "QA_RUNS")
    if not os.path.exists(qa_output_dir):
        os.makedirs(qa_output_dir)
    
    original_output_dir = config.OUTPUT_DIR
    config.OUTPUT_DIR = qa_output_dir


    # --- Loop through each group of scenarios ---
    for group in scenario_groups:
        
        # --- Set the window size for the current group ---
        current_window_sizes = group["WINDOW_SIZES"]
        config.WINDOW_SIZES = current_window_sizes
        print(f"\n{'#'*80}")
        print(f"--- Processing group with WINDOW_SIZES = {current_window_sizes} ---")
        print(f"{'#'*80}")

        # --- One-time Pre-computation for the current group ---
        print("\n--- Performing pre-computation for candidate points and graph... ---")
        overall_start_time = time.time()
        
        frequencies = target_job_template['frequencies']
        psd_values = target_job_template['psd_values']
        
        candidate_points = psd_utils.create_multi_scale_envelope(
            frequencies, psd_values, config.WINDOW_SIZES
        )

        if len(frequencies) > 0:
            first_point = np.array([[frequencies[0], psd_values[0]]])
            other_points_mask = candidate_points[:, 0] > frequencies[0]
            other_points = candidate_points[other_points_mask]
            candidate_points = np.vstack((first_point, other_points))
        
        # --- Dynamically select the correct problem definition based on the first scenario in the group ---
        # This is a simplification; assumes all scenarios in a group could use the same graph logic.
        # A more robust solution might need to check per scenario.
        temp_mode = group["scenarios"][0]['OPTIMIZATION_MODE']
        problem = problem_definition_points if temp_mode == 'TARGET_P' else problem_definition_area

        valid_jumps_graph = problem.build_valid_jumps_graph(
            candidate_points, frequencies, psd_values
        )

        base_ga_params = {
            'graph': valid_jumps_graph,
            'simplified_points': candidate_points,
            'original_psd_freqs': frequencies,
            'original_psd_values': psd_values,
            'prune_threshold': config.PRUNE_THRESHOLD
        }
        precomputation_end_time = time.time()
        print(f"--- Pre-computation finished in {precomputation_end_time - overall_start_time:.2f} seconds. ---")

        # --- Loop through each scenario in the current group and run the optimization ---
        scenarios = group["scenarios"]
        for i, scenario in enumerate(scenarios):
            print(f"\n{'='*60}")
            print(f"--- Running Scenario {i+1}/{len(scenarios)}: {scenario['name']} ---")
            print(f"{'='*60}")
            scenario_start_time = time.time()

            # --- Dynamically override config settings ---
            config.AREA_X_AXIS_MODE = scenario['AREA_X_AXIS_MODE']
            config.OPTIMIZATION_MODE = scenario['OPTIMIZATION_MODE']
            
            # --- Select the correct problem module for this specific scenario ---
            if config.OPTIMIZATION_MODE == 'TARGET_P':
                problem = problem_definition_points
            else:
                problem = problem_definition_area

            output_filename_base = f"{target_gauge}_{scenario['name']}"

            if config.OPTIMIZATION_MODE == 'TARGET_P':
                config.TARGET_POINTS = scenario['TARGET_POINTS']
                print(f"Mode: TARGET_POINTS, Target: {config.TARGET_POINTS} points")
            elif config.OPTIMIZATION_MODE == 'TARGET_A':
                config.TARGET_AREA_RATIO = scenario['TARGET_AREA_RATIO']
                print(f"Mode: TARGET_AREA_RATIO, Target: {config.TARGET_AREA_RATIO:.4f} ratio")
            
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

            # Critical check: if no solutions could be found, we cannot proceed with this scenario.
            if not population:
                print(f"\nFATAL: Could not generate any valid solutions for scenario '{scenario['name']}' after {max_attempts} attempts.")
                print("Skipping this scenario.")
                continue

            # Warning if the population is smaller than desired
            if len(population) < config.POPULATION_SIZE:
                print(f"Warning: Could only create {len(population)}/{config.POPULATION_SIZE} valid initial solutions.")

            best_solution_so_far, best_cost_so_far = None, float('inf')

            # --- Main Evolution Loop ---
            print("\n--- Starting Evolution ---")
            evolution_start_time = time.time()
            generations_without_improvement = 0
            last_best_cost = float('inf')
            generation = 0

            while generation < config.MAX_GENERATIONS:
                population = [p for p in population if p and len(p) > 1]
                if not population:
                    print("Population became empty. Exiting evolution.")
                    break

                all_metrics = [
                    problem.calculate_metrics(path, **base_ga_params, target_points=config.TARGET_POINTS)
                    for path in population
                ]
                costs = [m[0] for m in all_metrics]
                fitness_scores = [m[1] for m in all_metrics]

                current_min_cost = np.min(costs)
                if current_min_cost < best_cost_so_far:
                    best_cost_so_far = current_min_cost
                    best_solution_so_far = population[np.argmin(costs)]

                current_best_len = len(best_solution_so_far)

                if (generation + 1) % 10 == 0:
                    best_cost_report, best_fitness_report, best_len_report, best_ratio = problem.calculate_metrics(
                        best_solution_so_far, **base_ga_params, target_points=config.TARGET_POINTS
                    )
                    print(
                        f"Gen {generation + 1}/{config.MAX_GENERATIONS} | "
                        f"Area Ratio: {np.sqrt(best_ratio):.4f} | "
                        f"Points: {best_len_report} | "
                        f"Fitness: {best_fitness_report:.4f} | "
                        f"Cost: {best_cost_report:.4f}"
                    )

                if config.USE_CONVERGENCE_TERMINATION:
                    improvement = last_best_cost - best_cost_so_far
                    if improvement < config.CONVERGENCE_THRESHOLD:
                        generations_without_improvement += 1
                    else:
                        generations_without_improvement = 0
                    last_best_cost = best_cost_so_far
                    if generations_without_improvement >= config.CONVERGENCE_PATIENCE:
                        print(f"\n--- Terminating early at generation {generation + 1} due to convergence ---")
                        break
                
                new_population = []
                if config.ELITISM_SIZE > 0:
                    elite_indices = np.argsort(fitness_scores)[-config.ELITISM_SIZE:]
                    elite_solutions = [population[i] for i in elite_indices]
                    new_population.extend(elite_solutions)

                    for elite_sol in elite_solutions:
                        if len(elite_sol) < config.ADAPTIVE_MUTATION_THRESHOLD:
                            pruned_version = operators.mutate_prune_useless_points(elite_sol, **base_ga_params)
                            if pruned_version != elite_sol:
                                new_population.append(pruned_version)

                while len(new_population) < config.POPULATION_SIZE:
                    parent1 = operators.selection(population, fitness_scores)
                    parent2 = operators.selection(population, fitness_scores)
                    child = operators.crossover_multipoint_paths(parent1, parent2)

                    if random.random() < config.MUTATION_RATE:
                        mutated_child = operators.apply_mutations(
                            child, base_ga_params, current_best_len, config.ADAPTIVE_MUTATION_THRESHOLD
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
                    best_solution_so_far, **base_ga_params, target_points=config.TARGET_POINTS
                )
                print("\n--- Optimization Finished ---")
                print(f"Evolution process time: {end_time - evolution_start_time:.2f} seconds")
                print(f"Total scenario time: {end_time - scenario_start_time:.2f} seconds")
                
                print(f"Best solution has {final_len} points.")
                print(f"Final Internal Cost: {final_cost:.4f}")
                print(f"Final Area Ratio: {np.sqrt(final_ratio):.6f}")

                # --- Plot and Save the final solution ---
                final_points_coords = base_ga_params['simplified_points'][best_solution_so_far]
                psd_utils.plot_final_solution(
                    base_ga_params['original_psd_freqs'],
                    base_ga_params['original_psd_values'],
                    final_points_coords,
                    np.sqrt(final_ratio),
                    output_filename_base
                )

            else:
                print("\n--- No valid solution found ---")

    # --- Restore original config ---
    config.OUTPUT_DIR = original_output_dir

    print(f"\n{'='*60}")
    print("QA run complete.")
    print(f"All results have been saved in the '{qa_output_dir}' directory.")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_qa_scenarios()
