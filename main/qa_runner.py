import os
import time
import numpy as np
import random

from optimizer_core import config
from optimizer_core import data_loader
from optimizer_core import psd_utils
from optimizer_core import problem_definition as problem
from optimizer_core import ga_operators as operators

def run_qa_scenarios():
    """
    Runs a predefined set of QA scenarios to test the PSD optimization
    with different configurations.
    """
    # --- Define all test scenarios ---
    scenarios = [
        # מצב 1
        {'name': 'Linear_40Points', 'AREA_X_AXIS_MODE': 'Linear', 'OPTIMIZATION_MODE': 'TARGET_POINTS', 'TARGET_POINTS': 40},
        # מצב 2
        {'name': 'Linear_1.44Ratio', 'AREA_X_AXIS_MODE': 'Linear', 'OPTIMIZATION_MODE': 'TARGET_AREA_RATIO', 'TARGET_AREA_RATIO': 1.2**2},
        # מצב 3
        {'name': 'Log_40Points', 'AREA_X_AXIS_MODE': 'Log', 'OPTIMIZATION_MODE': 'TARGET_POINTS', 'TARGET_POINTS': 40},
        # מצב 4
        {'name': 'Log_1.44Ratio', 'AREA_X_AXIS_MODE': 'Log', 'OPTIMIZATION_MODE': 'TARGET_AREA_RATIO', 'TARGET_AREA_RATIO': 1.2**2},
        # מצב 5
        {'name': 'Log_1.96Ratio', 'AREA_X_AXIS_MODE': 'Log', 'OPTIMIZATION_MODE': 'TARGET_AREA_RATIO', 'TARGET_AREA_RATIO': 1.4**2},
        # מצב 6
        {'name': 'Log_1.21Ratio', 'AREA_X_AXIS_MODE': 'Log', 'OPTIMIZATION_MODE': 'TARGET_AREA_RATIO', 'TARGET_AREA_RATIO': 1.1**2},
        # מצב 7
        {'name': 'Log_6Points', 'AREA_X_AXIS_MODE': 'Log', 'OPTIMIZATION_MODE': 'TARGET_POINTS', 'TARGET_POINTS': 6},
        # מצב 8
        {'name': 'Log_20Points', 'AREA_X_AXIS_MODE': 'Log', 'OPTIMIZATION_MODE': 'TARGET_POINTS', 'TARGET_POINTS': 20},
        # מצב 9
        {'name': 'Log_60Points', 'AREA_X_AXIS_MODE': 'Log', 'OPTIMIZATION_MODE': 'TARGET_POINTS', 'TARGET_POINTS': 60},
        # מצב 10
        {'name': 'Log_90Points', 'AREA_X_AXIS_MODE': 'Log', 'OPTIMIZATION_MODE': 'TARGET_POINTS', 'TARGET_POINTS': 90},
    ]

    target_gauge = 'A1X'

    print("--- Loading all data to find the target gauge ---")
    all_jobs = data_loader.load_all_data_from_input_dir()
    
    target_job_template = None
    for job in all_jobs:
        # Check if the filename starts with the target gauge name,
        # because the data loader adds the source filename to it.
        if job['output_filename_base'].strip().startswith(target_gauge):
            target_job_template = job
            break

    if not target_job_template:
        print(f"Error: Could not find data for the target gauge '{target_gauge}'. Exiting.")
        return

    print(f"Successfully found data for '{target_gauge}'. Starting scenarios...")

    # --- One-time Pre-computation ---
    # To save time, we generate the candidate points and the graph only once.
    print("\n--- Performing one-time pre-computation for candidate points and graph... ---")
    overall_start_time = time.time()
    
    frequencies = target_job_template['frequencies']
    psd_values = target_job_template['psd_values']
    
    original_area = np.trapezoid(psd_values, x=frequencies)

    candidate_points = psd_utils.create_multi_scale_envelope(
        frequencies, psd_values, config.WINDOW_SIZES
    )

    if len(frequencies) > 0:
        first_point = np.array([[frequencies[0], psd_values[0]]])
        other_points_mask = candidate_points[:, 0] > frequencies[0]
        other_points = candidate_points[other_points_mask]
        candidate_points = np.vstack((first_point, other_points))
    
    valid_jumps_graph = problem.build_valid_jumps_graph(
        candidate_points, frequencies, psd_values
    )

    base_ga_params = {
        'graph': valid_jumps_graph,
        'simplified_points': candidate_points,
        'original_psd_freqs': frequencies,
        'original_psd_values': psd_values,
        'prune_threshold': config.PRUNE_THRESHOLD,
        'original_area': original_area
    }
    precomputation_end_time = time.time()
    print(f"--- Pre-computation finished in {precomputation_end_time - overall_start_time:.2f} seconds. ---")


    # --- Create a dedicated output directory for the QA run ---
    qa_output_dir = os.path.join(config.OUTPUT_DIR, "QA_RUNS")
    if not os.path.exists(qa_output_dir):
        os.makedirs(qa_output_dir)
    
    # Store the original output dir and change it for the QA run
    original_output_dir = config.OUTPUT_DIR
    config.OUTPUT_DIR = qa_output_dir


    # --- Loop through each scenario and run the optimization ---
    for i, scenario in enumerate(scenarios):
        print(f"\n{'='*60}")
        print(f"--- Running Scenario {i+1}/{len(scenarios)}: {scenario['name']} ---")
        print(f"{'='*60}")
        scenario_start_time = time.time()

        # --- Dynamically override config settings ---
        config.AREA_X_AXIS_MODE = scenario['AREA_X_AXIS_MODE']
        config.OPTIMIZATION_MODE = scenario['OPTIMIZATION_MODE']
        
        output_filename_base = f"{target_gauge}_{scenario['name']}"

        if config.OPTIMIZATION_MODE == 'TARGET_POINTS':
            config.TARGET_POINTS = scenario['TARGET_POINTS']
            print(f"Mode: TARGET_POINTS, Target: {config.TARGET_POINTS} points")
        elif config.OPTIMIZATION_MODE == 'TARGET_AREA_RATIO':
            config.TARGET_AREA_RATIO = scenario['TARGET_AREA_RATIO']
            print(f"Mode: TARGET_AREA_RATIO, Target: {config.TARGET_AREA_RATIO:.4f} ratio")
        
        # --- Run the Genetic Algorithm for the current scenario ---
        ga_params = base_ga_params.copy()
        
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
            print(f"Total scenario time: {end_time - scenario_start_time:.2f} seconds")

            if config.OPTIMIZATION_MODE == 'TARGET_AREA_RATIO':
                print(f"Target Area Ratio: {config.TARGET_AREA_RATIO:.4f}")
                print(f"Achieved Area Ratio: {final_ratio:.6f}")
                print(f"Final Point Count: {final_len}")
            else:
                print(f"Best solution has {final_len} points.")
                print(f"Final Area Ratio: {final_ratio:.6f}")
            
            print(f"Final Internal Cost: {final_cost:.4f}")
            
            final_points_coords = ga_params['simplified_points'][best_solution_so_far]
            
            psd_utils.plot_final_solution(
                frequencies,
                psd_values,
                final_points_coords,
                final_ratio,
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
