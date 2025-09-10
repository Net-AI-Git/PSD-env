import os
from optimizer_core import config
from run_code import run_optimization_process


def run_qa_scenarios():
    """
    Runs a predefined set of QA scenarios by calling the main optimization function
    with different configurations.

    This script is designed with templates to make it easy to add or modify
    bulk scenarios for testing different parameter combinations.
    """
    # --- Template 1: Varying Target Points with Log X-Axis ---
    # Keeps area ratio constant and tests different point counts.
    scenarios_vary_points_log = [
        {
            'name': f'Log_P{p}_A1_25',
            'area_x_axis_mode': 'Log',
            'target_points': p,
            'target_area_ratio': 1.25
        }
        for p in [20, 40, 60, 80]
    ]

    # --- Template 2: Varying Target Area Ratio with Log X-Axis ---
    # Keeps point count constant and tests different area ratios.
    scenarios_vary_area_log = [
        {
            'name': f'Log_P45_A{a:.2f}'.replace('.', '_'),
            'area_x_axis_mode': 'Log',
            'target_points': 45,
            'target_area_ratio': a
        }
        for a in [1.1, 1.2, 1.3, 1.4, 1.5]
    ]

    # --- Template 3: Varying Target Points with Linear X-Axis ---
    scenarios_vary_points_linear = [
        {
            'name': f'Linear_P{p}_A1_25',
            'area_x_axis_mode': 'Linear',
            'target_points': p,
            'target_area_ratio': 1.25
        }
        for p in [20, 40, 60, 80]
    ]

    # --- Template 4: Varying Target Area Ratio with Linear X-Axis ---
    scenarios_vary_area_linear = [
        {
            'name': f'Linear_P45_A{a:.2f}'.replace('.', '_'),
            'area_x_axis_mode': 'Linear',
            'target_points': 45,
            'target_area_ratio': a
        }
        for a in [1.1, 1.2, 1.3, 1.4, 1.5]
    ]

    # --- Template 5: Custom Combined Scenarios ---
    # Use this section for specific, one-off test cases.
    scenarios_custom = [
        {'name': 'Log_P30_A1_40', 'area_x_axis_mode': 'Log', 'target_points': 30, 'target_area_ratio': 1.4},
        {'name': 'Linear_P50_A1_15', 'area_x_axis_mode': 'Linear', 'target_points': 50, 'target_area_ratio': 1.15},
    ]

    # --- Master list of all scenarios to run ---
    # Add or remove scenario groups from this list to control the QA run.
    all_scenarios = (
        scenarios_vary_points_log +
        scenarios_vary_area_log +
        scenarios_vary_points_linear +
        scenarios_vary_area_linear +
        scenarios_custom
    )

    # --- Execution Loop ---
    original_output_dir = config.OUTPUT_DIR
    qa_base_dir = os.path.join(original_output_dir, "QA_RUNS")

    print(f"{'#'*80}")
    print(f"--- Starting QA Run: {len(all_scenarios)} scenarios total ---")
    print(f"Results will be saved in subdirectories under: {qa_base_dir}")
    print(f"{'#'*80}")

    for i, scenario in enumerate(all_scenarios):
        print(f"\n{'='*80}")
        print(f"--- Running Scenario {i+1}/{len(all_scenarios)}: {scenario['name']} ---")
        print(f"Parameters: {scenario}")
        print(f"{'='*80}\n")

        # Create a dedicated output directory for this scenario to prevent overwriting results
        scenario_output_dir = os.path.join(qa_base_dir, scenario['name'])
        if not os.path.exists(scenario_output_dir):
            os.makedirs(scenario_output_dir)
        
        config.OUTPUT_DIR = scenario_output_dir

        # Call the main optimization process from run_code.py
        run_optimization_process(
            min_frequency_hz=5,
            max_frequency_hz=2000,
            target_points=scenario['target_points'],
            target_area_ratio=scenario['target_area_ratio'],
            stab_wide="narrow",  # Using "narrow" as a default for QA
            area_x_axis_mode=scenario['area_x_axis_mode']
        )

    # Restore the original output directory configuration
    config.OUTPUT_DIR = original_output_dir

    print(f"\n{'#'*80}")
    print("--- QA run complete. ---")
    print(f"All results saved in '{qa_base_dir}'")
    print(f"{'#'*80}")


if __name__ == "__main__":
    run_qa_scenarios()
