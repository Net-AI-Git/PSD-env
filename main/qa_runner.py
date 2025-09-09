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
from run_code import run_optimization_process


def run_qa_scenarios():
    """
    Runs a predefined set of QA scenarios by calling the main optimization function
    with different configurations.
    """
    scenarios = [
        # 1. Linear axis, target 45 points
        {'area_x_axis_mode': 'Linear', 'target_points': 45},
        # 2. Log axis, target 20 points
        {'area_x_axis_mode': 'Log', 'target_points': 20},
        # 3. Log axis, target 60 points
        {'area_x_axis_mode': 'Log', 'target_points': 60},
    ]

    print(f"{'#'*80}")
    print("--- Starting QA Scenarios ---")
    print(f"{'#'*80}")

    for i, scenario in enumerate(scenarios):
        print(f"\n{'='*60}")
        print(f"--- Running Scenario {i+1}/{len(scenarios)} ---")
        print(f"Parameters: {scenario}")
        print(f"{'='*60}\n")

        # Call the main optimization process from run_code.py
        run_optimization_process(
            min_frequency_hz=5,
            max_frequency_hz=2000,
            target_points=scenario['target_points'],
            stab_wide="narrow",  # Using "narrow" as a default for QA
            area_x_axis_mode=scenario['area_x_axis_mode']
        )

    print(f"\n{'='*60}")
    print("--- QA run complete. ---")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_qa_scenarios()
