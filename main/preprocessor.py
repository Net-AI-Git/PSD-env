import matplotlib.pyplot as plt
import numpy as np
import os
from optimizer_core import data_loader
from optimizer_core import psd_utils
from optimizer_core import config

def main():
    """
    This script loads a specific job (A1X), generates its candidate 
    points pool, and displays them visually on a plot for debugging.
    """
    print("--- Running Preprocessor for A1X Visualization ---")

    # 1. Load all data from the input directory
    all_jobs = data_loader.load_all_data_from_input_dir()

    if not all_jobs:
        print("No data found in the input directory. Exiting.")
        return

    # 2. Find the first job corresponding to "A1X"
    target_job_name_part = "A1Z"
    target_job = None
    for job in all_jobs:
        if target_job_name_part in job['output_filename_base']:
            target_job = job
            break

    if not target_job:
        print(f"Could not find a job containing '{target_job_name_part}'.")
        return

    print(f"\nFound target job to process: {target_job['output_filename_base']}")

    # 3. Extract the frequency and PSD value data from the selected job
    frequencies = target_job['frequencies']
    psd_values = target_job['psd_values']

    # 4. Generate the full candidate points pool using the same logic as the main script
    print("\n--- Generating Candidate Points ---")
    candidate_points = psd_utils.create_multi_scale_envelope(
        frequencies,
        psd_values,
        config.WINDOW_SIZES
    )
    print(f"\nFound a total of {len(candidate_points)} unique candidate points to visualize.")

    # 5. Create the visual plot
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    fig.suptitle(f"Visualization of All Candidate Points for {target_job['output_filename_base']}", fontsize=18)

    for ax, x_scale in zip(axes, ["log", "linear"]):
        ax.set_title(f"Candidate Points Pool ({x_scale.capitalize()} X-axis)", fontsize=14)
        ax.set_xscale(x_scale)
        ax.set_yscale('log')

        # Plot the original PSD signal in blue
        ax.plot(frequencies, psd_values, 'b-', label='Original PSD', alpha=0.7)

        # Plot all candidate points from the pool as red dots
        ax.plot(candidate_points[:, 0], candidate_points[:, 1], 'ro',
                label=f'All Candidate Points ({len(candidate_points)})', markersize=4)

        ax.set_xlabel('Frequency [Hz]', fontsize=12)
        ax.set_ylabel('PSD [g²/Hz]', fontsize=12)
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()
