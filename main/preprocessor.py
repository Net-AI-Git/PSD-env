import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Import modules from the optimizer_core package
from optimizer_core import psd_utils as utils
from optimizer_core import config


def main():
    """
    This script is for visualization and debugging purposes only.
    It loads a single PSD data file from the input directory, generates the
    candidate points pool using the exact same method as the main algorithm,
    and displays them visually on a plot.
    """
    # --- Step 1: Find a file to process ---
    if not os.path.exists(config.INPUT_DIR):
        print(f"Error: Input directory '{config.INPUT_DIR}' not found.")
        print("Please create it and add your PSD files.")
        return

    files_to_process = [f for f in os.listdir(config.INPUT_DIR) if f.endswith(config.INPUT_FILE_EXTENSION)]
    if not files_to_process:
        print(f"Error: No '{config.INPUT_FILE_EXTENSION}' files found in the '{config.INPUT_DIR}' directory.")
        return

    # Process the first file found for visualization
    filename_to_process = files_to_process[0]
    filepath = os.path.join(config.INPUT_DIR, filename_to_process)
    print(f"--- Visualizing candidate points for: {filename_to_process} ---")


    # --- Step 2: Load data and create candidate pool ---
    frequencies, psd_values = utils.read_psd_data(filepath)
    if frequencies is None:
        return

    # This is the same function the main algorithm uses.
    # It will now correctly use the settings from config.py (like LIFT_FACTOR).
    candidate_points = utils.create_multi_scale_envelope(
        frequencies, psd_values, config.WINDOW_SIZES
    )

    print(f"\nFound a total of {len(candidate_points)} unique candidate points to visualize.")

    # --- Step 3: Create the visual plot ---
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    fig.suptitle("Visualization of All Candidate Points", fontsize=18)

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
