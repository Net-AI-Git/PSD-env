import matplotlib.pyplot as plt
from optimizer_core import data_loader
import numpy as np


def filter_local_minimums(frequencies, psd_values):
    """
    Filters out points that are local minimums from a PSD signal.

    A point is defined as a local minimum if its value is strictly lower than
    the values of its immediate neighbors (the point before and the point after).
    The first and last points of the signal are never considered minimums.

    Args:
        frequencies (np.ndarray): The frequency values of the signal.
        psd_values (np.ndarray): The PSD amplitude values of the signal.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the frequencies and
                                       psd_values of the points that are NOT
                                       local minimums.
    """
    if len(frequencies) < 3:
        return frequencies, psd_values

    # Create lists to hold the filtered points
    filtered_freqs = [frequencies[0]]
    filtered_psd = [psd_values[0]]

    # Iterate through the interior points
    for i in range(1, len(frequencies) - 1):
        is_minimum = psd_values[i-1] > psd_values[i] and psd_values[i] < psd_values[i+1]
        # Keep the point only if it is NOT a local minimum
        if not is_minimum:
            filtered_freqs.append(frequencies[i])
            filtered_psd.append(psd_values[i])

    # Add the last point
    filtered_freqs.append(frequencies[-1])
    filtered_psd.append(psd_values[-1])

    return np.array(filtered_freqs), np.array(filtered_psd)


def lift_points_logarithmically(psd_values, lift_factor):
    """
    Increases the PSD values by a given factor on a logarithmic scale.

    This is equivalent to multiplying the values in linear scale, but the
    operation is done in the log domain to match the visual representation
    of the plots.

    Args:
        psd_values (np.ndarray): The PSD amplitude values.
        lift_factor (float): The factor by which to lift the points (e.g., 1.5).

    Returns:
        np.ndarray: A new array with the lifted PSD values.
    """
    epsilon = 1e-12  # To prevent log(0) errors
    log_values = np.log10(psd_values + epsilon)
    log_lift = np.log10(lift_factor)
    lifted_log_values = log_values + log_lift
    return 10 ** lifted_log_values


def generate_custom_candidate_points(frequencies, psd_values, lift_factor):
    """
    Generates a unified pool of candidate points for the GA.

    This function combines multiple point generation strategies into a single,
    sorted, and unique list of candidate points, matching the output format
    required by the genetic algorithm's main workflow.

    Args:
        frequencies (np.ndarray): The original frequency values.
        psd_values (np.ndarray): The original PSD amplitude values.
        lift_factor (float): The factor for the logarithmic lift.

    Returns:
        np.ndarray: A 2D numpy array of [frequency, psd_value] points,
                    sorted by frequency, with all duplicates removed.
    """
    # 1. Get the non-minimum points (our "green" points)
    non_min_freqs, non_min_psd = filter_local_minimums(frequencies, psd_values)
    non_min_points = np.column_stack((non_min_freqs, non_min_psd))

    # 2. Get the logarithmically lifted points (our "red" points)
    lifted_psd = lift_points_logarithmically(psd_values, lift_factor)
    lifted_points = np.column_stack((frequencies, lifted_psd))

    # 3. Combine all point groups into a single array
    combined_points = np.vstack([non_min_points, lifted_points])
    print(f"\n--- Generating Custom Candidate Pool ---")
    print(f"Combined {len(non_min_points)} non-minimum points and {len(lifted_points)} lifted points.")

    # 4. Remove duplicates and sort by frequency
    unique_points = np.unique(combined_points, axis=0)
    sorted_points = unique_points[np.argsort(unique_points[:, 0])]
    
    print(f"Final pool contains {len(sorted_points)} unique candidate points.")
    return sorted_points


def main():
    """
    This script loads a specific PSD job, generates a custom candidate point
    pool, and visualizes the components and the final result.
    """
    print("--- Running Custom Point Generator Visualization ---")

    # 1. Load all data from the input directory using the existing data_loader
    all_jobs = data_loader.load_all_data_from_input_dir()

    if not all_jobs:
        print("No data found in the input directory. Exiting.")
        return

    # 2. Find the first job that corresponds to "A1X" for demonstration
    target_job_name_part = "A1X"
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

    # 4. Filter out the local minimum points to get the points to display
    non_minimum_freqs, non_minimum_psd = filter_local_minimums(frequencies, psd_values)
    print(f"Removed {len(frequencies) - len(non_minimum_freqs)} local minimum points.")

    # 5. Lift all original points by a factor of 1.3 in log scale
    lifted_psd = lift_points_logarithmically(psd_values, 1.3)
    print("Created a new set of points lifted by a factor of 1.3.")

    # 6. Generate the final, unified candidate point pool
    final_candidate_points = generate_custom_candidate_points(frequencies, psd_values, 1.3)
    print("First 5 candidate points:\n", final_candidate_points[:5])

    # 7. Create the visual plot with two subplots (log and linear x-axis)
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    fig.suptitle(f"PSD Signal Analysis for {target_job['output_filename_base']}", fontsize=18)

    for ax, x_scale in zip(axes, ["log", "linear"]):
        # Plot the original PSD signal in blue on the current subplot
        ax.plot(frequencies, psd_values, 'b-', label='Original PSD', alpha=0.6)

        # Plot the non-minimum points in green
        ax.scatter(non_minimum_freqs, non_minimum_psd, c='green', s=15, zorder=5,
                   label=f'Non-Minimum Points ({len(non_minimum_freqs)})')

        # Plot the lifted points in red
        ax.scatter(frequencies, lifted_psd, c='red', s=15, zorder=4,
                   label=f'Lifted Points (x1.3) ({len(frequencies)})')

        # Plot the final candidate pool for verification
        ax.scatter(final_candidate_points[:, 0], final_candidate_points[:, 1],
                   c='yellow', s=5, zorder=6, alpha=0.8,
                   label=f'Final Candidate Pool ({len(final_candidate_points)})')

        # Configure the plot scales and labels
        ax.set_title(f"PSD with {x_scale.capitalize()} X-axis", fontsize=14)
        ax.set_xscale(x_scale)
        ax.set_yscale('log')
        ax.set_xlabel('Frequency [Hz]', fontsize=12)
        ax.set_ylabel('PSD [g²/Hz]', fontsize=12)
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend()

    # 8. Show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()
