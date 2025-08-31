# -*- coding: utf-8 -*-
"""
Standalone script to load a complex MATLAB (.mat) file containing multiple
PSD measurements, process them, and save them as individual .txt files.

This script performs the following actions:
1.  Loads a specific .mat file using scipy.
2.  Parses the MATLAB struct to extract a common frequency vector and a list
    of individual PSD measurements.
3.  Iterates through each measurement, extracting its name and PSD values.
4.  Filters the data to keep only the frequency range between 5 and 2000 Hz.
5.  Saves each filtered measurement into a separate .txt file, named after
    the measurement's identifier (e.g., 'A10X.txt').
6.  Plots the first processed measurement for visual verification.
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os

def process_and_split_mat_file(filepath, output_dir='converted_txt_files'):
    """
    Loads a .mat file with a specific structure, splits it into individual
    PSD measurements, filters them, and saves them as text files.

    Args:
        filepath (str): The path to the .mat file.
        output_dir (str): The name of the directory to save the output .txt files.
    """
    try:
        mat_data = scipy.io.loadmat(filepath)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # Verify that the expected variables exist in the .mat file
    if 'fvec' not in mat_data or 'FFTpsd' not in mat_data:
        print("Error: The .mat file is missing the required 'fvec' or 'FFTpsd' variables.")
        return

    # Extract the common frequency vector and the array of measurement structs
    # The [0] indexing is used to access the actual array within the loaded structure
    freq_vector = mat_data['fvec']
    measurements = mat_data['FFTpsd'][0]

    # Create the output directory if it doesn't already exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: '{output_dir}'")

    print(f"\nFound {len(measurements)} measurements to process...")

    first_measurement_data = None
    first_measurement_name = ""

    # Iterate over each measurement struct in the array
    for i, measurement in enumerate(measurements):
        try:
            # Extract the name and PSD values. These are often nested in arrays.
            name = measurement['name'][0]
            psd_values = measurement['psd']

            # Combine the common frequency vector with the specific PSD values
            # np.hstack requires 1D arrays, so we flatten them if necessary.
            combined_data = np.hstack((freq_vector.flatten()[:, np.newaxis], psd_values.flatten()[:, np.newaxis]))

            # Create a boolean mask to filter for frequencies between 5 and 2000 Hz
            frequency_filter_mask = (combined_data[:, 0] >= 5) & (combined_data[:, 0] <= 2000)
            filtered_data = combined_data[frequency_filter_mask]

            # Define the output filename and path
            output_filename = f"{name}.txt"
            output_filepath = os.path.join(output_dir, output_filename)

            # Save the filtered data to a tab-delimited text file
            np.savetxt(output_filepath, filtered_data, fmt='%.8g', delimiter='\t')
            print(f"  - Successfully processed and saved '{output_filename}'")

            # Keep the first measurement's data for plotting after the loop
            if i == 0:
                first_measurement_data = filtered_data
                first_measurement_name = name

        except Exception as e:
            # Handle potential errors if a single measurement is malformed
            print(f"  - Could not process measurement #{i}. Error: {e}")

    # After processing all files, plot the first one for verification
    if first_measurement_data is not None:
        print("\nPlotting the first processed measurement for verification...")
        plot_single_psd(first_measurement_data, first_measurement_name)


def plot_single_psd(data, name):
    """
    Generates and displays a log-log plot for a single PSD measurement.

    Args:
        data (np.ndarray): A 2D numpy array with frequency and PSD columns.
        name (str): The name of the measurement for the plot title.
    """
    if data.shape[1] != 2:
        print("Error plotting: Data must have two columns.")
        return

    frequencies = data[:, 0]
    psd_values = data[:, 1]

    plt.figure(figsize=(12, 7))
    plt.plot(frequencies, psd_values, label='PSD Data')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"PSD for Measurement: {name} (Filtered 5-2000 Hz)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [g²/Hz]")
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Define the path to the .mat file that needs to be processed.
    # Make sure this file is in the same directory as the script.
    file_to_process = 'Qmax Env w 1_2 UF.res.mat'
    process_and_split_mat_file(file_to_process)
