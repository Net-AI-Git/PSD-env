# -*- coding: utf-8 -*-
"""
Module responsible for loading all PSD data from the input directory.

It detects file types (.txt, .mat), processes them into a unified in-memory
list of "jobs", and sorts them in a natural, numerical order. Each job is a
dictionary containing all necessary data for a single optimization run.
"""

import os
import numpy as np
import scipy.io
import re  # <-- IMPORT THE REGULAR EXPRESSION MODULE
from . import config


def _natural_sort_key(job):
    """
    Creates a key for natural sorting of measurement names like 'A1X', 'A10Z'.
    It sorts by the number first, then by the axis character (X, Y, Z).
    """
    name = job['output_filename_base']
    # Use the part of the name before the source file for sorting
    measurement_part = name.split(' - ')[0]

    # This regex is designed to capture names like A1X, A10Z, etc.
    match = re.match(r'A(\d+)([XYZ])$', measurement_part, re.IGNORECASE)
    if match:
        number = int(match.group(1))
        axis = match.group(2).upper()
        # Create a sort order for axes: X=0, Y=1, Z=2
        axis_order = {'X': 0, 'Y': 1, 'Z': 2}.get(axis, 3)
        return (number, axis_order)
    else:
        # For any other names (like simple TXT files), sort them alphabetically at the end
        return (float('inf'), name)


def _read_txt_file(filepath):
    """
    Loads data from a simple two-column .txt file.

    Args:
        filepath (str): The full path to the .txt file.

    Returns:
        list: A list containing a single job dictionary for the measurement.
    """
    try:
        data = np.loadtxt(filepath)
        if data.ndim != 2 or data.shape[1] != 2:
            print(f"Warning: Skipping malformed TXT file (not 2 columns): {filepath}")
            return []

        base_filename = os.path.basename(filepath)
        filename_no_ext = os.path.splitext(base_filename)[0]

        # Filter data to the required frequency range
        mask = (data[:, 0] >= config.MIN_FREQUENCY_HZ) & (data[:, 0] <= config.MAX_FREQUENCY_HZ)
        filtered_data = data[mask]

        if filtered_data.shape[0] == 0:
            print(f"Warning: No data within the {config.MIN_FREQUENCY_HZ}-{config.MAX_FREQUENCY_HZ} Hz range in {filepath}. Skipping.")
            return []

        job = {
            'frequencies': filtered_data[:, 0],
            'psd_values': filtered_data[:, 1],
            'output_filename_base': filename_no_ext
        }
        return [job]  # Return as a list for consistency
    except Exception as e:
        print(f"Warning: Could not process TXT file '{filepath}'. Error: {e}")
        return []


def _read_mat_file(filepath):
    """
    Loads data from a complex .mat file containing multiple measurements.

    Args:
        filepath (str): The full path to the .mat file.

    Returns:
        list: A list of job dictionaries, one for each measurement found.
    """
    jobs = []
    try:
        mat_data = scipy.io.loadmat(filepath)
        if 'fvec' not in mat_data or 'FFTpsd' not in mat_data:
            print(f"Warning: Skipping MAT file with missing 'fvec' or 'FFTpsd': {filepath}")
            return []

        freq_vector = mat_data['fvec']
        measurements = mat_data['FFTpsd'][0]
        source_filename = os.path.splitext(os.path.basename(filepath))[0]

        for measurement in measurements:
            name = measurement['name'][0]
            psd_values = measurement['psd']

            combined_data = np.hstack((freq_vector.flatten()[:, np.newaxis], psd_values.flatten()[:, np.newaxis]))

            # Filter data to the required frequency range
            mask = (combined_data[:, 0] >= config.MIN_FREQUENCY_HZ) & (combined_data[:, 0] <= config.MAX_FREQUENCY_HZ)
            filtered_data = combined_data[mask]

            if filtered_data.shape[0] == 0:
                print(f"Warning: No data for measurement '{name}' within the {config.MIN_FREQUENCY_HZ}-{config.MAX_FREQUENCY_HZ} Hz range. Skipping.")
                continue

            job = {
                'frequencies': filtered_data[:, 0],
                'psd_values': filtered_data[:, 1],
                'output_filename_base': f"{name} - {source_filename}"
            }
            jobs.append(job)
        return jobs
    except Exception as e:
        print(f"Warning: Could not process MAT file '{filepath}'. Error: {e}")
        return []


def load_all_data_from_input_dir():
    """
    Scans the input directory, loads all supported files (.txt, .mat),
    and returns a single list of all measurements, sorted naturally,
    ready for processing.

    Returns:
        list: A sorted list of "job" dictionaries. Returns an empty list if no
              valid files or measurements are found.
    """
    all_jobs = []
    input_dir = config.INPUT_DIR

    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        return []

    print("--- Starting Data Loading Phase ---")
    for filename in sorted(os.listdir(input_dir)):  # Sorted for consistent order
        filepath = os.path.join(input_dir, filename)

        if filename.lower().endswith('.mat'):
            print(f"Reading MAT file: {filename}")
            all_jobs.extend(_read_mat_file(filepath))
        elif filename.lower().endswith(config.INPUT_FILE_EXTENSION):
            print(f"Reading TXT file: {filename}")
            all_jobs.extend(_read_txt_file(filepath))

    # --- Sort the jobs using the natural sort key ---
    all_jobs.sort(key=_natural_sort_key)

    print(f"--- Data Loading Complete. Found and sorted {len(all_jobs)} total measurements. ---")
    return all_jobs

