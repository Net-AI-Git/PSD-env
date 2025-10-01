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
import matplotlib.pyplot as plt
from . import config


def natural_sort_key(job):
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
        mask = (data[:, 0] >= 5) & (data[:, 0] <= 2000)
        filtered_data = data[mask]

        if filtered_data.shape[0] == 0:
            print(f"Warning: No data within the 5-2000 Hz range in {filepath}. Skipping.")
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
            mask = (combined_data[:, 0] >= 5) & (combined_data[:, 0] <= 2000)
            filtered_data = combined_data[mask]

            if filtered_data.shape[0] == 0:
                print(f"Warning: No data for measurement '{name}' within the 5-2000 Hz range. Skipping.")
                continue

            job = {
                'frequencies': filtered_data[:, 0],
                'psd_values': filtered_data[:, 1],
                'output_filename_base': name
            }
            jobs.append(job)
        return jobs
    except Exception as e:
        print(f"Warning: Could not process MAT file '{filepath}'. Error: {e}")
        return []


def plot_envelope_comparison(original_jobs, envelope_job, channel_name, output_path):
    """
    Creates a log-log plot comparing original PSD data with the envelope.
    
    This function plots all original PSD measurements for a specific channel
    in different colors, and overlays the envelope (maximum values) in red.
    The plot is saved as a PNG file in the specified output path.
    
    Args:
        original_jobs (list): List of job dictionaries containing original PSD data.
        envelope_job (dict): Job dictionary containing the envelope PSD data.
        channel_name (str): Name of the channel being plotted.
        output_path (str): Full path where the plot should be saved.
    """
    plt.figure(figsize=(20, 8))
    
    # Plot original PSD data in different colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(original_jobs)))
    for i, job in enumerate(original_jobs):
        plt.loglog(job['frequencies'], job['psd_values'], 
                  color=colors[i], alpha=0.7, linewidth=1.5,
                  label=f"Original {i+1}")
    
    # Plot envelope in red
    plt.loglog(envelope_job['frequencies'], envelope_job['psd_values'], 
              color='red', linewidth=2.5, label='Envelope (Max)')
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.title(f'Envelope Comparison for Channel: {channel_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved envelope comparison plot: {output_path}")


def load_full_envelope_data(input_dir):
    """
    Loads all files from the input directory and creates envelope data
    by taking the maximum PSD values for each frequency across all files
    with matching channel names.

    This function processes all files in the directory, groups measurements
    by channel name, and creates envelope PSD data by taking the maximum
    value at each frequency across all files for each channel group.

    Args:
        input_dir (str): The directory containing all input files.

    Returns:
        tuple: A tuple containing:
            - list: Envelope job dictionaries
            - dict: Original jobs grouped by channel name
    """
    all_jobs = []
    
    # Load all files from the directory
    for filename in sorted(os.listdir(input_dir)):
        filepath = os.path.join(input_dir, filename)
        if os.path.isfile(filepath):
            jobs_from_file = load_data_from_file(filepath)
            all_jobs.extend(jobs_from_file)
    
    if not all_jobs:
        print("Warning: No valid jobs found in the input directory.")
        return [], {}
    
    # Group jobs by channel name
    channel_groups = {}
    for job in all_jobs:
        channel_name = job['output_filename_base']
        if channel_name not in channel_groups:
            channel_groups[channel_name] = []
        channel_groups[channel_name].append(job)
    
    # Create envelope jobs for each channel group
    envelope_jobs = []
    for channel_name, jobs in channel_groups.items():
        if len(jobs) == 1:
            # Only one file has this channel, use it as is
            envelope_jobs.append(jobs[0])
            continue
        
        print(f"Creating envelope for channel '{channel_name}' from {len(jobs)} files")
        
        # Find common frequency range
        all_frequencies = []
        for job in jobs:
            all_frequencies.extend(job['frequencies'])
        
        # Create a unified frequency grid
        unique_frequencies = np.unique(np.concatenate([job['frequencies'] for job in jobs]))
        unique_frequencies = np.sort(unique_frequencies)
        
        # Interpolate all PSD data to the common frequency grid
        interpolated_psds = []
        for job in jobs:
            # Interpolate to common frequency grid
            interpolated_psd = np.interp(unique_frequencies, job['frequencies'], job['psd_values'])
            interpolated_psds.append(interpolated_psd)
        
        # Take maximum value at each frequency
        envelope_psd = np.max(interpolated_psds, axis=0)
        
        # Create envelope job
        envelope_job = {
            'frequencies': unique_frequencies,
            'psd_values': envelope_psd,
            'output_filename_base': f"{channel_name}_envelope"
        }
        envelope_jobs.append(envelope_job)
    
    print(f"Created {len(envelope_jobs)} envelope jobs from {len(all_jobs)} original jobs")
    return envelope_jobs, channel_groups


def load_data_from_file(filepath):
    """
    Loads all measurement jobs from a single specified file (.txt or .mat).

    This function acts as a dispatcher, determining the correct parsing function
    based on the file extension.

    Args:
        filepath (str): The full path to the input file.

    Returns:
        list: A list of "job" dictionaries found in the file. Returns an
              empty list if the file is unsupported or processing fails.
    """
    filename = os.path.basename(filepath)
    if filename.lower().endswith('.mat'):
        print(f"Reading MAT file: {filename}")
        return _read_mat_file(filepath)
    elif filename.lower().endswith(config.INPUT_FILE_EXTENSION):
        print(f"Reading TXT file: {filename}")
        return _read_txt_file(filepath)
    else:
        print(f"Warning: Skipping unsupported file type: {filename}")
        return []

