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
from enum import Enum
from . import config
from . import file_saver


class FileType(Enum):
    """Enumeration for different file types supported by the data loader."""
    TESTLAB = "testlab"  # TestLab file - _read_testlab_file
    MATLAB = "matlab"    # Matlab file - _read_mat_file
    TXT = "txt"          # TXT file - _read_txt_file


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
            'output_filename_base': filename_no_ext,
            'source_filename': filename_no_ext
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
                'output_filename_base': name,
                'source_filename': source_filename
            }
            jobs.append(job)
        return jobs
    except Exception as e:
        print(f"Warning: Could not process MAT file '{filepath}'. Error: {e}")
        return []


def _create_full_envelope_data(combined_data, min_freq, max_freq):
    """
    Creates full envelope data by adding interpolated boundary points if needed.
    
    This function isolates the full envelope functionality that was previously
    embedded within _read_testlab_file. It handles boundary interpolation
    to ensure complete frequency coverage.
    
    Args:
        combined_data (np.ndarray): Combined frequency and PSD data array.
        min_freq (float): Minimum frequency threshold.
        max_freq (float): Maximum frequency threshold.
        
    Returns:
        np.ndarray: Processed data with interpolated boundary points if needed.
    """
    # First filter to the target range to see what we actually have
    mask = (combined_data[:, 0] >= min_freq) & (combined_data[:, 0] <= max_freq)
    filtered_data = combined_data[mask]
    
    # Check if we need to add interpolated points at the boundaries
    needs_min_interpolation = filtered_data[0, 0] > min_freq
    needs_max_interpolation = filtered_data[-1, 0] < max_freq
    
    # Add interpolated points at boundaries if needed
    if needs_min_interpolation or needs_max_interpolation:
        # Create interpolation function
        from scipy.interpolate import interp1d
        
        # Use linear interpolation for PSD data
        interp_func = interp1d(combined_data[:, 0], combined_data[:, 1], 
                            kind='linear', bounds_error=False, fill_value='extrapolate')
        
        # Add boundary points if needed
        boundary_points = []
        
        if needs_min_interpolation:
            min_psd_value = interp_func(min_freq)
            boundary_points.append([min_freq, min_psd_value])
        
        if needs_max_interpolation:
            max_psd_value = interp_func(max_freq)
            boundary_points.append([max_freq, max_psd_value])
        
        # Add boundary points to the data
        if boundary_points:
            boundary_array = np.array(boundary_points)
            combined_data = np.vstack([boundary_array, combined_data])
            # Sort by frequency to maintain order
            combined_data = combined_data[combined_data[:, 0].argsort()]
    
    # Filter data to the required frequency range
    mask = (combined_data[:, 0] >= min_freq) & (combined_data[:, 0] <= max_freq)
    filtered_data = combined_data[mask]
    
    return filtered_data


def _read_testlab_file(filepath):
    """
    Loads data from the new PSD format MATLAB file (PSD_A01X, PSD_A01Y, PSD_A01Z).
    
    This function reads MATLAB files containing PSD data in the new format where
    each measurement is stored as a structured array with x_values, y_values, and
    function_record fields.
    
    Args:
        filepath (str): The full path to the .mat file.
        
    Returns:
        list: A list of job dictionaries, one for each PSD measurement found.
    """
    jobs = []
    source_filename = os.path.splitext(os.path.basename(filepath))[0]
    try:
        print(f"Reading TestLab MATLAB file: {filepath}")
        mat_data = scipy.io.loadmat(filepath)
        
        # Find all PSD variables (PSD_A01X, PSD_A01Y, PSD_A01Z, etc.)
        psd_variables = [key for key in mat_data.keys() if key.startswith('PSD_')]
        
        if not psd_variables:
            print(f"Warning: No PSD variables found in {filepath}")
            return []
        
        print(f"Found PSD variables: {psd_variables}")
        
        for psd_var in psd_variables:
            try:
                print(f"Processing {psd_var}...")
                
                # Extract PSD data structure
                psd_data = mat_data[psd_var][0, 0]
                
                # Extract x_values (frequency information)
                x_vals = psd_data['x_values'][0, 0]
                freq_start = x_vals[0][0][0]      # Starting frequency (0 Hz)
                freq_increment = x_vals[1][0][0]  # Frequency step (3.49895031 Hz)
                num_points = x_vals[2][0][0]      # Number of points (1430)
                
                # Generate frequency array
                frequencies = np.arange(num_points) * freq_increment + freq_start
                
                # Extract y_values (PSD data)
                y_vals = psd_data['y_values'][0, 0]
                psd_values = y_vals[0][0].flatten()
                
                # Take real part if complex
                if np.iscomplexobj(psd_values):
                    psd_values = np.real(psd_values)
                
                # Ensure we have the right number of points
                if len(psd_values) != num_points:
                    print(f"Warning: Mismatch in number of points for {psd_var}. Expected {num_points}, got {len(psd_values)}")
                    min_len = min(len(frequencies), len(psd_values))
                    frequencies = frequencies[:min_len]
                    psd_values = psd_values[:min_len]
                
                # Combine frequency and PSD data
                combined_data = np.column_stack((frequencies, psd_values))
                
                # Get frequency range from config
                min_freq = getattr(config, 'MIN_FREQUENCY_HZ', None) or 5
                max_freq = getattr(config, 'MAX_FREQUENCY_HZ', None) or 2000
                
                # Use the isolated full envelope functionality
                filtered_data = _create_full_envelope_data(combined_data, min_freq, max_freq)
                
                if filtered_data.shape[0] == 0:
                    print(f"Warning: No data for {psd_var} within the {min_freq}-{max_freq} Hz range. Skipping.")
                    continue
                
                # Create job dictionary
                job = {
                    'frequencies': filtered_data[:, 0],
                    'psd_values': filtered_data[:, 1],
                    'output_filename_base': psd_var,
                    'source_filename': source_filename
                }
                jobs.append(job)
                
                print(f"Successfully processed {psd_var}: {len(filtered_data)} points in range {min_freq}-{max_freq} Hz")
                
            except Exception as e:
                print(f"Warning: Could not process {psd_var} from {filepath}. Error: {e}")
                continue
        
        print(f"Successfully loaded {len(jobs)} PSD measurements from {filepath}")
        return jobs
        
    except Exception as e:
        print(f"Warning: Could not process TestLab MAT file '{filepath}'. Error: {e}")
        return []


def plot_envelope_comparison(original_jobs, envelope_job, channel_name, output_path):
    """
    Creates a log-log plot comparing original PSD data with the envelope.
    
    This function plots all original PSD measurements for a specific channel
    in different colors, and overlays the envelope (maximum values) in red.
    The plot includes RMS value of the envelope and proper labels with source filenames.
    
    Args:
        original_jobs (list): List of job dictionaries containing original PSD data.
        envelope_job (dict): Job dictionary containing the envelope PSD data.
        channel_name (str): Name of the channel being plotted.
        output_path (str): Full path where the plot should be saved.
    """
    plt.figure(figsize=(20, 8))
    
    # Calculate RMS of the envelope
    envelope_rms = np.sqrt(np.mean(envelope_job['psd_values']**2))
    
    # Plot original PSD data in different colors with proper labels
    colors = plt.cm.tab10(np.linspace(0, 1, len(original_jobs)))
    for i, job in enumerate(original_jobs):
        source_filename = job.get('source_filename', 'Unknown')
        label = f"{source_filename} , {job['output_filename_base']}"
        plt.loglog(job['frequencies'], job['psd_values'], 
                  color=colors[i], alpha=0.7, linewidth=1.5,
                  label=label)
    
    # Plot envelope in red
    plt.loglog(envelope_job['frequencies'], envelope_job['psd_values'], 
              color='red', linewidth=2.5, label=f'Envelope (Max) - RMS: {envelope_rms:.2e}')
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.title(f'Envelope Comparison for Channel: {channel_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close('all')  # Close all figures to prevent tkinter warnings
    print(f"Saved envelope comparison plot: {output_path}")
    
    # Save envelope data as text file using existing function
    text_output_path = output_path.replace('.png', '.txt')
    envelope_points = np.column_stack((envelope_job['frequencies'], envelope_job['psd_values']))
    file_saver.save_results_to_text_file(text_output_path, envelope_points)


def load_full_envelope_data(input_dir, file_type=None):
    """
    Loads all files from the input directory and creates envelope data
    by taking the maximum PSD values for each frequency across all files
    with matching channel names.

    This function processes all files in the directory, groups measurements
    by channel name, and creates envelope PSD data by taking the maximum
    value at each frequency across all files for each channel group.

    Args:
        input_dir (str): The directory containing all input files.
        file_type (FileType, optional): The type of file to process. If None,
                                       will attempt to determine from extension.

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
            jobs_from_file = load_data_from_file(filepath, file_type)
            all_jobs.extend(jobs_from_file)
    
    if not all_jobs:
        print("Warning: No valid jobs found in the input directory.")
        return [], {}
    
    # Group jobs by full channel name (A01X, A01Y, A01Z are separate groups)
    channel_groups = {}
    for job in all_jobs:
        channel_name = job['output_filename_base']  # This is the full channel name (A01X, A01Y, A01Z)
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
        
        # Create envelope job with source filename from first job
        envelope_job = {
            'frequencies': unique_frequencies,
            'psd_values': envelope_psd,
            'output_filename_base': f"{channel_name}_envelope",
            'source_filename': jobs[0].get('source_filename', 'envelope')
        }
        envelope_jobs.append(envelope_job)
    
    print(f"Created {len(envelope_jobs)} envelope jobs from {len(all_jobs)} original jobs")
    return envelope_jobs, channel_groups


def load_data_from_file(filepath, file_type=None):
    """
    Loads all measurement jobs from a single specified file (.txt, .mat, or .res.mat).

    This function acts as a dispatcher, determining the correct parsing function
    based on the file type parameter or file extension as fallback.

    Args:
        filepath (str): The full path to the input file.
        file_type (FileType, optional): The type of file to process. If None,
                                       will attempt to determine from extension.

    Returns:
        list: A list of "job" dictionaries found in the file. Returns an
              empty list if the file is unsupported or processing fails.
    """
    filename = os.path.basename(filepath)
    
    # If file_type is provided, use it directly
    if file_type is not None:
        if file_type == FileType.TESTLAB:
            print(f"Reading TestLab file: {filename}")
            return _read_testlab_file(filepath)
        elif file_type == FileType.MATLAB:
            print(f"Reading MATLAB file: {filename}")
            return _read_mat_file(filepath)
        elif file_type == FileType.TXT:
            print(f"Reading TXT file: {filename}")
            return _read_txt_file(filepath)
        else:
            print(f"Warning: Unsupported file type: {file_type}")
            return []
    
    # Fallback to extension-based detection for backward compatibility
    if filename.lower().endswith('.res.mat'):
        print(f"Reading RES MAT file: {filename}")
        return _read_mat_file(filepath)
    elif filename.lower().endswith('.mat'):
        print(f"Reading TestLab MAT file: {filename}")
        return _read_testlab_file(filepath)
    elif filename.lower().endswith(config.INPUT_FILE_EXTENSION):
        print(f"Reading TXT file: {filename}")
        return _read_txt_file(filepath)
    else:
        print(f"Warning: Skipping unsupported file type: {filename}")
        return []

