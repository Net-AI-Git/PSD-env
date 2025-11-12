# -*- coding: utf-8 -*-

import os
import sys
from typing import Dict

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import scipy.io
from utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)


def load_mat_file(filepath: str) -> Dict:
    """
    Loads a MATLAB file and returns its data structure.
    
    Why (Purpose and Necessity):
    Provides a centralized way to load MATLAB files, ensuring consistent
    file loading across all extraction functions and avoiding code duplication.
    
    What (Implementation Details):
    Uses scipy.io.loadmat to read the MATLAB file and return the complete
    data structure containing all variables and their values.
    
    Args:
        filepath (str): The full path to the .mat file to load.
        
    Returns:
        Dict: Dictionary containing all variables from the MATLAB file.
    """
    return scipy.io.loadmat(filepath)


def _extract_field_values(mat_data: Dict, field_name: str) -> Dict[str, np.ndarray]:
    """
    Helper function to extract values from a specific field in MATLAB data structure.
    
    Why (Purpose and Necessity):
    Eliminates code duplication between x_values and y_values extraction by
    providing a generic function that can extract any field from the MATLAB structure.
    This follows DRY principle and ensures consistent extraction logic.
    
    What (Implementation Details):
    Iterates through all keys in mat_data, extracts the specified field from
    each variable's structure using the specific format: mat_data[key][0, 0][field_name][0, 0],
    flattens arrays, and handles complex numbers.
    
    Args:
        mat_data (Dict): Loaded MATLAB data structure from load_mat_file.
        field_name (str): Name of the field to extract (e.g., 'y_values', 'x_values').
        
    Returns:
        Dict[str, np.ndarray]: Dictionary with variable names as keys and extracted
                              field values as numpy arrays.
    """
    field_dict = {}
    
    for key in mat_data.keys():
        if not key.startswith('__'):
            try:
                psd_data = mat_data[key][0, 0]
                field_vals = psd_data[field_name][0, 0]
                # field_vals is a tuple, first element is the array with shape (4001, 1)
                field_values = field_vals[0].flatten()
                field_dict[key] = np.real(field_values) if np.iscomplexobj(field_values) else field_values
            except Exception:
                continue
    
    return field_dict


def extract_y_values_to_dict(mat_data: Dict) -> Dict[str, np.ndarray]:
    """
    Extracts all y_values from MATLAB file and stores them in a dictionary.
    
    Why (Purpose and Necessity):
    PSD values (y_values) represent the amplitude data for each frequency point.
    This function provides easy access to all y_values across all PSD variables
    in the MATLAB file for further processing and analysis.
    
    What (Implementation Details):
    Uses the generic _extract_field_values helper function to extract 'y_values'
    field from all variables in the MATLAB data structure using the specific format.
    Returns a dictionary with key 'Y' containing the first y_values found.
    
    Args:
        mat_data (Dict): Loaded MATLAB data structure from load_mat_file.
        
    Returns:
        Dict[str, np.ndarray]: Dictionary with key 'Y' and y_values array as value.
    """
    field_dict = _extract_field_values(mat_data, 'y_values')
    if field_dict:
        # Return dictionary with 'Y' as key, using the first value found
        return {'Y': list(field_dict.values())[0]}
    return {}


def _calculate_frequency_vector(x_vals: np.ndarray) -> np.ndarray:
    """
    Calculates frequency vector from x_values structure parameters.
    
    Why (Purpose and Necessity):
    x_values in MATLAB structure contains metadata (start_value, increment, number_of_values)
    rather than the actual frequency array. This function converts the metadata into
    a complete frequency vector for analysis and plotting.
    
    What (Implementation Details):
    Extracts start_value, increment, and number_of_values from the x_values structure,
    then generates a frequency array using numpy.arange with the formula:
    frequencies = np.arange(number_of_values) * increment + start_value
    
    Args:
        x_vals (np.ndarray): x_values structure from MATLAB containing frequency metadata.
        
    Returns:
        np.ndarray: Complete frequency vector as a numpy array.
    """
    freq_start = float(x_vals[0][0][0].item())
    freq_increment = float(x_vals[1][0][0].item())
    num_points = int(x_vals[2][0][0].item())
    
    return np.arange(num_points) * freq_increment + freq_start


def extract_x_values_to_dict(mat_data: Dict) -> Dict[str, np.ndarray]:
    """
    Extracts frequency vectors (x_values) from MATLAB file.
    
    Why (Purpose and Necessity):
    Frequency values (x_values) represent the X-axis data points for PSD measurements.
    Different MATLAB files may store x_values in different formats:
    - Some files store x_values as metadata (start, increment, count) that must be calculated.
    - Other files store x_values directly as arrays (like y_values).
    This function handles both formats automatically by trying calculation first (checking for
    increment parameter), then falling back to direct extraction if calculation is not possible.
    
    What (Implementation Details):
    First attempts to calculate x_values by extracting the metadata structure and checking
    if it contains increment parameter (indicating calculation method). If increment is found,
    calculates the frequency vector using _calculate_frequency_vector helper. If calculation
    fails or increment is not found, falls back to direct extraction using _extract_field_values
    helper (same method as y_values extraction), and returns a dictionary with key 'X'.
    
    Args:
        mat_data (Dict): Loaded MATLAB data structure from load_mat_file.
        
    Returns:
        Dict[str, np.ndarray]: Dictionary with key 'X' and frequency vector as numpy array.
    """
    # Try calculation method first (for files like "1 PSD.mat")
    # Check if x_values contains increment parameter (metadata structure)
    x_values_dict = {}
    
    for key in mat_data.keys():
        if not key.startswith('__'):
            try:
                psd_data = mat_data[key][0, 0]
                x_vals = psd_data['x_values'][0, 0]
                
                # Check if x_vals has the structure for calculation (has increment at index 1)
                # If it has at least 3 elements and we can access increment, use calculation
                if len(x_vals) >= 3:
                    try:
                        # Try to access increment to verify it's a metadata structure
                        increment = x_vals[1][0][0].item()
                        frequencies = _calculate_frequency_vector(x_vals)
                        x_values_dict[key] = frequencies
                    except (IndexError, AttributeError, ValueError):
                        # If we can't access increment, it's not a metadata structure
                        continue
                else:
                    # Not enough elements for calculation structure
                    continue
            except Exception:
                continue
    
    if x_values_dict:
        return {'X': list(x_values_dict.values())[0]}
    
    # Fall back to direct extraction (for files like "2 PSD.mat")
    field_dict = _extract_field_values(mat_data, 'x_values')
    if field_dict:
        return {'X': list(field_dict.values())[0]}
    
    return {}


def extract_name_from_function_record(mat_data: Dict) -> Dict[str, str]:
    """
    Extracts the 'name' field from function_record in MATLAB file.
    
    Why (Purpose and Necessity):
    The 'name' field in function_record contains the measurement name/identifier
    (e.g., 'PSD A01X'). This function provides easy access to this name.
    
    What (Implementation Details):
    Iterates through all keys in mat_data, extracts function_record structure
    using the specific format: mat_data[key][0, 0]['function_record'][0, 0],
    and extracts only the 'name' field.
    
    Args:
        mat_data (Dict): Loaded MATLAB data structure from load_mat_file.
        
    Returns:
        Dict[str, str]: Dictionary with key 'name' and the name value as string.
    """
    for key in mat_data.keys():
        if not key.startswith('__'):
            try:
                psd_data = mat_data[key][0, 0]
                function_record = psd_data['function_record'][0, 0]
                name_value = function_record['name']
                
                # Remove 'PSD' if it appears and strip whitespace
                name_str = str(name_value)
                # Replace 'PSD' with space, then remove all spaces and strip
                if 'PSD' in name_str:
                    name_str = name_str.replace('PSD', ' ')
                    name_str = name_str.replace(' ', '')
                    name_str = name_str.strip()
                
                return {'name': name_str}
            except Exception:
                continue
    
    return {}


if __name__ == "__main__":
    import sys
    input_dir = os.path.join(project_root, "input")
    
    # Test both files
    test_files = ["1 PSD.mat", "2 PSD.mat"]
    
    for filename in test_files:
        filepath = os.path.join(input_dir, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            continue
        
        logger.info(f"Processing file: {filename}")
        mat_data = load_mat_file(filepath)
        
        # Debug: Check what keys exist in the file
        all_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        logger.debug(f"Keys in mat_data: {all_keys}")
        
        # Check if there are PSD_ variables
        psd_vars = [k for k in mat_data.keys() if k.startswith('PSD_')]
        logger.debug(f"PSD_ variables found: {psd_vars}")
        
        y_values_dict = extract_y_values_to_dict(mat_data)
        x_values_dict = extract_x_values_to_dict(mat_data)
        name_dict = extract_name_from_function_record(mat_data)
        
        logger.info(f"File: {filename}")
        logger.info(f"Y values dict keys: {list(y_values_dict.keys())}")
        logger.info(f"X values dict keys: {list(x_values_dict.keys())}")
        if y_values_dict and 'Y' in y_values_dict:
            logger.info(f"Y values shape: {y_values_dict['Y'].shape}")
            logger.info(f"Y values: {y_values_dict['Y']}")
        if x_values_dict and 'X' in x_values_dict:
            logger.info(f"X values shape: {x_values_dict['X'].shape}")
            logger.info(f"X values: {x_values_dict['X']}")
        logger.info(f"Name: {name_dict}")
        logger.info("-" * 50)
