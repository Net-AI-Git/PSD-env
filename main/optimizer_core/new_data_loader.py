# -*- coding: utf-8 -*-

import os
import sys
import re
from typing import Dict, List, Any, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from utils.logger import get_logger
from optimizer_core import file_saver
from optimizer_core import config

# Initialize logger for this module
logger = get_logger(__name__)


def natural_sort_key(job: Dict[str, Any]) -> Tuple[float, int]:
    """
    Creates a key for natural sorting of measurement names like 'A1X', 'A10Z'.
    
    Why (Purpose and Necessity):
    Measurement names follow patterns like 'A1X', 'A10Z', etc., where we need to
    sort by the numeric part first (1, 10, 2 should be 1, 2, 10), then by the
    axis character (X, Y, Z). Standard string sorting would produce incorrect
    order (A1X, A10Z, A2Y instead of A1X, A2Y, A10Z). This function provides
    a natural sorting key that handles numeric parts correctly.
    
    What (Implementation Details):
    Extracts the measurement part from output_filename_base (before ' - ' if present),
    uses regex to match pattern A<number><axis>, extracts number and axis, creates
    a tuple (number, axis_order) where axis_order is X=0, Y=1, Z=2. For names that
    don't match the pattern, returns (inf, name) to sort them alphabetically at the end.
    
    Args:
        job (Dict[str, Any]): Job dictionary containing 'output_filename_base' field.
        
    Returns:
        Tuple[float, int]: Sorting key tuple (number, axis_order) for matching names,
                          or (float('inf'), name) for non-matching names.
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
                
                # Extract string value from numpy array or list if needed
                if isinstance(name_value, np.ndarray):
                    if name_value.size > 0:
                        name_str = str(name_value.item() if name_value.size == 1 else name_value[0])
                    else:
                        continue
                elif isinstance(name_value, (list, tuple)):
                    name_str = str(name_value[0]) if len(name_value) > 0 else ''
                else:
                    name_str = str(name_value)
                
                # Remove 'PSD' if it appears and strip whitespace
                # Replace 'PSD' with space, then remove all spaces and strip
                if 'PSD' in name_str:
                    name_str = name_str.replace('PSD', ' ')
                    name_str = name_str.replace(' ', '')
                    name_str = name_str.strip()
                
                return {'name': name_str}
            except Exception:
                continue
    
    return {}


def _create_full_envelope_data(combined_data: np.ndarray, min_freq: float, max_freq: float) -> np.ndarray:
    """
    Creates full envelope data by adding interpolated boundary points if needed.
    
    Why (Purpose and Necessity):
    When filtering PSD data to a specific frequency range, the data may not contain
    exact boundary points (min_freq and max_freq). This function ensures complete
    frequency coverage by interpolating boundary points when they are missing, then
    filters the data to the exact required range. This ensures consistent data
    structure across all PSD measurements regardless of their original frequency grid.
    
    What (Implementation Details):
    First filters data to the target range to identify what frequencies are available.
    Checks if boundary points (min_freq and max_freq) are missing by comparing the
    first and last filtered frequencies. If boundaries are missing, creates a linear
    interpolation function from the original combined_data, interpolates the missing
    boundary points, adds them to the data, sorts by frequency, and finally filters
    to the exact required range.
    
    Args:
        combined_data (np.ndarray): Combined frequency and PSD data array with shape (N, 2),
                                   where column 0 is frequencies and column 1 is PSD values.
        min_freq (float): Minimum frequency threshold for filtering.
        max_freq (float): Maximum frequency threshold for filtering.
        
    Returns:
        np.ndarray: Processed data array with shape (M, 2) containing frequencies and PSD
                   values filtered to the range [min_freq, max_freq], with interpolated
                   boundary points if they were missing.
    """
    # First filter to the target range to see what we actually have
    mask = (combined_data[:, 0] >= min_freq) & (combined_data[:, 0] <= max_freq)
    filtered_data = combined_data[mask]
    
    if filtered_data.shape[0] == 0:
        return filtered_data
    
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


def _create_single_psd_mat_data(all_mat_data: Dict, psd_key: str) -> Dict:
    """
    Creates a temporary mat_data structure containing only a single PSD variable.
    
    Why (Purpose and Necessity):
    The existing extraction functions (extract_y_values_to_dict, extract_x_values_to_dict,
    extract_name_from_function_record) are designed to work with mat_data dictionaries
    containing PSD variables. When processing 'all PSD.mat' file, we need to isolate each
    PSD variable and process it individually using these existing functions. This helper
    function creates a compatible mat_data structure for a single PSD, enabling code reuse.
    
    What (Implementation Details):
    Creates a new dictionary containing only the specified psd_key and its corresponding
    value from all_mat_data, preserving the original structure. This allows the existing
    extraction functions to process the single PSD as if it were in a separate file.
    
    Args:
        all_mat_data (Dict): Complete MATLAB data structure loaded from 'all PSD.mat' file.
        psd_key (str): The key name of the PSD variable to isolate (e.g., 'PSD_A01X').
        
    Returns:
        Dict: Dictionary containing only the specified PSD variable, compatible with
              existing extraction functions.
    """
    return {psd_key: all_mat_data[psd_key]}


def extract_single_psd_from_mat(mat_data: Dict, psd_key: str = None, source_filename: str = None) -> Dict[str, Any]:
    """
    Extracts a single PSD from MATLAB data structure with unified output format.
    
    Why (Purpose and Necessity):
    Provides a unified interface to extract a single PSD from MATLAB files, ensuring
    consistent output structure across all file types. This function combines the
    separate extraction functions into a single call that returns a standardized format
    matching data_loader.py structure.
    
    What (Implementation Details):
    If psd_key is provided, creates a temporary mat_data with only that PSD. Otherwise,
    uses the first PSD found in mat_data. Calls extract_y_values_to_dict, extract_x_values_to_dict,
    and extract_name_from_function_record, then combines results into a unified dictionary
    with frequencies, psd_values, output_filename_base, and source_filename fields.
    
    Args:
        mat_data (Dict): Loaded MATLAB data structure from load_mat_file.
        psd_key (str, optional): Specific PSD variable key to extract. If None, uses first PSD found.
        source_filename (str, optional): Source filename without extension. If None, uses empty string.
        
    Returns:
        Dict[str, Any]: Dictionary with unified structure:
                       - 'frequencies' (np.ndarray): Frequency vector
                       - 'psd_values' (np.ndarray): PSD values array
                       - 'output_filename_base' (str): Name from function_record
                       - 'source_filename' (str): Source filename without extension
    """
    if psd_key is None:
        psd_keys = [key for key in mat_data.keys() if not key.startswith('__')]
        if not psd_keys:
            return {}
        psd_key = psd_keys[0]
        single_psd_mat_data = mat_data
    else:
        single_psd_mat_data = _create_single_psd_mat_data(mat_data, psd_key)
    
    y_values_dict = extract_y_values_to_dict(single_psd_mat_data)
    x_values_dict = extract_x_values_to_dict(single_psd_mat_data)
    name_dict = extract_name_from_function_record(single_psd_mat_data)
    
    frequencies = x_values_dict.get('X')
    psd_values = y_values_dict.get('Y')
    
    if frequencies is None or psd_values is None:
        return {}
    
    # Combine frequency and PSD data for filtering and interpolation
    combined_data = np.column_stack((frequencies, psd_values))
    
    # Get frequency range from config (from run_code.py)
    min_freq = getattr(config, 'MIN_FREQUENCY_HZ', None) or 5
    max_freq = getattr(config, 'MAX_FREQUENCY_HZ', None) or 2000
    
    # Apply frequency filtering and boundary interpolation
    filtered_data = _create_full_envelope_data(combined_data, min_freq, max_freq)
    
    if filtered_data.shape[0] == 0:
        logger.warning(f"No data within the {min_freq}-{max_freq} Hz range. Skipping.")
        return {}
    
    return {
        'frequencies': filtered_data[:, 0],
        'psd_values': filtered_data[:, 1],
        'output_filename_base': name_dict.get('name', ''),
        'source_filename': source_filename if source_filename is not None else ''
    }


def extract_all_psds_from_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Extracts all PSD measurements from 'all PSD.mat' file.
    
    Why (Purpose and Necessity):
    The 'all PSD.mat' file contains multiple PSD measurements in a single file, each
    stored as a separate variable (e.g., PSD_A01X, PSD_A01Y, PSD_A01Z). This function
    processes each PSD individually by reusing the existing extraction functions,
    providing a unified interface to access all PSDs from the collection file.
    
    What (Implementation Details):
    Loads the MATLAB file, extracts source filename, identifies all PSD variables
    (keys not starting with '__'), and for each PSD: calls extract_single_psd_from_mat
    function with source_filename to extract all data by reusing existing extraction
    functions, and collects all results into a list. Each result dictionary contains
    frequencies, psd_values, output_filename_base, and source_filename fields.
    
    Args:
        filepath (str): Full path to the 'all PSD.mat' file.
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries, one for each PSD found. Each dictionary
                             contains:
                             - 'frequencies' (np.ndarray): Frequency vector
                             - 'psd_values' (np.ndarray): PSD values array
                             - 'output_filename_base' (str): Name from function_record
                             - 'source_filename' (str): Source filename without extension
                             
    Raises:
        FileNotFoundError: If the specified file does not exist (handled internally,
                          returns empty list with warning).
    """
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return []
    
    source_filename = _extract_source_filename(filepath)
    all_results = []
    mat_data = load_mat_file(filepath)
    psd_keys = [key for key in mat_data.keys() if not key.startswith('__')]
    
    if not psd_keys:
        logger.warning(f"No PSD variables found in file: {filepath}")
        return []
    
    logger.info(f"Processing {len(psd_keys)} PSD variables from: {filepath}")
    
    for psd_key in psd_keys:
        try:
            psd_result = extract_single_psd_from_mat(mat_data, psd_key, source_filename)
            all_results.append(psd_result)
            logger.debug(f"Successfully extracted PSD: {psd_key}")
        except Exception as e:
            logger.warning(f"Failed to extract PSD '{psd_key}': {e}")
            continue
    
    logger.info(f"Successfully processed {len(all_results)} out of {len(psd_keys)} PSDs")
    return all_results


def _extract_filename_base(filepath: str) -> str:
    """
    Extracts the base filename (first word before first dot) from a file path.
    
    Why (Purpose and Necessity):
    For TXT files like 'A1X.spc.txt', we need to extract 'A1X' as the identifier.
    This function provides a consistent way to extract the base name from various
    file naming conventions, ensuring the name matches the expected format.
    
    What (Implementation Details):
    Extracts the filename from the full path using os.path.basename, then splits
    by the first dot ('.') and returns the first part. This handles cases like
    'A1X.spc.txt' -> 'A1X' and 'A1X.txt' -> 'A1X'.
    
    Args:
        filepath (str): Full path to the file.
        
    Returns:
        str: The base filename (first word before first dot).
    """
    filename = os.path.basename(filepath)
    return filename.split('.')[0]


def _extract_source_filename(filepath: str) -> str:
    """
    Extracts the source filename (without extension) from a file path.
    
    Why (Purpose and Necessity):
    Provides a centralized way to extract source filename from filepath, ensuring
    consistent source filename extraction across all extraction functions. This
    follows DRY principle and matches the behavior in data_loader.py.
    
    What (Implementation Details):
    Extracts the filename from the full path using os.path.basename, then removes
    the extension using os.path.splitext. This handles cases like '1 PSD.mat' -> '1 PSD'
    and 'all PSD.mat' -> 'all PSD'.
    
    Args:
        filepath (str): Full path to the file.
        
    Returns:
        str: The source filename without extension.
    """
    filename = os.path.basename(filepath)
    return os.path.splitext(filename)[0]


def _validate_psd_result(result: Dict[str, Any]) -> bool:
    """
    Validates that a PSD extraction result contains valid data.
    
    Why (Purpose and Necessity):
    Before accepting a PSD extraction result, we need to ensure it contains valid
    frequency and PSD value arrays. This prevents empty or malformed data from
    being processed, ensuring that only files with valid X and Y data are considered
    successful reads.
    
    What (Implementation Details):
    Checks if result is a dictionary, contains required keys ('frequencies' and
    'psd_values'), both values are numpy arrays, not None, not empty, and have
    matching lengths. Returns False if any validation fails.
    
    Args:
        result (Dict[str, Any]): PSD extraction result dictionary to validate.
        
    Returns:
        bool: True if result is valid, False otherwise.
    """
    if not isinstance(result, dict):
        return False
    
    if 'frequencies' not in result or 'psd_values' not in result:
        return False
    
    frequencies = result.get('frequencies')
    psd_values = result.get('psd_values')
    
    if frequencies is None or psd_values is None:
        return False
    
    if not isinstance(frequencies, np.ndarray) or not isinstance(psd_values, np.ndarray):
        return False
    
    if len(frequencies) == 0 or len(psd_values) == 0:
        return False
    
    if len(frequencies) != len(psd_values):
        return False
    
    return True


def _convert_single_to_list(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Converts a single PSD result (Dict) to list format for consistency.
    
    Why (Purpose and Necessity):
    Some extraction functions return a single dictionary, while others return a list.
    This function normalizes single results to list format, ensuring consistent
    return types across all extraction functions and enabling uniform processing.
    
    What (Implementation Details):
    If result is an empty dictionary, returns empty list. Otherwise, wraps the
    result dictionary in a list and returns it.
    
    Args:
        result (Dict[str, Any]): Single PSD extraction result dictionary.
        
    Returns:
        List[Dict[str, Any]]: List containing the result dictionary, or empty list
                            if result is empty.
    """
    if not result:
        return []
    return [result]


def extract_txt_file(filepath: str) -> Dict[str, Any]:
    """
    Extracts PSD data from a TXT file in two-column format (X, Y).
    
    Why (Purpose and Necessity):
    Some PSD data is stored in simple TXT files with two columns: frequency (X)
    and PSD values (Y). This function provides a way to read such files and
    convert them to the same data structure format used by MATLAB file readers,
    ensuring consistent data handling across different file formats. The function
    also applies frequency filtering and boundary interpolation to match the
    behavior of data_loader.py.
    
    What (Implementation Details):
    Uses np.loadtxt to read the file (more reliable than manual parsing), validates
    that the file has exactly 2 columns, combines frequency and PSD data, applies
    frequency filtering and boundary interpolation using _create_full_envelope_data,
    and returns a dictionary with the same structure as other extraction functions
    matching data_loader.py format. The filename base (without extensions) is used
    as both output_filename_base and source_filename.
    
    Args:
        filepath (str): Full path to the .txt file to read.
        
    Returns:
        Dict[str, Any]: Dictionary containing:
                       - 'frequencies' (np.ndarray): Frequency values array (filtered)
                       - 'psd_values' (np.ndarray): PSD values array (filtered)
                       - 'output_filename_base' (str): Base filename without extensions
                       - 'source_filename' (str): Source filename without extension
                       
    Raises:
        FileNotFoundError: If the specified file does not exist (handled internally,
                          returns empty dictionary with warning).
    """
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return {}
    
    filename_base = _extract_filename_base(filepath)
    source_filename = _extract_source_filename(filepath)
    
    try:
        data = np.loadtxt(filepath)
        if data.ndim != 2 or data.shape[1] != 2:
            logger.warning(f"Skipping malformed TXT file (not 2 columns): {filepath}")
            return {}
        
        # Combine frequency and PSD data for filtering and interpolation
        combined_data = data
        
        # Get frequency range from config (from run_code.py)
        min_freq = getattr(config, 'MIN_FREQUENCY_HZ', None) or 5
        max_freq = getattr(config, 'MAX_FREQUENCY_HZ', None) or 2000
        
        # Apply frequency filtering and boundary interpolation
        filtered_data = _create_full_envelope_data(combined_data, min_freq, max_freq)
        
        if filtered_data.shape[0] == 0:
            logger.warning(f"No data within the {min_freq}-{max_freq} Hz range in {filepath}. Skipping.")
            return {}
        
        logger.info(f"Successfully extracted {len(filtered_data)} data points from {filepath}")
        
        return {
            'frequencies': filtered_data[:, 0],
            'psd_values': filtered_data[:, 1],
            'output_filename_base': filename_base,
            'source_filename': source_filename
        }
    except Exception as e:
        logger.warning(f"Failed to read TXT file '{filepath}': {e}")
        return {}


def _interpolate_psd_at_frequency(psd_dict: Dict[str, Any], target_frequency: float) -> float:
    """
    Interpolates a PSD value at a specific frequency that doesn't exist in the PSD's frequency array.
    
    Why (Purpose and Necessity):
    When creating envelopes from multiple PSDs with different frequency grids, we need to compare
    values at the same frequency points. This function provides linear interpolation to estimate
    PSD values at frequencies that exist in one PSD but not in another, enabling fair comparison
    and maximum value selection across all PSDs.
    
    What (Implementation Details):
    Extracts frequencies and psd_values from the PSD dictionary, finds the two closest frequencies
    to the target (one below and one above), and uses numpy's linear interpolation to estimate
    the PSD value at the target frequency. If the target frequency is outside the PSD's range,
    uses extrapolation with the closest boundary values.
    
    Args:
        psd_dict (Dict[str, Any]): PSD dictionary containing 'frequencies' and 'psd_values' arrays.
        target_frequency (float): The frequency at which to interpolate the PSD value.
        
    Returns:
        float: Interpolated PSD value at the target frequency.
        
    Raises:
        ValueError: If psd_dict doesn't contain required keys or arrays are empty.
    """
    frequencies = psd_dict.get('frequencies')
    psd_values = psd_dict.get('psd_values')
    
    if frequencies is None or psd_values is None:
        raise ValueError("PSD dictionary must contain 'frequencies' and 'psd_values'")
    
    if len(frequencies) == 0 or len(psd_values) == 0:
        raise ValueError("PSD arrays cannot be empty")
    
    # Use numpy's interp which handles extrapolation automatically
    interpolated_value = np.interp(target_frequency, frequencies, psd_values)
    return float(interpolated_value)


def _group_psds_by_output_filename(psd_list: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Groups PSD dictionaries by their output_filename_base field.
    
    Why (Purpose and Necessity):
    Envelopes should be created separately for each channel/measurement type. This function
    organizes PSDs into groups based on their output_filename_base, enabling independent
    envelope creation for each group.
    
    What (Implementation Details):
    Iterates through all PSDs in the list, extracts the output_filename_base from each,
    and groups them into a dictionary where keys are output_filename_base values and values
    are lists of PSD dictionaries sharing that same output_filename_base.
    
    Args:
        psd_list (List[Dict[str, Any]]): List of PSD dictionaries to group.
        
    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary mapping output_filename_base to lists
                                        of PSD dictionaries.
    """
    groups = {}
    for psd in psd_list:
        output_name = psd.get('output_filename_base', '')
        if output_name not in groups:
            groups[output_name] = []
        groups[output_name].append(psd)
    return groups


def create_envelope_from_psds(psd_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Creates envelope data by grouping PSDs and taking maximum values at each frequency.
    
    Why (Purpose and Necessity):
    When multiple PSD measurements exist for the same channel (same output_filename_base),
    we need to create an envelope that represents the maximum PSD value at each frequency
    point across all measurements. This envelope ensures conservative analysis by using
    the worst-case (highest) PSD values at each frequency.
    
    What (Implementation Details):
    Groups PSDs by output_filename_base, then for each group: if only one PSD exists, returns
    it as-is. If multiple PSDs exist, creates a unified frequency grid from all unique
    frequencies, interpolates each PSD to this grid, takes the maximum value at each
    frequency point, and creates an envelope dictionary with the same structure as input PSDs.
    
    Args:
        psd_list (List[Dict[str, Any]]): List of PSD dictionaries, each containing:
                                        - 'frequencies' (np.ndarray): Frequency vector
                                        - 'psd_values' (np.ndarray): PSD values array
                                        - 'output_filename_base' (str): Channel/measurement name
                                        - 'source_filename' (str): Source file identifier
        
    Returns:
        List[Dict[str, Any]]: List of envelope dictionaries, one per unique output_filename_base.
                             Each envelope has the same structure as input PSDs, with maximum
                             PSD values at each frequency point.
                             
    Raises:
        None: All exceptions are caught and logged, function returns empty list on critical errors.
    """
    if not psd_list:
        return []
    
    groups = _group_psds_by_output_filename(psd_list)
    envelope_results = []
    
    for output_name, psd_group in groups.items():
        if len(psd_group) == 1:
            logger.info(f"Creating envelope for channel '{output_name}' from 1 file")
        else:
            logger.info(f"Creating envelope for channel '{output_name}' from {len(psd_group)} files")
        
        envelope_dict = _create_single_envelope(psd_group, output_name)
        if envelope_dict:
            envelope_results.append(envelope_dict)
    
    return envelope_results


def _create_single_envelope(psd_group: List[Dict[str, Any]], output_name: str) -> Dict[str, Any]:
    """
    Creates a single envelope from a group of PSDs with the same output_filename_base.
    
    Why (Purpose and Necessity):
    Isolates the envelope creation logic for a single group, keeping the main function
    short and focused. This follows the principle of short, focused functions.
    
    What (Implementation Details):
    Creates a unified frequency grid from all unique frequencies in the group, interpolates
    each PSD to this grid, takes the maximum value at each frequency, and constructs the
    envelope dictionary with appropriate metadata.
    
    Args:
        psd_group (List[Dict[str, Any]]): List of PSD dictionaries with same output_filename_base.
        output_name (str): The output_filename_base shared by all PSDs in the group.
        
    Returns:
        Dict[str, Any]: Envelope dictionary with frequencies, psd_values, output_filename_base,
                       and source_filename, or empty dict if creation fails.
    """
    try:
        # Create unified frequency grid
        all_frequencies = np.concatenate([psd['frequencies'] for psd in psd_group])
        unique_frequencies = np.unique(all_frequencies)
        unique_frequencies = np.sort(unique_frequencies)
        
        # Interpolate all PSDs to unified grid and collect values
        interpolated_psds = []
        for psd in psd_group:
            interpolated_psd = np.interp(unique_frequencies, psd['frequencies'], psd['psd_values'])
            interpolated_psds.append(interpolated_psd)
        
        # Take maximum at each frequency
        envelope_psd = np.max(interpolated_psds, axis=0)
        
        # Create envelope dictionary
        return {
            'frequencies': unique_frequencies,
            'psd_values': envelope_psd,
            'output_filename_base': output_name,
            'source_filename': psd_group[0].get('source_filename', 'envelope')
        }
    except Exception as e:
        logger.warning(f"Failed to create envelope for '{output_name}': {e}")
        return {}


def plot_envelope_comparison(original_jobs: List[Dict[str, Any]], envelope_job: Dict[str, Any], 
                            channel_name: str, output_path: str) -> None:
    """
    Creates a log-log plot comparing original PSD data with the envelope.
    
    Why (Purpose and Necessity):
    Visual comparison between original PSD measurements and their envelope is essential
    for validating that the envelope correctly captures the maximum values. This function
    provides a graphical representation that helps users understand how the envelope
    relates to the original data, showing all original measurements in different colors
    and the envelope in red for easy identification.
    
    What (Implementation Details):
    Creates a matplotlib figure with log-log scale, plots all original PSD measurements
    in different colors with labels showing source filename and output filename base,
    overlays the envelope in red with RMS value in the label, adds proper axis labels,
    title, legend, and grid, saves the plot as PNG file, and saves the envelope data
    as a text file using the existing file_saver utility.
    
    Args:
        original_jobs (List[Dict[str, Any]]): List of job dictionaries containing
                                             original PSD data with 'frequencies',
                                             'psd_values', 'source_filename', and
                                             'output_filename_base' fields.
        envelope_job (Dict[str, Any]): Job dictionary containing the envelope PSD data
                                      with 'frequencies' and 'psd_values' fields.
        channel_name (str): Name of the channel being plotted (used in title).
        output_path (str): Full path where the plot should be saved (PNG format).
        
    Returns:
        None: Function saves files but doesn't return a value.
        
    Raises:
        None: All exceptions are caught and logged internally.
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
    logger.info(f"Saved envelope comparison plot: {output_path}")
    
    # Save envelope data as text file using existing function
    text_output_path = output_path.replace('.png', '.txt')
    envelope_points = np.column_stack((envelope_job['frequencies'], envelope_job['psd_values']))
    file_saver.save_results_to_text_file(text_output_path, envelope_points)


def load_data_from_file(filepath: str, file_type=None) -> List[Dict[str, Any]]:
    """
    Loads all measurement jobs from a single specified file (.txt, .mat).
    
    Why (Purpose and Necessity):
    Different files may have different internal structures or formats. This function
    automatically attempts to read the file using multiple extraction methods in
    sequence until one succeeds, providing a robust interface that handles various
    file formats without requiring explicit specification.
    
    What (Implementation Details):
    Tries extraction methods in the following order:
    1. Multiple PSDs extraction (for files like "all PSD.mat", "example.mat")
    2. Single PSD extraction (for files like "1 PSD.mat", "2 PSD.mat")
    3. TXT file extraction (for .txt files)
    Each attempt validates that the result contains valid X and Y data. If all
    attempts fail, returns an empty list.
    Note: The file_type parameter is accepted for compatibility with data_loader.py
    interface, but this function uses auto-detection and does not use this parameter.
    
    Args:
        filepath (str): The full path to the input file.
        file_type (optional): The type of file to process. Currently not used as
                             this function uses auto-detection. Kept for compatibility.
        
    Returns:
        List[Dict[str, Any]]: A list of job dictionaries found in the file. Each
                             dictionary contains:
                             - 'frequencies' (np.ndarray): Frequency vector
                             - 'psd_values' (np.ndarray): PSD values array
                             - 'output_filename_base' (str): Measurement name/identifier
                             - 'source_filename' (str): Source filename without extension
                             Returns an empty list if the file is unsupported or processing fails.
    """
    filename = os.path.basename(filepath)
    
    # Try multiple PSDs extraction first (for files like "all PSD.mat", "example.mat")
    try:
        results = extract_all_psds_from_file(filepath)
        if results and all(_validate_psd_result(r) for r in results):
            logger.info(f"Successfully read {filename} as multiple PSDs ({len(results)} PSDs)")
            return results
    except Exception as e:
        logger.debug(f"Multiple PSDs extraction failed for {filename}: {e}")
    
    # Try single PSD extraction (for files like "1 PSD.mat", "2 PSD.mat")
    try:
        mat_data = load_mat_file(filepath)
        source_filename = _extract_source_filename(filepath)
        result = extract_single_psd_from_mat(mat_data, source_filename=source_filename)
        if _validate_psd_result(result):
            logger.info(f"Successfully read {filename} as single PSD")
            return _convert_single_to_list(result)
    except Exception as e:
        logger.debug(f"Single PSD extraction failed for {filename}: {e}")
    
    # Try TXT file extraction
    try:
        result = extract_txt_file(filepath)
        if _validate_psd_result(result):
            logger.info(f"Successfully read {filename} as TXT file")
            return _convert_single_to_list(result)
    except Exception as e:
        logger.debug(f"TXT file extraction failed for {filename}: {e}")
    
    logger.warning(f"Could not read {filename} with any supported format. Skipping.")
    return []


def load_full_envelope_data(input_dir: str, file_type=None) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """
    Loads all files from the input directory and creates envelope data
    by taking the maximum PSD values for each frequency across all files
    with matching channel names.
    
    Why (Purpose and Necessity):
    When processing multiple files containing PSD measurements, we need to
    group measurements by channel name and create envelope data representing
    the maximum PSD value at each frequency point. This function provides
    the same interface as data_loader.load_full_envelope_data, ensuring
    compatibility with existing code that expects envelope jobs and channel groups.
    
    What (Implementation Details):
    Loads all files from input_dir using load_data_from_file for each file,
    collects all PSDs into a single list, groups them by output_filename_base
    using _group_psds_by_output_filename helper, creates envelopes using
    create_envelope_from_psds function which handles interpolation and maximum
    value selection, and returns both envelope jobs and original channel groups
    in the same structure as data_loader.load_full_envelope_data.
    
    Args:
        input_dir (str): The directory containing all input files.
        file_type (optional): The type of file to process. If None, will attempt
                             to determine from extension. Currently not used but
                             kept for compatibility with data_loader interface.
        
    Returns:
        Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]: A tuple containing:
            - List[Dict[str, Any]]: Envelope job dictionaries, one per unique
                                   output_filename_base, with frequencies, psd_values,
                                   output_filename_base, and source_filename fields.
            - Dict[str, List[Dict[str, Any]]]: Dictionary mapping output_filename_base
                                              to lists of original job dictionaries,
                                              grouped by channel name.
    """
    all_jobs = []
    
    # Load all files from the directory
    for filename in sorted(os.listdir(input_dir)):
        filepath = os.path.join(input_dir, filename)
        if os.path.isfile(filepath):
            jobs_from_file = load_data_from_file(filepath, file_type)
            all_jobs.extend(jobs_from_file)
    
    if not all_jobs:
        logger.warning("No valid jobs found in the input directory.")
        return [], {}
    
    # Group jobs by output_filename_base using existing helper function
    channel_groups = _group_psds_by_output_filename(all_jobs)
    
    # Create envelope jobs using existing function
    envelope_jobs = create_envelope_from_psds(all_jobs)
    
    logger.info(f"Created {len(envelope_jobs)} envelope jobs from {len(all_jobs)} original jobs")
    return envelope_jobs, channel_groups


if __name__ == "__main__":
    input_dir = os.path.join(project_root, "input")
    
    if not os.path.exists(input_dir):
        logger.warning(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    logger.info("=" * 70)
    logger.info(f"Processing all files from input directory: {input_dir}")
    logger.info("=" * 70)
    
    # Get all files from input directory
    all_files = []
    for filename in sorted(os.listdir(input_dir)):
        filepath = os.path.join(input_dir, filename)
        if os.path.isfile(filepath):
            all_files.append(filepath)
    
    if not all_files:
        logger.warning(f"No files found in input directory: {input_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(all_files)} file(s) to process")
    logger.info("-" * 70)
    
    total_psds = 0
    successful_files = 0
    failed_files = 0
    
    # Collect all PSDs from all files
    all_psds = []
    
    # Process each file using load_data_from_file
    for filepath in all_files:
        filename = os.path.basename(filepath)
        logger.info(f"\nProcessing file: {filename}")
        
        try:
            results = load_data_from_file(filepath)
            
            if results:
                successful_files += 1
                total_psds += len(results)
                all_psds.extend(results)
                logger.info(f"Successfully extracted {len(results)} PSD measurement(s) from {filename}")
                
                # Display details for each PSD
                for i, psd_result in enumerate(results, 1):
                    logger.info(f"  PSD {i}:")
                    logger.info(f"    Source Filename: {psd_result.get('source_filename', 'N/A')}")
                    logger.info(f"    Output Filename Base: {psd_result.get('output_filename_base', 'N/A')}")
                    if psd_result.get('frequencies') is not None:
                        logger.info(f"    Frequencies shape: {psd_result['frequencies'].shape}")
                    if psd_result.get('psd_values') is not None:
                        logger.info(f"    PSD values shape: {psd_result['psd_values'].shape}")
            else:
                failed_files += 1
                logger.warning(f"Failed to extract any PSD data from {filename}")
        except Exception as e:
            failed_files += 1
            logger.warning(f"Error processing {filename}: {e}")
        
        logger.info("-" * 70)
    
    # Summary of file processing
    logger.info("=" * 70)
    logger.info("File Processing Summary:")
    logger.info(f"  Total files processed: {len(all_files)}")
    logger.info(f"  Successful files: {successful_files}")
    logger.info(f"  Failed files: {failed_files}")
    logger.info(f"  Total PSD measurements extracted: {total_psds}")
    logger.info("=" * 70)
    
    # Create envelopes from all PSDs
    if all_psds:
        logger.info("\n" + "=" * 70)
        logger.info("Creating Envelopes")
        logger.info("=" * 70)
        
        envelope_results = create_envelope_from_psds(all_psds)
        
        logger.info(f"Created {len(envelope_results)} envelope(s) from {len(all_psds)} PSD measurement(s)")
        
        # Group original PSDs by output_filename_base for plotting
        channel_groups = _group_psds_by_output_filename(all_psds)
        
        # Create output directory for envelope results
        results_dir = os.path.join(project_root, "results")
        envelope_output_dir = os.path.join(results_dir, "SPEC")
        if not os.path.exists(envelope_output_dir):
            os.makedirs(envelope_output_dir)
            logger.info(f"Created output directory: {envelope_output_dir}")
        
        # Create envelop subdirectory for comparison plots
        envelop_plots_dir = os.path.join(envelope_output_dir, "envelop")
        if not os.path.exists(envelop_plots_dir):
            os.makedirs(envelop_plots_dir)
            logger.info(f"Created envelop plots directory: {envelop_plots_dir}")
        
        # Create opt SECS subdirectory for optimization results
        opt_secs_dir = os.path.join(envelope_output_dir, "opt SECS")
        if not os.path.exists(opt_secs_dir):
            os.makedirs(opt_secs_dir)
            logger.info(f"Created opt SECS directory: {opt_secs_dir}")
        
        # Create comparison plots for all channels
        logger.info("\n" + "=" * 70)
        logger.info("Creating Envelope Comparison Plots")
        logger.info("=" * 70)
        
        plots_created = 0
        for envelope_job in envelope_results:
            channel_name = envelope_job.get('output_filename_base', 'Unknown')
            original_jobs = channel_groups.get(channel_name, [])
            
            if original_jobs:
                # Create plot filename
                plot_filename = f"{channel_name}.png"
                plot_path = os.path.join(envelop_plots_dir, plot_filename)
                
                try:
                    plot_envelope_comparison(original_jobs, envelope_job, channel_name, plot_path)
                    plots_created += 1
                    logger.info(f"Created plot for channel: {channel_name}")
                except Exception as e:
                    logger.warning(f"Failed to create plot for channel '{channel_name}': {e}")
        
        logger.info("\n" + "=" * 70)
        logger.info("Envelope Processing Summary:")
        logger.info(f"  Total envelopes created: {len(envelope_results)}")
        logger.info(f"  Comparison plots created: {plots_created}")
        logger.info(f"  Output directory: {envelope_output_dir}")
        logger.info("=" * 70)
    else:
        logger.warning("No PSD data available to create envelopes")
