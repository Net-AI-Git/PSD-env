# -*- coding: utf-8 -*-

import os
import sys
from typing import Dict, List, Any

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
    
    return {
        'frequencies': x_values_dict.get('X'),
        'psd_values': y_values_dict.get('Y'),
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
    ensuring consistent data handling across different file formats.
    
    What (Implementation Details):
    Reads the file line by line, parses each line to extract X and Y values
    (separated by whitespace or tab), converts them to numpy arrays, and returns
    a dictionary with the same structure as other extraction functions matching
    data_loader.py format. The filename base (without extensions) is used as both
    output_filename_base and source_filename. Empty lines are skipped.
    
    Args:
        filepath (str): Full path to the .txt file to read.
        
    Returns:
        Dict[str, Any]: Dictionary containing:
                       - 'frequencies' (np.ndarray): Frequency values array
                       - 'psd_values' (np.ndarray): PSD values array
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
    x_values = []
    y_values = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                x_values.append(float(parts[0]))
                y_values.append(float(parts[1]))
        
        x_array = np.array(x_values)
        y_array = np.array(y_values)
        
        logger.info(f"Successfully extracted {len(x_array)} data points from {filepath}")
        
        return {
            'frequencies': x_array,
            'psd_values': y_array,
            'output_filename_base': filename_base,
            'source_filename': source_filename
        }
    except Exception as e:
        logger.warning(f"Failed to read TXT file '{filepath}': {e}")
        return {}


def load_data_from_file(filepath: str) -> List[Dict[str, Any]]:
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
    
    Args:
        filepath (str): The full path to the input file.
        
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
    
    # Process each file using load_data_from_file
    for filepath in all_files:
        filename = os.path.basename(filepath)
        logger.info(f"\nProcessing file: {filename}")
        
        try:
            results = load_data_from_file(filepath)
            
            if results:
                successful_files += 1
                total_psds += len(results)
                logger.info(f"Successfully extracted {len(results)} PSD measurement(s) from {filename}")
                
                # Display details for each PSD
                for i, psd_result in enumerate(results, 1):
                    logger.info(f"  PSD {i}:")
                    logger.info(f"    Source Filename: {psd_result.get('source_filename', 'N/A')}")
                    logger.info(f"    Output Filename Base: {psd_result.get('output_filename_base', 'N/A')}")
                    if psd_result.get('frequencies') is not None:
                        logger.info(f"    Frequencies: {psd_result['frequencies']}")
                    if psd_result.get('psd_values') is not None:
                        logger.info(f"    PSD values: {psd_result['psd_values']}")
            else:
                failed_files += 1
                logger.warning(f"Failed to extract any PSD data from {filename}")
        except Exception as e:
            failed_files += 1
            logger.warning(f"Error processing {filename}: {e}")
        
        logger.info("-" * 70)
    
    # Summary
    logger.info("=" * 70)
    logger.info("Processing Summary:")
    logger.info(f"  Total files processed: {len(all_files)}")
    logger.info(f"  Successful files: {successful_files}")
    logger.info(f"  Failed files: {failed_files}")
    logger.info(f"  Total PSD measurements extracted: {total_psds}")
    logger.info("=" * 70)
