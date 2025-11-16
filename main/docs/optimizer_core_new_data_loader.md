# optimizer_core/new_data_loader.py

## Role in the System

`optimizer_core/new_data_loader.py` provides a unified data loading system that handles multiple file formats (.mat with various structures, .txt) and provides consistent output format. This module eliminates code duplication and ensures consistent behavior across the system.

## Responsibilities

- Loads PSD data from MATLAB (.mat) files with various internal structures
- Loads PSD data from text (.txt) files
- Creates envelopes from multiple PSD files by taking maximum values
- Matches PSD files with envelope files for visualization
- Handles frequency filtering and boundary interpolation
- Provides natural sorting for measurement names

## Dependencies

**Imports:**
- `os`, `sys` - File and path operations
- `re` - Regular expressions for name matching
- `typing` - Type hints
- `numpy` - Array operations
- `scipy.io` - MATLAB file loading
- `matplotlib.pyplot` - Plotting for envelope comparisons
- `scipy.interpolate.interp1d` - Interpolation for boundary points
- `utils.logger` - Logging
- `optimizer_core.file_saver` - Text file saving
- `optimizer_core.config` - Configuration parameters

**Used In:**
- `run_code.py` - Uses `load_data_from_file()`, `load_full_envelope_data()`
- `app/gui_utils.py` - Uses `find_data_pairs_unified()` via wrapper
- `app/tab_visualizer.py` - Uses `find_data_pairs_unified()` indirectly

## Key Functions

### Function: `load_data_from_file(filepath, file_type=None)`

**Location:** `optimizer_core/new_data_loader.py`

**Purpose:**  
Loads all measurement jobs from a single specified file (.txt, .mat). Automatically attempts multiple extraction methods until one succeeds, providing robust interface for various file formats.

**Parameters:**
- `filepath (str)` - Full path to the input file
- `file_type (optional)` - File type (currently not used, kept for compatibility)

**Returns:**
- `list[dict]` - List of job dictionaries, each containing:
  - `'frequencies' (np.ndarray)` - Frequency vector
  - `'psd_values' (np.ndarray)` - PSD values array
  - `'output_filename_base' (str)` - Measurement name/identifier
  - `'source_filename' (str)` - Source filename without extension
  - Returns empty list if file is unsupported or processing fails

**Side Effects:**
- Logs extraction attempts and results
- Applies frequency filtering based on config
- Applies boundary interpolation if needed

**Error Handling:**
- Tries multiple extraction methods in sequence (multiple PSDs, single PSD, TXT)
- Validates results before returning
- Returns empty list if all methods fail
- Logs warnings for failed attempts

**Used In:**
- `run_code.py::main()` - Called for each file in input directory
- `load_full_envelope_data()` - Called for each file
- `find_data_pairs_unified()` - Called to load envelope files

### Function: `load_full_envelope_data(input_dir, file_type=None)`

**Location:** `optimizer_core/new_data_loader.py`

**Purpose:**  
Loads all files from input directory and creates envelope data by taking maximum PSD values for each frequency across all files with matching channel names. Provides same interface as legacy data_loader for compatibility.

**Parameters:**
- `input_dir (str)` - Directory containing all input files
- `file_type (optional)` - File type (currently not used)

**Returns:**
- `tuple[list[dict], dict[str, list[dict]]]` - Tuple containing:
  - List of envelope job dictionaries (one per unique output_filename_base)
  - Dictionary mapping output_filename_base to lists of original job dictionaries (grouped by channel)

**Side Effects:**
- Loads all files from directory using `load_data_from_file()`
- Groups PSDs by output_filename_base
- Creates envelopes using `create_envelope_from_psds()`
- Logs processing progress

**Error Handling:**
- Returns empty lists if no valid jobs found
- Handles file loading errors gracefully (continues with next file)

**Used In:**
- `run_code.py::main()` - Called when FULL_ENVELOPE mode is enabled

### Function: `find_data_pairs_unified(source_directory, envelope_directory)`

**Location:** `optimizer_core/new_data_loader.py`

**Purpose:**  
Scans directories to find pairs of original PSD data and their corresponding envelope files. Uses unified loading system and multiple matching strategies to robustly pair files.

**Parameters:**
- `source_directory (str)` - Path to directory containing source PSD files (.mat or .txt)
- `envelope_directory (str)` - Path to directory containing envelope files (.spc.txt)

**Returns:**
- `list[dict]` - List of dictionaries, each containing:
  - `'name' (str)` - Envelope filename
  - `'psd_data' (np.ndarray)` - PSD data as array with shape (N, 2)
  - `'envelope_data' (np.ndarray)` - Envelope data as array with shape (M, 2)

**Side Effects:**
- Loads all PSD files from source directory
- Groups PSDs and creates envelopes
- Loads envelope files from envelope directory
- Matches envelopes to PSDs using multiple strategies
- Logs matching results and warnings for unmatched files

**Error Handling:**
- Validates that directories exist
- Handles unmatched envelope files gracefully (logs warning, continues)
- Validates envelope data format (must have 2 columns)

**Used In:**
- `app/gui_utils.py::find_data_pairs()` - Called via wrapper function

### Function: `create_envelope_from_psds(psd_list)`

**Location:** `optimizer_core/new_data_loader.py`

**Purpose:**  
Creates envelope data by grouping PSDs and taking maximum values at each frequency. When multiple PSD measurements exist for the same channel, creates envelope representing worst-case (highest) PSD values.

**Parameters:**
- `psd_list (list[dict])` - List of PSD dictionaries, each containing frequencies, psd_values, output_filename_base, source_filename

**Returns:**
- `list[dict]` - List of envelope dictionaries, one per unique output_filename_base, with same structure as input PSDs but with maximum PSD values

**Side Effects:**
- Groups PSDs by output_filename_base
- Creates unified frequency grid from all unique frequencies
- Interpolates each PSD to unified grid
- Takes maximum value at each frequency point
- Logs envelope creation progress

**Error Handling:**
- Returns empty list if input list is empty
- Handles interpolation errors gracefully (logs warning, continues with next group)
- Returns single PSD as-is if only one exists for a channel

**Used In:**
- `load_full_envelope_data()` - Called to create envelopes from loaded PSDs
- `find_data_pairs_unified()` - Called to create PSD envelopes for matching

### Function: `plot_envelope_comparison(original_jobs, envelope_job, channel_name, output_path)`

**Location:** `optimizer_core/new_data_loader.py`

**Purpose:**  
Creates a log-log plot comparing original PSD data with the envelope. Shows all original measurements in different colors and envelope in red for visual validation.

**Parameters:**
- `original_jobs (list[dict])` - List of job dictionaries containing original PSD data
- `envelope_job (dict)` - Job dictionary containing envelope PSD data
- `channel_name (str)` - Name of channel being plotted (used in title)
- `output_path (str)` - Full path where plot should be saved (PNG format)

**Returns:**
None

**Side Effects:**
- Creates matplotlib figure
- Saves PNG image to output_path
- Saves envelope data as text file using `file_saver.save_results_to_text_file()`
- Closes matplotlib figures
- Logs save operations

**Error Handling:**
- All exceptions are caught and logged internally

**Used In:**
- `run_code.py::main()` - Called when creating envelope comparison plots in FULL_ENVELOPE mode

### Function: `natural_sort_key(job)`

**Location:** `optimizer_core/new_data_loader.py`

**Purpose:**  
Creates a key for natural sorting of measurement names like 'A1X', 'A10Z'. Handles numeric parts correctly (1, 2, 10 instead of 1, 10, 2) and sorts by axis (X, Y, Z).

**Parameters:**
- `job (dict)` - Job dictionary containing 'output_filename_base' field

**Returns:**
- `tuple[float, int]` - Sorting key tuple (number, axis_order) for matching names, or (inf, name) for non-matching names

**Side Effects:**
None

**Error Handling:**
- Returns (inf, name) for names that don't match pattern (sorts them alphabetically at end)

**Used In:**
- `run_code.py::main()` - Used as key function for sorting jobs
- Called indirectly when sorting envelope jobs and channel groups

## Helper Functions

The module contains many helper functions for extracting data from MATLAB files:
- `load_mat_file()` - Loads MATLAB file
- `_extract_field_values()` - Generic field extraction
- `extract_y_values_to_dict()` - Extracts PSD values
- `extract_x_values_to_dict()` - Extracts frequency values (handles both metadata and direct arrays)
- `extract_name_from_function_record()` - Extracts measurement name
- `extract_single_psd_from_mat()` - Unified single PSD extraction
- `extract_all_psds_from_file()` - Extracts all PSDs from "all PSD.mat" files
- `extract_txt_file()` - Extracts from TXT files
- `_create_full_envelope_data()` - Handles frequency filtering and boundary interpolation
- `_normalize_name()` - Normalizes names for matching
- `_find_matching_psd_name()` - Implements multiple matching strategies

These helpers are used internally by the main functions and handle the complexity of different file formats and structures.

