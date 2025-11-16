# optimizer_core/psd_utils.py

## Role in the System

`optimizer_core/psd_utils.py` provides utility functions for handling PSD data, including generating candidate points for the envelope and plotting final optimization results.

## Responsibilities

- Generates candidate points using multi-scale moving window maximum
- Simplifies envelopes by removing redundant points
- Creates "lifted" candidate points to enrich search space
- Plots and saves final optimization results

## Dependencies

**Imports:**
- `matplotlib.pyplot` - Plot creation
- `numpy` - Array operations
- `os` - File path operations
- `optimizer_core.config` - Configuration parameters
- `optimizer_core.file_saver.save_results_to_text_file` - Text file saving
- `utils.logger` - Logging

**Used In:**
- `run_code.py` - Uses `create_multi_scale_envelope()` and `plot_final_solution()`

## Functions

### Function: `moving_window_maximum(psd_values, window_size)`

**Location:** `optimizer_core/psd_utils.py`

**Purpose:**  
Calculates the moving window maximum of a 1D array. For each point, finds the maximum value within a symmetric window centered at that point. This is a key step in generating a coarse envelope.

**Parameters:**
- `psd_values (np.ndarray)` - Input array of PSD amplitudes
- `window_size (int)` - Total size of the moving window

**Returns:**
- `np.ndarray` - Array of same shape as input, containing moving window maximum values

**Side Effects:**
None

**Error Handling:**
None (assumes valid input)

**Used In:**
- `create_multi_scale_envelope()` - Called for each window size

### Function: `simplify_envelope(frequencies, psd_values)`

**Location:** `optimizer_core/psd_utils.py`

**Purpose:**  
Reduces the number of points in an envelope by keeping only "corner points" (points where slope changes). Removes points that lie on straight horizontal lines.

**Parameters:**
- `frequencies (np.ndarray)` - Frequency values
- `psd_values (np.ndarray)` - Amplitude values of the envelope

**Returns:**
- `np.ndarray` - 2D array of simplified [frequency, psd_value] points

**Side Effects:**
None

**Error Handling:**
- Returns all points if less than 3 points (can't simplify)

**Used In:**
- `create_multi_scale_envelope()` - Called after moving window maximum for each window size

### Function: `create_multi_scale_envelope(frequencies, psd_values, window_sizes)`

**Location:** `optimizer_core/psd_utils.py`

**Purpose:**  
Creates a rich set of candidate points for the genetic algorithm. The process: (1) generates base points using multi-scale moving window, (2) creates "lifted" version if enabled, (3) enriches with low-frequency points if enabled, (4) combines all into sorted, unique pool.

**Parameters:**
- `frequencies (np.ndarray)` - Frequency values of original PSD
- `psd_values (np.ndarray)` - PSD amplitude values
- `window_sizes (list[int])` - List of window sizes for multi-scale generation

**Returns:**
- `np.ndarray` - 2D array of candidate points with shape (N, 2), sorted by frequency

**Side Effects:**
- Logs generation progress and statistics

**Error Handling:**
None (assumes valid input)

**Used In:**
- `run_code.py::process_psd_job()` - Called once per job to generate candidate points

### Function: `plot_final_solution(original_freqs, original_psd, solution_points, final_area_ratio, output_filename_base, output_directory)`

**Location:** `optimizer_core/psd_utils.py`

**Purpose:**  
Renders and saves a dual view (log and linear X-axis) of the final optimized envelope solution. Calculates RMS values and saves both image and text file.

**Parameters:**
- `original_freqs (np.ndarray)` - Original frequency data
- `original_psd (np.ndarray)` - Original PSD amplitude data
- `solution_points (np.ndarray)` - Coordinates of final envelope points
- `final_area_ratio (float)` - Calculated area ratio for title
- `output_filename_base (str)` - Base name for output files (without extension)
- `output_directory (str)` - Directory where results will be saved

**Returns:**
None

**Side Effects:**
- Creates matplotlib figure with two subplots
- Saves PNG image: `{output_filename_base}.png`
- Saves text file: `{output_filename_base}.spc.txt` using `file_saver.save_results_to_text_file()`
- Closes matplotlib figures
- Logs save operations

**Error Handling:**
None (assumes valid input and writable directory)

**Used In:**
- `run_code.py::process_psd_job()` - Called at end of optimization to save final results

