# optimizer_core/file_saver.py

## Role in the System

`optimizer_core/file_saver.py` provides a centralized function for saving optimization results to text files in a standardized format.

## Responsibilities

- Saves envelope point data to text files with specific formatting
- Provides consistent file format across the system

## Dependencies

**Imports:**
- `numpy` - Array operations and file I/O
- `typing.NoReturn` - Type hints
- `utils.logger` - Logging

**Used In:**
- `optimizer_core/psd_utils.py` - Called by `plot_final_solution()` to save results
- `app/save_utils.py` - Called by `save_matplotlib_plot_and_data()` to save envelope data
- `optimizer_core/new_data_loader.py` - Called by `plot_envelope_comparison()` to save envelope data

## Functions

### Function: `save_results_to_text_file(file_path, points)`

**Location:** `optimizer_core/file_saver.py`

**Purpose:**  
Saves the resulting points of optimization to a text file in a simple, human-readable format. Provides a way to persist envelope data for external analysis, archiving, and comparison.

**Parameters:**
- `file_path (str)` - Full path including filename and extension where results will be saved
- `points (np.ndarray)` - 2D NumPy array with shape (n, 2), where column 0 is frequency and column 1 is amplitude

**Returns:**
None

**Side Effects:**
- Writes data to file at `file_path`
- Uses tab delimiter between columns
- Formats frequency as integer, amplitude as floating-point with 10 decimal precision
- Logs success or error messages

**Error Handling:**
- Catches `IOError` and logs error message (doesn't raise to avoid crashing main process)
- Catches general exceptions and logs error message
- Does not raise exceptions (designed to fail gracefully)

**Used In:**
- `optimizer_core/psd_utils.py::plot_final_solution()` - Saves final optimization results
- `app/save_utils.py::save_matplotlib_plot_and_data()` - Saves modified envelope data
- `optimizer_core/new_data_loader.py::plot_envelope_comparison()` - Saves envelope comparison data

