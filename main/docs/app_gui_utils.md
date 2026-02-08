# app/gui_utils.py

## Role in the System

`app/gui_utils.py` provides utility functions for the GUI layer, including RMS calculation and interactive Bokeh plot creation for PSD visualization.

## Responsibilities

- Calculates RMS (Root Mean Square) values from frequency and PSD data
- Creates interactive Bokeh plots with editing capabilities (drag, add, delete points)
- Provides data loading wrapper that delegates to the unified data loader

## Dependencies

**Imports:**
- `numpy` - Array operations and integration
- `bokeh.plotting.figure` - Plot creation
- `bokeh.models` - Data sources, tools, and widgets
- `bokeh.layouts` - Layout containers
- `optimizer_core.new_data_loader` - Unified data loading system
- `optimizer_core.psd_utils.calculate_rms_from_psd` - Centralized RMS calculation
- `utils.logger` - Logging

**Used In:**
- `app/tab_visualizer.py` - Uses `find_data_pairs()`, `create_psd_plot()`, `_calculate_rms()`

## Functions

### Function: `_calculate_rms(frequencies, psd_values)`

**Location:** `app/gui_utils.py`

**Purpose:**  
Wrapper function that delegates to the centralized RMS calculation function `optimizer_core.psd_utils.calculate_rms_from_psd`. This function maintains backward compatibility with existing code that calls `_calculate_rms()` while using the centralized implementation to ensure consistent RMS calculation across the entire codebase.

**Parameters:**
- `frequencies (np.ndarray)` - Frequency values array
- `psd_values (np.ndarray)` - PSD amplitude values array

**Returns:**
- `float` - RMS value in g units, or 0.0 if input is invalid

**Side Effects:**
None (delegates to `calculate_rms_from_psd`)

**Error Handling:**
- All error handling is delegated to `calculate_rms_from_psd`, which:
  - Returns 0.0 if frequencies or psd_values are None
  - Returns 0.0 if arrays have less than 2 points
  - Returns 0.0 if calculated mean square is negative (physical impossibility)
  - Sorts data by frequency before integration to ensure correct results

**Used In:**
- `app/tab_visualizer.py` - Called to calculate RMS for legend display and save operations
- `create_psd_plot()` - Called internally to calculate RMS for legend (though `create_psd_plot` now receives pre-calculated RMS values via `rms_info` parameter)

### Function: `create_psd_plot(psd_data, envelope_data, plot_title, on_change_callback=None, rms_info=None)`

**Location:** `app/gui_utils.py`

**Purpose:**  
Creates a Bokeh layout containing two interactive plots (log and linear X-axis) of PSD and envelope data. Includes editing tools for manual point manipulation.

**Parameters:**
- `psd_data (np.ndarray)` - Original PSD data with shape (N, 2), column 0 = frequencies, column 1 = PSD values
- `envelope_data (np.ndarray)` - Envelope data with shape (M, 2), column 0 = frequencies, column 1 = envelope values
- `plot_title (str)` - Title to display on plots
- `on_change_callback (callable, optional)` - Callback function triggered when envelope points are edited. Signature: `(attr, old, new) -> None`
- `rms_info (dict, optional)` - Dictionary with keys 'psd_rms', 'env_rms', 'ratio' for legend display

**Returns:**
- `tuple[bokeh.layouts.Column, list]` - Tuple containing:
  - Layout column with both plots and controls
  - List of plot container objects for direct manipulation

**Side Effects:**
- Creates Bokeh figure objects
- Creates ColumnDataSource objects for plot data
- Attaches JavaScript callbacks for data sorting and point deletion
- Attaches Python callback for point editing (if provided)
- Creates interactive editing tools (PointDrawTool, TapTool)

**Error Handling:**
- JavaScript callback includes guard clause to prevent infinite sorting loops
- Handles missing legend gracefully

**Used In:**
- `app/tab_visualizer.py` - Called by `update_plot_and_controls()` to create/refresh plots

### Function: `find_data_pairs(source_directory, envelope_directory)`

**Location:** `app/gui_utils.py`

**Purpose:**  
Wrapper function that delegates to the unified data loading system to find matching pairs of PSD source files and envelope files. This ensures all file loading uses the centralized system.

**Parameters:**
- `source_directory (str)` - Path to directory containing source PSD files (.mat or .txt)
- `envelope_directory (str)` - Path to directory containing envelope files (.spc.txt)

**Returns:**
- `list[dict]` - List of dictionaries, each containing:
  - `'name' (str)` - Envelope filename
  - `'psd_data' (np.ndarray)` - PSD data with shape (N, 2)
  - `'envelope_data' (np.ndarray)` - Envelope data with shape (M, 2)

**Side Effects:**
- Calls `new_data_loader.find_data_pairs_unified()` which may log warnings for unmatched files

**Error Handling:**
- Delegates error handling to `new_data_loader.find_data_pairs_unified()`

**Used In:**
- `app/tab_visualizer.py` - Called by `load_data_callback()` to load and match files

