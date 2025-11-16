# app/save_utils.py

## Role in the System

`app/save_utils.py` provides functions for saving visualization results, including plot images, data files, and document generation coordination.

## Responsibilities

- Saves matplotlib plots comparing PSD and envelope data
- Creates detailed images with envelope data tables
- Saves envelope data to text files
- Coordinates Word document generation from image directories

## Dependencies

**Imports:**
- `matplotlib.pyplot` - Plot creation
- `numpy` - Array operations
- `os` - File path operations
- `math` - Mathematical operations
- `typing` - Type hints
- `optimizer_core.file_saver.save_results_to_text_file` - Text file saving
- `app.word_generator.create_images_document` - Word document creation

**Used In:**
- `app/tab_visualizer.py` - Uses `save_matplotlib_plot_and_data()`, `generate_word_document_from_images()`

## Functions

### Function: `_create_envelope_with_table_image(envelope_data, output_path, title)`

**Location:** `app/save_utils.py`

**Purpose:**  
Creates a special image containing a log-log plot of the envelope with a table of all envelope points listed below. The table is wrapped into multiple rows and columns if needed. The layout uses a maximum of 4 columns per row (changed from 5) and a figure width of 14.0 inches (increased from 12.8) to accommodate wider value columns.

**Parameters:**
- `envelope_data (np.ndarray)` - Envelope data with shape (N, 2), column 0 = frequencies, column 1 = envelope values
- `output_path (str)` - Full path where the image should be saved
- `title (str)` - Title to display on the plot

**Returns:**
None

**Side Effects:**
- Creates matplotlib figure with dynamic sizing based on number of points:
  - Base figure height: 5.0 (for plot area)
  - Table row height: 4.0 (estimated per row of tables)
  - Total height: base + (table_row_height * num_table_rows)
  - Figure width: 14.0 inches
- Uses GridSpec with max_cols_per_row=4 for table layout
- Saves PNG image to `output_path`
- Closes matplotlib figure to free memory

**Error Handling:**
None (assumes valid input data)

**Used In:**
- `save_matplotlib_plot_and_data()` - Called to create details image

### Function: `save_matplotlib_plot_and_data(original_psd_data, modified_envelope_data, output_filename_base, output_directory, rms_info=None)`

**Location:** `app/save_utils.py`

**Purpose:**  
Saves a dual-view comparison plot (log and linear X-axis) of original PSD and modified envelope, saves envelope data to text file, and creates a details image with table. This is the main saving function for visualization results. Returns the paths of created images for use in document generation.

**Parameters:**
- `original_psd_data (np.ndarray)` - Original PSD data with shape (N, 2)
- `modified_envelope_data (np.ndarray)` - Modified envelope data with shape (M, 2)
- `output_filename_base (str)` - Base filename (without extension) for all output files
- `output_directory (str)` - Directory where files should be saved
- `rms_info (dict, optional)` - Dictionary with keys 'psd_rms', 'env_rms', 'ratio' for legend

**Returns:**
- `tuple[str, str]` - Tuple containing absolute paths to:
  - Main plot image path: `{output_directory}/{output_filename_base}.png`
  - Details plot image path: `{output_directory}/{output_filename_base}_details.png`

**Side Effects:**
- Creates matplotlib figure with two subplots (log and linear X-axis) using `create_dual_axis_psd_subplots()`
- Saves main plot as PNG: `{output_filename_base}.png`
- Saves envelope data as text file: `{output_filename_base}.spc.txt` using `file_saver.save_results_to_text_file()`
- Creates and saves details image: `{output_filename_base}_details.png` using `_create_envelope_with_table_image()`
- Closes matplotlib figures
- Prints save confirmation messages to console

**Error Handling:**
None (assumes valid input data and writable directory)

**Used In:**
- `app/tab_visualizer.py` - Called by `save_changes_callback()` for each data pair. The returned paths are collected and used to generate PowerPoint and Word documents.

### Function: `generate_word_document_from_images(directory)`

**Location:** `app/save_utils.py`

**Purpose:**  
Finds all images in a directory and generates a Word document from them. This function serves as a bridge between file-saving logic and Word document generation.

**Parameters:**
- `directory (str)` - Path to directory containing image files

**Returns:**
None

**Side Effects:**
- Scans directory for image files (.png, .jpg, .jpeg, .gif)
- Calls `word_generator.create_images_document()` with collected image paths
- Prints error messages if directory not found or no images found

**Error Handling:**
- Catches `FileNotFoundError` and prints error message
- Catches general exceptions and prints error message
- Returns early if no images found

**Used In:**
- `app/tab_visualizer.py` - Called by `save_changes_callback()` after saving all images
- `run_code.py` - Called after optimization to create Word documents

