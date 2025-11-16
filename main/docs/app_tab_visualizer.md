# app/tab_visualizer.py

## Role in the System

`app/tab_visualizer.py` creates the "PSD Visualizer" tab UI for the Bokeh application. This tab provides an interactive interface for loading, viewing, editing, and saving PSD data and envelope files.

## Responsibilities

- Loads and pairs PSD source files with envelope files from user-specified directories
- Displays interactive Bokeh plots comparing original PSD data with envelopes
- Allows manual editing of envelope points (drag, add, delete)
- Applies uncertainty and safety factors to envelopes with real-time preview
- Saves modified envelopes and generates reports (PowerPoint, Word)

## Dependencies

**Imports:**
- `bokeh.models.widgets` - UI widgets (Button, TextInput, Div, Spinner)
- `bokeh.layouts` - Layout containers (column, row)
- `numpy` - Array operations
- `math.log10` - Logarithmic calculations for dB conversion
- `os` - Directory and file path operations
- `app.gui_utils.find_data_pairs, create_psd_plot, _calculate_rms` - GUI utilities
- `app.save_utils.save_matplotlib_plot_and_data, generate_word_document_from_images` - File saving
- `app.powerpoint_generator.create_presentation_from_images` - PowerPoint generation

**Used In:**
- `app/gui.py` - Imported and called to create the visualizer tab

## Classes

### Class: `VisualizerTab`

**Location:** `app/tab_visualizer.py`

**Purpose:**  
Manages the complete state and UI for the PSD Visualizer tab. Encapsulates all widgets, data, and callback functions.

**Instance Variables:**
- `data_pairs (list[dict])` - List of matched PSD/envelope pairs, each containing 'name', 'psd_data', 'envelope_data'
- `current_index (int)` - Index of currently displayed data pair
- `graph_modifications (dict[int, np.ndarray])` - Stores manually edited envelope data by index
- `plots (list)` - References to Bokeh plot objects for direct manipulation
- `plot_layout (bokeh.layouts.Column)` - Container for plot widgets
- `_is_updating_factors (bool)` - Flag to prevent callback loops during factor conversion

**Methods:**

#### Method: `__init__(self)`

**Purpose:**  
Initializes the VisualizerTab instance, creating all UI widgets and setting up callback connections.

**Parameters:**  
None

**Returns:**  
None

**Side Effects:**  
- Creates all Bokeh widget objects
- Sets up callback connections for buttons and factor inputs
- Initializes instance variables to default values

**Error Handling:**  
None (initialization only)

**Used In:**
- `create_visualizer_tab()` - Called when creating tab instance

#### Method: `_on_envelope_change(self, attr, old, new)`

**Purpose:**  
Callback function triggered when user edits envelope points in the Bokeh plot. Stores modifications and updates legend with new RMS values.

**Parameters:**  
- `attr (str)` - Attribute name that changed (always 'data' for this callback)
- `old (dict)` - Previous data source values
- `new (dict)` - New data source values with keys 'freq' and 'val'

**Returns:**  
None

**Side Effects:**  
- Stores modified envelope data in `self.graph_modifications[self.current_index]`
- Updates legend labels on all plot objects with new RMS values
- Updates status message to indicate manual edits are active
- Enables save button if modifications exist

**Error Handling:**  
None (assumes valid data structure from Bokeh)

**Used In:**
- Passed as `on_change_callback` parameter to `gui_utils.create_psd_plot()`
- Called automatically by Bokeh when plot data source changes

#### Method: `uncertainty_ratio_to_db(self, attr, old, new)`

**Purpose:**  
Converts uncertainty factor from ratio to dB when ratio input changes. Prevents infinite callback loops using flag.

**Parameters:**  
- `attr (str)` - Attribute name ('value')
- `old (float)` - Previous value
- `new (float)` - New value (ratio)

**Returns:**  
None

**Side Effects:**  
- Updates `self.uncertainty_db_input.value` with converted dB value
- Calls `update_plot_and_controls()` to refresh plot
- Uses `_is_updating_factors` flag to prevent loop

**Error Handling:**  
- Checks `new > 0` before conversion to avoid log(0) errors

**Used In:**
- Connected to `self.uncertainty_input.on_change('value', ...)` callback

#### Method: `uncertainty_db_to_ratio(self, attr, old, new)`

**Purpose:**  
Converts uncertainty factor from dB to ratio when dB input changes.

**Parameters:**  
- `attr (str)` - Attribute name ('value')
- `old (float)` - Previous value
- `new (float)` - New value (dB)

**Returns:**  
None

**Side Effects:**  
- Updates `self.uncertainty_input.value` with converted ratio value
- Calls `update_plot_and_controls()` to refresh plot
- Uses `_is_updating_factors` flag to prevent loop

**Error Handling:**  
None (dB values can be negative)

**Used In:**
- Connected to `self.uncertainty_db_input.on_change('value', ...)` callback

#### Method: `safety_ratio_to_db(self, attr, old, new)`

**Purpose:**  
Converts safety factor from ratio to dB when ratio input changes.

**Parameters:**  
- `attr (str)` - Attribute name ('value')
- `old (float)` - Previous value
- `new (float)` - New value (ratio)

**Returns:**  
None

**Side Effects:**  
- Updates `self.safety_db_input.value` with converted dB value
- Calls `update_plot_and_controls()` to refresh plot
- Uses `_is_updating_factors` flag to prevent loop

**Error Handling:**  
- Checks `new > 0` before conversion to avoid log(0) errors

**Used In:**
- Connected to `self.safety_input.on_change('value', ...)` callback

#### Method: `safety_db_to_ratio(self, attr, old, new)`

**Purpose:**  
Converts safety factor from dB to ratio when dB input changes.

**Parameters:**  
- `attr (str)` - Attribute name ('value')
- `old (float)` - Previous value
- `new (float)` - New value (dB)

**Returns:**  
None

**Side Effects:**  
- Updates `self.safety_input.value` with converted ratio value
- Calls `update_plot_and_controls()` to refresh plot
- Uses `_is_updating_factors` flag to prevent loop

**Error Handling:**  
None (dB values can be negative)

**Used In:**
- Connected to `self.safety_db_input.on_change('value', ...)` callback

#### Method: `save_changes_callback(self)`

**Purpose:**  
Saves all modified envelopes, applying factors and manual edits, to disk. Creates both "with factors" and "without factors" versions if factors are applied. Generates PowerPoint and Word documents.

**Parameters:**  
None (accesses instance variables)

**Returns:**  
None

**Side Effects:**  
- Creates output directories based on envelope source directory
- Saves PNG images and .spc.txt files for each data pair
- Applies uncertainty and safety factors (factor^2 to PSD values)
- Creates PowerPoint presentations using `create_presentation_from_images()`
- Creates Word documents using `generate_word_document_from_images()`
- Updates status message with save location links

**Error Handling:**  
- Checks if there are changes to save before proceeding
- Catches exceptions during save and displays error in status
- Validates that data_pairs exist before saving

**Used In:**
- Connected to `self.save_button.on_click()` callback

#### Method: `update_suffix_preview(self)`

**Purpose:**  
Updates the file suffix preview field based on current uncertainty and safety factor values. Shows what suffix will be added to filenames.

**Parameters:**  
None (accesses widget values via instance)

**Returns:**  
None

**Side Effects:**  
- Updates `self.suffix_preview_input.value` with suffix string (e.g., "SF1.20 uncertainty1.10")
- Only includes factors that are not 1.0

**Error Handling:**  
None

**Used In:**
- Called by `update_plot_and_controls()` after factor changes
- Called during initialization

#### Method: `update_plot_and_controls(self)`

**Purpose:**  
Refreshes the plot display with current data pair, applying factors and manual edits. Updates navigation buttons and status.

**Parameters:**  
None (accesses instance variables)

**Returns:**  
None

**Side Effects:**  
- Replaces `self.plot_layout.children` with new plot
- Updates `self.plots` with new plot references
- Applies uncertainty and safety factors to envelope data (factor^2)
- Updates status message with current pair information
- Enables/disables Previous/Next buttons based on index
- Updates suffix preview
- Enables/disables save button based on modifications

**Error Handling:**  
- Checks if data_pairs exist, shows message if empty
- Handles case where current_index is out of bounds

**Used In:**
- `load_data_callback()` - After loading data
- `show_previous_callback()` - After navigating
- `show_next_callback()` - After navigating
- Factor conversion callbacks - After factor changes

#### Method: `load_data_callback(self)`

**Purpose:**  
Loads PSD and envelope files from user-specified directories, finds matching pairs, and displays the first pair.

**Parameters:**  
None (reads widget values via instance)

**Returns:**  
None

**Side Effects:**  
- Calls `find_data_pairs()` to load and match files
- Resets `self.current_index` to 0
- Clears `self.graph_modifications`
- Clears `self.plots`
- Updates status message
- Calls `update_plot_and_controls()` to display data

**Error Handling:**  
- Validates that both directory paths are provided
- Shows error message if directories are empty or invalid

**Used In:**
- Connected to `self.load_button.on_click()` callback

#### Method: `show_previous_callback(self)`

**Purpose:**  
Navigates to the previous data pair in the list.

**Parameters:**  
None

**Returns:**  
None

**Side Effects:**  
- Decrements `self.current_index` if > 0
- Calls `update_plot_and_controls()` to refresh display

**Error Handling:**  
- Checks bounds before decrementing

**Used In:**
- Connected to `self.prev_button.on_click()` callback

#### Method: `show_next_callback(self)`

**Purpose:**  
Navigates to the next data pair in the list.

**Parameters:**  
None

**Returns:**  
None

**Side Effects:**  
- Increments `self.current_index` if < len(data_pairs) - 1
- Calls `update_plot_and_controls()` to refresh display

**Error Handling:**  
- Checks bounds before incrementing

**Used In:**
- Connected to `self.next_button.on_click()` callback

#### Method: `get_layout(self)`

**Purpose:**  
Assembles and returns the complete tab layout with all widgets arranged.

**Parameters:**  
None

**Returns:**  
- `bokeh.layouts.Column` - Complete tab layout

**Side Effects:**  
None

**Error Handling:**  
None

**Used In:**
- `create_visualizer_tab()` - Called to get layout for tab panel

## Functions

### Function: `create_visualizer_tab()`

**Location:** `app/tab_visualizer.py`

**Purpose:**  
Factory function that creates a VisualizerTab instance and returns its layout. This is the entry point called by `app/gui.py`.

**Parameters:**  
None

**Returns:**  
- `bokeh.layouts.Column` - The complete visualizer tab layout

**Side Effects:**  
- Creates VisualizerTab instance (which creates all widgets)

**Error Handling:**  
None

**Used In:**
- `app/gui.py` - Called during application initialization to create the tab

