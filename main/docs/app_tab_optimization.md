# app/tab_optimization.py

## Role in the System

`app/tab_optimization.py` creates the "Optimization" tab UI for the Bokeh application. This tab provides controls for configuring and running the genetic algorithm optimization process.

## Responsibilities

- Creates and configures all input widgets (frequency ranges, target points, area ratio, etc.)
- Validates user input before starting optimization
- Manages UI state during optimization (disabling widgets, showing progress)
- Handles optimization execution in a background thread to prevent UI blocking
- Provides stop functionality to terminate running optimizations
- Saves and restores parameter values when optimization is stopped

## Dependencies

**Imports:**
- `bokeh.models.widgets` - UI widgets (Button, Spinner, RadioButtonGroup, TextInput, Div, Slider, Checkbox, Paragraph)
- `bokeh.layouts.column` - Layout containers
- `bokeh.plotting.curdoc` - Bokeh document context for thread-safe UI updates
- `threading` - For background thread execution
- `os` - For directory path validation
- `run_code.run_optimization_process` - Main optimization execution function

**Used In:**
- `app/gui.py` - Imported and called to create the optimization tab

## Functions

### Function: `create_optimization_tab()`

**Location:** `app/tab_optimization.py`

**Purpose:**  
Creates and returns the complete layout for the Optimization tab, including all widgets, callbacks, and event handlers.

**Parameters:**  
None

**Returns:**  
- `bokeh.layouts.Column` - A Bokeh layout containing all optimization controls

**Side Effects:**  
- Creates Bokeh widget objects
- Sets up callback functions that will be triggered by user interactions
- Uses global `stop_event` (threading.Event) to coordinate with optimization process

**Error Handling:**  
- Input validation prevents invalid parameters from being passed to optimization
- Exception handling in background thread catches optimization errors and updates UI status
- Directory existence is validated before optimization starts

**Used In:**
- `app/gui.py` - Called during application initialization to create the tab

### Function: `validate_inputs()`

**Location:** `app/tab_optimization.py` (nested inside `create_optimization_tab`)

**Purpose:**  
Validates all user input parameters to ensure they are within acceptable ranges and logically consistent.

**Parameters:**  
None (accesses widget values via closure)

**Returns:**  
- `tuple[bool, str]` - (is_valid, error_message) tuple

**Side Effects:**  
None

**Error Handling:**  
- Catches exceptions during validation and returns error message
- Validates: frequency range (min < max), target points (> 0), area ratio (> 0), directory existence

**Used In:**
- `run_optimization_callback()` - Called before starting optimization

### Function: `run_optimization_callback()`

**Location:** `app/tab_optimization.py` (nested inside `create_optimization_tab`)

**Purpose:**  
Main callback function executed when user clicks "Run Optimization". Validates inputs, saves current parameters, starts optimization in background thread, and updates UI state.

**Parameters:**  
None (accesses widget values via closure)

**Returns:**  
None

**Side Effects:**  
- Sets `stop_event.clear()` to allow optimization to run
- Disables all input widgets
- Hides "Run" button, shows "Stop" button
- Updates status and progress messages
- Starts background thread running `run_in_background()`
- Calls `run_code.run_optimization_process()` with all parameters including `strict_points`

**Error Handling:**  
- Input validation errors are caught and displayed in status
- Optimization exceptions are caught in background thread and displayed to user
- UI state is always restored (widgets re-enabled, buttons toggled) even on error

**Used In:**
- Connected to `run_button.on_click()` event handler

### Function: `stop_optimization_callback()`

**Location:** `app/tab_optimization.py` (nested inside `create_optimization_tab`)

**Purpose:**  
Handles user request to stop running optimization. Signals stop event, restores UI state, and resets parameters to saved values.

**Parameters:**  
None

**Returns:**  
None

**Side Effects:**  
- Sets `stop_event.set()` to signal optimization to stop
- Restores all widget values from `saved_parameters`
- Re-enables all input widgets
- Shows "Run" button, hides "Stop" button
- Updates status message

**Error Handling:**  
None (operation is always safe)

**Used In:**
- Connected to `stop_button.on_click()` event handler

### Function: `save_current_parameters()`

**Location:** `app/tab_optimization.py` (nested inside `create_optimization_tab`)

**Purpose:**  
Captures current state of all input widgets into a dictionary for later restoration.

**Parameters:**  
None (accesses widget values via closure)

**Returns:**  
- `dict` - Dictionary containing all widget values:
  - `min_freq`, `max_freq`, `target_points`, `target_area_ratio`
  - `stab_wide`, `area_x_axis_mode` (as active indices)
  - `full_envelope`, `strict_points` (as booleans)
  - `input_dir` (as string)

**Note:** The `strict_points` parameter is saved and restored to maintain UI state consistency.

**Side Effects:**  
None

**Error Handling:**  
None

**Used In:**
- `run_optimization_callback()` - Called before starting optimization
- `stop_optimization_callback()` - Uses saved parameters to restore UI

### Function: `restore_parameters(params)`

**Location:** `app/tab_optimization.py` (nested inside `create_optimization_tab`)

**Purpose:**  
Restores all widget values from a saved parameters dictionary.

**Parameters:**  
- `params (dict)` - Dictionary containing widget values (from `save_current_parameters()`)

**Returns:**  
None

**Side Effects:**  
- Updates all widget values to match saved parameters (including `strict_points`)
- Modifies UI state

**Error Handling:**  
None (assumes valid parameter dictionary structure)

**Used In:**
- `stop_optimization_callback()` - Restores UI after stopping optimization

### Function: `update_status(message, color)`

**Location:** `app/tab_optimization.py` (nested inside `create_optimization_tab`)

**Purpose:**  
Updates the status display div with a colored message.

**Parameters:**  
- `message (str)` - Status message text
- `color (str)` - CSS color name (e.g., "blue", "red", "green", "orange")

**Returns:**  
None

**Side Effects:**  
- Updates `status_div.text` with HTML-formatted message

**Error Handling:**  
None

**Used In:**
- `run_optimization_callback()` - Sets "in progress" status
- `stop_optimization_callback()` - Sets "stopped" status
- Background thread success/error handlers - Sets completion/error status

### Function: `set_widgets_enabled(enabled)`

**Location:** `app/tab_optimization.py` (nested inside `create_optimization_tab`)

**Purpose:**  
Enables or disables all input widgets to prevent user interaction during optimization.

**Parameters:**  
- `enabled (bool)` - True to enable widgets, False to disable

**Returns:**  
None

**Side Effects:**  
- Sets `disabled` property on all input widgets (min_freq_input, max_freq_input, target_points_input, strict_points_input, etc.)
- Sets `disabled` property on run_button

**Error Handling:**  
None

**Used In:**
- `run_optimization_callback()` - Disables widgets when starting
- Background thread success/error handlers - Re-enables widgets when finished
- `stop_optimization_callback()` - Re-enables widgets when stopped

## Global Variables

### `stop_event`
**Type:** `threading.Event`  
**Purpose:** Global event object used to signal optimization process to stop. Set by `stop_optimization_callback()`, checked by `run_code.py` optimization loop.

### `saved_parameters`
**Type:** `dict`  
**Purpose:** Stores widget values before optimization starts, used to restore UI state when optimization is stopped. Includes all parameters including `strict_points`.

