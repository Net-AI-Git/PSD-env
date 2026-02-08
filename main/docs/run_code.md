# run_code.py

## Role in the System

`run_code.py` is the main execution script that orchestrates the entire PSD optimization process. It coordinates data loading, genetic algorithm execution, multiprocessing, and result saving.

## Responsibilities

- Configures optimization parameters from user input
- Loads PSD data from input directory
- Runs genetic algorithm optimization for each measurement
- Manages multiprocessing for parallel job execution
- Handles stop events for user-initiated termination
- Saves optimization results and generates reports
- Creates envelope data when in FULL_ENVELOPE mode

## Dependencies

**Imports:**
- `time` - Timing measurements
- `numpy` - Array operations
- `random` - Random number generation
- `os` - File and directory operations
- `sys` - System operations
- `threading` - Thread management for stop event monitoring
- `multiprocessing` - Parallel job processing
- `multiprocessing.Manager` - Shared objects for multiprocessing
- `typing.Literal` - Type hints
- `optimizer_core.config` - Configuration parameters
- `optimizer_core.psd_utils` - PSD utilities
- `optimizer_core.ga_operators` - Genetic operators
- `optimizer_core.new_data_loader` - Data loading
- `optimizer_core.problem_definition_points` - Problem definition
- `utils.logger` - Logging
- `app.powerpoint_generator` - PowerPoint generation
- `app.save_utils` - Word document generation

**Used In:**
- `app/tab_optimization.py` - Calls `run_optimization_process()` to start optimization

## Functions

### Function: `process_job_wrapper(job_data)`

**Location:** `run_code.py`

**Purpose:**  
Wrapper function for multiprocessing pool to process a single job. Multiprocessing.Pool.map() requires a function that takes a single argument, so this wrapper unpacks the job data tuple and updates the local config module with values from config_dict.

**Parameters:**
- `job_data (tuple)` - Tuple containing (job, output_directory, config_dict, mp_stop_event)

**Returns:**
None

**Side Effects:**
- Updates local config module with values from config_dict (necessary because other modules access config directly)
- This includes POINTS_WEIGHT which is calculated in `run_optimization_process()` based on strict_points
- Calls `process_psd_job()` with unpacked parameters
- Logs errors if job fails

**Error Handling:**
- Catches exceptions during job processing and logs error with traceback
- Does not raise exceptions (allows other jobs to continue)

**Used In:**
- `main()` - Passed to multiprocessing.Pool.map() for parallel execution

### Function: `process_psd_job(job, output_directory, config_dict, stop_event=None)`

**Location:** `run_code.py`

**Purpose:**  
Runs the complete genetic algorithm optimization for a single measurement job. This is the core optimization function that executes the full GA process from candidate generation to final solution.

**Parameters:**
- `job (dict)` - Dictionary containing measurement data and metadata (frequencies, psd_values, output_filename_base)
- `output_directory (str)` - Directory to save results in
- `config_dict (dict)` - Dictionary containing all configuration values needed for optimization
- `stop_event (threading.Event|multiprocessing.Event, optional)` - Event to signal stop request

**Returns:**
None

**Side Effects:**
- Generates candidate points using `psd_utils.create_multi_scale_envelope()`
- Builds valid jumps graph using `problem.build_valid_jumps_graph()`
- Creates initial population of random solutions
- Runs evolution loop with selection, crossover, and mutation
- Evaluates solutions using `problem.calculate_metrics()`
- Saves final solution using `psd_utils.plot_final_solution()`
- Logs progress and timing information
- Checks stop_event periodically to allow graceful termination

**Error Handling:**
- Returns early if job has no data
- Returns early if no valid solutions can be generated
- Checks stop_event at multiple points to allow termination
- Handles empty population gracefully
- Logs warnings for edge cases (smaller than desired population, etc.)

**Used In:**
- `process_job_wrapper()` - Called for each job in multiprocessing pool
- Can be called directly for single-job execution

### Function: `main(stop_event=None, config_dict=None)`

**Location:** `run_code.py`

**Purpose:**  
Main batch processing function with multiprocessing support. Manages directories, loads all jobs, and processes them in parallel using multiprocessing pool.

**Parameters:**
- `stop_event (threading.Event, optional)` - Event to signal stop request (from GUI)
- `config_dict (dict, optional)` - Dictionary containing all configuration values. If None, created from current config module state.

**Returns:**
None

**Side Effects:**
- Creates output directory if it doesn't exist
- Loads all jobs from input directory (either file-by-file or full envelope mode)
- Creates subdirectories for envelope results if in FULL_ENVELOPE mode
- Creates comparison plots for envelopes if in FULL_ENVELOPE mode
- Processes jobs in parallel using multiprocessing.Pool
- Creates PowerPoint and Word documents from results
- Logs progress and completion messages
- Syncs stop_event from GUI to multiprocessing using monitor thread

**Error Handling:**
- Validates input directory exists
- Handles empty job lists gracefully
- Creates config_dict from config module if not provided (for backward compatibility)
- Handles multiprocessing stop event synchronization

**Used In:**
- `run_optimization_process()` - Called after configuration setup

### Function: `run_optimization_process(min_frequency_hz, max_frequency_hz, target_points, target_area_ratio, stab_wide, area_x_axis_mode, input_dir, full_envelope, stop_event, strict_points)`

**Location:** `run_code.py`

**Purpose:**  
Sets up the configuration and runs the entire PSD optimization process. This function serves as the main entry point for running optimization with specific parameters, allowing programmatic execution without manually editing configuration files.

**Parameters:**
- `min_frequency_hz (int)` - Minimum frequency for data filtering
- `max_frequency_hz (int)` - Maximum frequency for data filtering
- `target_points (int)` - Target number of points for the envelope
- `target_area_ratio (float)` - Target area ratio (will be squared internally)
- `stab_wide (Literal["narrow", "wide"])` - Stability analysis mode, affects WINDOW_SIZES
- `area_x_axis_mode (Literal["Log", "Linear"])` - X-axis domain for area integration
- `input_dir (str, optional)` - Overrides default input directory. If None, uses config.INPUT_DIR
- `full_envelope (bool)` - If True, creates envelope from maximum PSD values across matching channels
- `stop_event (threading.Event, optional)` - Event to signal stop request
- `strict_points (bool)` - If True, sets POINTS_WEIGHT to 80.0 for strict points constraint. If False, uses default POINTS_WEIGHT value (2.5). Defaults to False.

**Returns:**
None

**Side Effects:**
- Updates config module with all parameter values
- Calculates POINTS_WEIGHT based on strict_points parameter:
  - If `strict_points=True`: sets `points_weight = 80.0` (strict constraint)
  - If `strict_points=False`: sets `points_weight = 2.5` (default)
- Sets WINDOW_SIZES and ENRICH_LOW_FREQUENCIES based on stab_wide
- Creates config_dict for multiprocessing, including the calculated POINTS_WEIGHT value
- Calls `main()` to execute optimization
- Logs configuration summary including strict_points setting

**Error Handling:**
- Uses provided input_dir or falls back to config.INPUT_DIR
- Calculates POINTS_WEIGHT based on strict_points parameter (no error handling needed, boolean check)

**Used In:**
- `app/tab_optimization.py` - Called when user clicks "Run Optimization" button
- Can be called directly from command line (if __name__ == "__main__")

## Execution Flow

1. **Configuration**: `run_optimization_process()` sets up all config values
2. **Data Loading**: `main()` loads all jobs from input directory
3. **Job Processing**: For each job, `process_psd_job()` runs:
   - Candidate point generation
   - Graph building
   - Initial population creation
   - Evolution loop (selection, crossover, mutation, evaluation)
   - Final solution saving
4. **Report Generation**: After all jobs complete, PowerPoint and Word documents are created

## Multiprocessing Details

- Uses `multiprocessing.Pool` with number of processes = CPU count
- Each worker process gets its own config_dict to avoid module state issues
- Stop events are synchronized between GUI thread and multiprocessing using a monitor thread
- Manager.Event is used for shared stop event across processes

