# optimizer_core/problem_definition_points.py

## Role in the System

`optimizer_core/problem_definition_points.py` defines the specifics of the PSD envelope optimization problem. It contains the fitness function (how to evaluate a solution), which is the core of the genetic algorithm's optimization process.

## Responsibilities

- Calculates cost and fitness for solution paths
- Implements multi-objective optimization (area ratio and point count)
- Supports both Linear and Log X-axis modes for area calculation
- Handles low-frequency area weighting

## Dependencies

**Imports:**
- `numpy` - Array operations and integration
- `random` - Random number generation (not used in this file, but imported for consistency)
- `time` - Timing (not used in this file)
- `optimizer_core.config` - Configuration parameters
- `optimizer_core.problem_definition_base` - Base functions (imports all: `is_segment_valid`, `build_valid_jumps_graph`, `create_random_solution`, `prune_dead_end_nodes`)

**Used In:**
- `run_code.py` - Uses `calculate_metrics()` to evaluate solutions and `build_valid_jumps_graph()`, `prune_dead_end_nodes()`, `create_random_solution()` from base module

## Functions

### Function: `calculate_metrics(path, simplified_points, original_psd_freqs, original_psd_values, target_area_ratio, target_points, X_AXIS_MODE, **kwargs)`

**Location:** `optimizer_core/problem_definition_points.py`

**Purpose:**  
Calculates the cost and fitness of a given solution path. Computes a multi-objective cost, prioritizing the tightness of the envelope (measured by area ratio) while also penalizing solutions that deviate from the target number of points. Supports both Linear and Log scaling for the X-axis during area calculation.

**Parameters:**
- `path (list[int])` - List of indices representing points in the solution from `simplified_points` array
- `simplified_points (np.ndarray)` - Array of all candidate points for the envelope
- `original_psd_freqs (np.ndarray)` - Frequency values of the original signal
- `original_psd_values (np.ndarray)` - PSD values of the original signal
- `target_area_ratio (float)` - Target for the linear area ratio
- `target_points (int)` - Desired number of points for the envelope
- `X_AXIS_MODE (str)` - X-axis scale for area integration: 'Log' or 'Linear'
- `**kwargs` - Additional parameters, must include `POINTS_WEIGHT (float)` - weight for points error component

**Returns:**
- `tuple[float, float, int, float]` - Tuple containing:
  - `total_cost (float)` - Combined cost of the solution
  - `fitness (float)` - Fitness score (higher is better)
  - `len(path) (int)` - Number of points in the solution
  - `linear_area_ratio (float)` - Calculated area ratio in linear space

**Side Effects:**
None

**Error Handling:**
- Returns `(float('inf'), 0, 0, float('inf'))` if path is empty or has less than 2 points
- Raises `ValueError` if `POINTS_WEIGHT` not provided in kwargs
- Handles division by zero in area ratio calculations (returns `float('inf')`)
- Uses epsilon to avoid log(0) errors

**Used In:**
- `run_code.py::process_psd_job()` - Called for every solution in every generation to evaluate fitness

## Algorithm Details

The function implements a multi-objective cost function with three components:

1. **Area Cost (Log-based)**: Calculated in log space with optional low-frequency weighting. The envelope and original PSD are shifted to be non-negative, then integrated. The ratio of envelope area to original area is computed.

2. **Area Cost (Linear-based)**: Calculated in linear space as a secondary constraint. Provides an additional area ratio metric.

3. **Points Penalty**: Penalizes deviation from target point count using squared relative error.

The total cost combines these with weights:
- `AREA_WEIGHT * area_error` (log-based)
- `AREA_WEIGHT_LINEAR * linear_area_error`
- `POINTS_WEIGHT * points_error`

Fitness is calculated as `1.0 / (1.0 + total_cost)` to convert cost (lower is better) to fitness (higher is better).

