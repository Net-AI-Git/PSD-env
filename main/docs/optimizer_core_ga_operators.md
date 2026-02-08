# optimizer_core/ga_operators.py

## Role in the System

`optimizer_core/ga_operators.py` contains the core evolutionary operators for the genetic algorithm: selection, crossover, and mutations. These functions are responsible for evolving the population of solutions from one generation to the next.

## Responsibilities

- Tournament selection for parent selection
- Multi-point crossover for combining parent solutions
- Various mutation operators (pruning, point removal, point replacement)
- Adaptive mutation rate based on solution quality

## Dependencies

**Imports:**
- `random` - Random number generation
- `numpy` - Array operations

**Used In:**
- `run_code.py` - Uses all operators in the evolution loop

## Functions

### Function: `selection(population, fitness_scores, tournament_size=3)`

**Location:** `optimizer_core/ga_operators.py`

**Purpose:**  
Selects a single parent from the population using tournament selection. In tournament selection, a few individuals are chosen at random, and the one with the best fitness is selected.

**Parameters:**
- `population (list[list[int]])` - Current population of solutions (paths)
- `fitness_scores (list[float])` - Fitness score of each solution
- `tournament_size (int, optional)` - Number of individuals to compete (default: 3)

**Returns:**
- `list[int]` - Winning solution (chromosome) to be used as a parent

**Side Effects:**
None

**Error Handling:**
None (assumes valid input)

**Used In:**
- `run_code.py::process_psd_job()` - Called twice per child generation to select parents

### Function: `crossover_multipoint_paths(parent1, parent2)`

**Location:** `optimizer_core/ga_operators.py`

**Purpose:**  
Creates a child solution by combining segments from two parent solutions. Identifies common points between parents and uses them as crossover points, then splices segments from alternating parents.

**Parameters:**
- `parent1 (list[int])` - First parent solution (list of point indices)
- `parent2 (list[int])` - Second parent solution (list of point indices)

**Returns:**
- `list[int]` - Resulting child solution

**Side Effects:**
None

**Error Handling:**
- Falls back to single-point crossover if not enough common points
- Falls back to random parent selection if no valid crossover points
- Handles ValueError if segment not found in chosen donor (uses fallback donor)

**Used In:**
- `run_code.py::process_psd_job()` - Called once per child generation to create offspring

### Function: `mutate_prune_useless_points(path, **all_ga_params)`

**Location:** `optimizer_core/ga_operators.py`

**Purpose:**  
Intelligently removes points from a path that have negligible impact on area cost. This is a "refinement" mutation that helps reduce the number of points in a solution.

**Parameters:**
- `path (list[int])` - Solution path to be pruned
- `**all_ga_params (dict)` - Dictionary of all GA parameters, must include:
  - `graph` - Valid jumps graph
  - `simplified_points` - Candidate points array
  - `original_psd_freqs` - Original frequency array
  - `original_psd_values` - Original PSD values array
  - `prune_threshold` - Threshold for point removal

**Returns:**
- `list[int]` - New, potentially shorter, path

**Side Effects:**
None

**Error Handling:**
- Checks if direct jump is possible before removing point
- Only removes point if area change is below threshold

**Used In:**
- `run_code.py::process_psd_job()` - Called for elite solutions and percentage of population

### Function: `mutate_remove_point(path, graph)`

**Location:** `optimizer_core/ga_operators.py`

**Purpose:**  
Simple mutation that removes a random point from the path, if valid.

**Parameters:**
- `path (list[int])` - Solution path
- `graph (list[list[int]])` - Valid jumps graph

**Returns:**
- `list[int]` - Mutated path (or original if mutation not possible)

**Side Effects:**
None

**Error Handling:**
- Returns original path if length <= 2 (can't remove from very short paths)
- Checks if direct jump exists before removing point

**Used In:**
- `apply_mutations()` - Called as one mutation option

### Function: `mutate_replace_point(path, graph, candidate_points)`

**Location:** `optimizer_core/ga_operators.py`

**Purpose:**  
Simple mutation that replaces a random point with another valid point.

**Parameters:**
- `path (list[int])` - Solution path
- `graph (list[list[int]])` - Valid jumps graph
- `candidate_points (np.ndarray)` - All candidate points

**Returns:**
- `list[int]` - Mutated path (or original if mutation not possible)

**Side Effects:**
None

**Error Handling:**
- Returns original path if length <= 2
- Tries multiple potential replacements until finding valid one
- Returns original path if no valid replacement found

**Used In:**
- `apply_mutations()` - Called as one mutation option

### Function: `apply_mutations(path, all_ga_params, best_solution_len, adaptive_mutation_threshold)`

**Location:** `optimizer_core/ga_operators.py`

**Purpose:**  
Applies an adaptive number of mutations based on the current best solution's length. Implements "turbo mode" - when solutions get close to target point count, mutation rate increases to encourage exploration and escape local optima.

**Parameters:**
- `path (list[int])` - Child solution to mutate
- `all_ga_params (dict)` - Dictionary of all GA parameters
- `best_solution_len (int)` - Number of points in current best solution
- `adaptive_mutation_threshold (int)` - Threshold to activate "turbo mode"

**Returns:**
- `list[int]` - Mutated path

**Side Effects:**
None

**Error Handling:**
None (mutation functions handle their own errors)

**Used In:**
- `run_code.py::process_psd_job()` - Called for each child that should be mutated (based on MUTATION_RATE)

### Function: `calculate_segment_area_cost_linear(decoded_segment, simplified_points, original_psd_freqs, original_psd_values)`

**Location:** `optimizer_core/ga_operators.py`

**Purpose:**  
Calculates the area cost for a segment using LINEAR calculations. Used by pruning mutation to evaluate point importance.

**Parameters:**
- `decoded_segment (np.ndarray)` - Segment points (decoded from indices)
- `simplified_points (np.ndarray)` - All candidate points (not used, kept for signature consistency)
- `original_psd_freqs (np.ndarray)` - Original frequency array
- `original_psd_values (np.ndarray)` - Original PSD values array

**Returns:**
- `float` - Area cost (difference between envelope and PSD areas)

**Side Effects:**
None

**Error Handling:**
- Returns 0 if segment has less than 2 points
- Returns 0 if no frequencies in range

**Used In:**
- `calculate_segment_area_cost()` - Called when AREA_X_AXIS_MODE is "Linear"

### Function: `calculate_segment_area_cost_log(decoded_segment, simplified_points, original_psd_freqs, original_psd_values)`

**Location:** `optimizer_core/ga_operators.py`

**Purpose:**  
Calculates the area cost for a segment using LOGARITHMIC calculations. Used by pruning mutation to evaluate point importance.

**Parameters:**
- `decoded_segment (np.ndarray)` - Segment points (decoded from indices)
- `simplified_points (np.ndarray)` - All candidate points (not used, kept for signature consistency)
- `original_psd_freqs (np.ndarray)` - Original frequency array
- `original_psd_values (np.ndarray)` - Original PSD values array

**Returns:**
- `float` - Area cost in log space

**Side Effects:**
None

**Error Handling:**
- Returns 0 if segment has less than 2 points
- Returns 0 if no frequencies in range
- Uses epsilon to avoid log(0) errors

**Used In:**
- `calculate_segment_area_cost()` - Called when AREA_X_AXIS_MODE is "Log"

### Function: `calculate_segment_area_cost(decoded_segment, simplified_points, original_psd_freqs, original_psd_values)`

**Location:** `optimizer_core/ga_operators.py`

**Purpose:**  
Main function that calls the appropriate segment area calculation method based on configuration.

**Parameters:**
- `decoded_segment (np.ndarray)` - Segment points
- `simplified_points (np.ndarray)` - All candidate points
- `original_psd_freqs (np.ndarray)` - Original frequency array
- `original_psd_values (np.ndarray)` - Original PSD values array

**Returns:**
- `float` - Area cost

**Side Effects:**
- Imports config module (to avoid circular imports)

**Error Handling:**
None (delegates to linear or log functions)

**Used In:**
- `mutate_prune_useless_points()` - Called to evaluate point importance

