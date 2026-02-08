# optimizer_core/problem_definition_base.py

## Role in the System

`optimizer_core/problem_definition_base.py` contains common, shared logic for problem definitions, including constraint validation, graph creation, and base solution generation. This module provides reusable components that can be used by different problem-specific implementations.

## Responsibilities

- Validates line segments to ensure they don't intersect the original PSD
- Builds directed acyclic graph of valid "jumps" between candidate points
- Prunes dead-end nodes from the graph
- Creates random valid solutions (paths through the graph)

## Dependencies

**Imports:**
- `numpy` - Array operations and interpolation
- `random` - Random number generation
- `time` - Timing measurements
- `optimizer_core.config` - Configuration parameters
- `utils.logger` - Logging

**Used In:**
- `optimizer_core/problem_definition_points.py` - Imports all functions from this module
- `run_code.py` - Uses `build_valid_jumps_graph()`, `prune_dead_end_nodes()`, `create_random_solution()`

## Functions

### Function: `is_segment_valid(p1, p2, original_psd_freqs, original_psd_values)`

**Location:** `optimizer_core/problem_definition_base.py`

**Purpose:**  
Robustly checks if a line segment between two points is valid (doesn't intersect the original PSD signal). The check is performed in semi-log space (linear X-axis, logarithmic Y-axis) to perfectly match the visual representation and guarantee no visual intersections are missed.

**Parameters:**
- `p1 (tuple)` - (x, y) coordinates of the starting point
- `p2 (tuple)` - (x, y) coordinates of the ending point
- `original_psd_freqs (np.ndarray)` - Frequency points of the original PSD
- `original_psd_values (np.ndarray)` - Amplitude values of the original PSD

**Returns:**
- `bool` - True if segment is valid (doesn't intersect), False otherwise

**Side Effects:**
None

**Error Handling:**
- Returns True if not enough points to form a line (edge case)
- Uses epsilon to avoid log(0) errors
- Uses tolerance for floating-point comparisons

**Used In:**
- `build_valid_jumps_graph()` - Called for each potential jump to validate it

### Function: `build_valid_jumps_graph(simplified_points, original_psd_freqs, original_psd_values)`

**Location:** `optimizer_core/problem_definition_base.py`

**Purpose:**  
Pre-computes a directed acyclic graph of all valid "jumps" between candidate points. This is a critical pre-computation step that significantly speeds up the evolutionary process. The result is an adjacency list where graph[i] contains all indices j > i such that the segment from point i to point j is valid.

**Parameters:**
- `simplified_points (np.ndarray)` - Pool of candidate points for the envelope
- `original_psd_freqs (np.ndarray)` - Frequency points of the original PSD
- `original_psd_values (np.ndarray)` - Amplitude values of the original PSD

**Returns:**
- `list[list[int]]` - Adjacency list representation of the valid jumps graph

**Side Effects:**
- Logs graph building start and completion with timing information
- Uses "smart break" optimization: stops checking if many consecutive jumps are invalid (based on `config.BREAK_THRESHOLD`)

**Error Handling:**
None (assumes valid input arrays)

**Used In:**
- `run_code.py::process_psd_job()` - Called once per optimization job to build graph

### Function: `prune_dead_end_nodes(graph)`

**Location:** `optimizer_core/problem_definition_base.py`

**Purpose:**  
Prunes the graph by iteratively removing nodes that have no outgoing connections (dead-end nodes). This process repeats until no more dead-ends exist, ensuring all paths can reach the final node. The result is a cleaner, more efficient graph for the GA to traverse.

**Parameters:**
- `graph (list[list[int]])` - Adjacency list of the graph

**Returns:**
- `list[list[int]]` - The pruned graph

**Side Effects:**
- Modifies the input graph in place (removes connections)
- Logs pruning progress and statistics

**Error Handling:**
None (assumes valid graph structure)

**Used In:**
- `run_code.py::process_psd_job()` - Called after building graph to clean it up

### Function: `create_random_solution(graph, target_points)`

**Location:** `optimizer_core/problem_definition_base.py`

**Purpose:**  
Creates a single, valid random solution (path) through the pre-computed graph. The choice of the next node is weighted to prefer shorter jumps, encouraging initial solutions to follow the PSD shape more closely rather than making large, unrealistic jumps.

**Parameters:**
- `graph (list[list[int]])` - Valid jumps graph (adjacency list)
- `target_points (int)` - Target number of points (used for initial guidance, though not strictly enforced)

**Returns:**
- `list[int]|None` - List of node indices representing a valid path, or None if a valid path to the end could not be found

**Side Effects:**
None

**Error Handling:**
- Returns None if no valid forward jumps available (invalid solution)
- Ensures path is strictly increasing and ends at the last node

**Used In:**
- `run_code.py::process_psd_job()` - Called repeatedly to create initial population

