# optimizer_core/config.py

## Role in the System

`optimizer_core/config.py` centralizes all tunable hyperparameters and configuration values for the genetic algorithm optimization process. This allows easy experimentation and tuning without modifying core algorithm logic.

## Responsibilities

- Defines all configuration parameters with default values
- Provides single source of truth for optimization settings
- Documents parameter purposes and valid ranges

## Dependencies

**Imports:**
None (pure configuration file)

**Used In:**
- All `optimizer_core/` modules import and use config values
- `run_code.py` reads and updates config values before optimization

## Configuration Parameters

### File and Directory Settings
- `INPUT_DIR (str)` - Directory for input PSD data files (default: "input")
- `OUTPUT_DIR (str)` - Directory where output result images will be saved (default: "results")
- `INPUT_FILE_EXTENSION (str)` - File extension to look for (default: ".txt")
- `FULL_ENVELOPE (bool)` - Enable full envelope mode - loads all files and creates envelope from maximum values (default: False)

### Frequency Filtering Settings
- `MIN_FREQUENCY_HZ (int|None)` - Minimum frequency for data filtering in Hz (default: None, set at runtime)
- `MAX_FREQUENCY_HZ (int|None)` - Maximum frequency for data filtering in Hz (default: None, set at runtime)

### Optimization Strategy Settings
- `OPTIMIZATION_MODE (str)` - Controls optimization strategy: "points" (minimize area while targeting point count) or "area" (minimize points while targeting area ratio) (default: "points")
- `TARGET_POINTS (int|None)` - Ideal number of points for final envelope (default: None, set at runtime)
- `TARGET_AREA_RATIO (float|None)` - Target area ratio between envelope and original PSD (default: None, set at runtime)
- `AREA_WEIGHT (float)` - Weight for area error component in cost function (default: 120.0)
- `AREA_WEIGHT_LINEAR (float)` - Weight for linear area error component (default: 100.0)
- `POINTS_WEIGHT (float)` - Weight for points error component in cost function (default: 2.5). **Note:** This value can be dynamically overridden by the `strict_points` parameter in `run_code.py`. When `strict_points=True`, POINTS_WEIGHT is set to 80.0 for strict points constraint. When `strict_points=False`, the default value of 2.5 is used. The actual value used during optimization is passed in `config_dict` to worker processes to avoid multiprocessing issues.

### Candidate Point Generation Settings
- `WINDOW_SIZES (list[int])` - List of window sizes for multi-scale candidate point generation (default: [10, 20, 30])
- `LIFT_FACTOR (float)` - Factor for lifting points to enrich search space (default: 1.3)
- `ENRICH_LOW_FREQUENCIES (bool)` - Add all original PSD points below threshold to candidate pool (default: True)
- `LOW_FREQ_ENRICHMENT_FACTORS (list[float])` - Scaling factors for lifted low-frequency points (default: [1.2, 1.5])
- `LOW_FREQUENCY_THRESHOLD (float)` - Frequency in Hz below which points are enriched (default: 100.0)

### Genetic Algorithm Core Settings
- `POPULATION_SIZE (int)` - Number of individual solutions in each generation (default: 1000)
- `MAX_GENERATIONS (int)` - Maximum number of generations (default: 2000)
- `MUTATION_RATE (float)` - Probability that a child solution will undergo mutation (default: 0.9)
- `ELITISM_SIZE (int)` - Number of best solutions carried over to next generation (default: 2)

### Area Integration Settings
- `AREA_X_AXIS_MODE (str|None)` - X-axis domain for area integration: "Linear" or "Log" (default: None, set at runtime)
- `LOW_FREQ_AREA_WEIGHT (float)` - Weight for area cost in low-frequency region (default: 1.0)

### Advanced Mutation Strategy Settings
- `PRUNE_THRESHOLD (float)` - Relative area cost change below which a point is considered "useless" (default: 0.05)
- `PRUNE_PERCENTAGE_OF_POPULATION (float)` - Percentage of non-elite population to apply pruning mutation (default: 0.1)
- `ADAPTIVE_MUTATION_THRESHOLD (int)` - Point count threshold below which "turbo mode" activates (default: 80)
- `BREAK_THRESHOLD (int)` - Stop checking if this many consecutive invalid jumps found during graph building (default: 100)

### Termination Criteria Settings
- `USE_CONVERGENCE_TERMINATION (bool)` - Enable early stopping when solution converges (default: True)
- `CONVERGENCE_PATIENCE (int)` - Number of consecutive generations with no improvement before termination (default: 80)
- `CONVERGENCE_THRESHOLD (float)` - Minimum change in total_cost to be considered significant improvement (default: 1e-7)

## Usage Notes

- Many parameters are set to `None` initially and must be set at runtime by `run_code.py` based on user input
- Parameters are accessed directly as module attributes (e.g., `config.POPULATION_SIZE`)
- For multiprocessing, config values are passed in `config_dict` to worker processes to avoid module state issues
- **POINTS_WEIGHT** is calculated dynamically in `run_code.py` based on the `strict_points` parameter:
  - `strict_points=True` → POINTS_WEIGHT = 80.0 (strict constraint)
  - `strict_points=False` → POINTS_WEIGHT = 2.5 (default)
  - This calculated value is included in `config_dict` and passed to `calculate_metrics()` via kwargs to ensure consistency across multiprocessing workers

