# ===================================================================
#
#           Configuration File for the PSD Optimization GA
#
# This file centralizes all tunable hyperparameters for the genetic
# algorithm. This allows for easy experimentation and tuning without
# modifying the core logic of the algorithm.
#
# ===================================================================

# --- File and Directory Settings ---
# Directory for input PSD data files.
INPUT_DIR = "input"
# Directory where the output result images will be saved.
OUTPUT_DIR = "results"
# The file extension to look for in the input directory.
INPUT_FILE_EXTENSION = ".txt"


# --- Optimization Strategy Settings ---
# Controls which optimization strategy to use:
#   - "points": minimize area cost while targeting specific number of points (original behavior)
#   - "area": minimize number of points while targeting specific area ratio
OPTIMIZATION_MODE = "points"

# The ideal number of points for the final envelope. The fitness function
# will penalize solutions that deviate from this target.
TARGET_P = 45
TARGET_POINTS =  TARGET_P * 0.9

# For area optimization, what is the target area ratio between the
# envelope and the original PSD? (e.g., 1.2 means 20% larger area)
TARGET_A = 1.25
TARGET_AREA_RATIO = (TARGET_A**2) * 0.95

# Weight for the area error component in the multi-objective cost function.
# This acts as a multiplier for the area error cost, ensuring that meeting the
# area ratio target is prioritized over minimizing points. A large value (e.g., 10000)
# makes the area target a hard constraint. The points cost has an implicit weight of 1.0.
AREA_LOG_AWEIGHT = 10000.0

# Weight for the area error when in 'points' optimization mode.
# This ensures that a tight envelope is prioritized over meeting the target point count.
POINTS_LOG_WEIGHT = 1000.0


# --- Candidate Point Generation Settings ---
# A list of window sizes for the multi-scale candidate point generation.
WINDOW_SIZES = [10, 20, 30]
# Factor for lifting points to enrich the search space.
# This creates an additional set of candidate points by scaling their Y-value.
# Set to 0 to disable. A good value to try is 1.1 (for a 10% lift).
LIFT_FACTOR = 1.05

# Set to True to add all original PSD points below a certain frequency
# to the candidate pool. This can improve the fit at low frequencies.
ENRICH_LOW_FREQUENCIES = True
# A list of scaling factors used to create additional "lifted" candidate
# points in the low-frequency range. For example, [1.05, 1.1] would create
# two extra sets of points, lifted by 5% and 10% respectively.
# Set to an empty list [] to disable.
LOW_FREQ_ENRICHMENT_FACTORS = [1.2, 1.5]
# The frequency (in Hz) below which all original PSD points will be
# added to the candidate pool if the above setting is enabled.
LOW_FREQUENCY_THRESHOLD = 100.0


# --- Genetic Algorithm Core Settings ---
# The number of individual solutions (chromosomes) in each generation.
POPULATION_SIZE = 1000
# The maximum number of generations the evolution will run for.
# This acts as a safeguard if convergence is not met.
MAX_GENERATIONS = 2000
# The probability that a newly created child solution will undergo mutation.
MUTATION_RATE = 0.9
# The number of the best solutions from one generation to be carried over
# directly to the next, ensuring the best-found solution is never lost.
ELITISM_SIZE = 2


# --- Area Integration Settings ---
# Controls the X-axis domain used for integrating the area cost in calculate_metrics.
# Allowed values:
#   - "Linear": integrate over original frequencies (current behavior, default)
#   - "Log": integrate over log10(frequency) to match Y's log domain
AREA_X_AXIS_MODE = "Log"

# The weight to apply to the area cost calculation for the low-frequency
# region. A value of 2.0 means the area cost in this region is twice as important.
LOW_FREQ_AREA_WEIGHT = 1.0


# --- Advanced Mutation Strategy Settings ---
# The relative area cost change below which a point is considered "useless"
# by the pruning mutation and can be removed.
PRUNE_THRESHOLD = 0.02
# The percentage of the non-elite population to apply the pruning mutation to.
PRUNE_PERCENTAGE_OF_POPULATION = 0.1
# The point count threshold below which "turbo mode" for mutations activates,
# increasing the mutation rate to escape local optima.
ADAPTIVE_MUTATION_THRESHOLD = 80
# Stop checking if we find this many consecutive invalid jumps during graph building
BREAK_THRESHOLD = 100


# --- Termination Criteria Settings ---
# Set to True to enable early stopping when the solution converges.
# If False, the algorithm will run for the full MAX_GENERATIONS.
USE_CONVERGENCE_TERMINATION = True

# The number of consecutive generations with no significant improvement
# before the algorithm terminates early.
CONVERGENCE_PATIENCE = 80

# The minimum change in 'total_cost' to be considered a significant
# improvement. If the improvement is less than this, it's counted as a
# generation without improvement.
CONVERGENCE_THRESHOLD = 1e-7