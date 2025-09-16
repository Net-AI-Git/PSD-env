import numpy as np
import logging
from typing import NoReturn


logger = logging.getLogger(__name__)

def save_results_to_text_file(file_path: str, points: np.ndarray) -> None:
    """
    Saves the resulting points of the optimization to a text file.

    Why (Purpose and Necessity):
    This function provides a way to persist the final envelope data in a simple,
    human-readable, and machine-parsable text format. This allows for external
    analysis, archiving, and comparison of results, complementing the graphical output.

    What (Implementation Details):
    The function takes a NumPy array where each row represents a point (frequency, amplitude).
    It uses numpy.savetxt to efficiently write these points to the specified file path.
    The data is formatted as two columns separated by a tab character, with no header.
    The function includes error handling to log any issues that occur during the file-saving process.

    Args:
        file_path (str): The full path, including the filename and extension,
                         where the results file will be saved.
        points (np.ndarray): A 2D NumPy array of shape (n, 2), where n is the
                             number of points. The first column represents frequency,
                             and the second column represents amplitude.

    Returns:
        None

    Raises:
        This function logs errors but does not raise exceptions itself,
        to avoid crashing the main process for a file-saving issue.
        Potential underlying exceptions from numpy.savetxt (e.g., IOError)
        are caught and logged.
    """
    try:
        logger.info(f"Saving results to text file: {file_path}")
        # Use a specific format: integer for the first column (frequency) and
        # a floating-point number with high precision for the second to avoid scientific notation.
        np.savetxt(file_path, points, delimiter='\t', fmt=['%d', '%.10f'])
        logger.info(f"Successfully saved results to {file_path}")
    except IOError as e:
        logger.error(f"Failed to save results to {file_path}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving to {file_path}: {e}")
