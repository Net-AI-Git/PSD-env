import matplotlib.pyplot as plt
import numpy as np
import os
from optimizer_core.file_saver import save_results_to_text_file
import math
from typing import Tuple, List
from app.word_generator import create_images_document


def _create_envelope_with_table_image(envelope_data, output_path, title):
    """
    Creates a special image containing a log-log plot of the envelope only,
    with a table of its points listed below, wrapped into multiple rows if needed.
    """
    points = envelope_data
    num_points = len(points)
    points_per_column = 20
    
    # --- New Layout Logic ---
    max_cols_per_row = 4 # Changed from 5 to 4
    num_total_cols = math.ceil(num_points / points_per_column)
    num_table_rows = math.ceil(num_total_cols / max_cols_per_row)

    # --- Dynamic Figure Sizing ---
    base_fig_height = 5.0  # Height for the plot area
    table_row_height = 4.0 # Estimated height for one row of tables
    fig_height = base_fig_height + (table_row_height * num_table_rows)

    fig = plt.figure(figsize=(14.0, fig_height)) # Increased width from 12.8 to 14.0
    
    # Create a GridSpec: one row for the plot, and then N rows for the tables
    # The plot gets a height relative to the number of table rows
    height_ratios = [base_fig_height] + [table_row_height] * num_table_rows
    # Let tight_layout handle the spacing automatically
    gs = fig.add_gridspec(1 + num_table_rows, max_cols_per_row, 
                          height_ratios=height_ratios)

    # --- 1. Create the Plot in the first row ---
    ax_plot = fig.add_subplot(gs[0, :])
    ax_plot.plot(points[:, 0], points[:, 1], 'r-', linewidth=2)
    ax_plot.set_xscale('log')
    ax_plot.set_yscale('log')
    ax_plot.set_title(f'{title} ({num_points} points)')
    ax_plot.set_xlabel('Frequency [Hz]')
    ax_plot.set_ylabel('Envelope Value [g²/Hz]')
    ax_plot.grid(True, which="both", ls="--", alpha=0.5)

    # --- 2. Create the Tables in the subsequent rows ---
    chunks = [points[i:i + points_per_column] for i in range(0, num_points, points_per_column)]
    
    for i, chunk in enumerate(chunks):
        # Calculate the grid position for the current table chunk
        current_row = 1 + (i // max_cols_per_row)
        current_col = i % max_cols_per_row
        
        ax_table = fig.add_subplot(gs[current_row, current_col])
        ax_table.axis('off')

        # Prepare text for one column with new header and wider value column
        header = f"{'Index':<8}{'Frequency':<15}{'g^2/Hz':<20}\n"
        separator = f"{'-'*6:<8}{'-'*13:<15}{'-'*18:<20}\n"
        table_text = header + separator
        
        # Correctly calculate the starting index for each chunk
        start_index = i * points_per_column
        for j, point in enumerate(chunk):
            index = start_index + j + 1
            freq, val = point
            # Use a wider format specifier for the value to prevent overlap
            table_text += f"{index:<8}{freq:<15.2f}{val:<20.8f}\n"

        # Place the text block in the correct subplot cell
        ax_table.text(0, 1, table_text, ha='left', va='top', family='monospace', fontsize=9)

    # Use tight_layout to automatically adjust subplot params to fit the figure area.
    # This is more robust than manual adjustment for preventing text cutoff.
    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Add a bit of top padding for the main title
    plt.savefig(output_path)
    plt.close()


def save_matplotlib_plot_and_data(original_psd_data, modified_envelope_data, output_filename_base, output_directory) -> Tuple[str, str]:
    """
    Renders and saves a dual view of the PSD and its modified envelope using Matplotlib.
    It also saves the modified envelope data to a text file and creates a details image.

    Returns:
        Tuple[str, str]: A tuple containing the absolute paths to the two saved images:
                         (main_plot_path, details_plot_path).
    """
    # --- 1. Save the standard comparison plot ---
    # Set figsize to produce an output image of 1280x600 pixels (at 100 DPI)
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 6.0))

    title_prefix = output_filename_base

    for ax, x_scale in zip(axes, ["log", "linear"]):
        ax.plot(original_psd_data[:, 0], original_psd_data[:, 1], 'b-', label='Original PSD', linewidth=1.5, alpha=0.7)
        ax.plot(modified_envelope_data[:, 0], modified_envelope_data[:, 1], 'r-',
                label=f'Modified Envelope ({len(modified_envelope_data)} points)', linewidth=2)
        # ax.scatter(modified_envelope_data[:, 0], modified_envelope_data[:, 1], c='red', s=20, zorder=5)

        ax.set_xscale(x_scale)
        ax.set_yscale('log')
        ax.set_title(f'Result for {title_prefix} ({x_scale.capitalize()} X-axis)')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('PSD [g²/Hz]')
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend()

    # Manually set subplot parameters for consistent layout
    plt.subplots_adjust(left=0.065, bottom=0.083, right=0.997, top=0.944, wspace=0.2, hspace=0.332)

    # --- Save the figure ---
    img_output_filename = f"{output_filename_base}.png"
    img_output_path = os.path.join(output_directory, img_output_filename)
    plt.savefig(img_output_path)
    print(f"Result image saved to: {img_output_path}")
    plt.close() # Close the first plot

    # --- 2. Save the modified envelope data using the centralized saver ---
    text_output_filename = f"{output_filename_base}.spc.txt"
    text_output_path = os.path.join(output_directory, text_output_filename)
    save_results_to_text_file(text_output_path, modified_envelope_data)

    # --- 3. Save the new "details" image with the table ---
    details_img_filename = f"{output_filename_base}_details.png"
    details_img_path = os.path.join(output_directory, details_img_filename)
    _create_envelope_with_table_image(modified_envelope_data, details_img_path, title=output_filename_base)
    print(f"Details image saved to: {details_img_path}")

    # Return the paths of the created images
    return img_output_path, details_img_path


def generate_word_document_from_images(directory: str) -> None:
    """
    Finds all images in a directory and generates a Word document from them.

    The 'Why': This function serves as a bridge between the file-saving logic and
    the Word document generation logic. It automates the process of collecting
    the final image outputs and passing them to the document creator.

    The 'What': The function scans the specified directory for files with common
    image extensions (png, jpg, jpeg, gif). It constructs a list of full paths
    to these images and then calls the `create_images_document` function to
    handle the actual document creation.

    Args:
        directory (str): The path to the directory containing the image files.
    """
    image_extensions = {".png", ".jpg", ".jpeg", ".gif"}
    image_paths = []
    try:
        for filename in os.listdir(directory):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                full_path = os.path.join(directory, filename)
                image_paths.append(full_path)

        if not image_paths:
            print(f"No images found in directory: {directory}")
            return

        create_images_document(image_paths, directory)

    except FileNotFoundError:
        print(f"Error: The directory '{directory}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
