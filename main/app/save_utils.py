import matplotlib.pyplot as plt
import numpy as np
import os
from optimizer_core.file_saver import save_results_to_text_file

def save_matplotlib_plot_and_data(original_psd_data, modified_envelope_data, output_filename_base, output_directory):
    """
    Renders and saves a dual view of the PSD and its modified envelope using Matplotlib.
    It also saves the modified envelope data to a text file.

    Args:
        original_psd_data (np.array): The original PSD data ([frequency, value]).
        modified_envelope_data (np.array): The coordinates of the modified envelope points.
        output_filename_base (str): The base name for the output file (without extension).
        output_directory (str): The path to the directory where results will be saved.
    """
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

    # --- Save the modified envelope data using the centralized saver ---
    text_output_filename = f"{output_filename_base}.spc.txt"
    text_output_path = os.path.join(output_directory, text_output_filename)
    save_results_to_text_file(text_output_path, modified_envelope_data)

    # Close the plot to free up memory
    plt.close()
