import gradio as gr
from typing import Literal
import tkinter as tk
from tkinter import filedialog
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import re

# =================================================================================
# ISOLATED LOGIC FROM optimizer_core FOR VISUALIZATION
# This code is copied/adapted here to avoid modifying the original files.
# =================================================================================

def load_psds_from_source_file(filepath):
    """
    Loads all raw PSD measurements from a single source file (.mat or .txt).
    Returns a dictionary mapping measurement names to their data for easy lookup.
    """
    psd_map = {}
    if not filepath or not os.path.isfile(filepath):
        return psd_map

    jobs = []
    if filepath.lower().endswith('.mat'):
        jobs = _read_mat_file_isolated(filepath)
    elif filepath.lower().endswith('.txt'):
        job = _read_txt_file_isolated(filepath)
        if job:
            jobs.append(job)

    for job in jobs:
        psd_map[job['output_filename_base']] = job
    return psd_map

def load_envelopes_from_results_dir(directory_path):
    """
    Loads all envelope data from .txt files in a results directory.
    Returns a dictionary mapping measurement names to their envelope data.
    """
    envelope_map = {}
    if not directory_path or not os.path.isdir(directory_path):
        return envelope_map
        
    for filename in os.listdir(directory_path):
        # Assuming result files are .txt and their name matches the measurement name
        if filename.lower().endswith('.spc.txt'):
            try:
                base_name = filename.replace('.spc.txt', '')
                filepath = os.path.join(directory_path, filename)
                data = np.loadtxt(filepath)
                if data.ndim == 2 and data.shape[1] == 2:
                    envelope_map[base_name] = data
            except Exception as e:
                print(f"Could not read envelope file {filename}: {e}")
    return envelope_map

def calculate_rms_isolated(frequencies, psd_values):
    """Calculates the RMS value from area under the curve in linear axes."""
    if len(frequencies) < 2:
        return 0.0
    area = np.trapezoid(psd_values, x=frequencies)
    return np.sqrt(area)


# --- Copied from data_loader.py (and adapted) ---
def _read_txt_file_isolated(filepath):
    try:
        data = np.loadtxt(filepath)
        if data.ndim != 2 or data.shape[1] != 2: return None
        mask = (data[:, 0] >= 5) & (data[:, 0] <= 2000)
        filtered_data = data[mask]
        if filtered_data.shape[0] == 0: return None
        return {'frequencies': filtered_data[:, 0], 'psd_values': filtered_data[:, 1], 'output_filename_base': os.path.splitext(os.path.basename(filepath))[0]}
    except Exception:
        return None

def _read_mat_file_isolated(filepath):
    jobs = []
    try:
        mat_data = scipy.io.loadmat(filepath)
        if 'fvec' not in mat_data or 'FFTpsd' not in mat_data: return []
        freq_vector = mat_data['fvec']
        measurements = mat_data['FFTpsd'][0]
        for measurement in measurements:
            name = measurement['name'][0]
            psd_values = measurement['psd']
            combined_data = np.hstack((freq_vector.flatten()[:, np.newaxis], psd_values.flatten()[:, np.newaxis]))
            mask = (combined_data[:, 0] >= 5) & (combined_data[:, 0] <= 2000)
            filtered_data = combined_data[mask]
            if filtered_data.shape[0] > 0:
                jobs.append({'frequencies': filtered_data[:, 0], 'psd_values': filtered_data[:, 1], 'output_filename_base': name})
        return jobs
    except Exception:
        return []

# --- New Plotting function for the UI ---
def create_psd_envelope_plot(source_psd_filepath, results_directory_path):
    """
    Loads raw PSDs from a source file and their corresponding pre-calculated
    envelopes from a results directory, then plots the matched pairs.
    Also returns a status message detailing the process.
    """
    status_messages = []
    
    # --- Input Validation ---
    if not source_psd_filepath or not results_directory_path:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Please provide paths for BOTH the source PSD file and the results directory.",
                ha='center', va='center', transform=ax.transAxes)
        status_messages.append("Error: Missing one or both paths.")
        return fig, "\n".join(status_messages)

    status_messages.append(f"Attempting to load source file: {os.path.basename(source_psd_filepath)}")
    psd_map = load_psds_from_source_file(source_psd_filepath)
    status_messages.append(f"-> Found {len(psd_map)} PSD measurements in the source file.")
    if psd_map:
        status_messages.append(f"   (e.g., {list(psd_map.keys())[0]})")


    status_messages.append(f"\nAttempting to load envelopes from: {os.path.basename(results_directory_path)}")
    envelope_map = load_envelopes_from_results_dir(results_directory_path)
    status_messages.append(f"-> Found {len(envelope_map)} envelope files (.spc.txt) in the results directory.")
    if envelope_map:
        status_messages.append(f"   (e.g., {list(envelope_map.keys())[0]})")

    if not psd_map:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Could not load any valid PSD data from:\n{os.path.basename(source_psd_filepath)}",
                ha='center', va='center', transform=ax.transAxes)
        return fig, "\n".join(status_messages)
    
    # --- Matching ---
    status_messages.append("\nMatching PSDs to Envelopes by name...")
    matched_pairs = []
    for name, psd_data in psd_map.items():
        if name in envelope_map:
            matched_pairs.append({
                'name': name,
                'psd': psd_data,
                'envelope': envelope_map[name]
            })
    
    status_messages.append(f"-> Successfully matched {len(matched_pairs)} pairs.")

    if not matched_pairs:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No matching envelopes found in the results directory for the PSDs in the source file.",
                ha='center', va='center', transform=ax.transAxes)
        return fig, "\n".join(status_messages)

    # --- Plotting ---
    status_messages.append("\nGenerating plot...")
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(matched_pairs)))

    for i, pair in enumerate(matched_pairs):
        ax.plot(pair['psd']['frequencies'], pair['psd']['psd_values'],
                label=f"{pair['name']} (Original)", color=colors[i], alpha=0.7)
        ax.plot(pair['envelope'][:, 0], pair['envelope'][:, 1], '--',
                label=f"{pair['name']} (Envelope)", color=colors[i], linewidth=2)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f'Matched PSDs and Envelopes ({len(matched_pairs)} of {len(psd_map)} found)')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('PSD [g²/Hz]')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    
    status_messages.append("Done.")
    return fig, "\n".join(status_messages)

def _generate_plot_for_pair(pair_data):
    """Generates a single matplotlib plot for one PSD/Envelope pair."""
    # Use a modern style for the plot for a cleaner, more innovative look
    plt.style.use('seaborn-v0_8-whitegrid')
    # Create a figure with two subplots, one for log scale, one for linear
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), dpi=150)

    # --- Calculations ---
    original_freqs = pair_data['psd']['frequencies']
    original_psd = pair_data['psd']['psd_values']
    envelope_points = pair_data['envelope']
    
    # Calculate RMS values
    original_rms = calculate_rms_isolated(original_freqs, original_psd)
    
    # To calculate envelope RMS, we must interpolate it onto the original frequency grid
    interp_envelope_values = np.interp(original_freqs, envelope_points[:, 0], envelope_points[:, 1])
    envelope_rms = calculate_rms_isolated(original_freqs, interp_envelope_values)
    
    # Calculate RMS Ratio
    rms_ratio = (envelope_rms / original_rms) if original_rms > 0 else 0.0
    
    num_points = len(envelope_points)

    # --- Main Title for the whole figure ---
    main_title = f"Displaying: {pair_data['name']} ({num_points} Points) | RMS Ratio: {rms_ratio:.4f}"
    fig.suptitle(main_title, fontsize=16, fontweight='bold')

    # --- Plotting on both subplots ---
    for ax, x_scale in zip(axes, ['log', 'linear']):
        # Plot with specified colors and solid red line for envelope
        ax.plot(original_freqs, original_psd, 'b-',
                label=f"Original PSD - RMS: {original_rms:.4f}", linewidth=1.5)
        ax.plot(envelope_points[:, 0], envelope_points[:, 1], 'r-',
                label=f"Envelope - RMS: {envelope_rms:.4f}", linewidth=2)

        ax.set_xscale(x_scale)
        ax.set_yscale('log')
        ax.set_title(f'{x_scale.capitalize()} X-Axis View', fontsize=12)
        ax.set_xlabel('Frequency [Hz]', fontsize=10)
        ax.set_ylabel('PSD [g²/Hz]', fontsize=10)
        
        # Set grid lines to be black but with some transparency to not be overpowering
        ax.grid(True, which="both", ls="--", color='black', alpha=0.35)
        
        ax.legend(facecolor='white', framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout to make space for the main title
    
    return fig

def load_and_display_initial_plot(source_psd_filepath, results_directory_path):
    """
    Loads and matches all data, stores it in state, and returns the
    first plot to be displayed.
    """
    status_messages = []
    
    # --- Input Validation and Data Loading ---
    if not source_psd_filepath or not results_directory_path:
        status_messages.append("Error: Missing one or both paths.")
        # Return empty plot and state
        return None, "Error: Missing paths.", [], 0, gr.update(interactive=False), gr.update(interactive=False)

    status_messages.append(f"Loading source file: {os.path.basename(source_psd_filepath)}")
    psd_map = load_psds_from_source_file(source_psd_filepath)
    status_messages.append(f"-> Found {len(psd_map)} PSDs.")

    status_messages.append(f"Loading envelopes from: {os.path.basename(results_directory_path)}")
    envelope_map = load_envelopes_from_results_dir(results_directory_path)
    status_messages.append(f"-> Found {len(envelope_map)} envelopes.")
    
    # --- Matching ---
    status_messages.append("\nMatching PSDs to Envelopes...")
    matched_pairs = []
    # Natural sort the keys so the display order is intuitive (A1, A2, A10...)
    for name in sorted(psd_map.keys()):
        if name in envelope_map:
            matched_pairs.append({
                'name': name,
                'psd': psd_map[name],
                'envelope': envelope_map[name]
            })
    status_messages.append(f"-> Successfully matched {len(matched_pairs)} pairs.")

    if not matched_pairs:
        status_messages.append("No matches found. Cannot display plot.")
        return None, "\n".join(status_messages), [], 0, gr.update(interactive=False), gr.update(interactive=False)

    # --- Generate First Plot and UI State ---
    status_messages.append("Done. Displaying first plot.")
    first_plot = _generate_plot_for_pair(matched_pairs[0])
    title = f"### Displaying 1 of {len(matched_pairs)}: `{matched_pairs[0]['name']}`"
    
    # Next button is interactive if there's more than one plot
    next_btn_interactive = len(matched_pairs) > 1

    return first_plot, "\n".join(status_messages), matched_pairs, 0, gr.update(interactive=False), gr.update(interactive=next_btn_interactive)


def navigate_plots(all_pairs, current_index, direction):
    """
    Navigates to the next or previous plot in the state list.
    """
    # Calculate new index
    if direction == "next":
        new_index = current_index + 1
    else: # direction == "prev"
        new_index = current_index - 1
    
    # Get the plot data for the new index
    pair_to_display = all_pairs[new_index]
    new_plot = _generate_plot_for_pair(pair_to_display)
    new_title = f"### Displaying {new_index + 1} of {len(all_pairs)}: `{pair_to_display['name']}`"
    
    # Determine button states
    prev_btn_interactive = new_index > 0
    next_btn_interactive = new_index < len(all_pairs) - 1
    
    return new_plot, new_title, new_index, gr.update(interactive=prev_btn_interactive), gr.update(interactive=next_btn_interactive)


# =================================================================================
# END OF ISOLATED LOGIC
# =================================================================================


# --- UI Backend Logic ---

def select_directory() -> str:
    """Opens a dialog for the user to select a directory."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    directory_path = filedialog.askdirectory(title="Select a Directory")
    root.destroy()
    return directory_path if directory_path else ""

def select_file() -> str:
    """Opens a dialog for the user to select a file."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    filepath = filedialog.askopenfilename(title="Select a File")
    root.destroy()
    return filepath if filepath else ""


def run_optimization_process_ui(
    input_directory: str,
    min_frequency_hz: int = 5,
    max_frequency_hz: int = 2000,
    target_points: int = 45,
    target_area_ratio: float = 1.25,
    stab_wide: Literal["narrow", "wide"] = "narrow",
    area_x_axis_mode: Literal["Log", "Linear"] = "Log"
) -> str:
    """
    This is a placeholder function for the Gradio UI.
    It takes the parameters from the UI and returns a string representation.
    In the future, this will be connected to the actual optimization process.
    """
    # For now, just format the inputs into a string and return it for display.
    return f"""
    Optimization Parameters Received:
    ---------------------------------
    Min Frequency (Hz): {min_frequency_hz}
    Max Frequency (Hz): {max_frequency_hz}
    Target Points: {target_points}
    Target RMS Ratio: {target_area_ratio}
    Stability Wide: {stab_wide}
    Area X-Axis Mode: {area_x_axis_mode}
    Input Directory: {input_directory}
    """

# --- Gradio UI Layout (using gr.Blocks for more control) ---

with gr.Blocks() as demo:
    gr.Markdown("# Optimization Process Controller")
    gr.Markdown("Set the parameters below to run the optimization process.")

    with gr.Tabs():
        with gr.TabItem("Optimizer"):
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        optimizer_input_dir_textbox = gr.Textbox(
                            label="Input Directory",
                            placeholder="Select directory containing source PSD files...",
                            interactive=True, scale=4)
                        optimizer_browse_button = gr.Button("Browse...", scale=1)
                    
                    min_freq_slider = gr.Number(label="Min Frequency (Hz)", value=5)
                    max_freq_slider = gr.Number(label="Max Frequency (Hz)", value=2000)
                    target_points_slider = gr.Slider(label="Target Points", minimum=10, maximum=100, step=1, value=45)
                    target_rms_slider = gr.Slider(label="Target RMS Ratio", minimum=1.0, maximum=2.0, step=0.01, value=1.25)
                    stab_wide_radio = gr.Radio(label="Stability Wide", choices=["narrow", "wide"], value="narrow")
                    area_mode_radio = gr.Radio(label="Area X-Axis Mode", choices=["Log", "Linear"], value="Log")

                with gr.Column(scale=1):
                    optimizer_submit_button = gr.Button("Submit", variant="primary")
                    optimizer_output_textbox = gr.Textbox(label="Output Status", interactive=False)
        
        with gr.TabItem("PSD Envelope Viewer"):
            gr.Markdown("Select a source PSD file and the directory containing the calculated envelope results to view them.")
            with gr.Row():
                viewer_source_file_textbox = gr.Textbox(label="Source PSD File (.mat or .txt)", placeholder="Select source file...", scale=4)
                viewer_browse_file_button = gr.Button("Browse File...", scale=1)
            with gr.Row():
                viewer_results_dir_textbox = gr.Textbox(label="Results Directory (containing .spc.txt files)", placeholder="Select results directory...", scale=4)
                viewer_browse_dir_button = gr.Button("Browse Dir...", scale=1)

            viewer_submit_button = gr.Button("Load and Display Matched Envelopes", variant="primary")
            
            with gr.Column(visible=False) as viewer_display_area:
                plot_title = gr.Markdown("### Title")
                with gr.Row():
                    prev_button = gr.Button("◀ Previous", interactive=False)
                    next_button = gr.Button("Next ▶", interactive=False)
                psd_plot = gr.Plot(label="PSD Envelope Plot", show_label=False)

            viewer_status_textbox = gr.Textbox(label="Status Log", interactive=False, lines=5)
            
            # Invisible state storage
            matched_pairs_state = gr.State([])
            current_index_state = gr.State(0)

    # --- UI Event Handling ---

    def on_browse_dir_click():
        directory = select_directory()
        return gr.update(value=directory)

    def on_browse_file_click():
        filepath = select_file()
        return gr.update(value=filepath)

    optimizer_browse_button.click(fn=on_browse_dir_click, inputs=None, outputs=optimizer_input_dir_textbox)
    viewer_browse_file_button.click(fn=on_browse_file_click, inputs=None, outputs=viewer_source_file_textbox)
    viewer_browse_dir_button.click(fn=on_browse_dir_click, inputs=None, outputs=viewer_results_dir_textbox)

    def show_viewer_area(plot, status, pairs, index, prev_btn, next_btn):
        # This function acts as a wrapper to also make the display area visible
        # once the first plot is successfully generated.
        is_visible = plot is not None
        return gr.update(visible=is_visible), plot, status, pairs, index, prev_btn, next_btn

    optimizer_submit_button.click(
        fn=run_optimization_process_ui,
        inputs=[
            optimizer_input_dir_textbox,
            min_freq_slider,
            max_freq_slider,
            target_points_slider,
            target_rms_slider,
            stab_wide_radio,
            area_mode_radio
        ],
        outputs=optimizer_output_textbox
    )

    viewer_submit_button.click(
        fn=load_and_display_initial_plot,
        inputs=[viewer_source_file_textbox, viewer_results_dir_textbox],
        outputs=[psd_plot, viewer_status_textbox, matched_pairs_state, current_index_state, prev_button, next_button]
    ).then(
        fn=lambda p: gr.update(visible=p is not None),
        inputs=[psd_plot],
        outputs=[viewer_display_area]
    )

    prev_button.click(
        fn=navigate_plots,
        inputs=[matched_pairs_state, current_index_state, gr.State("prev")],
        outputs=[psd_plot, plot_title, current_index_state, prev_button, next_button]
    )

    next_button.click(
        fn=navigate_plots,
        inputs=[matched_pairs_state, current_index_state, gr.State("next")],
        outputs=[psd_plot, plot_title, current_index_state, prev_button, next_button]
    )


# Launch the interface
if __name__ == "__main__":
    demo.launch()
