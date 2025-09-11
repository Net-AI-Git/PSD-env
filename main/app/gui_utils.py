import os
import re
import numpy as np
import scipy.io
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.layouts import column

# ===================================================================
#
#           Plotting Utilities
#
# ===================================================================

def create_psd_plot(psd_data, envelope_data, plot_title):
    """
    Creates a Bokeh layout containing two plots of a PSD and its envelope:
    one with a logarithmic X-axis and one with a linear X-axis.
    """
    psd_source = ColumnDataSource(data=dict(freq=psd_data[:, 0], val=psd_data[:, 1]))
    env_source = ColumnDataSource(data=dict(freq=envelope_data[:, 0], val=envelope_data[:, 1]))

    tooltips = [
        ("Frequency", "@freq{0.00} Hz"),
        ("PSD", "@val{0.00e0} g²/Hz"),
    ]
    
    plots = []
    for x_axis_type in ["log", "linear"]:
        hover = HoverTool(tooltips=tooltips)
        
        p = figure(
            height=300,  # Further reduced height
            sizing_mode="stretch_width",
            title=f"{plot_title} ({x_axis_type.capitalize()} X-axis)",
            x_axis_type=x_axis_type,
            y_axis_type="log",
            x_axis_label="Frequency (Hz)",
            y_axis_label="PSD (g²/Hz)",
            tools=[hover, "pan,wheel_zoom,box_zoom,reset,save"]
        )

        p.line(x='freq', y='val', source=psd_source, legend_label="Original PSD", color="blue", line_width=1)
        p.line(x='freq', y='val', source=env_source, legend_label="Envelope", color="red", line_width=2)
        p.scatter(x='freq', y='val', source=env_source, color="red", size=4)

        p.legend.location = "bottom_right"
        p.legend.click_policy = "hide"
        p.grid.grid_line_alpha = 0.3
        plots.append(p)

    # Return a column containing both plots
    return column(plots, sizing_mode="stretch_width")

# ===================================================================
#
#           Data Loading Utilities
#
# The following functions are adapted from the project's data_loader.py
# to create a self-contained GUI module.
# ===================================================================

def _load_all_psds_from_mat(filepath):
    """Loads all PSDs from a .mat file into a dictionary."""
    psd_dict = {}
    try:
        mat_data = scipy.io.loadmat(filepath)
        if 'fvec' not in mat_data or 'FFTpsd' not in mat_data:
            return {}

        freq_vector = mat_data['fvec'].flatten()
        measurements = mat_data['FFTpsd'][0]

        for measurement in measurements:
            name = measurement['name'][0]
            psd_values = measurement['psd'].flatten()
            combined_data = np.column_stack((freq_vector, psd_values))
            
            mask = (combined_data[:, 0] >= 5) & (combined_data[:, 0] <= 2000)
            psd_dict[name] = combined_data[mask]
            
        return psd_dict
    except Exception as e:
        print(f"Warning: Could not process MAT file '{filepath}'. Error: {e}")
        return {}

def _read_txt_psd(filepath):
    """Loads a single PSD from a .txt file."""
    filename_no_ext = os.path.splitext(os.path.basename(filepath))[0]
    try:
        data = np.loadtxt(filepath)
        if data.ndim != 2 or data.shape[1] != 2: return None, None
        mask = (data[:, 0] >= 5) & (data[:, 0] <= 2000)
        filtered_data = data[mask]
        return (filename_no_ext, filtered_data) if filtered_data.shape[0] > 0 else (None, None)
    except Exception:
        return None, None

def find_data_pairs(source_directory, envelope_directory):
    """
    Scans directories to find pairs of original PSD data and their corresponding envelope files.
    """
    if not os.path.isdir(source_directory):
        print(f"Error: Source is not a directory: {source_directory}")
        return []
    if not os.path.isdir(envelope_directory):
        print(f"Error: Envelopes is not a directory: {envelope_directory}")
        return []

    all_psds = {}
    # --- Step 1: Scan the source directory for all possible PSDs ---
    for filename in os.listdir(source_directory):
        filepath = os.path.join(source_directory, filename)
        if filename.lower().endswith('.mat'):
            all_psds.update(_load_all_psds_from_mat(filepath))
        elif filename.lower().endswith('.txt') and not filename.lower().endswith('.spc.txt'):
            # Make sure we don't accidentally read an envelope as a source
            name, data = _read_txt_psd(filepath)
            if name is not None: all_psds[name] = data

    if not all_psds:
        print(f"Warning: No source PSDs found in {source_directory}.")
        return []
        
    # --- Step 2: Scan the envelope directory and find matches ---
    data_pairs = []
    envelope_files = [f for f in os.listdir(envelope_directory) if f.lower().endswith('.spc.txt')]
    
    for env_filename in sorted(envelope_files):
        base_name = env_filename.split(' ')[0].split('.spc')[0]
        
        if base_name in all_psds:
            try:
                env_filepath = os.path.join(envelope_directory, env_filename)
                envelope_data = np.loadtxt(env_filepath)
                if envelope_data.ndim != 2 or envelope_data.shape[1] != 2: continue

                data_pairs.append({
                    'name': env_filename,
                    'psd_data': all_psds[base_name],
                    'envelope_data': envelope_data
                })
            except Exception as e:
                print(f"Warning: Could not load envelope file {env_filename}. Error: {e}")
    
    print(f"Found {len(data_pairs)} matching PSD/Envelope pairs.")
    return data_pairs
