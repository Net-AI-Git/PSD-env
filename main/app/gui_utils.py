import os
import re
import numpy as np
import scipy.io
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, PointDrawTool, RadioButtonGroup, CustomJS, TapTool
from bokeh.layouts import column, row

# ===================================================================
#
#           Plotting Utilities
#
# ===================================================================

def create_psd_plot(psd_data, envelope_data, plot_title, on_change_callback=None):
    """
    Creates a Bokeh layout containing two plots of a PSD and its envelope,
    each with its own set of interactive editing controls.
    """
    psd_source = ColumnDataSource(data=dict(freq=psd_data[:, 0], val=psd_data[:, 1]))
    env_source = ColumnDataSource(data=dict(freq=envelope_data[:, 0], val=envelope_data[:, 1]))

    # JS callback to sort the data source by frequency after any change
    sorting_callback = CustomJS(args=dict(source=env_source), code="""
        const data = source.data;
        const freqs = data['freq'];
        
        // Guard clause: Check if the data is already sorted to prevent infinite loops.
        let is_sorted = true;
        for (let i = 0; i < freqs.length - 1; i++) {
            if (freqs[i] > freqs[i+1]) {
                is_sorted = false;
                break;
            }
        }
        if (is_sorted) return;

        const vals = data['val'];

        // Combine into an array of points for sorting
        let points = freqs.map((freq, i) => ({freq: freq, val: vals[i]}));

        // Sort by frequency (the x-axis value)
        points.sort((a, b) => a.freq - b.freq);

        // Unpack back into separate arrays
        const sorted_freqs = points.map(p => p.freq);
        const sorted_vals = points.map(p => p.val);

        // Replace the data entirely. This is a robust way to trigger a full redraw.
        source.data = { 'freq': sorted_freqs, 'val': sorted_vals };
    """)
    env_source.js_on_change('data', sorting_callback)

    # If a Python callback is provided for data changes, attach it.
    # This allows the GUI to be notified of user edits in the plot.
    if on_change_callback:
        env_source.on_change('data', on_change_callback)

    tooltips = [
        ("Frequency", "@freq{0.00} Hz"),
        ("PSD", "@val{0.00e0} g²/Hz"),
    ]
    
    plots_with_controls = []
    for x_axis_type in ["log", "linear"]:
        p = figure(
            height=300,
            sizing_mode="stretch_width",
            title=f"{plot_title} ({x_axis_type.capitalize()} X-axis)",
            x_axis_type=x_axis_type,
            y_axis_type="log",
            x_axis_label="Frequency (Hz)",
            y_axis_label="PSD (g²/Hz)",
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )
        
        p.line(x='freq', y='val', source=psd_source, legend_label="Original PSD", color="blue", line_width=1)
        # Define a selection glyph to give visual feedback on tap
        envelope_points = p.scatter(x='freq', y='val', source=env_source, color="red", size=6,
                                    selection_color="orange", selection_fill_alpha=1.0)
        p.line(x='freq', y='val', source=env_source, legend_label="Envelope", color="red", line_width=2)
        
        # --- Create separate tools for editing ---
        drag_tool = PointDrawTool(renderers=[envelope_points], drag=True, add=False)
        add_tool = PointDrawTool(renderers=[envelope_points], drag=False, add=True)
        tap_tool = TapTool(renderers=[envelope_points]) # For selecting points to delete
        p.add_tools(drag_tool, add_tool, tap_tool, HoverTool(tooltips=tooltips, renderers=[envelope_points]))

        # --- Custom UI Controls using Radio Buttons ---
        edit_mode_rb = RadioButtonGroup(
            labels=["Drag Points", "Add Point", "Delete Point", "Off"], 
            active=3, # Off by default
            width=400
        )
        
        tool_activation_callback = CustomJS(args=dict(plot=p, drag_tool=drag_tool, add_tool=add_tool, tap_tool=tap_tool), code="""
            const active_mode = cb_obj.active;
            plot.toolbar.active_drag = null;
            plot.toolbar.active_tap = null;
            
            if (active_mode === 0) { // Drag mode
                plot.toolbar.active_drag = drag_tool;
            } else if (active_mode === 1) { // Add mode
                plot.toolbar.active_tap = add_tool;
            } else if (active_mode === 2) { // Delete mode
                plot.toolbar.active_tap = tap_tool;
            }
        """)
        edit_mode_rb.js_on_change('active', tool_activation_callback)

        # --- JS Callback for deleting selected points ---
        deletion_callback = CustomJS(args=dict(source=env_source, mode_button=edit_mode_rb), code="""
            // Only run deletion logic if the 'Delete Point' button is active (index 2)
            if (mode_button.active !== 2) {
                return;
            }
            const indices = source.selected.indices;
            if (indices.length === 0) return;

            const data = source.data;
            const new_freq = [];
            const new_val = [];
            for (let i = 0; i < data.freq.length; i++) {
                if (!indices.includes(i)) {
                    new_freq.push(data.freq[i]);
                    new_val.push(data.val[i]);
                }
            }
            source.data = {'freq': new_freq, 'val': new_val};
            // Clear selection after deletion
            source.selected.indices = [];
        """)
        env_source.selected.js_on_change('indices', deletion_callback)
        
        controls = row(edit_mode_rb)
        plots_with_controls.append(column(controls, p, sizing_mode="stretch_width"))

    return column(plots_with_controls, sizing_mode="stretch_width")

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
