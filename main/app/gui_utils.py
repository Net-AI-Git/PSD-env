import os
import numpy as np
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, PointDrawTool, RadioButtonGroup, CustomJS, TapTool, Label
from bokeh.layouts import column, row
from optimizer_core import new_data_loader
from utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

# ===================================================================
#
#           Calculation Utilities
#
# ===================================================================

def _calculate_rms(frequencies, psd_values):
    """Calculates the RMS value from frequency and PSD data using trapezoidal integration."""
    if frequencies is None or psd_values is None or len(frequencies) < 2:
        return 0.0
    # Ensure data is sorted by frequency before integration
    sort_indices = np.argsort(frequencies)
    sorted_freqs = frequencies[sort_indices]
    sorted_psd = psd_values[sort_indices]
    
    # Perform trapezoidal integration to find the area (Mean Square)
    mean_square = np.trapz(sorted_psd, sorted_freqs)
    if mean_square < 0:
        return 0.0 # Physical values cannot be negative
    # Return the square root of the area (Root Mean Square)
    return np.sqrt(mean_square)

# ===================================================================
#
#           Plotting Utilities
#
# ===================================================================

def create_psd_plot(psd_data, envelope_data, plot_title, on_change_callback=None, rms_info=None):
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
    
    # --- Prepare RMS labels if data is available ---
    psd_label = "Original PSD"
    env_label = "SPEC" # Changed from "Envelope"
    if rms_info:
        psd_label += f" (RMS: {rms_info['psd_rms']:.3f} g)"
        env_label += f" ({len(envelope_data)} points, RMS: {rms_info['env_rms']:.3f} g)"

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
        
        p.line(x='freq', y='val', source=psd_source, legend_label=psd_label, color="blue", line_width=1)
        # Define a selection glyph to give visual feedback on tap
        envelope_points = p.scatter(x='freq', y='val', source=env_source, color="red", size=6,
                                    selection_color="orange", selection_fill_alpha=1.0)
        p.line(x='freq', y='val', source=env_source, legend_label=env_label, color="red", line_width=2)
        
        # --- Add RMS Ratio as a dummy glyph to include it in the legend ---
        if rms_info:
            ratio_label = f"RMS Ratio: {rms_info['ratio']:.3f}"
            # Add an invisible glyph that the legend can reference
            p.scatter([], [], legend_label=ratio_label, visible=False)

        # --- Configure the legend ---
        p.legend.location = "top_left"
        p.legend.background_fill_color = "white"
        p.legend.background_fill_alpha = 0.8
        p.legend.border_line_color = "black"
        p.legend.label_text_font_size = "8pt" # Reduced font size
        p.legend.spacing = -9 # Further reduced spacing between legend items
        p.legend.padding = 2 # Reduced padding inside the legend box

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

    # Return both the final layout and the list of plot objects for direct manipulation
    return column(plots_with_controls, sizing_mode="stretch_width"), plots_with_controls

# ===================================================================
#
#           Data Loading Utilities
#
# All file loading is now handled by new_data_loader.py to ensure
# consistency and eliminate code duplication.
# ===================================================================

def find_data_pairs(source_directory, envelope_directory):
    """
    Scans directories to find pairs of original PSD data and their corresponding envelope files.
    
    This function is a wrapper that delegates to new_data_loader.find_data_pairs_unified()
    to ensure all file loading uses the centralized, unified loading system.
    
    The matching algorithm uses multiple strategies to find pairs:
    1. Exact match of base name (part before first space and .spc)
    2. Exact match after removing .spc.txt extension
    3. Normalized matching (ignoring spaces, underscores, hyphens, case)
    4. Partial matching (checking if one name starts with the other)
    
    Args:
        source_directory: Path to directory containing source PSD files (.mat or .txt).
        envelope_directory: Path to directory containing envelope files (.spc.txt).
    
    Returns:
        list: List of dictionaries, each containing:
            - 'name': The envelope filename
            - 'psd_data': The matched PSD data (numpy array with shape (N, 2))
            - 'envelope_data': The envelope data (numpy array with shape (M, 2))
    """
    return new_data_loader.find_data_pairs_unified(source_directory, envelope_directory)
