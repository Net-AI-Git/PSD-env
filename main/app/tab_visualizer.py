from bokeh.models.widgets import Button, TextInput, Div, Spinner
from bokeh.layouts import column, row
import numpy as np
from math import log10
import os

from .gui_utils import find_data_pairs, create_psd_plot, _calculate_rms
from .save_utils import save_matplotlib_plot_and_data, generate_word_document_from_images
from .powerpoint_generator import create_presentation_from_images


class VisualizerTab:
    """A class to manage the state and layout of the PSD Visualizer tab."""

    def __init__(self):
        self.data_pairs = []
        self.current_index = 0
        self.graph_modifications = {}  # To store per-graph point edits
        self.plots = [] # To hold references to the Bokeh plot objects for direct updates
        self.plot_layout = column(name="plot_layout", sizing_mode="stretch_width")
        
        # --- Create all widgets ---
        self.source_dir_input = TextInput(title="Source Data Directory (contains .mat or .txt files):", placeholder="Enter path to source data...", width=600)
        self.envelope_dir_input = TextInput(title="Envelopes Directory (contains .spc.txt files):", placeholder="Enter path to envelope files...", width=600)
        self.load_button = Button(label="Load Data", button_type="primary", width=150)
        
        self.prev_button = Button(label="< Previous", width=100, disabled=True)
        self.next_button = Button(label="Next >", width=100, disabled=True)
        
        self.status_div = Div(text="<i>Status: Enter directory paths and click 'Load Data'</i>", width=400, height_policy="min")
        
        # --- Factor inputs with dB conversion ---
        self.uncertainty_input = Spinner(title="Uncertainty Factor (Ratio)", value=1.0, step=0.01, width=150, format="0.00")
        self.uncertainty_db_input = Spinner(title="Uncertainty Factor (dB)", value=0.0, step=0.1, width=150, format="0.00")
        self.safety_input = Spinner(title="Safety Factor (Ratio)", value=1.0, step=0.01, width=150, format="0.00")
        self.safety_db_input = Spinner(title="Safety Factor (dB)", value=0.0, step=0.1, width=150, format="0.00")
        
        self.suffix_preview_input = TextInput(title="File Suffix Preview:", value="", disabled=True, width=400)
        self.save_button = Button(label="Save All Changes", button_type="success", width=150, disabled=True)

        # --- Assign callbacks ---
        self.load_button.on_click(self.load_data_callback)
        self.prev_button.on_click(self.show_previous_callback)
        self.next_button.on_click(self.show_next_callback)
        self.save_button.on_click(self.save_changes_callback)
        
        # Bi-directional callbacks
        self.uncertainty_input.on_change('value', self.uncertainty_ratio_to_db)
        self.uncertainty_db_input.on_change('value', self.uncertainty_db_to_ratio)
        self.safety_input.on_change('value', self.safety_ratio_to_db)
        self.safety_db_input.on_change('value', self.safety_db_to_ratio)

        self.update_suffix_preview() # Set initial state
        self._is_updating_factors = False # Flag to prevent callback loops

    def _on_envelope_change(self, attr, old, new):
        """Callback that fires when Bokeh plot points are changed by the user."""
        # The data source has been modified in the browser. We store the new
        # state (as a numpy array) in our modifications dictionary.
        modified_data = np.column_stack((new['freq'], new['val']))
        self.graph_modifications[self.current_index] = modified_data

        # --- Live update of RMS values in the legend without redrawing the plot ---
        # 1. Get the original PSD data for comparison
        original_psd_data = self.data_pairs[self.current_index]['psd_data']
        
        # 2. Recalculate RMS values
        psd_rms = _calculate_rms(original_psd_data[:, 0], original_psd_data[:, 1])
        env_rms = _calculate_rms(modified_data[:, 0], modified_data[:, 1])
        ratio = env_rms / psd_rms if psd_rms > 0 else 0
        
        # 3. Create the new label strings
        new_env_label = f"SPEC ({len(modified_data)} points, RMS: {env_rms:.3f} g)"
        new_ratio_label = f"RMS Ratio: {ratio:.3f}"
        
        # 4. Update the legend items on the existing plots directly.
        #    This is done by rebuilding the list of legend items and reassigning it,
        #    which forces a full refresh of the legend including the color glyphs.
        for plot_container in self.plots:
            # The actual plot is the second child of the column (after controls)
            plot = plot_container.children[1] 
            if plot.legend:
                # Get the existing items
                legend_items = plot.legend.items
                
                # Update the labels for the SPEC and Ratio items
                legend_items[1].label = {'value': new_env_label}
                legend_items[2].label = {'value': new_ratio_label}
                
                # Reassign the modified list to trigger a full legend refresh
                plot.legend.items = legend_items

        # Disable factor inputs to indicate manual override
        if not self.safety_input.disabled:
            self.safety_input.disabled = True
            self.uncertainty_input.disabled = True
            self.safety_db_input.disabled = True
            self.uncertainty_db_input.disabled = True
            # Update status to show manual edit mode is active
            self.status_div.text += " <i>(Manual edits active)</i>"

    # --- Factor Conversion Callbacks ---
    def uncertainty_ratio_to_db(self, attr, old, new):
        if self._is_updating_factors:
            return
        if new > 0:
            self._is_updating_factors = True
            self.uncertainty_db_input.value = 20 * log10(new)
            self._is_updating_factors = False
        self.update_plot_and_controls()

    def uncertainty_db_to_ratio(self, attr, old, new):
        if self._is_updating_factors:
            return
        self._is_updating_factors = True
        self.uncertainty_input.value = 10**(new / 20)
        self._is_updating_factors = False
        # Manually trigger the plot update, as the programmatic change above won't.
        self.update_plot_and_controls()

    def safety_ratio_to_db(self, attr, old, new):
        if self._is_updating_factors:
            return
        if new > 0:
            self._is_updating_factors = True
            self.safety_db_input.value = 20 * log10(new)
            self._is_updating_factors = False
        self.update_plot_and_controls()

    def safety_db_to_ratio(self, attr, old, new):
        if self._is_updating_factors:
            return
        self._is_updating_factors = True
        self.safety_input.value = 10**(new / 20)
        self._is_updating_factors = False
        # Manually trigger the plot update, as the programmatic change above won't.
        self.update_plot_and_controls()

    def save_changes_callback(self):
        """Saves all modified envelopes and their plot snapshots to a single new directory."""
        # Use a more descriptive check for savable work
        has_factor_changes = self.suffix_preview_input.value != ""
        has_manual_edits = bool(self.graph_modifications)

        if not self.data_pairs or (not has_factor_changes and not has_manual_edits):
            self.status_div.text = "<b>Status:</b> Nothing to save. Apply a factor or edit a graph first."
            return

        # Determine the suffix for factor-based changes
        suffix = self.suffix_preview_input.value if has_factor_changes else "MODIFIED"
        
        # --- Construct the new directory path based on the envelope source directory ---
        current_envelope_dir = self.envelope_dir_input.value
        parent_dir = os.path.dirname(current_envelope_dir)
        source_dir_name = os.path.basename(current_envelope_dir)
        
        new_dir_name = f"{source_dir_name} {suffix}"
        new_dir_path = os.path.join(parent_dir, new_dir_name)

        try:
            os.makedirs(new_dir_path, exist_ok=True)
            
            # --- Get the global factors once ---
            uncertainty = self.uncertainty_input.value
            safety = self.safety_input.value
            
            saved_image_paths = [] # To collect paths for the presentation

            # --- Loop through ALL data pairs and save each one ---
            for i, pair in enumerate(self.data_pairs):
                base_name = os.path.splitext(pair['name'])[0]
                
                # Check if there are manual modifications for this specific graph
                if i in self.graph_modifications:
                    # Use the manually modified data directly
                    modified_envelope_data = self.graph_modifications[i]
                    output_filename_base = f"{base_name} MODIFIED"
                else:
                    # If no manual edits, apply global factors (if any were set)
                    modified_envelope_data = pair['envelope_data'].copy()
                    if has_factor_changes:
                        modified_envelope_data[:, 1] *= (uncertainty**2) * (safety**2)
                    output_filename_base = f"{base_name} {suffix}"
                
                # --- Calculate RMS info for the final data ---
                psd_rms = _calculate_rms(pair['psd_data'][:, 0], pair['psd_data'][:, 1])
                env_rms = _calculate_rms(modified_envelope_data[:, 0], modified_envelope_data[:, 1])
                ratio = env_rms / psd_rms if psd_rms > 0 else 0
                
                rms_info = {
                    'psd_rms': psd_rms,
                    'env_rms': env_rms,
                    'ratio': ratio
                }

                # The save function now returns the paths of the generated images
                img_path, details_path = save_matplotlib_plot_and_data(
                    original_psd_data=pair['psd_data'],
                    modified_envelope_data=modified_envelope_data,
                    output_filename_base=output_filename_base,
                    output_directory=new_dir_path,
                    rms_info=rms_info
                )
                saved_image_paths.extend([img_path, details_path])

            # --- After saving all files, create the PowerPoint presentation ---
            if saved_image_paths:
                create_presentation_from_images(
                    image_paths=saved_image_paths,
                    output_dir=new_dir_path
                )

            # --- After saving all files, create the Word document ---
            if saved_image_paths:
                generate_word_document_from_images(new_dir_path)

            self.status_div.text = f"<b>Status:</b> All files saved successfully to <a href='file:///{new_dir_path}'>{new_dir_path}</a>"

        except Exception as e:
            self.status_div.text = f"<b>Status:</b> Error during save: {e}"


    def update_suffix_preview(self):
        """Updates the file suffix preview based on current factor values."""
        parts = []
        safety = self.safety_input.value
        uncertainty = self.uncertainty_input.value

        if safety != 1.0:
            parts.append(f"SF{safety:.2f}")
        
        if uncertainty != 1.0:
            parts.append(f"uncertainty{uncertainty:.2f}")

        self.suffix_preview_input.value = " ".join(parts)

    def update_plot_and_controls(self):
        if not self.data_pairs:
            self.plot_layout.children = []
            self.status_div.text = "<b>Status:</b> No matching data pairs found."
            self.prev_button.disabled = True
            self.next_button.disabled = True
            return

        # Check if the current graph has been manually edited
        is_manually_edited = self.current_index in self.graph_modifications
        
        # Disable factor inputs if manual edits exist for this graph
        self.safety_input.disabled = is_manually_edited
        self.uncertainty_input.disabled = is_manually_edited
        self.safety_db_input.disabled = is_manually_edited
        self.uncertainty_db_input.disabled = is_manually_edited

        pair = self.data_pairs[self.current_index]
        original_psd_data = pair['psd_data']
        
        # Use the modified envelope data if it exists, otherwise use original
        if is_manually_edited:
            display_envelope_data = self.graph_modifications[self.current_index]
        else:
            display_envelope_data = pair['envelope_data'].copy()
            # Apply global factors only if not manually edited
            uncertainty = self.uncertainty_input.value
            safety = self.safety_input.value
            display_envelope_data[:, 1] *= (uncertainty**2) * (safety**2)
        
        # --- Calculate RMS info for the data being displayed ---
        psd_rms = _calculate_rms(original_psd_data[:, 0], original_psd_data[:, 1])
        env_rms = _calculate_rms(display_envelope_data[:, 0], display_envelope_data[:, 1])
        ratio = env_rms / psd_rms if psd_rms > 0 else 0

        rms_info = {
            'psd_rms': psd_rms,
            'env_rms': env_rms,
            'ratio': ratio
        }

        layout, self.plots = create_psd_plot(
            original_psd_data,
            display_envelope_data,
            pair['name'],
            on_change_callback=self._on_envelope_change,
            rms_info=rms_info
        )
        self.plot_layout.children = [layout]
        
        status_message = f"<b>Displaying {self.current_index + 1}/{len(self.data_pairs)}:</b> {pair['name']}"
        if is_manually_edited:
            status_message += " <i>(Manual edits active)</i>"
        self.status_div.text = status_message
        
        self.prev_button.disabled = (self.current_index == 0)
        self.next_button.disabled = (self.current_index >= len(self.data_pairs) - 1)

        self.update_suffix_preview()
        # Enable save if either factors have been changed or manual edits have been made
        self.save_button.disabled = not (self.suffix_preview_input.value or self.graph_modifications)

    def load_data_callback(self):
        source_path = self.source_dir_input.value
        envelope_path = self.envelope_dir_input.value

        if not source_path or not envelope_path:
            self.status_div.text = "<b>Status:</b> Please enter both directory paths."
            return
            
        self.status_div.text = "<i>Status: Scanning directories...</i>"
        self.data_pairs = find_data_pairs(source_path, envelope_path)
        self.current_index = 0
        self.graph_modifications = {} # Reset modifications on new data load
        self.plots = [] # Reset plot references
        self.update_plot_and_controls()

    def show_previous_callback(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_plot_and_controls()

    def show_next_callback(self):
        if self.current_index < len(self.data_pairs) - 1:
            self.current_index += 1
            self.update_plot_and_controls()

    def get_layout(self):
        """Returns the final assembled layout for the tab."""
        controls_row = row(self.prev_button, self.status_div, self.next_button, align="center")
        input_layout = column(self.source_dir_input, self.envelope_dir_input)
        
        uncertainty_row = row(Div(text="<b>Uncertainty Factor:</b>"), self.uncertainty_input, self.uncertainty_db_input, align="center")
        safety_row = row(Div(text="<b>Safety Factor:</b>"), self.safety_input, self.safety_db_input, align="center")
        factor_note = Div(text="<i>Note: Factors are applied to the RMS value (multiplication on PSD is factor^2).</i>")
        factor_controls = column(uncertainty_row, safety_row, self.suffix_preview_input, factor_note)
        
        layout = column(
            Div(text="<h2>PSD & Envelope Visualizer</h2>"),
            row(input_layout, self.load_button, self.save_button, align="start"),
            factor_controls,
            controls_row,
            self.plot_layout,
            sizing_mode="stretch_width"
        )
        return layout

def create_visualizer_tab():
    visualizer = VisualizerTab()
    return visualizer.get_layout()
