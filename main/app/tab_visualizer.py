from bokeh.models.widgets import Button, TextInput, Div, Spinner
from bokeh.layouts import column, row
import numpy as np
from math import log10
import os

from .gui_utils import find_data_pairs, create_psd_plot
from .save_utils import save_matplotlib_plot_and_data

class VisualizerTab:
    """A class to manage the state and layout of the PSD Visualizer tab."""

    def __init__(self):
        self.data_pairs = []
        self.current_index = 0
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

    # --- Factor Conversion Callbacks ---
    def uncertainty_ratio_to_db(self, attr, old, new):
        if new > 0:
            # Temporarily remove the other callback to prevent a loop
            self.uncertainty_db_input.remove_on_change('value', self.uncertainty_db_to_ratio)
            self.uncertainty_db_input.value = 20 * log10(new)
            self.uncertainty_db_input.on_change('value', self.uncertainty_db_to_ratio)
        self.update_plot_and_controls()

    def uncertainty_db_to_ratio(self, attr, old, new):
        # Temporarily remove the other callback to prevent a loop
        self.uncertainty_input.remove_on_change('value', self.uncertainty_ratio_to_db)
        self.uncertainty_input.value = 10**(new / 20)
        self.uncertainty_input.on_change('value', self.uncertainty_ratio_to_db)
        # Manually trigger the plot update, as the programmatic change above won't.
        self.update_plot_and_controls()

    def safety_ratio_to_db(self, attr, old, new):
        if new > 0:
            self.safety_db_input.remove_on_change('value', self.safety_db_to_ratio)
            self.safety_db_input.value = 20 * log10(new)
            self.safety_db_input.on_change('value', self.safety_db_to_ratio)
        self.update_plot_and_controls()

    def safety_db_to_ratio(self, attr, old, new):
        self.safety_input.remove_on_change('value', self.safety_ratio_to_db)
        self.safety_input.value = 10**(new / 20)
        self.safety_input.on_change('value', self.safety_ratio_to_db)
        # Manually trigger the plot update, as the programmatic change above won't.
        self.update_plot_and_controls()

    def save_changes_callback(self):
        """Saves all modified envelopes and their plot snapshots to a single new directory."""
        if not self.data_pairs or not self.suffix_preview_input.value:
            self.status_div.text = "<b>Status:</b> Nothing to save. Apply a factor first."
            return

        suffix = self.suffix_preview_input.value
        
        # --- Construct the new directory path based on the envelope source directory ---
        current_envelope_dir = self.envelope_dir_input.value
        parent_dir = os.path.dirname(current_envelope_dir)
        source_dir_name = os.path.basename(current_envelope_dir)
        
        new_dir_name = f"{source_dir_name} {suffix}"
        new_dir_path = os.path.join(parent_dir, new_dir_name)

        try:
            os.makedirs(new_dir_path, exist_ok=True)
            
            # --- Get the factors once ---
            uncertainty = self.uncertainty_input.value
            safety = self.safety_input.value

            # --- Loop through ALL data pairs and save each one ---
            for pair in self.data_pairs:
                base_name = os.path.splitext(pair['name'])[0]
                
                # Calculate the modified envelope data for the current pair
                modified_envelope_data = pair['envelope_data'].copy()
                modified_envelope_data[:, 1] *= (uncertainty**2) * (safety**2)
                
                # Call the external save function for the current pair
                output_filename_base = f"{base_name} {suffix}"
                save_matplotlib_plot_and_data(
                    original_psd_data=pair['psd_data'],
                    modified_envelope_data=modified_envelope_data,
                    output_filename_base=output_filename_base,
                    output_directory=new_dir_path
                )

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

        pair = self.data_pairs[self.current_index]
        original_psd_data = pair['psd_data']
        original_envelope_data = pair['envelope_data']
        
        # Factors are read from the primary (ratio) spinners
        uncertainty = self.uncertainty_input.value
        safety = self.safety_input.value
        
        modified_envelope_data = original_envelope_data.copy()
        # Apply the factors squared, as they relate to RMS, not PSD.
        modified_envelope_data[:, 1] *= (uncertainty**2) * (safety**2)
        
        plot = create_psd_plot(original_psd_data, modified_envelope_data, pair['name'])
        self.plot_layout.children = [plot]
        
        self.status_div.text = f"<b>Displaying {self.current_index + 1}/{len(self.data_pairs)}:</b> {pair['name']}"
        self.prev_button.disabled = (self.current_index == 0)
        self.next_button.disabled = (self.current_index >= len(self.data_pairs) - 1)

        self.update_suffix_preview()
        self.save_button.disabled = not self.suffix_preview_input.value

    def load_data_callback(self):
        source_path = self.source_dir_input.value
        envelope_path = self.envelope_dir_input.value

        if not source_path or not envelope_path:
            self.status_div.text = "<b>Status:</b> Please enter both directory paths."
            return
            
        self.status_div.text = "<i>Status: Scanning directories...</i>"
        self.data_pairs = find_data_pairs(source_path, envelope_path)
        self.current_index = 0
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
