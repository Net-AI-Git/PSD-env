from bokeh.models.widgets import Button, TextInput, Div
from bokeh.layouts import column, row

from .gui_utils import find_data_pairs, create_psd_plot

class VisualizerTab:
    """A class to manage the state and layout of the PSD Visualizer tab."""

    def __init__(self):
        self.data_pairs = []
        self.current_index = 0
        self.plot_layout = column(name="plot_layout", sizing_mode="stretch_width")
        
        # Create all widgets and assign callbacks to methods of this class
        self.source_dir_input = TextInput(title="Source Data Directory (contains .mat or .txt files):", placeholder="Enter path to source data...", width=600)
        self.envelope_dir_input = TextInput(title="Envelopes Directory (contains .spc.txt files):", placeholder="Enter path to envelope files...", width=600)
        self.load_button = Button(label="Load Data", button_type="primary", width=150)
        
        self.prev_button = Button(label="< Previous", width=100, disabled=True)
        self.next_button = Button(label="Next >", width=100, disabled=True)
        
        self.status_div = Div(text="<i>Status: Enter directory paths and click 'Load Data'</i>", width=400, height_policy="min")
        
        self.load_button.on_click(self.load_data_callback)
        self.prev_button.on_click(self.show_previous_callback)
        self.next_button.on_click(self.show_next_callback)

    def update_plot_and_controls(self):
        if not self.data_pairs:
            self.plot_layout.children = []
            self.status_div.text = "<b>Status:</b> No matching data pairs found."
            self.prev_button.disabled = True
            self.next_button.disabled = True
            return

        pair = self.data_pairs[self.current_index]
        plot = create_psd_plot(pair['psd_data'], pair['envelope_data'], pair['name'])
        self.plot_layout.children = [plot]
        
        self.status_div.text = f"<b>Displaying {self.current_index + 1}/{len(self.data_pairs)}:</b> {pair['name']}"
        self.prev_button.disabled = (self.current_index == 0)
        self.next_button.disabled = (self.current_index >= len(self.data_pairs) - 1)

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
        
        layout = column(
            Div(text="<h2>PSD & Envelope Visualizer</h2>"),
            row(input_layout, self.load_button, align="start"),
            controls_row,
            self.plot_layout,
            sizing_mode="stretch_width"
        )
        return layout

def create_visualizer_tab():
    """
    Creates an instance of the VisualizerTab class and returns its layout.
    This function is the entry point called by the main gui.py file.
    """
    visualizer = VisualizerTab()
    return visualizer.get_layout()
