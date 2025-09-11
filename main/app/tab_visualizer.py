from bokeh.models.widgets import Button, TextInput, Div
from bokeh.layouts import column, row

# Use a relative import to access the utility functions
from .gui_utils import find_data_pairs, create_psd_plot

# --- Module-level state ---
# Storing the data and current index here allows callbacks to access and modify them.
data_pairs = []
current_index = 0

def create_visualizer_tab():
    """
    Creates the layout and functionality for the 'PSD Visualizer' tab.
    
    This tab allows users to load a directory of PSD data and navigate
    through plots of each PSD and its corresponding envelope.
    """

    # This layout object will hold the currently displayed plot
    plot_layout = column(name="plot_layout", sizing_mode="stretch_width")

    # --- Widgets Definition ---
    source_dir_input = TextInput(title="Source Data Directory (contains .mat or .txt files):", placeholder="Enter path to source data...", width=600)
    envelope_dir_input = TextInput(title="Envelopes Directory (contains .spc.txt files):", placeholder="Enter path to envelope files...", width=600)
    load_button = Button(label="Load Data", button_type="primary", width=150)
    
    prev_button = Button(label="< Previous", width=100, disabled=True)
    next_button = Button(label="Next >", width=100, disabled=True)
    
    status_div = Div(text="<i>Status: Enter a directory path and click 'Load Data'</i>", width=400, height_policy="min")

    # --- Callback Functions ---
    
    def update_plot_and_controls():
        """
        Main function to refresh the UI. It clears the old plot, creates a new one
        based on the current_index, and updates the button states and status text.
        """
        global data_pairs, current_index
        
        if not data_pairs:
            plot_layout.children = []  # Clear the layout by assigning an empty list
            status_div.text = "<b>Status:</b> No matching data pairs found in the specified directory."
            prev_button.disabled = True
            next_button.disabled = True
            return

        # Get the current data pair and create a new plot
        pair = data_pairs[current_index]
        plot = create_psd_plot(pair['psd_data'], pair['envelope_data'], pair['name'])
        plot_layout.children = [plot] # Replace the layout's children with a new list containing just the plot
        
        # Update the status text and enable/disable navigation buttons
        status_div.text = f"<b>Displaying {current_index + 1}/{len(data_pairs)}:</b> {pair['name']}"
        prev_button.disabled = (current_index == 0)
        next_button.disabled = (current_index >= len(data_pairs) - 1)

    def load_data_callback():
        """
        Triggered by the 'Load Data' button. It reads the path, calls the utility
        function to find data, resets the index, and calls the update function.
        """
        global data_pairs, current_index
        source_path = source_dir_input.value
        envelope_path = envelope_dir_input.value

        if not source_path or not envelope_path:
            status_div.text = "<b>Status:</b> Please enter both directory paths."
            return
            
        status_div.text = f"<i>Status: Scanning directories...</i>"
        # The core logic is in our utility function, now with two paths
        data_pairs = find_data_pairs(source_path, envelope_path)
        current_index = 0
        update_plot_and_controls()

    def show_previous_callback():
        """Navigates to the previous plot in the list."""
        global current_index
        if current_index > 0:
            current_index -= 1
            update_plot_and_controls()

    def show_next_callback():
        """Navigates to the next plot in the list."""
        global data_pairs, current_index
        if current_index < len(data_pairs) - 1:
            current_index += 1
            update_plot_and_controls()
            
    # --- Assign Callbacks to Widgets ---
    load_button.on_click(load_data_callback)
    prev_button.on_click(show_previous_callback)
    next_button.on_click(show_next_callback)

    # --- Final Layout Assembly ---
    controls_row = row(prev_button, status_div, next_button, align="center")
    
    input_layout = column(source_dir_input, envelope_dir_input)

    layout = column(
        Div(text="<h2>PSD & Envelope Visualizer</h2>"),
        row(input_layout, load_button, align="start"),
        controls_row,
        plot_layout, # The plot will be dynamically inserted here
        sizing_mode="stretch_width" # Ensure the main column also stretches
    )
    
    return layout
