from bokeh.models.widgets import Button, Spinner, RadioButtonGroup, TextInput, Div, Slider
from bokeh.layouts import column

def create_optimization_tab():
    """
    Creates the layout for the 'Optimization' tab.
    
    All widgets and their configurations for the optimization process are defined here.
    """
    # --- Widgets Definition ---

    # Title
    title = Div(text="<h1>Optimization Process Controller</h1>")

    # Numeric Inputs
    min_freq_input = Spinner(title="Min Frequency (Hz)", low=0, high=1000, step=1, value=5, width=200)
    max_freq_input = Spinner(title="Max Frequency (Hz)", low=1, high=10000, step=1, value=2000, width=200)
    target_points_input = Slider(title="Target Points", start=1, end=200, step=1, value=45, width=400)
    target_area_ratio_input = Slider(title="Target Area Ratio", start=0.1, end=5.0, step=0.01, value=1.25, width=400)

    # Radio Buttons for categorical inputs
    stab_wide_input = RadioButtonGroup(labels=["narrow", "wide"], active=0)
    area_x_axis_mode_input = RadioButtonGroup(labels=["Log", "Linear"], active=0)

    # Text Input for directory path
    input_dir_input = TextInput(value="", title="Input Directory:", width=400)

    # Run Button
    run_button = Button(label="Run Optimization", button_type="success", width=200)

    # --- Layout ---
    layout = column(
        title,
        min_freq_input,
        max_freq_input,
        target_points_input,
        target_area_ratio_input,
        Div(text="<b>Stab Wide:</b>"),
        stab_wide_input,
        Div(text="<b>Area X-Axis Mode:</b>"),
        area_x_axis_mode_input,
        input_dir_input,
        run_button,
        name="optimization_layout" # Add a name for potential future access
    )
    
    return layout
