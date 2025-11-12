from bokeh.models.widgets import Button, Spinner, RadioButtonGroup, TextInput, Div, Slider, Checkbox, Paragraph
from bokeh.layouts import column
from bokeh.plotting import curdoc
import threading
import os
from run_code import run_optimization_process

# Global stop event for controlling optimization execution
stop_event = threading.Event()
saved_parameters = {}  # Store parameters before run for restoration

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
    
    # New widgets for missing parameters
    full_envelope_input = Checkbox(label="Perform envelope on all files by matching channel names", active=False)
    
    # Advanced optimization parameters
    strict_points_input = Checkbox(label="Strict points constraint", active=False)

    # Text Input for directory path
    input_dir_input = TextInput(value="", title="Input Directory:", width=400)

    # Status and Progress widgets
    status_div = Div(text="<div style='color: blue; font-weight: bold;'>Ready to optimize</div>", 
                     width=400, height=50)
    progress_paragraph = Paragraph(text="", width=400, height=100)

    # Run Button
    run_button = Button(label="Run Optimization", button_type="success", width=200)
    
    # Stop Button
    stop_button = Button(label="Stop Optimization", button_type="danger", width=200, visible=False)

    # --- Validation Function ---
    def validate_inputs():
        """Validates all input parameters and returns (is_valid, error_message)"""
        try:
            min_freq = min_freq_input.value
            max_freq = max_freq_input.value
            target_points = target_points_input.value
            target_area_ratio = target_area_ratio_input.value
            input_dir = input_dir_input.value.strip()
            
            # Check frequency range
            if min_freq >= max_freq:
                return False, "Min frequency must be less than max frequency"
            
            # Check target points
            if target_points <= 0:
                return False, "Target points must be greater than 0"
            
            # Check target area ratio
            if target_area_ratio <= 0:
                return False, "Target area ratio must be greater than 0"
            
            # Check input directory if provided
            if input_dir and not os.path.exists(input_dir):
                return False, f"Input directory does not exist: {input_dir}"
            
            return True, ""
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    # --- Status Management Functions ---
    def update_status(message, color="blue"):
        """Updates the status div with a colored message"""
        status_div.text = f"<div style='color: {color}; font-weight: bold;'>{message}</div>"
    
    def set_widgets_enabled(enabled):
        """Enables or disables all input widgets"""
        min_freq_input.disabled = not enabled
        max_freq_input.disabled = not enabled
        target_points_input.disabled = not enabled
        target_area_ratio_input.disabled = not enabled
        stab_wide_input.disabled = not enabled
        area_x_axis_mode_input.disabled = not enabled
        full_envelope_input.disabled = not enabled
        strict_points_input.disabled = not enabled
        input_dir_input.disabled = not enabled
        run_button.disabled = not enabled

    # --- Parameter Save/Restore Functions ---
    def save_current_parameters():
        """
        Saves all current widget values for restoration after stop.
        
        Why: When user stops optimization, we need to restore the exact UI state
        they had before clicking Run, simulating a page refresh.
        
        Returns:
            dict: Dictionary containing all widget values
        """
        return {
            'min_freq': min_freq_input.value,
            'max_freq': max_freq_input.value,
            'target_points': target_points_input.value,
            'target_area_ratio': target_area_ratio_input.value,
            'stab_wide': stab_wide_input.active,
            'area_x_axis_mode': area_x_axis_mode_input.active,
            'full_envelope': full_envelope_input.active,
            'strict_points': strict_points_input.active,
            'input_dir': input_dir_input.value
        }

    def restore_parameters(params):
        """
        Restores all widget values from saved parameters.
        
        Why: After stopping optimization, restore UI to exact pre-run state.
        
        Args:
            params (dict): Dictionary containing saved widget values
        """
        min_freq_input.value = params['min_freq']
        max_freq_input.value = params['max_freq']
        target_points_input.value = params['target_points']
        target_area_ratio_input.value = params['target_area_ratio']
        stab_wide_input.active = params['stab_wide']
        area_x_axis_mode_input.active = params['area_x_axis_mode']
        full_envelope_input.active = params['full_envelope']
        strict_points_input.active = params['strict_points']
        input_dir_input.value = params['input_dir']

    # --- Callback Function ---
    def run_optimization_callback():
        """Main callback function that runs the optimization process"""
        # Validate inputs first
        is_valid, error_msg = validate_inputs()
        if not is_valid:
            update_status(f"Error: {error_msg}", "red")
            return
        
        # Save current parameters for restoration after stop
        global saved_parameters
        saved_parameters = save_current_parameters()
        
        # Clear stop event flag
        stop_event.clear()
        
        # Update status and disable widgets
        update_status("Optimization in progress... Please wait", "orange")
        set_widgets_enabled(False)
        progress_paragraph.text = "Starting optimization process..."
        
        # Toggle buttons: hide Run, show Stop
        run_button.visible = False
        stop_button.visible = True
        
        # Get all parameter values
        min_freq = min_freq_input.value
        max_freq = max_freq_input.value
        target_points = target_points_input.value
        target_area_ratio = target_area_ratio_input.value
        stab_wide = "narrow" if stab_wide_input.active == 0 else "wide"
        area_x_axis_mode = "Log" if area_x_axis_mode_input.active == 0 else "Linear"
        full_envelope = full_envelope_input.active
        input_dir = input_dir_input.value.strip() if input_dir_input.value.strip() else None
        strict_points = strict_points_input.active
        
        def run_in_background():
            """Runs the optimization in a background thread"""
            try:
                # Update progress
                def update_progress(msg):
                    curdoc().add_next_tick_callback(lambda: setattr(progress_paragraph, 'text', msg))
                
                # Run the optimization process with all parameters
                update_progress("Configuring optimization parameters...")
                
                run_optimization_process(
                    min_frequency_hz=min_freq,
                    max_frequency_hz=max_freq,
                    target_points=target_points,
                    target_area_ratio=target_area_ratio,
                    stab_wide=stab_wide,
                    area_x_axis_mode=area_x_axis_mode,
                    input_dir=input_dir,
                    full_envelope=full_envelope,
                    stop_event=stop_event,
                    strict_points=strict_points
                )
                
                # Success - update status and re-enable widgets
                def on_success():
                    update_status("Optimization completed successfully! Results saved to results directory.", "green")
                    set_widgets_enabled(True)
                    progress_paragraph.text = "Optimization completed. Check the results directory for output files."
                    # Toggle buttons back: show Run, hide Stop
                    run_button.visible = True
                    stop_button.visible = False
                
                curdoc().add_next_tick_callback(on_success)
                
            except Exception as e:
                # Error - update status and re-enable widgets
                def on_error():
                    update_status(f"Error: {str(e)}", "red")
                    set_widgets_enabled(True)
                    progress_paragraph.text = f"Optimization failed: {str(e)}"
                    # Toggle buttons back: show Run, hide Stop
                    run_button.visible = True
                    stop_button.visible = False
                
                curdoc().add_next_tick_callback(on_error)
        
        # Start background thread
        thread = threading.Thread(target=run_in_background, daemon=True)
        thread.start()

    def stop_optimization_callback():
        """
        Handles stop button click - terminates optimization and resets UI.
        
        Why: Allows user to forcefully stop optimization at any point and
        return to ready state with all settings preserved.
        """
        global saved_parameters
        
        # Signal the optimization thread to stop
        stop_event.set()
        
        # Restore UI state
        restore_parameters(saved_parameters)
        update_status("Optimization stopped by user", "orange")
        progress_paragraph.text = "Optimization was stopped. You can start a new run."
        
        # Toggle buttons
        run_button.visible = True
        stop_button.visible = False
        
        # Re-enable all widgets
        set_widgets_enabled(True)

    # Connect callbacks to buttons
    run_button.on_click(run_optimization_callback)
    stop_button.on_click(stop_optimization_callback)

    # --- Layout ---
    layout = column(
        title,
        min_freq_input,
        max_freq_input,
        target_points_input,
        target_area_ratio_input,
        strict_points_input,
        Div(text="<b>Stab Wide:</b>"),
        stab_wide_input,
        Div(text="<b>Area X-Axis Mode:</b>"),
        area_x_axis_mode_input,
        Div(text="<b>Full Envelope Mode:</b>"),
        full_envelope_input,
        input_dir_input,
        status_div,
        progress_paragraph,
        run_button,
        stop_button,
        name="optimization_layout", # Add a name for potential future access
    )
    
    return layout
