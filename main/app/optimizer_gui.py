import gradio as gr
from typing import Literal
import tkinter as tk
from tkinter import filedialog

# --- Backend Logic ---

def select_directory() -> str:
    """
    Opens a dialog for the user to select a directory.
    
    Returns:
        The selected directory path, or an empty string if canceled.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    root.attributes("-topmost", True)  # Bring the dialog to the front
    directory_path = filedialog.askdirectory(title="Select Input Directory")
    root.destroy()
    return directory_path if directory_path else ""

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

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                input_dir_textbox = gr.Textbox(
                    label="Input Directory",
                    placeholder="e.g., C:/Users/user/Desktop/data",
                    interactive=True,
                    scale=4
                )
                browse_button = gr.Button("Browse...", scale=1)

            min_freq_slider = gr.Number(label="Min Frequency (Hz)", value=5)
            max_freq_slider = gr.Number(label="Max Frequency (Hz)", value=2000)
            target_points_slider = gr.Slider(label="Target Points", minimum=10, maximum=100, step=1, value=45)
            target_rms_slider = gr.Slider(label="Target RMS Ratio", minimum=1.0, maximum=2.0, step=0.01, value=1.25)
            stab_wide_radio = gr.Radio(label="Stability Wide", choices=["narrow", "wide"], value="narrow")
            area_mode_radio = gr.Radio(label="Area X-Axis Mode", choices=["Log", "Linear"], value="Log")

        with gr.Column(scale=1):
            submit_button = gr.Button("Submit", variant="primary")
            output_status_textbox = gr.Textbox(label="Output Status", interactive=False)


    # --- UI Event Handling ---

    def on_browse_click():
        directory = select_directory()
        # Return a Gradio update object to change the textbox value
        return gr.update(value=directory)

    browse_button.click(
        fn=on_browse_click,
        inputs=None,
        outputs=input_dir_textbox
    )

    submit_button.click(
        fn=run_optimization_process_ui,
        inputs=[
            input_dir_textbox,
            min_freq_slider,
            max_freq_slider,
            target_points_slider,
            target_rms_slider,
            stab_wide_radio,
            area_mode_radio
        ],
        outputs=output_status_textbox
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch()
