import sys
import os

# Add the project root directory to the Python path.
# This is necessary so that Python knows where to find the 'app' module
# when we run the bokeh server from the root directory.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bokeh.models import Tabs, TabPanel
from bokeh.plotting import curdoc

# Import tab creation functions from their respective files
from app.tab_optimization import create_optimization_tab
from app.tab_visualizer import create_visualizer_tab

# --- Create Each Tab ---

# 1. Optimization Tab
optimization_layout = create_optimization_tab()
tab1 = TabPanel(child=optimization_layout, title="Optimization")

# 2. Visualizer Tab
visualizer_layout = create_visualizer_tab()
tab2 = TabPanel(child=visualizer_layout, title="PSD Visualizer")

# --- Combine Tabs ---
tabs = Tabs(tabs=[tab1, tab2], sizing_mode="stretch_width")

# --- Add to Document ---
curdoc().add_root(tabs)
curdoc().title = "PSD Analysis Tool"
