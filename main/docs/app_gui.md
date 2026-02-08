# app/gui.py

## Role in the System

`app/gui.py` is the main entry point for the Bokeh web server. It serves as the root document that combines all GUI tabs into a single application interface.

## Responsibilities

- Sets up the Python path to ensure module imports work correctly
- Imports tab creation functions from `tab_optimization.py` and `tab_visualizer.py`
- Creates tab panels and combines them into a Bokeh `Tabs` widget
- Attaches the tabs to the Bokeh document for server rendering

## Dependencies

**Imports:**
- `bokeh.models.Tabs, TabPanel` - Bokeh widgets for tabbed interface
- `bokeh.plotting.curdoc` - Bokeh document context
- `app.tab_optimization.create_optimization_tab` - Creates optimization tab layout
- `app.tab_visualizer.create_visualizer_tab` - Creates visualizer tab layout

**Used In:**
- Called by Bokeh server when starting the application (via `bokeh serve app/gui.py`)

## Code Structure

The file performs three main operations:
1. **Path Setup**: Adds project root to `sys.path` to enable relative imports
2. **Tab Creation**: Calls tab creation functions to build UI layouts
3. **Document Assembly**: Combines tabs and attaches to Bokeh document

## Implementation Details

The file executes at module import time (when Bokeh server loads it). All code runs in the Bokeh document context, meaning widget creation happens during server initialization.

