# app/powerpoint_generator.py

## Role in the System

`app/powerpoint_generator.py` creates PowerPoint presentations from collections of image files, using a template for consistent formatting.

## Responsibilities

- Loads PowerPoint template from `Data/format.pptx`
- Creates slides from images with titles extracted from filenames
- Saves final presentation to output directory

## Dependencies

**Imports:**
- `logging` - Logging system
- `pptx` (python-pptx) - PowerPoint file manipulation
- `pptx.util.Inches` - Size units
- `os` - File path operations
- `copy.deepcopy` - Deep copying for slide duplication

**Used In:**
- `app/tab_visualizer.py` - Called by `save_changes_callback()` after saving images
- `run_code.py` - Called after optimization to create presentations

## Functions

### Function: `_duplicate_slide(prs, slide_to_copy)`

**Location:** `app/powerpoint_generator.py`

**Purpose:**  
Duplicates a slide in a presentation by manually copying all shapes. This is necessary because python-pptx doesn't provide a direct slide duplication method.

**Parameters:**
- `prs (pptx.presentation.Presentation)` - The presentation object
- `slide_to_copy (pptx.slide.Slide)` - The slide to be duplicated

**Returns:**
- `pptx.slide.Slide` - The newly created duplicate slide

**Side Effects:**
- Adds new slide to presentation
- Copies all shapes from source slide to new slide

**Error Handling:**
None (assumes valid presentation and slide objects)

**Used In:**
- `create_presentation_from_images()` - Called for each image to create a slide

### Function: `create_presentation_from_images(image_paths, output_dir)`

**Location:** `app/powerpoint_generator.py`

**Purpose:**  
Creates a PowerPoint presentation from a list of image paths using a template. Each image becomes a slide with the image centered and a title extracted from the filename.

**Parameters:**
- `image_paths (list[str])` - List of absolute paths to image files
- `output_dir (str)` - Directory where the final presentation will be saved

**Returns:**
None

**Side Effects:**
- Loads template from `Data/format.pptx` (or creates blank presentation if template not found)
- Creates slides by duplicating master slide for each image
- Updates title placeholder ("A title") with filename (without extension)
- Adds and centers images on slides
- Removes original template slide
- Saves presentation as `optimization_summary.pptx` in output directory
- Logs progress and errors

**Error Handling:**
- Handles missing template by creating blank presentation
- Handles empty presentation template by creating at least one slide
- Skips images that don't exist
- Catches and logs exceptions during processing
- Sorts images to show main images before details images

**Used In:**
- `app/tab_visualizer.py` - Called by `save_changes_callback()` after saving images
- `run_code.py` - Called after optimization to create presentations

