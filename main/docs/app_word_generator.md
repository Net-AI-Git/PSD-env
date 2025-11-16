# app/word_generator.py

## Role in the System

`app/word_generator.py` creates Word documents from collections of image files, using a template for consistent formatting with captions and spacing.

## Responsibilities

- Loads Word template from `Data/format.docx`
- Analyzes template to extract caption format and spacing
- Creates document with images, captions, and proper formatting
- Applies RTL (right-to-left) formatting for Hebrew text support

## Dependencies

**Imports:**
- `os` - File path operations
- `copy` - Deep copying for template elements
- `traceback` - Error traceback printing
- `typing` - Type hints
- `docx` (python-docx) - Word document manipulation
- `docx.shared.Inches` - Size units
- `docx.text.paragraph.Paragraph` - Paragraph objects
- `docx.enum.text.WD_ALIGN_PARAGRAPH` - Alignment constants

**Used In:**
- `app/save_utils.py` - Called by `generate_word_document_from_images()`
- `run_code.py` - Called after optimization to create Word documents

## Functions

### Function: `_add_image_block(doc, image_path, caption_template, num_spacers)`

**Location:** `app/word_generator.py`

**Purpose:**  
Adds a complete "image block" (image, caption, and spacing) to a Word document, mirroring the structure found in the template.

**Parameters:**
- `doc (Document)` - The Word document object
- `image_path (str)` - Full path to the image file to add
- `caption_template (Paragraph)` - Template paragraph for the caption (from template analysis)
- `num_spacers (int)` - Number of empty paragraphs to add after the caption

**Returns:**
None

**Side Effects:**
- Adds image to document (width 6.0 inches)
- Adds caption paragraph by deep copying template and replacing "A title" placeholder
- Adds spacer paragraphs for visual spacing
- Prints error messages if image processing fails

**Error Handling:**
- Catches exceptions during image block creation and prints detailed error with traceback
- Handles missing placeholder text by appending title to end of caption

**Used In:**
- `create_images_document()` - Called for each image in the sorted lists

### Function: `_analyze_template(template_path)`

**Location:** `app/word_generator.py`

**Purpose:**  
Analyzes the template file to extract the caption paragraph and count spacer paragraphs that follow it. This information is used to replicate the template structure.

**Parameters:**
- `template_path (str)` - Path to the template Word document

**Returns:**
- `tuple[Paragraph, int]` - Tuple containing:
  - Caption template paragraph (or None if not found)
  - Number of spacer paragraphs following the caption

**Side Effects:**
- Loads template document
- Searches for caption field identifier in XML
- Counts consecutive empty paragraphs after caption

**Error Handling:**
- Returns `(None, 0)` if template cannot be loaded
- Returns `(None, 0)` if caption template not found
- Prints error message if template loading fails

**Used In:**
- `create_images_document()` - Called at start to analyze template

### Function: `create_images_document(image_paths, output_path, document_name="Image_Document.docx")`

**Location:** `app/word_generator.py`

**Purpose:**  
Creates a Word document from a template, following a precise plan. Analyzes template, validates contents, and builds document from scratch based on that analysis.

**Parameters:**
- `image_paths (list[str])` - Full paths to the image files
- `output_path (str)` - Directory where the Word document will be saved
- `document_name (str, optional)` - Filename for the output Word document (default: "Image_Document.docx")

**Returns:**
None

**Side Effects:**
- Analyzes template `Data/format.docx` to find caption template and spacer count
- Creates new blank document
- Sorts images into 'main' and 'details' lists
- Adds image blocks for all images (main first, then details)
- Applies document-wide formatting (center align, RTL)
- Saves document to `{output_path}/{document_name}`
- Prints progress messages and errors

**Error Handling:**
- Exits early if template analysis fails (caption template not found)
- Prints error messages if document save fails
- Handles exceptions during image block addition (continues with next image)

**Used In:**
- `app/save_utils.generate_word_document_from_images()` - Called with collected image paths
- `run_code.py` - Called after optimization to create Word documents

