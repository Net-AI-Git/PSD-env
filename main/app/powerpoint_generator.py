import logging
from typing import List
from pptx import Presentation
from pptx.util import Inches
import os

# Configure logger
logger = logging.getLogger(__name__)


def create_presentation_from_images(image_paths: List[str], output_dir: str) -> None:
    """
    Creates a PowerPoint presentation from a list of image paths.

    Why (Purpose and Necessity):
    This function gathers all the generated plot images into a single
    PowerPoint file. This provides a convenient way to review and present
    the visual results of the optimization process.

    What (Implementation Details):
    The function first separates images into two groups: those containing
    'details' in their filename and those that do not. It then sorts them
    to ensure a consistent order, with 'details' images appearing last.
    A new presentation is created, and each image is added to its own slide,
    centered and scaled appropriately. Finally, the presentation is saved
    to the specified output directory.

    Args:
        image_paths (List[str]): A list of absolute paths to the image files.
        output_dir (str): The directory where the final presentation
                          file will be saved.

    Returns:
        None

    Raises:
        Exception: Catches and logs any general exceptions during the
                   presentation creation process.
    """
    logger.info("Starting PowerPoint presentation generation...")
    try:
        # Separate 'details' images from others
        details_images = sorted([p for p in image_paths if 'details' in os.path.basename(p).lower()])
        other_images = sorted([p for p in image_paths if 'details' not in os.path.basename(p).lower()])

        # Combine lists, with 'details' images last
        sorted_image_paths = other_images + details_images

        if not sorted_image_paths:
            logger.warning("No images found to add to the presentation.")
            return

        prs = Presentation()
        # Use a blank slide layout (layout index 6 is typically blank)
        blank_slide_layout = prs.slide_layouts[6]

        for image_path in sorted_image_paths:
            if not os.path.exists(image_path):
                logger.warning(f"Image not found, skipping: {image_path}")
                continue

            slide = prs.slides.add_slide(blank_slide_layout)

            # --- Image Scaling and Centering ---
            # Define slide dimensions in EMUs (English Metric Units)
            slide_height = prs.slide_height
            slide_width = prs.slide_width

            # Add the picture to the slide
            pic = slide.shapes.add_picture(image_path, 0, 0)

            # Calculate the optimal width and height to fit the image
            # to the slide while maintaining aspect ratio.
            img_aspect_ratio = pic.width / pic.height
            slide_aspect_ratio = slide_width / slide_height

            if img_aspect_ratio > slide_aspect_ratio:
                # Image is wider than the slide, so width determines scale
                new_width = slide_width
                new_height = int(new_width / img_aspect_ratio)
            else:
                # Image is taller than the slide, so height determines scale
                new_height = slide_height
                new_width = int(new_height * img_aspect_ratio)

            # Center the image on the slide
            left = (slide_width - new_width) // 2
            top = (slide_height - new_height) // 2

            # Apply the calculated dimensions and position
            pic.left = left
            pic.top = top
            pic.width = new_width
            pic.height = new_height

        output_path = os.path.join(output_dir, "optimization_summary.pptx")
        prs.save(output_path)
        logger.info(f"Successfully saved presentation to {output_path}")

    except Exception as e:
        logger.error(f"Failed to create PowerPoint presentation. Error: {e}", exc_info=True)
