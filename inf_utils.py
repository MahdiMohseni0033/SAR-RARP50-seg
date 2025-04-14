# Need to check
import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
import torch
logger = logging.getLogger(__name__)

# Define Class Information (as provided)
NUM_CLASSES = 9
CLASS_NAMES = [
    'Tool shaft', 'Tool clasper', 'Tool wrist', 'Thread', 'Clamps',
    'Suturing needle', 'Suction tool', 'Catheter', 'Needle Holder'
]

# --- Palette Generation ---

def get_palette(num_classes: int) -> List[Tuple[int, int, int]]:
    """
    Generates a visually distinct color palette for segmentation masks.
    Index 0 is always black (background).
    """
    if num_classes <= 0:
        return []

    # Start with black for background (index 0)
    palette = [(0, 0, 0)]

    # Define specific colors for the first few classes for consistency
    predefined_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),     # Red, Green, Blue
        (255, 255, 0), (255, 0, 255), (0, 255, 255), # Yellow, Magenta, Cyan
        (128, 0, 0), (0, 128, 0), (0, 0, 128),     # Dark Red, Green, Blue
        (255, 128, 0), (128, 255, 0), (0, 255, 128), # Orange variants
        (0, 128, 255), (128, 0, 255), (255, 0, 128), # Purple/Pink variants
        (192, 192, 192), (128, 128, 128), (64, 64, 64) # Grays
    ]

    num_predefined = len(predefined_colors)
    num_to_generate = num_classes - 1 # Exclude background

    # Use predefined colors first
    palette.extend(predefined_colors[:num_to_generate])

    # Generate random colors if more are needed
    remaining = num_to_generate - len(palette) + 1
    if remaining > 0:
        # Set a seed for reproducibility if needed, otherwise remove/change it
        # np.random.seed(42)
        random_colors = [
            (np.random.randint(50, 256), np.random.randint(50, 256), np.random.randint(50, 256))
            for _ in range(remaining)
        ]
        palette.extend(random_colors)

    # Ensure the palette has exactly num_classes entries
    return palette[:num_classes]

# --- Mask Processing ---

def colorize_mask(mask: np.ndarray, palette: List[Tuple[int, int, int]]) -> np.ndarray:
    """
    Colorizes a single-channel mask image using the provided palette.
    Assumes mask contains class indices (0 for background).
    """
    if mask.ndim != 2:
        raise ValueError(f"Input mask must be 2D, but got shape {mask.shape}")
    if not issubclass(mask.dtype.type, np.integer):
         logger.warning(f"Mask dtype is {mask.dtype}, converting to uint8.")
         mask = mask.astype(np.uint8)

    num_classes_palette = len(palette)
    colorized_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    unique_mask_values = np.unique(mask)

    for class_id in unique_mask_values:
        if class_id == 0: # Skip background for explicit coloring if needed elsewhere
            continue
        if class_id < num_classes_palette:
            colorized_mask[mask == class_id] = palette[class_id]
        else:
            logger.warning(
                f"Mask contains class index {class_id} which is out of bounds "
                f"for the palette size {num_classes_palette}. Assigning black."
            )
            # Keep it black (default) or assign a default color like white:
            # colorized_mask[mask == class_id] = (255, 255, 255)

    return colorized_mask


def overlay_mask(image: np.ndarray, colorized_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlays a colorized mask onto an image with transparency.
    """
    if image.shape[:2] != colorized_mask.shape[:2]:
        logger.warning(
            f"Image shape {image.shape[:2]} and colorized mask shape "
            f"{colorized_mask.shape[:2]} differ. Resizing mask to image size."
        )
        # Use INTER_NEAREST to avoid introducing new colors/classes
        colorized_mask = cv2.resize(
            colorized_mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    if image.dtype != np.uint8:
         image = image.astype(np.uint8)
    if colorized_mask.dtype != np.uint8:
        colorized_mask = colorized_mask.astype(np.uint8)

    # Create a boolean mask where the colorized mask is not black ([0, 0, 0])
    # This ensures we only blend where there's an actual detection mask
    foreground_mask = np.any(colorized_mask != [0, 0, 0], axis=-1)

    # Apply overlay only on the foreground pixels
    blended_image = image.copy()
    blended_image[foreground_mask] = cv2.addWeighted(
        image[foreground_mask],
        1 - alpha,
        colorized_mask[foreground_mask],
        alpha,
        0  # gamma value
    )

    return blended_image


def create_combined_mask(
    masks_data: Optional[torch.Tensor],
    boxes_data: Optional[torch.Tensor],
    orig_shape: Tuple[int, int],
    conf_thres: float = 0.5
) -> Optional[np.ndarray]:
    """
    Processes raw mask and box data from YOLO results to create a single
    combined mask where pixel values correspond to class IDs (starting from 1).

    Args:
        masks_data: Tensor of masks from YOLO result (e.g., result.masks.data).
        boxes_data: Tensor of boxes from YOLO result (e.g., result.boxes).
                    Needed for confidence scores and class IDs.
        orig_shape: Tuple (height, width) of the original image.
        conf_thres: Confidence threshold to filter detections.

    Returns:
        A 2D numpy array (uint8) representing the combined mask, or None if no
        valid masks are found after filtering.
    """
    if masks_data is None or boxes_data is None or len(masks_data) == 0 or len(boxes_data) == 0:
        logger.debug("No masks or boxes data found in the result.")
        return None

    confidences = boxes_data.conf.cpu().numpy()
    class_ids = boxes_data.cls.cpu().numpy().astype(int)
    masks_cpu = masks_data.cpu().numpy()

    # Filter based on confidence threshold
    conf_mask = confidences >= conf_thres
    if not np.any(conf_mask):
        logger.debug(f"No detections above confidence threshold {conf_thres}.")
        return None

    filtered_masks = masks_cpu[conf_mask]
    filtered_class_ids = class_ids[conf_mask]

    if filtered_masks.shape[0] == 0:
        logger.debug("No masks remained after confidence filtering.")
        return None

    orig_h, orig_w = orig_shape
    # Initialize combined mask with 0 (background)
    combined_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

    # Process masks in descending order of confidence? (Optional, might matter for overlaps)
    # sorted_indices = np.argsort(confidences[conf_mask])[::-1]
    # for i in sorted_indices:

    for i in range(filtered_masks.shape[0]):
        mask_instance = filtered_masks[i]
        # Class IDs from model are 0-based. Add 1 for the combined mask
        # so that background is 0 and classes start from 1.
        class_id = filtered_class_ids[i] + 1

        # Resize mask to original image dimensions using nearest neighbor
        mask_instance_resized = cv2.resize(
            mask_instance,
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST
        )

        # Binarize the resized mask (sometimes they are float masks)
        binary_mask = (mask_instance_resized > 0.5).astype(np.uint8)

        # Add the mask to the combined mask, assigning the class ID
        # Use np.maximum to handle overlaps: higher confidence masks added later
        # might overwrite lower confidence ones if sorted, otherwise it depends on loop order.
        # Or simply let the last mask drawn win in case of overlap:
        combined_mask[binary_mask == 1] = class_id

        # Alternative: Prioritize higher class IDs in overlaps (if desired)
        # combined_mask = np.maximum(combined_mask, binary_mask * class_id)

    # Check if any masks were actually added
    if np.max(combined_mask) == 0:
         logger.debug("Combined mask is empty after processing all instances.")
         return None

    return combined_mask


def process_single_result(
    result: object, # Ultralytics result object
    original_image: np.ndarray,
    palette: List[Tuple[int, int, int]],
    conf_thres: float
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Processes a single YOLO result object to generate masks and overlays.

    Args:
        result: The result object from model.predict().
        original_image: The corresponding input image (numpy array).
        palette: The color palette for visualization.
        conf_thres: Confidence threshold for filtering detections.

    Returns:
        A tuple containing:
        - combined_mask (np.ndarray or None): Raw class ID mask.
        - colorized_mask (np.ndarray or None): Colorized visualization mask.
        - overlayed_image (np.ndarray or None): Image with mask overlayed.
    """
    if not hasattr(result, 'masks') or not hasattr(result, 'boxes'):
        logger.warning("Result object lacks 'masks' or 'boxes' attribute.")
        return None, None, None

    orig_h, orig_w = original_image.shape[:2]

    combined_mask = create_combined_mask(
        result.masks.data if result.masks else None,
        result.boxes if result.boxes else None,
        (orig_h, orig_w),
        conf_thres
    )

    if combined_mask is None:
        logger.debug("No combined mask generated.")
        # Return the original image if no overlay can be made
        return None, None, original_image

    # Colorize the combined mask
    colorized_mask = colorize_mask(combined_mask, palette)

    # Overlay the colorized mask on the original image
    overlayed_image = overlay_mask(original_image, colorized_mask, alpha=0.6) # Adjust alpha as needed

    return combined_mask, colorized_mask, overlayed_image