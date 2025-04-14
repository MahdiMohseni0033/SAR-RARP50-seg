# Need to check
import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import argparse
from pathlib import Path
import logging
from tqdm import tqdm  # For progress bar in video processing

# Import utility functions from inf_utils.py
from inf_utils import (
    get_palette,
    process_single_result,
    NUM_CLASSES as DEFAULT_NUM_CLASSES, # Use NUM_CLASSES from utils as default
    CLASS_NAMES
)

# Configure basic logging
# Define custom format including function name
log_format = '%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

# --- Constants ---
SUPPORTED_IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
SUPPORTED_VIDEO_FORMATS = ('.mp4', '.avi', '.mov', '.mkv')


# --- Core Processing Functions ---

def process_image(
    model: YOLO,
    image_path: Path,
    output_dir: Path,
    palette: list,
    conf_thres: float,
    iou_thres: float,
    visualize: bool
):
    """Runs inference on a single image and saves results."""
    logger.info(f"Processing image: {image_path}")
    try:
        original_image = cv2.imread(str(image_path))
        if original_image is None:
            raise IOError(f"Could not read image file: {image_path}")

        # Perform prediction
        results = model.predict(
            original_image,
            conf=conf_thres,
            iou=iou_thres,
            verbose=False # Keep console clean, rely on logging
        )

        if not results or len(results) == 0:
             logger.warning(f"Model returned no results for {image_path}.")
             return

        # Process the first (and likely only) result
        combined_mask, colorized_mask, overlayed_image = process_single_result(
            results[0], original_image, palette, conf_thres
        )

        # --- Save Outputs ---
        base_name = image_path.stem

        if combined_mask is not None:
            raw_mask_output_path = output_dir / f"{base_name}_raw_mask.png"
            saveable_raw_mask = combined_mask.astype(np.uint8) # Ensure correct type
            cv2.imwrite(str(raw_mask_output_path), saveable_raw_mask)
            logger.debug(f"Saved raw class ID mask to: {raw_mask_output_path}")

        if colorized_mask is not None:
            mask_output_path = output_dir / f"{base_name}_mask.png"
            cv2.imwrite(str(mask_output_path), colorized_mask)
            logger.debug(f"Saved colorized mask to: {mask_output_path}")

        if visualize and overlayed_image is not None:
            overlay_output_path = output_dir / f"{base_name}_overlay.png"
            cv2.imwrite(str(overlay_output_path), overlayed_image)
            logger.info(f"Saved overlay image to: {overlay_output_path}")

            # Optional: Save stacked image (overlay + colorized mask)
            if colorized_mask is not None:
                 # Ensure colorized mask has same height as overlay
                 if colorized_mask.shape[0] != overlayed_image.shape[0] or colorized_mask.shape[1] != overlayed_image.shape[1]:
                     colorized_mask_resized = cv2.resize(
                         colorized_mask,
                         (overlayed_image.shape[1], overlayed_image.shape[0]),
                         interpolation=cv2.INTER_NEAREST
                     )
                 else:
                     colorized_mask_resized = colorized_mask

                 stacked_image = np.concatenate((overlayed_image, colorized_mask_resized), axis=1)
                 stacked_output_path = output_dir / f"{base_name}_stacked.png"
                 cv2.imwrite(str(stacked_output_path), stacked_image)
                 logger.debug(f"Saved stacked image to: {stacked_output_path}")

        elif not visualize and overlayed_image is not None and combined_mask is None:
             # If visualize is false but we only got the original image back
             # (because no masks passed threshold), maybe save the original?
             # Or just log that no output was generated.
             logger.info(f"No segmentation masks met the threshold for {image_path}. No overlay generated.")


    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}", exc_info=True)


def process_video(
    model: YOLO,
    video_path: Path,
    output_dir: Path,
    palette: list,
    conf_thres: float,
    iou_thres: float
):
    """Runs inference on a video file and saves the output video with overlays."""
    logger.info(f"Processing video: {video_path}")
    output_video_path = output_dir / f"{video_path.stem}_overlay.mp4" # Or choose another format

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Error opening video file: {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video properties: {frame_width}x{frame_height} @ {fps:.2f} FPS, {total_frames} frames")

    # Define the codec and create VideoWriter object
    # common codecs: 'mp4v', 'XVID', 'MJPG', 'avc1'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use 'mp4v' for .mp4 output
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))

    logger.info(f"Output video will be saved to: {output_video_path}")

    frame_count = 0
    with tqdm(total=total_frames, desc="Processing video frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.info("Reached end of video or cannot read frame.")
                break

            frame_count += 1
            try:
                # Perform prediction on the frame
                results = model.predict(
                    frame,
                    conf=conf_thres,
                    iou=iou_thres,
                    verbose=False # Keep console clean
                )

                if not results or len(results) == 0:
                     logger.warning(f"Frame {frame_count}: Model returned no results.")
                     output_frame = frame # Write original frame if no results
                else:
                    # Process the result for the current frame
                    _, _, overlayed_frame = process_single_result(
                        results[0], frame, palette, conf_thres
                    )
                    # If processing failed or no masks found, overlayed_frame might be None or the original frame
                    output_frame = overlayed_frame if overlayed_frame is not None else frame

                # Write the frame (with or without overlay) to the output video
                out.write(output_frame)
                pbar.update(1)

            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {e}", exc_info=True)
                # Optionally write the original frame if processing fails
                out.write(frame)
                pbar.update(1)


    # Release everything when job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows() # Just in case any windows were opened
    logger.info(f"Finished processing video. Output saved to {output_video_path}")


# --- Main Execution ---

def main(args):
    """Main function to handle argument parsing and initiate processing."""
    model_path = Path(args.model_path)
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    conf_thres = args.conf_thres
    iou_thres = args.iou_thres
    visualize = args.visualize # Only relevant for image/directory processing

    # --- Validate Paths ---
    if not model_path.is_file():
        # Attempt to resolve relative to script location if not found initially
        script_dir = Path(__file__).parent
        potential_path = script_dir / model_path
        if potential_path.is_file():
            model_path = potential_path
            logger.info(f"Resolved model path to: {model_path}")
        else:
            logger.error(f"Model file not found at specified path: {args.model_path} or relative path: {potential_path}")
            return # Exit if model not found

    if not input_path.exists():
        logger.error(f"Input path not found: {input_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory set to: {output_dir}")

    # --- Load Model ---
    try:
        model = YOLO(model_path)
        logger.info(f"Successfully loaded YOLO model from: {model_path}")

        # Determine number of classes from the loaded model if possible
        if hasattr(model, 'names'):
            num_classes_model = len(model.names)
            model_class_names = model.names # This is often a dict {id: name}
            logger.info(f"Model reports {num_classes_model} classes: {model_class_names}")
            # You might want to validate num_classes_model against expected DEFAULT_NUM_CLASSES
            if num_classes_model != DEFAULT_NUM_CLASSES:
                 logger.warning(f"Model has {num_classes_model} classes, but expected {DEFAULT_NUM_CLASSES}. Using model's count for palette generation.")
                 effective_num_classes = num_classes_model
            else:
                 effective_num_classes = DEFAULT_NUM_CLASSES
        else:
            logger.warning("Could not automatically determine number of classes from model. Using default value.")
            effective_num_classes = DEFAULT_NUM_CLASSES

    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}", exc_info=True)
        return

    # --- Generate Color Palette ---
    # Add 1 for the background class (index 0)
    palette = get_palette(effective_num_classes + 1)
    logger.info(f"Generated color palette for {effective_num_classes + 1} entries (including background).")


    # --- Determine Input Type and Process ---
    input_suffix = input_path.suffix.lower()

    if input_path.is_file():
        if input_suffix in SUPPORTED_IMAGE_FORMATS:
            process_image(model, input_path, output_dir, palette, conf_thres, iou_thres, visualize)
        elif input_suffix in SUPPORTED_VIDEO_FORMATS:
            # Visualize flag is ignored for video, overlay is always created
            if visualize:
                 logger.info("Note: --visualize flag is ignored for video input. Overlay video is always generated.")
            process_video(model, input_path, output_dir, palette, conf_thres, iou_thres)
        else:
            logger.error(f"Unsupported file type: {input_suffix}. Please provide a supported image or video file.")

    elif input_path.is_dir():
        logger.info(f"Processing all supported images in directory: {input_path}")
        image_files = sorted([
            f for f in input_path.glob('*')
            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_FORMATS
        ])
        if not image_files:
            logger.warning(f"No supported image files found in directory: {input_path}")
        else:
            logger.info(f"Found {len(image_files)} images to process.")
            for image_file in image_files:
                process_image(model, image_file, output_dir, palette, conf_thres, iou_thres, visualize)
    else:
        logger.error(f"Input path {input_path} is neither a file nor a directory.")

    logger.info("Inference process finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 segmentation inference on images or videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help message
    )
    parser.add_argument(
        "--model_path", type=str, default="runs/segment/train/weights/best.pt",
        help="Path to the trained YOLO segmentation model (.pt file)."
    )
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Path to the input image, video file, or directory of images."
    )
    parser.add_argument(
        "--output_dir", type=str, default="output",
        help="Directory to save the output (masks, overlays, output video)."
    )
    parser.add_argument(
        "--conf_thres", type=float, default=0.5,
        help="Confidence threshold for filtering detections."
    )
    parser.add_argument(
        "--iou_thres", type=float, default=0.7,
        help="IoU threshold for Non-Maximum Suppression (NMS)."
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Save visualization overlay images (ignored for video input)."
    )

    args = parser.parse_args()
    main(args)