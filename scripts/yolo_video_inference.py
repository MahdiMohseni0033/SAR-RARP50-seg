import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import argparse
from pathlib import Path
import time
import random

# Class names
CLASS_NAMES = [
    'Tool shaft',
    'Tool clasper',
    'Tool wrist',
    'Thread',
    'Clamps',
    'Suturing needle',
    'Suction tool',
    'Catheter',
    'Needle Holder'
]


# Generate distinct colors for each class - using HSV to ensure good visual separation
def generate_colors(num_classes):
    colors = []
    for i in range(num_classes):
        # Use HSV color space for better distinctness
        hue = int(i * 255 / num_classes)
        # Full saturation and value for vibrant colors
        hsv_color = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        # Convert to BGR for OpenCV
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        # Convert to RGB and then to hex format
        colors.append((int(bgr_color[2]), int(bgr_color[1]), int(bgr_color[0])))
    return colors


def draw_legend(frame, class_names, colors):
    """Draw a legend with class names and their corresponding colors."""
    # Configuration
    start_x = 20
    start_y = 30
    box_size = 15
    text_offset = 5
    line_spacing = 5
    text_scale = 0.5
    text_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get the width of the longest class name to ensure the background is wide enough
    max_width = 0
    for name in class_names:
        (text_width, _), _ = cv2.getTextSize(name, font, text_scale, text_thickness)
        max_width = max(max_width, text_width)

    # Calculate background dimensions
    legend_height = len(class_names) * (box_size + line_spacing) + line_spacing
    legend_width = start_x + box_size + text_offset + max_width + 20

    # Create semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (legend_width, legend_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame, 0)

    # Draw class boxes and names
    for i, name in enumerate(class_names):
        y = start_y + i * (box_size + line_spacing)

        # Draw color box
        cv2.rectangle(frame, (start_x, y - box_size), (start_x + box_size, y), colors[i], -1)
        cv2.rectangle(frame, (start_x, y - box_size), (start_x + box_size, y), (0, 0, 0), 1)

        # Draw class name
        cv2.putText(frame, name, (start_x + box_size + text_offset, y - 2),
                    font, text_scale, (255, 255, 255), text_thickness)

    return frame


def process_video(model_path, video_path, output_path, conf_threshold=0.3):
    """Process video with YOLOv8 segmentation model and overlay masks."""
    # Load model
    model = YOLO(model_path)

    # Generate colors for classes
    colors = generate_colors(len(CLASS_NAMES))

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process frames
    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model.predict(frame, conf=conf_threshold, verbose=False)[0]

        # Create a copy for drawing
        overlay_frame = frame.copy()

        # Process each detected segment
        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()
            boxes = results.boxes.data.cpu().numpy()

            for i, mask in enumerate(masks):
                # Get class ID
                class_id = int(boxes[i][5])

                # Ensure the class_id is within our range
                if class_id < len(colors):
                    # Create binary mask
                    mask = mask.reshape(height, width)
                    binary_mask = (mask > 0.5).astype(np.uint8)

                    # Create colored mask
                    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
                    colored_mask[binary_mask == 1] = colors[class_id]

                    # Blend with original frame
                    alpha = 0.5  # Transparency factor
                    mask_area = (binary_mask == 1)
                    overlay_frame[mask_area] = cv2.addWeighted(
                        overlay_frame[mask_area],
                        1 - alpha,
                        colored_mask[mask_area],
                        alpha,
                        0
                    )

                    # Add contour for better visibility
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay_frame, contours, -1, colors[class_id], 2)

        # Add legend
        overlay_frame = draw_legend(overlay_frame, CLASS_NAMES, colors)

        # Add frame number and processing stats
        elapsed_time = time.time() - start_time
        fps_text = f"FPS: {frame_idx / elapsed_time:.1f}" if elapsed_time > 0 else "FPS: --"
        progress = f"Frame: {frame_idx}/{frame_count}"

        cv2.putText(overlay_frame, fps_text, (width - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overlay_frame, progress, (width - 200, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Write frame to output video
        out.write(overlay_frame)

        # Update frame index
        frame_idx += 1

        # Print progress
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx}/{frame_count} frames ({frame_idx / frame_count * 100:.1f}%)")

    # Release resources
    cap.release()
    out.release()

    duration = time.time() - start_time
    print(f"Video processing complete. Duration: {duration:.2f}s, Average FPS: {frame_idx / duration:.2f}")
    print(f"Output saved to {output_path}")

    return output_path


def main():
    """Parse arguments and process video."""
    parser = argparse.ArgumentParser(description="Process video with YOLOv8 segmentation model")
    parser.add_argument('--model', type=str, required=True, help='Path to YOLOv8 segmentation model (.pt file)')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, default='output_video.mp4', help='Path to output video file')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold for detections')

    args = parser.parse_args()

    process_video(args.model, args.video, args.output, args.conf)


if __name__ == "__main__":
    main()