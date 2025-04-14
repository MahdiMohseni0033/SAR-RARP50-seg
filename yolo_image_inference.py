import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import argparse
from pathlib import Path

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


# Generate distinct colors for each class using HSV color space
def generate_colors(num_classes):
    colors = []
    for i in range(num_classes):
        # Use HSV color space for better distinctness
        hue = int(i * 255 / num_classes)
        # Full saturation and value for vibrant colors
        hsv_color = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        # Convert to BGR for OpenCV
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        # Convert to RGB format
        colors.append((int(bgr_color[2]), int(bgr_color[1]), int(bgr_color[0])))
    return colors


def draw_legend(image, class_names, colors):
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

    # Get the width of the longest class name
    max_width = 0
    for name in class_names:
        (text_width, _), _ = cv2.getTextSize(name, font, text_scale, text_thickness)
        max_width = max(max_width, text_width)

    # Calculate background dimensions
    legend_height = len(class_names) * (box_size + line_spacing) + line_spacing
    legend_width = start_x + box_size + text_offset + max_width + 20

    # Create semi-transparent background
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (legend_width, legend_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image, 0)

    # Draw class boxes and names
    for i, name in enumerate(class_names):
        y = start_y + i * (box_size + line_spacing)

        # Draw color box
        cv2.rectangle(image, (start_x, y - box_size), (start_x + box_size, y), colors[i], -1)
        cv2.rectangle(image, (start_x, y - box_size), (start_x + box_size, y), (0, 0, 0), 1)

        # Draw class name
        cv2.putText(image, name, (start_x + box_size + text_offset, y - 2),
                    font, text_scale, (255, 255, 255), text_thickness)

    return image


def process_image(model_path, image_path, output_path=None, conf_threshold=0.3):
    """Process image with YOLOv8 segmentation model and overlay masks."""
    # Load model
    model = YOLO(model_path)

    # Generate colors for classes
    colors = generate_colors(len(CLASS_NAMES))

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    height, width = image.shape[:2]

    # Run inference
    results = model.predict(image, conf=conf_threshold, verbose=False)[0]

    # Create a copy for drawing
    overlay_image = image.copy()

    # Process masks
    if results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        boxes = results.boxes.data.cpu().numpy()

        for i, mask in enumerate(masks):
            # Get class ID
            class_id = int(boxes[i][5])

            # Ensure class_id is within our range
            if class_id < len(colors):
                # Create binary mask
                mask = mask.reshape(height, width)
                binary_mask = (mask > 0.5).astype(np.uint8)

                # Create colored mask
                colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
                colored_mask[binary_mask == 1] = colors[class_id]

                # Blend with original image
                alpha = 0.5  # Transparency factor
                mask_area = (binary_mask == 1)
                overlay_image[mask_area] = cv2.addWeighted(
                    overlay_image[mask_area],
                    1 - alpha,
                    colored_mask[mask_area],
                    alpha,
                    0
                )

                # Add contour for better visibility
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay_image, contours, -1, colors[class_id], 2)

    # Add legend
    overlay_image = draw_legend(overlay_image, CLASS_NAMES, colors)

    # Determine output path if not provided
    if output_path is None:
        base_path = os.path.splitext(image_path)[0]
        output_path = f"{base_path}_segmented.jpg"

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Save the output image
    cv2.imwrite(output_path, overlay_image)
    print(f"Segmentation saved to: {output_path}")

    return output_path


def process_directory(model_path, input_dir, output_dir=None, conf_threshold=0.3):
    """Process all images in a directory."""
    if output_dir is None:
        output_dir = os.path.join(input_dir, "segmented")

    os.makedirs(output_dir, exist_ok=True)

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    processed_count = 0

    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                process_image(model_path, input_path, output_path, conf_threshold)
                processed_count += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"Processed {processed_count} images. Results saved to {output_dir}")


def main():
    """Parse arguments and process images."""
    parser = argparse.ArgumentParser(description="Process images with YOLOv8 segmentation model")
    parser.add_argument('--model', type=str, required=True, help='Path to YOLOv8 segmentation model (.pt file)')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='Path to input image file')
    group.add_argument('--dir', type=str, help='Path to directory containing images')

    parser.add_argument('--output', type=str, help='Path to output image or directory')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold for detections')

    args = parser.parse_args()

    if args.image:
        process_image(args.model, args.image, args.output, args.conf)
    else:
        process_directory(args.model, args.dir, args.output, args.conf)


if __name__ == "__main__":
    main()