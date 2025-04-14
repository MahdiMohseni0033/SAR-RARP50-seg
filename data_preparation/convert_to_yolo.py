import os
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import shutil


def mask_to_polygons(mask, class_id):
    """
    Convert a binary mask to YOLO polygon format for a specific class.
    Returns a list of [class_id, x1, y1, x2, y2, ...] polygons.
    """
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    img_h, img_w = mask.shape[:2]

    for contour in contours:
        # Skip tiny contours that might be noise
        if cv2.contourArea(contour) < 25:  # Minimum area threshold
            continue

        # Simplify the contour to reduce the number of points
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Need at least 3 points to form a polygon
        if len(approx) < 3:
            continue

        # Format polygon as [class_id, x1, y1, x2, y2, ...] with normalized coordinates
        polygon = [class_id]
        for point in approx:
            x, y = point[0]
            # Normalize coordinates
            x_norm = x / img_w
            y_norm = y / img_h
            polygon.extend([x_norm, y_norm])

        polygons.append(polygon)

    return polygons


def process_mask(mask_path, output_dir, image_filename, classes):
    """
    Process a mask image and convert it to YOLO segmentation format.
    """
    # Read the mask image
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Warning: Could not read mask {mask_path}")
        return

    # The output label path will have the same name as the image but with .txt extension
    label_filename = os.path.splitext(image_filename)[0] + '.txt'
    label_path = output_dir / label_filename

    # Create a list to store all polygons
    all_polygons = []

    # For each class (1-9), extract polygons
    for class_id in range(1, len(classes) + 1):
        # Create a binary mask for this class
        class_mask = (mask == class_id).astype(np.uint8) * 255

        # If no pixels for this class, skip
        if not np.any(class_mask):
            continue

        # Convert to YOLO polygons (zero-indexed classes in YOLO)
        # Mask value 1 corresponds to class index 0, mask value 2 to class index 1, etc.
        polygons = mask_to_polygons(class_mask, class_id - 1)
        all_polygons.extend(polygons)

    # Write polygons to the label file
    with open(label_path, 'w') as f:
        for polygon in all_polygons:
            # Convert all values to strings
            line = ' '.join(map(str, polygon))
            f.write(line + '\n')


def create_dataset_structure(output_base, mode="train_val", split_names=['train', 'val']):
    """Create the YOLO dataset directory structure."""
    if mode == 'train_val':
        for split in split_names:
            os.makedirs(os.path.join(output_base, 'images', split), exist_ok=True)
            os.makedirs(os.path.join(output_base, 'labels', split), exist_ok=True)
        return True
    elif mode == 'test':
        os.makedirs(os.path.join(output_base, 'images', "test"), exist_ok=True)
        os.makedirs(os.path.join(output_base, 'labels', "test"), exist_ok=True)
        return True
    else:
        raise ValueError("Invalid mode. Use 'train_val' or 'test'.")


def copy_images(image_dir, output_base, split='train'):
    """Copy images to the YOLO dataset structure."""
    dest_dir = os.path.join(output_base, 'images', split)
    image_paths = list(Path(image_dir).glob('*.png')) + list(Path(image_dir).glob('*.jpg'))

    for img_path in tqdm(image_paths, desc=f"Copying {split} images"):
        shutil.copy(str(img_path), os.path.join(dest_dir, img_path.name))

    return [p.name for p in image_paths]


def convert_dataset(image_dir, mask_dir, output_dir, classes, split='train'):
    """
    Convert a dataset of images and masks to YOLO segmentation format.
    """
    # Create output directory for labels
    output_labels_dir = Path(output_dir) / 'labels' / split
    os.makedirs(output_labels_dir, exist_ok=True)

    # Get list of images (by name for matching with masks)
    image_paths = list(Path(image_dir).glob('*.png')) + list(Path(image_dir).glob('*.jpg'))
    image_filenames = [p.name for p in image_paths]

    # Process each mask
    mask_paths = list(Path(mask_dir).glob('*.png'))

    for mask_path in tqdm(mask_paths, desc=f"Processing {split} masks"):
        # Find corresponding image filename
        mask_name = mask_path.name
        # Assume mask and image have the same filename (adjust if needed)
        image_filename = mask_name

        if image_filename in image_filenames:
            process_mask(mask_path, output_labels_dir, image_filename, classes)
        else:
            print(f"Warning: No matching image found for mask {mask_name}")


def main():
    parser = argparse.ArgumentParser(description='Convert PNG masks to YOLO segmentation format')
    parser.add_argument('--mode', type=str, help='conversion for train_val, or test', choices=['train_val', 'test'],
                        default='train_val')
    parser.add_argument('--train-images', type=str, help='Directory containing training images')
    parser.add_argument('--train-masks', type=str, help='Directory containing training masks')
    parser.add_argument('--val-images', type=str, help='Directory containing validation images')
    parser.add_argument('--val-masks', type=str, help='Directory containing validation masks')
    parser.add_argument('--test-images', type=str, help='Directory containing test images')
    parser.add_argument('--test-masks', type=str, help='Directory containing test masks')
    parser.add_argument('--output-dir', type=str, help='Output directory for YOLO dataset')
    args = parser.parse_args()

    # Define classes from your YAML file
    classes = [
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

    # Create dataset structure
    create_dataset_structure(args.output_dir, mode=args.mode)

    if args.mode == 'test':
        # Process test set
        print("Processing test set...")
        copy_images(args.test_images, args.output_dir, 'test')
        convert_dataset(args.test_images, args.test_masks, args.output_dir, classes, 'test')
        print(f"Conversion complete. YOLO dataset created at {args.output_dir}")
        print("Directory structure:")
        print(f"{args.output_dir}/")
        print("├── images/")
        print("│   └── test/")
        print("└── labels/")
        print("    └── test/")



    elif args.mode == 'train_val':
        # Process training set
        print("Processing training set...")
        copy_images(args.train_images, args.output_dir, 'train')
        convert_dataset(args.train_images, args.train_masks, args.output_dir, classes, 'train')

        # Process validation set
        print("Processing validation set...")
        copy_images(args.val_images, args.output_dir, 'val')
        convert_dataset(args.val_images, args.val_masks, args.output_dir, classes, 'val')

        print(f"Conversion complete. YOLO dataset created at {args.output_dir}")
        print("Directory structure:")
        print(f"{args.output_dir}/")
        print("├── images/")
        print("│   ├── train/")
        print("│   └── val/")
        print("└── labels/")
        print("    ├── train/")
        print("    └── val/")
    else:
        raise ValueError("Invalid mode. Use 'train_val' or 'test'.")


if __name__ == "__main__":
    main()