import os
import random
import shutil
import argparse
from pathlib import Path

def split_dataset(image_dir, mask_dir, output_dir, split_ratio=0.8):
    """
    Randomly splits images and corresponding masks into train and validation sets
    and copies them into a YOLO-compatible directory structure.

    Args:
        image_dir (str): Path to the directory containing original images.
        mask_dir (str): Path to the directory containing original masks.
        output_dir (str): Path to the base directory where the split dataset
                          (train/val folders) will be created.
        split_ratio (float): Fraction of data to use for training (e.g., 0.8 for 80%).
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)

    if not image_dir.is_dir():
        print(f"Error: Image directory not found at {image_dir}")
        return
    if not mask_dir.is_dir():
        print(f"Error: Mask directory not found at {mask_dir}")
        return

    # --- Create output directories ---
    train_img_path = output_dir / 'images' / 'train'
    val_img_path = output_dir / 'images' / 'val'
    train_lbl_path = output_dir / 'labels' / 'train' # YOLO expects 'labels' for masks
    val_lbl_path = output_dir / 'labels' / 'val'

    train_img_path.mkdir(parents=True, exist_ok=True)
    val_img_path.mkdir(parents=True, exist_ok=True)
    train_lbl_path.mkdir(parents=True, exist_ok=True)
    val_lbl_path.mkdir(parents=True, exist_ok=True)

    # --- Get and shuffle image files ---
    image_files = sorted([f for f in image_dir.glob('*.png') if f.is_file()]) # Assuming png format
    if not image_files:
        print(f"Error: No PNG images found in {image_dir}")
        return

    random.seed(42) # for reproducibility
    random.shuffle(image_files)

    # --- Calculate split point ---
    split_index = int(len(image_files) * split_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    print(f"Total images: {len(image_files)}")
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")

    # --- Function to copy files ---
    def copy_files(file_list, img_dest_path, lbl_dest_path):
        copied_count = 0
        for img_path in file_list:
            mask_path = mask_dir / img_path.name # Assumes mask has the same name
            if mask_path.exists():
                try:
                    shutil.copy(img_path, img_dest_path / img_path.name)
                    shutil.copy(mask_path, lbl_dest_path / mask_path.name)
                    copied_count += 1
                except Exception as e:
                    print(f"Warning: Could not copy {img_path.name} or its mask. Error: {e}")
            else:
                print(f"Warning: Mask file not found for {img_path.name} at {mask_path}. Skipping.")
        return copied_count

    # --- Copy files to respective directories ---
    print("\nCopying training files...")
    train_copied = copy_files(train_files, train_img_path, train_lbl_path)
    print(f"Copied {train_copied} training image/mask pairs.")

    print("\nCopying validation files...")
    val_copied = copy_files(val_files, val_img_path, val_lbl_path)
    print(f"Copied {val_copied} validation image/mask pairs.")

    print(f"\nDataset splitting complete. Output structure created at: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Segmentation Dataset for YOLO")
    parser.add_argument("--image_dir", default="/home/ubuntu/projects/SAM/Final_Dataset/all_frames", help="Path to the ground truth image directory")
    parser.add_argument("--mask_dir", default="/home/ubuntu/projects/SAM/Final_Dataset/all_masks", help="Path to the mask directory")
    parser.add_argument("--output_dir", default="Yolo_dataset", help="Path to the output directory for split data")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Training split ratio (default: 0.8)")

    args = parser.parse_args()

    split_dataset(args.image_dir, args.mask_dir, args.output_dir, args.split_ratio)

    # Example Usage from terminal:
    # python split_data.py --image_dir /path/to/your_data/ground_truth_images \
    #                     --mask_dir /path/to/your_data/masks \
    #                     --output_dir /path/to/output_dataset \
    #                     --split_ratio 0.85