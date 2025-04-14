import cv2
import numpy as np
import os
import glob
import sys
import shutil # For potentially copying files if needed, though imwrite is fine
import argparse # Import argparse

def process_dataset_folders(dataset_base_path, output_mask_folder, output_frame_folder, output_format='png'):
    """
    Processes a dataset with multiple video subfolders, extracts frames
    corresponding to masks, renames/saves masks and frames with unique names
    to consolidated output folders, and counts total instances.

    Args:
        dataset_base_path (str): Path to the main dataset folder containing video_XX subfolders.
        output_mask_folder (str): Path to save all renamed .png mask images.
        output_frame_folder (str): Path to save all extracted frame images.
        output_format (str): Format for saved frames ('png' or 'jpg'). Defaults to 'png'.
    """
    print("--- Starting Dataset Processing ---")

    # --- Validate Base Path ---
    if not os.path.isdir(dataset_base_path):
        print(f"Error: Dataset base path not found: {dataset_base_path}")
        sys.exit(1)

    # --- Validate Output Format ---
    if output_format.lower() not in ['png', 'jpg', 'jpeg']:
        print(f"Warning: Invalid output_format '{output_format}'. Defaulting to 'png'.")
        output_format = 'png'
    else:
        # Normalize to handle 'jpeg'
        output_format = 'jpg' if output_format.lower() == 'jpeg' else output_format.lower()


    # --- Create Output Folders ---
    try:
        # Use exist_ok=True to avoid error if directories already exist
        os.makedirs(output_mask_folder, exist_ok=True)
        print(f"Output mask directory confirmed: {output_mask_folder}")

        os.makedirs(output_frame_folder, exist_ok=True)
        print(f"Output frame directory confirmed: {output_frame_folder}")
    except OSError as e:
        print(f"Error: Could not create or access output directories: {e}")
        sys.exit(1)

    # --- Find Video Subfolders ---
    video_subfolders = sorted(glob.glob(os.path.join(dataset_base_path, 'video_*')))
    video_subfolders = [f for f in video_subfolders if os.path.isdir(f)] # Ensure they are directories

    if not video_subfolders:
        print(f"Error: No 'video_*' subfolders found in '{dataset_base_path}'.")
        sys.exit(1)

    print(f"Found {len(video_subfolders)} potential video subfolders to process.")
    print("-" * 40)

    # --- Initialize Counters ---
    total_pairs_processed = 0
    total_instance_count = 0
    total_errors = 0

    # --- Process Each Video Subfolder ---
    for video_folder_path in video_subfolders:
        subfolder_name = os.path.basename(video_folder_path) # e.g., "video_01"
        print(f"Processing subfolder: {subfolder_name}...")

        # Construct paths within the subfolder
        mask_subfolder_path = os.path.join(video_folder_path, 'segmentation')
        video_file_path = os.path.join(video_folder_path, 'video_left.avi')

        # --- Validate paths within subfolder ---
        if not os.path.isdir(mask_subfolder_path):
            print(f"  Warning: 'segmentation' folder not found in {subfolder_name}. Skipping this subfolder.")
            total_errors += 1
            continue
        if not os.path.isfile(video_file_path):
            print(f"  Warning: 'video_left.avi' not found in {subfolder_name}. Skipping this subfolder.")
            total_errors += 1
            continue

        # --- Open Video for this subfolder ---
        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            print(f"  Error: Could not open video file: {video_file_path}. Skipping this subfolder.")
            total_errors += 1
            continue

        # --- Find mask files within this subfolder ---
        mask_files = sorted(glob.glob(os.path.join(mask_subfolder_path, '*.png')))
        if not mask_files:
            print(f"  Info: No '.png' mask files found in 'segmentation' folder for {subfolder_name}.")
            cap.release()
            continue

        print(f"  Found {len(mask_files)} masks in {subfolder_name}/segmentation.")

        # --- Process each mask in the current subfolder ---
        for mask_path in mask_files:
            original_mask_filename = os.path.basename(mask_path)
            # Robustly get frame number string, handling potential variations
            frame_num_str = os.path.splitext(original_mask_filename)[0]

            # --- Create Unique Filename Prefix ---
            unique_base_name = f"{subfolder_name}_{frame_num_str}" # e.g., "video_01_00000120"

            # --- Construct Full Output Paths ---
            output_mask_path = os.path.join(output_mask_folder, f"{unique_base_name}.png")
            output_frame_path = os.path.join(output_frame_folder, f"{unique_base_name}.{output_format}")

            # --- Read Mask & Count Instances ---
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"  Error: Could not read mask file: {mask_path}. Skipping.")
                total_errors += 1
                continue

            try:
                object_values = np.unique(mask)
                # Count non-zero unique values (potential instances)
                num_instances = len(object_values[object_values != 0])
                total_instance_count += num_instances
            except Exception as e:
                 print(f"  Warning: Could not count instances in mask {mask_path}: {e}")
                 # Continue processing even if count fails

            # --- Save Renamed Mask ---
            try:
                # Using high compression for PNG masks as they are often simple
                cv2.imwrite(output_mask_path, mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            except Exception as e:
                print(f"  Error: Failed to save renamed mask to {output_mask_path}: {e}")
                total_errors += 1
                continue # Skip frame extraction if mask saving failed

            # --- Extract Frame Number ---
            try:
                # Extract frame number assuming it's the numeric part of the filename
                # This might need adjustment if filenames are different
                frame_number = int(frame_num_str)
            except ValueError:
                print(f"  Error: Could not parse frame number from mask filename: {original_mask_filename} ('{frame_num_str}'). Skipping frame extraction.")
                total_errors += 1
                # Clean up the already saved mask? Or leave it? Let's leave it for now.
                # os.remove(output_mask_path) # Optional cleanup
                continue

            # --- Read Corresponding Video Frame ---
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret or frame is None:
                # Try reading the next frame as sometimes frame numbers might be off by 1
                print(f"  Warning: Failed to read frame {frame_number} directly. Trying frame {frame_number + 1}.")
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number + 1)
                ret, frame = cap.read()
                if not ret or frame is None:
                    print(f"  Error: Failed to read frame {frame_number} (or {frame_number + 1}) from {video_file_path}. Skipping frame.")
                    total_errors += 1
                    # Clean up saved mask?
                    # os.remove(output_mask_path) # Optionally remove mask if frame fails
                    continue

            # --- Save Extracted Frame ---
            try:
                save_params = []
                if output_format == 'jpg':
                    save_params = [cv2.IMWRITE_JPEG_QUALITY, 95] # Good quality JPG
                elif output_format == 'png':
                     save_params = [cv2.IMWRITE_PNG_COMPRESSION, 3] # Moderate PNG compression

                success = cv2.imwrite(output_frame_path, frame, save_params)
                if not success:
                    # This usually indicates a deeper issue (permissions, disk space, invalid path format)
                    print(f"  Error: Failed to save frame {frame_number} to {output_frame_path} (imwrite returned false). Check path and permissions.")
                    total_errors += 1
                    # Clean up saved mask?
                    # os.remove(output_mask_path)
                else:
                    total_pairs_processed += 1 # Count success only if both mask and frame likely saved

            except Exception as e:
                print(f"  Error: Exception occurred while saving frame {frame_number} ({output_frame_path}): {e}")
                total_errors += 1
                # Clean up saved mask?
                # os.remove(output_mask_path)

        # --- Release video capture for the current subfolder ---
        cap.release()
        print(f"  Finished processing {subfolder_name}.")
        print("-" * 40)

    # --- Final Summary ---
    print("--- Dataset Processing Complete ---")
    print(f"Total video subfolders scanned: {len(video_subfolders)}")
    print(f"Total mask/frame pairs successfully processed: {total_pairs_processed}")
    print(f"Total object instances counted across all processed masks: {total_instance_count}")
    print(f"Total errors/skips encountered: {total_errors}")
    print(f"Consolidated masks saved to: {output_mask_folder}")
    print(f"Consolidated frames saved to: {output_frame_folder}")
    print("-----------------------------------")


# --- Main Execution Block ---
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process video dataset: extract frames corresponding to masks, rename both, and save to consolidated folders.")

    # Define arguments
    parser.add_argument(
        "-i", "--input_dir",
        required=True,
        metavar="PATH",
        help="Path to the main dataset folder containing video_XX subfolders (e.g., 'test_set')."
    )
    parser.add_argument(
        "-m", "--masks_dir",
        required=True,
        metavar="PATH",
        help="Path to the NEW output folder where ALL renamed masks (.png) will be saved (e.g., 'Dataset/all_masks')."
    )
    parser.add_argument(
        "-f", "--frames_dir",
        required=True,
        metavar="PATH",
        help="Path to the NEW output folder where ALL extracted frames will be saved (e.g., 'Dataset/all_frames')."
    )
    parser.add_argument(
        "--format",
        default='png',
        choices=['png', 'jpg', 'jpeg'],
        help="Output format for the extracted frames ('png' or 'jpg'/'jpeg'). Default is 'png'."
    )

    # Parse arguments from command line
    args = parser.parse_args()

    # Run the processing function with parsed arguments
    process_dataset_folders(
        dataset_base_path=args.input_dir,
        output_mask_folder=args.masks_dir,
        output_frame_folder=args.frames_dir,
        output_format=args.format
    )

    print("Script finished.")