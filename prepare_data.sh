#!/bin/bash

# Exit on error
set -e

# Help message
function show_usage {
    echo "Usage: bash prepare_data.sh [train|test]"
    echo ""
    echo "Options:"
    echo "  train    Download and prepare training dataset"
    echo "  test     Download and prepare testing dataset"
    exit 1
}

# Check for correct number of arguments
if [ $# -ne 1 ]; then
    show_usage
fi

# Set mode based on argument
MODE=$1

if [ "$MODE" = "train" ]; then
    echo "=== Preparing training dataset ==="

    # Download train set
    echo "Downloading training dataset..."
    wget -O train_set.zip "https://rdr.ucl.ac.uk/ndownloader/articles/24932529/versions/1"

    # Extract files
    echo "Extracting files..."
    mkdir -p train_set
    unzip train_set.zip -d train_set
    rm -rf train_set.zip
    bash data_preparation/unzip_all.sh train_set

    # Process dataset
    echo "Processing dataset..."
    python data_preparation/dataset_manager2.py \
        --input_dir train_set \
        --masks_dir Dataset/all_masks \
        --frames_dir Dataset/all_frames \
        --format png

    # Clean up
    rm -rf train_set

    # Split data
    echo "Splitting dataset into train and validation sets..."
    python data_preparation/split_data.py \
        --image_dir Dataset/all_masks \
        --mask_dir Dataset/all_frames \
        --output_dir splited_dataset \
        --split_ratio 0.85

    rm -rf Dataset

    # Convert to YOLO format
    echo "Converting to YOLO format..."
    python data_preparation/convert_to_yolo.py \
        --mode train_val \
        --train-images splited_dataset/images/train \
        --train-masks splited_dataset/labels/train \
        --val-images splited_dataset/images/val \
        --val-masks splited_dataset/labels/val \
        --output-dir yolo_dataset2

    echo "✅ Training dataset preparation complete!"
    echo "Dataset available in: yolo_dataset2"

elif [ "$MODE" = "test" ]; then
    echo "=== Preparing testing dataset ==="

    # Download test set
    echo "Downloading testing dataset..."
    wget -O test_set.zip "https://rdr.ucl.ac.uk/ndownloader/articles/24932499/versions/1"

    # Extract files
    echo "Extracting files..."
    mkdir -p test_set
    unzip test_set.zip -d test_set
    rm -rf test_set.zip
    bash data_preparation/unzip_all.sh test_set

    # Process dataset
    echo "Processing dataset..."
    python data_preparation/dataset_manager2.py \
        --input_dir test_set \
        --masks_dir Dataset/all_masks \
        --frames_dir Dataset/all_frames \
        --format png

    # Convert to YOLO format
    echo "Converting to YOLO format..."
    python data_preparation/convert_to_yolo.py \
        --mode test \
        --test-images Dataset/all_frames \
        --test-masks Dataset/all_masks \
        --output-dir yolo_test_set

    # Clean up
    rm -rf test_set

    echo "✅ Testing dataset preparation complete!"
    echo "Dataset available in: yolo_test_set"

else
    echo "❌ Error: Invalid mode. Use 'train' or 'test'."
    show_usage
fi